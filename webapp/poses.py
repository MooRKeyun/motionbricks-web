# SPDX-License-Identifier: Apache-2.0
"""Preset postures sampled from the bundled clips, and helpers that place them in the scene.

Every preset is sampled once, converted to the runtime's Z-up world space, and
canonicalised: the root yaw is zeroed and the root is moved to the origin, so a pose can
be placed anywhere with one ground position and one heading.

The heading convention matches the model's: a pose faces the direction the body faces,
measured from the leg joints the SOMA contract names as heading joints. ``forward_yaw``
stores that intrinsic yaw for each pose, so the UI's heading is absolute.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import warp as wp

from motionbricks.io.bvh import load_bvh
from motionbricks.model.soma.contract import SOMA_N1_HEADING_JOINTS

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTIONS_ROOT = REPO_ROOT / "motionbricks" / "assets" / "motions" / "soma"


@dataclass(frozen=True)
class PoseSpec:
    """One entry of the pose library."""

    key: str
    label: str
    clip: str
    time: float
    kind: str
    hint: str = ""


POSE_SPECS: tuple[PoseSpec, ...] = (
    PoseSpec("idle", "站立待机", "locomotion/idle/idle_loop_001__A268.bvh", 1.0, "posture"),
    PoseSpec("relaxed", "放松站立", "locomotion/idle/legs_relax_002__A407.bvh", 1.0, "posture"),
    PoseSpec("crouch", "蹲伏", "locomotion/idle/crouch_idle_001__A529.bvh", 1.0, "posture"),
    PoseSpec("sit", "坐在椅子上", "sitting/sit_on_chair_loop.bvh", 1.0, "posture"),
    PoseSpec("look", "张望", "locomotion/idle/neutral_looking_around_R_001__A542.bvh", 2.0, "posture"),
    PoseSpec("walk", "行走", "locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh", 1.2, "travel"),
    PoseSpec("jog", "慢跑", "locomotion/bs_1d_linear/neutral_jog_ff_180_R_001__A534.bvh", 0.8, "travel"),
    PoseSpec("run", "奔跑", "locomotion/bs_1d_linear/run_loop_180_R_003__A326.bvh", 0.6, "travel"),
    PoseSpec("skip", "蹦跳前行", "variations/skipping_R_002__A360.bvh", 1.0, "travel"),
    PoseSpec("zombie", "僵尸走", "variations/zombie_walk_180_R_003__A330.bvh", 1.0, "travel"),
    PoseSpec("dance", "舞蹈", "variations/dance_vouge_vogue_sequence_180_R_002__A316.bvh", 2.0, "action"),
    PoseSpec("jump", "跃起", "jump/jump_over_obstacle_0_5m_R_001__A415.bvh", 0.9, "action"),
)

TRAVEL_POSE_KEYS = frozenset(
    spec.key for spec in POSE_SPECS if spec.kind == "travel"
)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two XYZW quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def quaternion_from_yaw(yaw: float) -> np.ndarray:
    """Quaternion of a rotation about +Z."""
    return np.array([0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)], dtype=np.float64)


def yaw_from_quaternion(q: np.ndarray) -> float:
    """Yaw of a quaternion about +Z, in radians."""
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def multiply_transform(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two (position, quaternion) transforms."""
    return np.concatenate([a[:3] + quaternion_rotate(a[3:], b[:3]), quaternion_multiply(a[3:], b[3:])])


def quaternion_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion."""
    x, y, z, w = q
    u = np.array([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


class PoseLibrary:
    """Sample, canonicalise, and place the preset postures."""

    def __init__(self, skeleton, model_to_runtime, stream=None) -> None:
        self.skeleton = skeleton
        self.model_to_runtime = model_to_runtime
        self.stream = stream
        self.device = skeleton.device
        self._clips: dict[str, object] = {}
        self._canonical: dict[str, np.ndarray] = {}
        self._forward_yaw: dict[str, float] = {}

    # -- sampling ---------------------------------------------------------

    def _clip(self, relative: str):
        clip = self._clips.get(relative)
        if clip is None:
            path = MOTIONS_ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(f"Motion clip not found: {path}")
            clip = load_bvh(str(path), self.skeleton)
            self._clips[relative] = clip
        return clip

    def canonical(self, key: str) -> np.ndarray:
        """Return the canonical local transforms of a preset pose, shaped [joints, 7]."""
        cached = self._canonical.get(key)
        if cached is not None:
            return cached
        spec = self._spec(key)
        clip = self._clip(spec.clip)
        pose = wp.zeros(self.skeleton.num_joints, dtype=wp.transform, device=self.device)
        clip.sample(spec.time, pose, root_offset=self.model_to_runtime, stream=self.stream)
        wp.synchronize()
        array = np.asarray(pose.numpy(), dtype=np.float64).reshape(self.skeleton.num_joints, 7)

        yaw = yaw_from_quaternion(array[0, 3:7])
        array[0, 3:7] = quaternion_multiply(quaternion_from_yaw(-yaw), array[0, 3:7])
        array[0, 0:2] = 0.0

        self._canonical[key] = array
        self._forward_yaw[key] = self._measure_forward_yaw(array)
        return array

    def _measure_forward_yaw(self, pose: np.ndarray) -> float:
        """Yaw of the direction the body faces, from the heading joints."""
        world = self.world_positions(pose)
        right = world[self.skeleton.joint_index(SOMA_N1_HEADING_JOINTS[0])]
        left = world[self.skeleton.joint_index(SOMA_N1_HEADING_JOINTS[1])]
        left_dir = np.array([left[0] - right[0], left[1] - right[1], 0.0])
        norm = np.linalg.norm(left_dir)
        if norm < 1e-6:
            return 0.0
        left_dir /= norm
        forward = np.cross(left_dir, np.array([0.0, 0.0, 1.0]))
        return math.atan2(float(forward[1]), float(forward[0]))

    def world_positions(self, pose: np.ndarray) -> np.ndarray:
        """Forward kinematics: world position of every joint, shaped [joints, 3]."""
        joints = self.skeleton.num_joints
        parents = [self.skeleton.joint_parent(i) for i in range(joints)]
        world = np.zeros((joints, 3), dtype=np.float64)
        rotations = np.zeros((joints, 4), dtype=np.float64)

        done = [False] * joints

        def resolve(index: int) -> None:
            if done[index]:
                return
            parent = parents[index]
            if parent < 0:
                world[index] = pose[index, :3]
                rotations[index] = pose[index, 3:7]
            else:
                resolve(parent)
                offset = quaternion_rotate(rotations[parent], pose[index, :3])
                world[index] = world[parent] + offset
                rotations[index] = quaternion_multiply(rotations[parent], pose[index, 3:7])
            done[index] = True

        for i in range(joints):
            resolve(i)
        return world

    # -- placement --------------------------------------------------------

    def forward_yaw(self, key: str) -> float:
        """Yaw of a preset pose's intrinsic facing."""
        self.canonical(key)
        return self._forward_yaw[key]

    def place(self, key: str, position, heading: float) -> np.ndarray:
        """Return the pose placed on the ground at position, facing heading (radians, from +X)."""
        pose = self.canonical(key).copy()
        yaw = heading - self._forward_yaw[key]
        root = pose[0].copy()
        placement = np.concatenate([[float(position[0]), float(position[1]), 0.0], quaternion_from_yaw(yaw)])
        pose[0] = multiply_transform(placement, root)
        return pose

    def kind(self, key: str) -> str:
        return self._spec(key).kind

    def keys(self) -> list[str]:
        return [spec.key for spec in POSE_SPECS]

    def _spec(self, key: str) -> PoseSpec:
        for spec in POSE_SPECS:
            if spec.key == key:
                return spec
        raise ValueError(f"未知姿态: {key}")


__all__ = [
    "MOTIONS_ROOT",
    "POSE_SPECS",
    "PoseLibrary",
    "PoseSpec",
    "TRAVEL_POSE_KEYS",
    "quaternion_from_yaw",
    "quaternion_multiply",
    "quaternion_rotate",
    "yaw_from_quaternion",
]
