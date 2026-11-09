"""Quantitative checks on a generated motion: continuity, ground contact, pose match."""

from __future__ import annotations

import argparse
import base64
import math

import numpy as np
import warp as wp

from motionbricks.io.usd import load_skeleton_from_usd
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file
from motionbricks.model.contract import ModelRig
from webapp.motion import MotionService, _model_to_runtime
from webapp.poses import PoseLibrary


def forward_kinematics(pose: np.ndarray, parents: list[int]) -> np.ndarray:
    """World transforms [J, 7] of one pose, given its local transforms."""
    joints = pose.shape[0]
    world = np.zeros((joints, 7), dtype=np.float64)

    def resolve(index: int) -> None:
        parent = parents[index]
        if parent < 0:
            world[index] = pose[index]
            return
        resolve(parent)
        base = world[parent]
        local = pose[index]
        bx, by, bz, bw = base[3:7]
        lx, ly, lz = local[:3]
        # rotate the local offset by the parent's world rotation
        tx = 2.0 * (by * lz - bz * ly)
        ty = 2.0 * (bz * lx - bx * lz)
        tz = 2.0 * (bx * ly - by * lx)
        world[index, 0] = base[0] + lx + bw * tx + (by * tz - bz * ty)
        world[index, 1] = base[1] + ly + bw * ty + (bz * tx - bx * tz)
        world[index, 2] = base[2] + lz + bw * tz + (bx * ty - by * tx)
        ax, ay, az, aw = base[3:7]
        qx, qy, qz, qw = local[3:7]
        world[index, 3] = aw * qx + ax * qw + ay * qz - az * qy
        world[index, 4] = aw * qy - ax * qz + ay * qw + az * qx
        world[index, 5] = aw * qz + ax * qy - ay * qx + az * qw
        world[index, 6] = aw * qw - ax * qx - ay * qy - az * qz

    for i in range(joints):
        resolve(i)
    return world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="idle")
    parser.add_argument("--end", default="idle")
    parser.add_argument("--start-xy", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--end-xy", nargs=2, type=float, default=[3.0, 1.0])
    parser.add_argument("--end-heading", type=float, default=45.0)
    parser.add_argument("--pace", default="normal")
    args = parser.parse_args()

    wp.init()
    skeleton = load_skeleton_from_usd(get_asset_file("human", "soma_base_skel_minimal_no_root.usd"), SOMA_SKELETON_PRIM_PATH)
    parents = [skeleton.joint_parent(i) for i in range(skeleton.num_joints)]
    names = skeleton.joint_names
    index = {name: i for i, name in enumerate(names)}

    service = MotionService()
    service.load()
    assert service.state == "ready", service.error

    response = service.generate(
        {
            "startPose": args.start,
            "endPose": args.end,
            "start": args.start_xy,
            "end": args.end_xy,
            "startHeading": 0.0,
            "endHeading": args.end_heading,
            "pace": args.pace,
            "step": 1.0,
        }
    )
    frames = np.frombuffer(base64.b64decode(response["frames"]), dtype=np.float32).reshape(
        response["numFrames"], response["numJoints"], 7
    ).astype(np.float64)
    print(f"frames={response['numFrames']} duration={response['duration']:.2f}s segments={len(response['segments'])}")

    # per-frame world transforms
    worlds = np.stack([forward_kinematics(frames[f], parents) for f in range(frames.shape[0])])
    root = worlds[:, 0, :3]
    steps = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1)
    print(f"root step  mean={steps.mean():.4f} m/frame  max={steps.max():.4f} m  (jump > 0.25 would be a pop)")

    feet = {name: worlds[:, index[name], :3] for name in ("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot")}
    min_foot_z = min(float(f[:, 2].min()) for f in feet.values())
    print(f"lowest foot point z={min_foot_z:.3f} m")

    # stance slip: horizontal speed of the toe while it is planted near its lowest point
    for name in ("LeftToeBase", "RightToeBase"):
        track = feet[name]
        floor = track[:, 2].min()
        grounded = track[:, 2] < floor + 0.03
        if grounded.sum() < 3:
            continue
        velocity = np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1) * 30.0
        stance = velocity[grounded[:-1]]
        print(f"{name:12s} planted frames={int(grounded.sum()):3d}  mean slip={stance.mean():.2f} m/s  p90={np.percentile(stance, 90):.2f}")

    # start and end poses must match the requested states
    library = PoseLibrary(skeleton, _model_to_runtime(ModelRig.SOMA))
    start_pose = library.place(args.start, args.start_xy, 0.0)
    end_pose = library.place(args.end, args.end_xy, math.radians(args.end_heading))

    def compare(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
        position = float(np.abs(actual[:, :3] - expected[:, :3]).max())
        dots = np.abs(np.sum(actual[:, 3:7] * expected[:, 3:7], axis=1))
        angle = float(np.degrees(2 * np.arccos(np.clip(dots, -1, 1))).max())
        return position, angle

    start_error = compare(frames[0], start_pose)
    end_error = compare(frames[-1], end_pose)
    print(f"first frame vs start state: max position {start_error[0]:.4f} m, max rotation {start_error[1]:.2f} deg")
    print(f"last frame  vs end state:   max position {end_error[0]:.4f} m, max rotation {end_error[1]:.2f} deg")

    # angular velocity sanity: no joint rotates more than 60 deg per frame
    quats = frames[:, :, 3:7]
    dots = np.abs(np.sum(quats[1:] * quats[:-1], axis=2))
    angles = 2 * np.degrees(np.arccos(np.clip(dots, -1, 1)))
    print(f"max joint rotation per frame: {angles.max():.1f} deg")


if __name__ == "__main__":
    main()
