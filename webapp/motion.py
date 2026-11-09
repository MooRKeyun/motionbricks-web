# SPDX-License-Identifier: Apache-2.0
"""Generation service: turn a start state and an end state into one continuous motion.

The MotionBricks model fills in 0.6 to 1.9 s of motion between four context poses and
one to four target poses. To cross a longer distance, the service splits the path into
waypoints and chains generations: each generated segment becomes the context of the next
one, exactly like the runtime's motion players do. The last waypoint carries the end
pose, so the motion settles on the state the user picked.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import threading
import time

import numpy as np
import warp as wp

from motionbricks.io.usd import load_skeleton_from_usd
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file
from motionbricks.model.config import MotionBricksInferenceConfig
from motionbricks.model.contract import MODEL_SAMPLE_RATE, ModelRig
from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH
from motionbricks.model.soma.engine import MotionBricksSOMAEngine
from motionbricks.inference.errors import TensorContractError
from motionbricks.inference.progress import TensorRTBuildProgressMonitor

from .poses import POSE_SPECS, PoseLibrary

LOGGER = logging.getLogger("webapp.motion")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "motionbricks" / ".trt-cache"
DEFAULT_STEP_METERS = 1.0

# Token counts set the length of one generation (4 frames per token, 6 fewer valid
# frames). Fewer tokens means the same distance is covered faster, so the three paces
# below act as a speed control for a fixed waypoint spacing.
PACE_TOKEN_COUNTS: dict[str, tuple[int, ...]] = {
    "fast": (6, 7, 8),
    "normal": (8, 9, 10),
    "slow": (11, 12, 13),
}


@dataclass
class SegmentInfo:
    index: int
    frames: int
    inference_ms: float


@dataclass
class MotionResult:
    frames: np.ndarray
    segments: list[SegmentInfo]
    start_position: tuple[float, float]
    end_position: tuple[float, float]
    elapsed_ms: float

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration(self) -> float:
        return max(0, self.num_frames - 1) / MODEL_SAMPLE_RATE


class MotionService:
    """Own the engine, the pose library, and one generation at a time."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR, status_callback=None) -> None:
        self.cache_dir = Path(cache_dir)
        self.device = "cuda:0"
        self._status_callback = status_callback or (lambda event: None)
        self._lock = threading.Lock()
        self._status_lock = threading.Lock()

        self.state = "starting"
        self.message = "正在加载 SOMA 骨架与姿态库……"
        self.progress = 0.0
        self.error: str | None = None
        self._warming = False

        self.skeleton = None
        self.library: PoseLibrary | None = None
        self.engine: MotionBricksSOMAEngine | None = None
        self.context = None
        self.target = None
        self.output = None
        self.model_to_runtime = wp.transform_identity()

    # -- lifecycle --------------------------------------------------------

    def _emit(self, **event) -> None:
        with self._status_lock:
            self.state = event.get("state", self.state)
            self.message = event.get("message", self.message)
            self.progress = event.get("progress", self.progress)
        self._status_callback(event)

    def status(self) -> dict:
        with self._status_lock:
            return {
                "state": self.state,
                "message": self.message,
                "progress": round(float(self.progress), 3),
                "error": self.error,
            }

    def load(self) -> None:
        """Build the engine and warm the pose library. Safe to run in a worker thread."""
        try:
            wp.init()
            wp.set_device(self.device)

            self._emit(state="loading", message="正在读取 SOMA 骨架……", progress=0.02)
            self.skeleton = load_skeleton_from_usd(
                get_asset_file("human", "soma_base_skel_minimal_no_root.usd"),
                SOMA_SKELETON_PRIM_PATH,
            )
            self.model_to_runtime = _model_to_runtime(ModelRig.SOMA)
            self.library = PoseLibrary(self.skeleton, self.model_to_runtime)

            self._emit(state="loading", message="正在采样预设姿态……", progress=0.06)
            for key in self.library.keys():
                self.library.canonical(key)

            monitor = TensorRTBuildProgressMonitor()
            stop = threading.Event()

            def poll_build() -> None:
                while not stop.is_set():
                    snapshot = monitor.snapshot()
                    fraction = snapshot.fraction
                    phase = snapshot.phase_name or "TensorRT 优化"
                    self._emit(
                        state="building",
                        message=f"正在构建 TensorRT 引擎（{phase}）……首次启动需要 1–3 分钟",
                        progress=min(0.95, 0.08 + 0.85 * fraction),
                    )
                    stop.wait(0.6)

            poller = threading.Thread(target=poll_build, name="trt-progress", daemon=True)
            poller.start()
            try:
                self._emit(state="building", message="正在构建 TensorRT 引擎……", progress=0.08)
                self.engine = MotionBricksSOMAEngine(
                    SOMA_N1_FP32_ONNX_PATH,
                    model_to_runtime=self.model_to_runtime,
                    cache_dir=str(self.cache_dir),
                    build_progress_monitor=monitor,
                )
            finally:
                stop.set()
                poller.join(timeout=2.0)

            self.context = self.engine.create_input(self.skeleton)
            self.target = self.engine.create_input(self.skeleton)
            self.output = self.engine.create_output(self.skeleton)

            self._emit(state="building", message="正在预热推理（烘焙 CUDA graph）……", progress=0.97)
            self._warm_up()

            self._emit(state="ready", message="引擎就绪，可以在场景中摆放起止状态了", progress=1.0)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.error = f"{type(exc).__name__}: {exc}"
            self._emit(state="error", message=self.error, progress=0.0)

    def _warm_up(self) -> None:
        """Bake the CUDA graph and warm the kernels before the first user request.

        Capture happens on the first generation, which also runs inference twice. Doing
        it here keeps the first click fast and avoids the one-off race that the capture
        can cause on the request that triggers it.
        """
        try:
            self._warming = True
            self._generate(
                {
                    "startPose": "idle",
                    "endPose": "idle",
                    "start": [0.0, 0.0],
                    "end": [0.6, 0.0],
                    "startHeading": 0.0,
                    "endHeading": 0.0,
                    "pace": "normal",
                    "step": 1.0,
                }
            )
        except Exception as exc:  # noqa: BLE001 - warm-up must never block startup
            LOGGER.warning("Warm-up generation failed: %s", exc)
        finally:
            self._warming = False

    # -- scene data -------------------------------------------------------
    def scene_info(self) -> dict:
        if self.library is None:
            raise RuntimeError("服务尚未就绪")
        return {
            "rig": "SOMA",
            "fps": MODEL_SAMPLE_RATE,
            "joints": list(self.skeleton.joint_names),
            "numJoints": self.skeleton.num_joints,
            "poses": [
                {
                    "key": spec.key,
                    "label": spec.label,
                    "kind": spec.kind,
                    "hint": spec.hint,
                }
                for spec in POSE_SPECS
            ],
            "modelUrl": "/static/soma.glb",
            "step": DEFAULT_STEP_METERS,
        }

    def preview_pose(self, key: str) -> dict:
        """Return a canonical pose for the ghost previews, as raw float32."""
        if self.library is None:
            raise RuntimeError("服务尚未就绪")
        pose = np.asarray(self.library.canonical(key), dtype=np.float32)
        return {
            "key": key,
            "numJoints": int(pose.shape[0]),
            "forwardYaw": float(self.library.forward_yaw(key)),
            "rootHeight": float(pose[0, 2]),
            "data": base64.b64encode(pose.tobytes()).decode("ascii"),
        }

    # -- generation -------------------------------------------------------

    def generate(self, request: dict) -> dict:
        if self.state != "ready" or self.engine is None:
            raise RuntimeError("引擎尚未就绪，请等待初始化完成")
        with self._lock:
            result = self._generate(request)
        return self._serialize(result, request)

    def _generate(self, request: dict) -> MotionResult:
        started = time.perf_counter()
        start_pose = str(request["startPose"])
        end_pose = str(request["endPose"])
        sx, sy = float(request["start"][0]), float(request["start"][1])
        ex, ey = float(request["end"][0]), float(request["end"][1])
        start_heading = math.radians(float(request.get("startHeading", 0.0)))
        end_heading = math.radians(float(request.get("endHeading", 0.0)))
        step = float(request.get("step") or DEFAULT_STEP_METERS)
        step = max(0.35, min(3.0, step))

        travel_pose = request.get("travelPose")
        if not travel_pose:
            travel_pose = start_pose if self.library.kind(start_pose) == "travel" else "walk"
        if self.library.kind(travel_pose) != "travel":
            travel_pose = "walk"

        pace = str(request.get("pace", "normal"))
        config = MotionBricksInferenceConfig(
            allowed_token_counts=PACE_TOKEN_COUNTS.get(pace, PACE_TOKEN_COUNTS["normal"])
        )

        start_local = self.library.place(start_pose, (sx, sy), start_heading)
        end_local = self.library.place(end_pose, (ex, ey), end_heading)

        dx, dy = ex - sx, ey - sy
        distance = math.hypot(dx, dy)
        segment_count = max(1, int(math.ceil(distance / step))) if distance > 1e-4 else 1
        travel_heading = math.atan2(dy, dx) if distance > 1e-4 else end_heading

        frames: list[np.ndarray] = []
        segments: list[SegmentInfo] = []
        context_poses = np.stack([start_local] * 4).astype(np.float32)

        for index in range(1, segment_count + 1):
            fraction = index / segment_count
            waypoint = (sx + dx * fraction, sy + dy * fraction)
            is_last = index == segment_count
            if is_last:
                target_pose = end_local
            else:
                target_pose = self.library.place(travel_pose, waypoint, travel_heading)

            self.context.update_from_poses(self._to_device(context_poses), stream=self.engine.stream)
            target_array = self._to_device(target_pose[None].astype(np.float32))
            self.target.update_from_poses(target_array, stream=self.engine.stream)
            self._generate_segment(config)
            frame_count = int(self.output.num_frames)
            segment = np.asarray(self.output.poses[:frame_count].numpy(), dtype=np.float32)
            if frames:
                segment = segment[1:]
            frames.append(segment)
            segments.append(
                SegmentInfo(index, int(segment.shape[0]), float(self.engine.timer.milliseconds.get("inference", 0.0)))
            )
            if not self._warming:
                self._emit(
                    state="ready",
                    message=f"已生成第 {index}/{segment_count} 段",
                    progress=1.0,
                )

            tail = np.asarray(self.output.poses[frame_count - 4 : frame_count].numpy(), dtype=np.float32)
            context_poses = tail

        all_frames = np.concatenate(frames, axis=0)
        return MotionResult(
            frames=all_frames,
            segments=segments,
            start_position=(sx, sy),
            end_position=(ex, ey),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _generate_segment(self, config: MotionBricksInferenceConfig, attempts: int = 2) -> None:
        """Run one generation, retrying with the full token range if the model strays.

        Right after the CUDA graph is captured, the model has been seen to report a
        token count outside the requested set. The retry keeps that from failing a
        user request.
        """
        for attempt in range(attempts):
            try:
                self.engine.generate_animation(self.context, self.target, self.output, config=config)
                return
            except TensorContractError as exc:
                if attempt == attempts - 1:
                    raise
                LOGGER.warning("generation reported an unexpected token count, retrying: %s", exc)
                config = MotionBricksInferenceConfig(
                    allowed_token_counts=tuple(range(6, 17)),
                    correct_generated_diff=config.correct_generated_diff,
                    heading_alignment=config.heading_alignment,
                )

    def _to_device(self, array: np.ndarray) -> wp.array:
        """Upload [poses, joints, 7] numpy transforms as a Warp transform array."""
        data = np.ascontiguousarray(array, dtype=np.float32)
        return wp.array(data, dtype=wp.transform, device=self.device)

    def _serialize(self, result: MotionResult, request: dict) -> dict:
        data = np.ascontiguousarray(result.frames, dtype=np.float32)
        root = data[:, 0, :3]
        return {
            "fps": MODEL_SAMPLE_RATE,
            "numJoints": int(data.shape[1]),
            "numFrames": result.num_frames,
            "duration": result.duration,
            "segments": [
                {"index": s.index, "frames": s.frames, "inferenceMs": round(s.inference_ms, 2)}
                for s in result.segments
            ],
            "elapsedMs": round(result.elapsed_ms, 1),
            "distance": float(
                math.hypot(result.end_position[0] - result.start_position[0],
                           result.end_position[1] - result.start_position[1])
            ),
            "requested": request,
            "frames": base64.b64encode(data.tobytes()).decode("ascii"),
            "rootPath": [[round(float(v), 4) for v in row] for row in root],
        }


def _model_to_runtime(rig) -> wp.transform:
    from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter

    if rig is ModelRig.SOMA:
        return SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(wp.transform_identity())
    return wp.transform_identity()


__all__ = ["DEFAULT_CACHE_DIR", "MotionResult", "MotionService", "SegmentInfo"]
