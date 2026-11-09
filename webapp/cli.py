# SPDX-License-Identifier: Apache-2.0
"""Generate one motion from a start state to an end state, without the browser.

Example:

    python -m webapp.cli --start idle --end sit --start-xy 0 0 --end-xy 3.2 1.2 --out out.npz
"""

from __future__ import annotations

import argparse
import base64
import math
from pathlib import Path

import numpy as np

from .motion import MotionService


def parse_pair(values: list[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise SystemExit(f"{name} needs two numbers")
    return float(values[0]), float(values[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="idle", help="start pose key")
    parser.add_argument("--end", default="walk", help="end pose key")
    parser.add_argument("--start-xy", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--end-xy", nargs=2, type=float, default=[0.0, 2.0])
    parser.add_argument("--start-heading", type=float, default=0.0, help="degrees, 0 faces +X")
    parser.add_argument("--end-heading", type=float, default=0.0, help="degrees, 0 faces +X")
    parser.add_argument("--step", type=float, default=1.0, help="meters per chained segment")
    parser.add_argument("--travel", default=None, help="pose used between waypoints")
    parser.add_argument("--pace", default="normal", choices=("fast", "normal", "slow"))
    parser.add_argument("--out", type=Path, default=None, help="write the frames to this .npz file")
    args = parser.parse_args()

    service = MotionService()
    print("loading engine ...")
    service.load()
    print("status:", service.status())
    if service.state != "ready":
        raise SystemExit(f"engine failed to load: {service.error}")

    request = {
        "startPose": args.start,
        "endPose": args.end,
        "start": list(parse_pair(args.start_xy, "--start-xy")),
        "end": list(parse_pair(args.end_xy, "--end-xy")),
        "startHeading": args.start_heading,
        "endHeading": args.end_heading,
        "step": args.step,
        "pace": args.pace,
    }
    if args.travel:
        request["travelPose"] = args.travel

    result = service.generate(request)
    frames = np.frombuffer(base64.b64decode(result["frames"]), dtype=np.float32).reshape(
        result["numFrames"], result["numJoints"], 7,
    )

    root = frames[:, 0, :3]
    requested_end = np.array([*request["end"]], dtype=np.float32)
    final_error = float(np.linalg.norm(root[-1, :2] - requested_end))
    final_height = float(root[-1, 2])
    step_lengths = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1)
    speed = step_lengths * result["fps"]

    print(f"segments: {len(result['segments'])}")
    print(f"frames: {result['numFrames']}  duration: {result['duration']:.2f} s")
    print(f"per-segment frames: {[s['frames'] for s in result['segments']]}")
    print(f"inference ms: {[s['inferenceMs'] for s in result['segments']]}")
    print(f"total wall time: {result['elapsedMs']:.0f} ms")
    print(f"root start: {root[0].round(3)}  end: {root[-1].round(3)}")
    print(f"final ground error: {final_error:.3f} m  (root height {final_height:.3f} m)")
    print(f"mean ground speed: {float(speed.mean()):.2f} m/s  max: {float(speed.max()):.2f} m/s")
    print(f"height range: {float(root[:, 2].min()):.3f} .. {float(root[:, 2].max()):.3f} m")

    if args.out:
        np.savez_compressed(args.out, frames=frames, fps=result["fps"])
        print(f"wrote {args.out}")

    if final_error > 0.05:
        raise SystemExit(f"motion ends {final_error:.3f} m from the requested end state")


if __name__ == "__main__":
    main()
