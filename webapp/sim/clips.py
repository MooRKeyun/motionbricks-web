"""Turn events into MotionBricks clips and pack them for the browser.

Every event becomes one generation: a start state (position, heading, pose key) and an
end state, exactly what the manual in-betweening page uses. Clips are stored as one
float32 blob (frames x joints x 7) plus a JSON index; the blob is what the day page
streams in.

Resumable: clips already present in ``clips.json`` are kept, so a failed run can be
continued with ``python -m webapp.sim.clips``.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
import time

import numpy as np

from ..motion import MotionService

DATA = Path(__file__).resolve().parent / "data"

# Poses are picked by kind: seated transitions take their time, meets are brisker.
KIND_PACE = {
    "sit": "slow",
    "stand": "slow",
    "meet_start": "normal",
    "meet_end": "normal",
}


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def request_for(event: dict) -> dict:
    """Build the /api/generate payload for one event."""
    start = event["start"]
    end = event["end"]
    distance = math.hypot(end["x"] - start["x"], end["y"] - start["y"])
    return {
        "startPose": start["pose"],
        "endPose": end["pose"],
        "start": [round(start["x"], 3), round(start["y"], 3)],
        "end": [round(end["x"], 3), round(end["y"], 3)],
        "startHeading": round(math.degrees(start["heading"]), 2),
        "endHeading": round(math.degrees(end["heading"]), 2),
        "pace": KIND_PACE.get(event["kind"], "normal"),
        # One segment per short transition: never split a 1-2 m move.
        "step": round(max(0.6, distance), 3),
    }


def decode_frames(payload: dict) -> np.ndarray:
    raw = base64.b64decode(payload["frames"])
    return np.frombuffer(raw, dtype=np.float32).reshape(
        payload["numFrames"], payload["numJoints"], 7,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", type=Path, default=DATA / "events.json")
    parser.add_argument("--limit", type=int, default=0, help="only generate the first N events")
    parser.add_argument("--rebuild", action="store_true", help="ignore clips.json and start over")
    args = parser.parse_args()

    events = json.loads(args.events.read_text(encoding="utf-8"))["events"]
    if args.limit:
        events = events[: args.limit]

    clips_path = DATA / "clips.json"
    blob_path = DATA / "clips.bin"
    done: dict[str, dict] = {}
    frames_by_id: dict[str, np.ndarray] = {}
    if clips_path.is_file() and not args.rebuild:
        existing = json.loads(clips_path.read_text(encoding="utf-8"))
        done = {clip["id"]: clip for clip in existing.get("clips", [])}
        if blob_path.is_file() and done:
            raw = np.fromfile(blob_path, dtype=np.float32)
            for clip in existing["clips"]:
                count = clip["frames"] * clip["joints"] * 7
                frames_by_id[clip["id"]] = raw[clip["offset"] : clip["offset"] + count].reshape(
                    clip["frames"], clip["joints"], 7,
                )

    service = MotionService()
    print("loading engine ...")
    service.load()
    if service.state != "ready":
        raise SystemExit(f"engine failed to load: {service.error}")

    started = time.perf_counter()
    failures: list[str] = []
    made = 0
    for index, event in enumerate(events):
        if event["id"] in done:
            continue
        try:
            payload = service.generate(request_for(event))
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures.append(f"{event['id']}: {exc}")
            continue
        frames = decode_frames(payload)
        done[event["id"]] = {
            "id": event["id"],
            "agent": event["agent"],
            "kind": event["kind"],
            "t0": event["t0"],
            "t1": event["t1"],
            "poi": event.get("poi"),
            "partner": event.get("partner"),
            "scenario": event.get("scenario"),
            "start": event["start"],
            "end": event["end"],
            "fps": payload["fps"],
            "frames": int(frames.shape[0]),
            "joints": int(frames.shape[1]),
            "elapsedMs": payload["elapsedMs"],
            "segments": len(payload["segments"]),
            "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        frames_by_id[event["id"]] = frames.astype(np.float32)
        made += 1
        if made % 10 == 0 or index == len(events) - 1:
            print(f"  {index + 1}/{len(events)} clips ({(time.perf_counter() - started):.1f}s)")

    # Rebuild the blob in a stable order (sorted clip id) and record each clip's offset.
    ordered = []
    offset = 0
    for clip_id in sorted(done):
        clip = done[clip_id]
        if clip_id not in frames_by_id:
            continue
        clip["offset"] = offset
        offset += clip["frames"] * clip["joints"] * 7
        ordered.append(clip)
    blob = (
        np.concatenate([frames_by_id[clip["id"]] for clip in ordered], axis=0)
        if ordered
        else np.zeros((0, 77, 7), dtype=np.float32)
    )

    blob.astype(np.float32).tofile(blob_path)
    clips_path.write_text(
        json.dumps(
            {
                "meta": {
                    "count": len(ordered),
                    "fps": 30,
                    "joints": 77,
                    "floats": int(blob.size),
                    "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "failures": failures,
                },
                "clips": ordered,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    print(
        f"clips={len(ordered)} new={made} failures={len(failures)} "
        f"blob={blob_path.stat().st_size / 1024 / 1024:.1f} MB in {time.perf_counter() - started:.1f}s"
    )
    if failures:
        print("first failures:", failures[:3])


if __name__ == "__main__":
    main()
