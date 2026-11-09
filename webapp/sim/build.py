"""Build everything the /day page needs, in order.

    python -m webapp.sim.build            # facility, day, events, staff, scenario, clips
    python -m webapp.sim.build --no-clips # skip the MotionBricks generation (fast)

Each step reads the previous step's output; nothing in ``demo/`` is modified.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


STEPS = [
    ("facility", "webapp.sim.facility"),
    ("day", "webapp.sim.day"),
    ("events", "webapp.sim.events"),
    ("staff", "webapp.sim.staff"),
    ("scenario", "webapp.sim.scenario"),
    ("clips", "webapp.sim.clips"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-clips", action="store_true", help="skip MotionBricks generation")
    parser.add_argument("--rebuild", action="store_true", help="regenerate cached outputs")
    args = parser.parse_args()

    for name, module in STEPS:
        if name == "clips" and args.no_clips:
            continue
        command = [sys.executable, "-m", module]
        if name == "clips" and args.rebuild:
            command.append("--rebuild")
        started = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        result = subprocess.run(command)
        if result.returncode != 0:
            raise SystemExit(f"{name} failed with exit code {result.returncode}")
        print(f"--- {name} done in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
