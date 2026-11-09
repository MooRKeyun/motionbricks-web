"""Convert one simulated day into the playback timeline the browser reads.

Inputs (read only, produced by ``demo/scripts/run_demo.py``):

* ``trajectories.csv`` – 12 agents, every 5 s, with zone / state / activity
* ``episodes.csv``     – dwell episodes per agent

Output ``day.json``: compact per-agent arrays (x, y, heading, zone index,
activity index) plus the episode list, each episode snapped to the furniture
anchor it happened at. Positions stay in the plan's metre coordinates.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
import sys

DEMO_ROOT = Path(r"C:\Users\13119\Desktop\工作\论文项目\调研\未来研究方向\社区\demo")
RUN = DEMO_ROOT / "runs" / "demo01"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "day.json"

AGENT_COLORS = {
    "resident": "#6ea8ff",
    "nurse": "#ffffff",
    "staff": "#ffd166",
}

# Which furniture an activity most likely happens at, by keyword.
ACTIVITY_SEAT_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("起床", "洗漱", "午休", "休息", "回房", "睡"), ("bed",)),
    (("早餐", "午餐", "晚餐", "茶", "餐"), ("dining_table",)),
    (("电视", "聊天", "集体", "活动"), ("sofa", "chair")),
    (("花园", "晒太阳", "散步"), ("bench",)),
    (("康复", "训练", "锻炼"), ("exercise_bike", "parallel_bars", "treatment_table")),
    (("会客", "门厅"), ("sofa", "chair", "reception")),
    (("走廊歇脚",), ("bench", "chair")),
)


def _angle_delta(a: float, b: float) -> float:
    """Shortest signed difference from b to a, in radians."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


def _compute_headings(xs: list[float], ys: list[float], dt: float) -> list[float]:
    """Heading per sample from the local velocity, smoothed, frozen while still."""
    n = len(xs)
    speed = [0.0] * n
    raw = [0.0] * n
    for i in range(n):
        i0 = max(0, i - 1)
        i1 = min(n - 1, i + 1)
        vx = (xs[i1] - xs[i0]) / max(1e-6, (i1 - i0) * dt)
        vy = (ys[i1] - ys[i0]) / max(1e-6, (i1 - i0) * dt)
        speed[i] = math.hypot(vx, vy)
        raw[i] = math.atan2(vy, vx)

    headings = [0.0] * n
    current = raw[0]
    for i in range(n):
        if speed[i] > 0.15:
            current = current + 0.45 * _angle_delta(raw[i], current)
        headings[i] = current
    return headings


def _load_trajectories(path: Path) -> tuple[dict[str, dict], list[str], list[str], list[str]]:
    per_agent: dict[str, dict] = collections.defaultdict(lambda: {"t": [], "x": [], "y": [], "zone": [], "state": [], "activity": []})
    zones: list[str] = []
    states: list[str] = []
    activities: list[str] = []

    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            entry = per_agent[row["agent_id"]]
            entry["t"].append(float(row["t_s"]))
            entry["x"].append(float(row["x"]))
            entry["y"].append(float(row["y"]))
            entry["zone"].append(row["zone_id"])
            entry["state"].append(row["state"])
            entry["activity"].append(row["activity"])
            for value, table in ((row["zone_id"], zones), (row["state"], states), (row["activity"], activities)):
                if value not in table:
                    table.append(value)
    return per_agent, zones, states, activities


def _load_episodes(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [
            {
                "agent": row["agent_id"],
                "activity": row["activity"],
                "zone": row["zone_id"],
                "zone_type": row["zone_type"],
                "t0": float(row["start_s"]),
                "t1": float(row["end_s"]),
                "start": row["start_clock"],
                "end": row["end_clock"],
                "minutes": float(row["duration_min"]),
            }
            for row in csv.DictReader(handle)
        ]


def _seat_candidates(activity: str) -> tuple[str, ...]:
    for keywords, types in ACTIVITY_SEAT_HINTS:
        if any(keyword in activity for keyword in keywords):
            return types
    return ()


def _snap_episode(episode: dict, sample_positions: list[tuple[float, float, float]], pois: list[dict]) -> str | None:
    """Pick the furniture anchor the episode happened at, if any."""
    inside = [(x, y) for t, x, y in sample_positions if episode["t0"] <= t <= episode["t1"]]
    if not inside:
        return None
    cx = sum(p[0] for p in inside) / len(inside)
    cy = sum(p[1] for p in inside) / len(inside)

    hints = _seat_candidates(episode["activity"])
    pool = [p for p in pois if p["zone"] == episode["zone"] and p.get("seat")]
    if hints:
        preferred = [p for p in pool if p["type"] in hints]
        if preferred:
            pool = preferred
    if not pool:
        return None

    best = min(pool, key=lambda p: (p["xy"][0] - cx) ** 2 + (p["xy"][1] - cy) ** 2)
    distance = math.hypot(best["xy"][0] - cx, best["xy"][1] - cy)
    if distance > 3.0:
        return None
    return best["id"]


def _population_names(plan_path: Path) -> dict[str, dict]:
    """Read the synthetic roster so the 3D agents carry the same names and rooms."""
    try:
        sys.path.insert(0, str(DEMO_ROOT / "src"))
        from eldercare.agents.population import build_population  # noqa: PLC0415
        from eldercare.space.floorplan import FloorPlan  # noqa: PLC0415

        plan = FloorPlan.load(str(plan_path))
        residents = build_population(plan, n=12, seed=42)
    except Exception as exc:  # noqa: BLE001 - names are cosmetic
        print(f"[day] roster unavailable ({exc}); falling back to ids")
        return {}
    return {
        f"R{index + 1:02d}": {"name": resident.name, "room": resident.room, "age": resident.age,
                             "mobility": getattr(resident, "mobility", "")}
        for index, resident in enumerate(residents)
    }


def build_day(run_dir: Path = RUN, facility_path: Path | None = None, dt_hint: float = 5.0) -> dict:
    facility = json.loads(Path(facility_path).read_text(encoding="utf-8")) if facility_path else None
    pois = facility["pois"] if facility else []

    per_agent, zones, states, activities = _load_trajectories(run_dir / "trajectories.csv")
    episodes = _load_episodes(run_dir / "episodes.csv")
    roster = _population_names(DEMO_ROOT / "assets" / "mock_plan.json")

    zone_index = {z: i for i, z in enumerate(zones)}
    state_index = {s: i for i, s in enumerate(states)}
    activity_index = {a: i for i, a in enumerate(activities)}

    agents = []
    for agent_id in sorted(per_agent):
        entry = per_agent[agent_id]
        times = entry["t"]
        dt = (times[1] - times[0]) if len(times) > 1 else dt_hint
        headings = _compute_headings(entry["x"], entry["y"], dt)
        meta = roster.get(agent_id, {})
        room = meta.get("room") or entry["zone"][0]
        agents.append(
            {
                "id": agent_id,
                "name": meta.get("name", agent_id),
                "room": room,
                "age": meta.get("age"),
                "role": "resident",
                "color": AGENT_COLORS["resident"],
                "dt": dt,
                "t0": times[0],
                "x": [round(v, 2) for v in entry["x"]],
                "y": [round(v, 2) for v in entry["y"]],
                "heading": [round(v, 3) for v in headings],
                "zone": [zone_index[z] for z in entry["zone"]],
                "state": [state_index[s] for s in entry["state"]],
                "activity": [activity_index[a] for a in entry["activity"]],
            }
        )

    samples = {
        agent["id"]: list(zip(per_agent[agent["id"]]["t"], per_agent[agent["id"]]["x"], per_agent[agent["id"]]["y"]))
        for agent in agents
    }
    for episode in episodes:
        episode["poi"] = _snap_episode(episode, samples.get(episode["agent"], []), pois)

    snaps = sum(1 for e in episodes if e["poi"])
    return {
        "meta": {
            "date": "demo01",
            "start": "06:30:00",
            "end": "21:30:00",
            "dt": dt_hint,
            "bounds": facility["meta"]["bounds"] if facility else None,
            "source": str(run_dir),
        },
        "zones": zones,
        "states": states,
        "activities": activities,
        "agents": agents,
        "episodes": episodes,
        "stats": {"agents": len(agents), "episodes": len(episodes), "episodesWithPoi": snaps},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--facility", type=Path, default=Path(__file__).resolve().parent / "data" / "facility.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    day = build_day(args.run, args.facility)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(day, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(
        f"agents={day['stats']['agents']} episodes={day['stats']['episodes']} "
        f"withPoi={day['stats']['episodesWithPoi']} -> {args.out} ({size_mb:.2f} MB)"
    )
    print("states:", day["states"])
    print("activities:", day["activities"])


if __name__ == "__main__":
    main()
