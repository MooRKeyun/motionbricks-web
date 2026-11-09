"""Synthesise the three staff agents for the day.

The demo only simulates residents, so the nurses and the porter are scripted here: a
short duty list of anchors through the day, walked along the floor plan's A* grid (the
same navigation the residents use, imported read-only from ``demo/src``). The chest-pain
scenario interrupts the nurse's list at 13:20.

Output ``staff.json`` mirrors the resident schema in ``day.json`` so the browser can
treat both the same way.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

DATA = Path(__file__).resolve().parent / "data"
DEMO_ROOT = Path(r"C:\Users\13119\Desktop\工作\论文项目\调研\未来研究方向\社区\demo")

DT = 5.0
DAY_END = 54000.0
WALK_SPEED = 1.05
NURSE_SPEED = 1.25

STAFF_COLORS = {"nurse": "#ffffff", "staff": "#ffd166"}


def _floorplan():
    sys.path.insert(0, str(DEMO_ROOT / "src"))
    from eldercare.space.floorplan import FloorPlan  # noqa: PLC0415

    return FloorPlan.load(str(DEMO_ROOT / "assets" / "mock_plan.json"))


def _walk(points: list[tuple[float, float]], speed: float, t0: float) -> list[tuple[float, float, float]]:
    """Turn a polyline into (t, x, y) samples at DT, with linear speed."""
    samples: list[tuple[float, float, float]] = []
    if not points:
        return samples
    t = t0
    samples.append((t, points[0][0], points[0][1]))
    for index in range(1, len(points)):
        x0, y0 = points[index - 1]
        x1, y1 = points[index]
        distance = math.hypot(x1 - x0, y1 - y0)
        if distance < 1e-6:
            continue
        steps = max(1, int(math.ceil(distance / (speed * DT))))
        for step in range(1, steps + 1):
            ratio = step / steps
            t += DT
            samples.append((t, x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio))
    return samples


def _heading(prev: tuple[float, float], cur: tuple[float, float], fallback: float) -> float:
    dx, dy = cur[0] - prev[0], cur[1] - prev[1]
    if math.hypot(dx, dy) < 1e-4:
        return fallback
    return math.atan2(dy, dx)


DUTY = {
    "N01": {
        "role": "nurse",
        "name": "护理员·张敏",
        "blocks": [
            (0, "护理站", "值守"),
            (90, "W-走廊", "巡视"),
            (180, "活动室", "巡查"),
            (250, "护理站", "值守"),
            (330, "餐厅", "协助就餐"),
            (400, "护理站", "值守"),
            (470, "门厅", "接待"),
            (540, "护理站", "值守"),
            (610, "活动室", "巡查"),
            (680, "E-走廊", "巡视"),
            (760, "护理站", "值守"),
            (820, "餐厅", "协助就餐"),
            (860, "护理站", "交接"),
        ],
    },
    "N02": {
        "role": "nurse",
        "name": "护理员·李婷",
        "blocks": [
            (0, "护理站", "值守"),
            (60, "E-走廊", "巡视"),
            (150, "W-走廊", "巡视"),
            (220, "康复训练室", "协助训练"),
            (300, "餐厅", "协助就餐"),
            (380, "护理站", "值守"),
            (460, "庭院花园", "陪同"),
            (520, "护理站", "值守"),
            (600, "E-走廊", "巡视"),
            (700, "餐厅", "协助就餐"),
            (800, "护理站", "值守"),
        ],
    },
    "S01": {
        "role": "staff",
        "name": "员工·王强",
        "blocks": [
            (0, "员工室", "准备"),
            (80, "门厅", "接待"),
            (160, "餐厅", "布置"),
            (260, "员工室", "休息"),
            (320, "餐厅", "协助就餐"),
            (420, "庭院花园", "维护"),
            (500, "员工室", "整理"),
            (580, "门厅", "接待"),
            (700, "餐厅", "协助就餐"),
            (820, "员工室", "收尾"),
        ],
    },
}

ANCHORS = {
    "护理站": "NURSE",
    "W-走廊": "W-COR",
    "E-走廊": "E-COR",
    "活动室": "ACTIVITY",
    "餐厅": "DINING",
    "门厅": "LOBBY",
    "庭院花园": "GARDEN",
    "康复训练室": "REHAB",
    "员工室": "STAFF",
}


def build_staff(facility: dict) -> list[dict]:
    plan = _floorplan()
    zones = {z["id"]: z for z in facility["zones"]}
    poi_by_zone: dict[str, list[dict]] = {}
    for poi in facility["pois"]:
        poi_by_zone.setdefault(poi["zone"], []).append(poi)

    agents = []
    for staff_id, spec in DUTY.items():
        color = STAFF_COLORS[spec["role"]]
        samples: list[tuple[float, float, float, str, str]] = []
        current = None
        for minute, place, activity in spec["blocks"]:
            zone_id = ANCHORS.get(place, place)
            zone = zones.get(zone_id)
            if zone is None:
                continue
            # Stand near the zone centre, anchored to a piece of furniture when there is one.
            target = (zone["centroid"][0], zone["centroid"][1])
            anchor_pois = poi_by_zone.get(zone_id, [])
            if anchor_pois:
                target = (anchor_pois[0]["xy"][0] + 0.9, anchor_pois[0]["xy"][1] + 0.6)

            t_start = max(0.0, minute * 60.0)
            if current is None:
                current = target
                samples.append((t_start, target[0], target[1], zone_id, activity))
                continue

            path = plan.astar(current, target) or [current, target]
            speed = NURSE_SPEED if spec["role"] == "nurse" else WALK_SPEED
            walk = _walk(path, speed, t_start)
            for t, x, y in walk:
                samples.append((t, x, y, zone_id, "移动"))
                if t >= DAY_END:
                    break
            samples.append((max(t_start, walk[-1][0] if walk else t_start), target[0], target[1], zone_id, activity))
            current = target

        samples.sort(key=lambda row: row[0])
        # resample to a strict 5 s grid so the browser can index by time
        grid: list[tuple[float, float, float, str, str]] = []
        index = 0
        last = samples[0]
        for step in range(int(DAY_END / DT) + 1):
            t = step * DT
            while index + 1 < len(samples) and samples[index + 1][0] <= t:
                index += 1
                last = samples[index]
            grid.append((t, last[1], last[2], last[3], last[4]))

        headings = [0.0] * len(grid)
        current_heading = 0.0
        for i in range(1, len(grid)):
            current_heading = _heading(grid[i - 1][1:3], grid[i][1:3], current_heading)
            headings[i] = current_heading

        zones_seq = [row[3] for row in grid]
        activities = [row[4] for row in grid]
        agents.append({
            "id": staff_id,
            "name": spec["name"],
            "role": spec["role"],
            "color": color,
            "room": zones_seq[0],
            "dt": DT,
            "t0": 0.0,
            "x": [round(row[1], 2) for row in grid],
            "y": [round(row[2], 2) for row in grid],
            "heading": [round(h, 3) for h in headings],
            "zone": zones_seq,
            "activity": activities,
        })
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DATA / "staff.json")
    args = parser.parse_args()

    facility = json.loads((DATA / "facility.json").read_text(encoding="utf-8"))
    agents = build_staff(facility)
    args.out.write_text(json.dumps({"agents": agents}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for agent in agents:
        print(f'{agent["id"]} {agent["name"]}: {len(agent["x"])} samples, zones {len(set(agent["zone"]))}')
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
