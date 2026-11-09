"""Convert the demo floor plan into scene data for the browser.

Reads ``demo/assets/mock_plan.json`` (read only) and produces a self-contained
``facility.json``:

* zones  – the 42 rectangles, with a display colour per zone type
* walls  – every zone edge, deduplicated, with the door gaps cut out
* doors  – kept for reference and for drawing thresholds
* pois   – the 102 furniture anchors, with a facing yaw and a display size

All geometry stays in the plan's own metre coordinates (origin at the lower-left
corner, x to the right, y upwards). The browser maps y to three.js z, so the plan
is played back without any rescaling.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEMO_ROOT = Path(r"C:\Users\13119\Desktop\工作\论文项目\调研\未来研究方向\社区\demo")
DEFAULT_PLAN = DEMO_ROOT / "assets" / "mock_plan.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "facility.json"

ZONE_COLORS = {
    "resident_room": "#dfe6f2",
    "corridor": "#efe7d6",
    "lobby": "#fdf6e3",
    "dining": "#f6e0c8",
    "activity": "#dfeedd",
    "garden": "#cfe8cf",
    "rehab": "#d9e9f2",
    "nurse_station": "#f6dfe4",
    "staff": "#e8e0f0",
}

ZONE_LABELS = {
    "resident_room": "居室",
    "corridor": "走廊",
    "lobby": "门厅",
    "dining": "餐厅",
    "activity": "活动室",
    "garden": "庭院花园",
    "rehab": "康复训练室",
    "nurse_station": "护理站",
    "staff": "员工室",
}

# Footprint (width, depth, height) in metres, used by the browser to draw the
# primitive that stands for each kind of furniture.
POI_SIZES = {
    "bed": [0.95, 2.0, 0.55],
    "chair": [0.5, 0.5, 0.85],
    "dining_table": [1.3, 1.3, 0.75],
    "sofa": [1.9, 0.85, 0.75],
    "bench": [1.6, 0.5, 0.45],
    "tv": [1.2, 0.15, 0.7],
    "piano": [1.5, 0.6, 1.0],
    "craft_table": [1.6, 0.9, 0.75],
    "bookshelf": [1.2, 0.35, 1.9],
    "planter": [0.7, 0.7, 0.9],
    "parallel_bars": [2.4, 0.6, 1.3],
    "exercise_bike": [0.7, 1.3, 1.1],
    "treatment_table": [0.9, 1.9, 0.75],
    "reception": [2.2, 0.8, 1.1],
    "notice_board": [1.4, 0.08, 1.0],
    "desk": [1.3, 0.7, 0.75],
}

# Furniture that a character can sit on or lie in; used to pick sit/stand clips.
SEAT_TYPES = {"chair", "sofa", "bench", "bed", "dining_table", "treatment_table"}


def _bbox(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _zone_edges(x0: float, y0: float, x1: float, y1: float) -> list[tuple[str, float, float, float]]:
    """Return the four axis-aligned edges of a rectangle as (axis, fixed, start, end)."""
    return [
        ("v", round(x0, 3), y0, y1),
        ("v", round(x1, 3), y0, y1),
        ("h", round(y0, 3), x0, x1),
        ("h", round(y1, 3), x0, x1),
    ]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union a list of [start, end) intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _subtract(intervals: list[tuple[float, float]], gaps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Remove gap intervals from wall intervals."""
    result = list(intervals)
    for gap_start, gap_end in sorted(gaps):
        updated: list[tuple[float, float]] = []
        for start, end in result:
            if gap_end <= start + 1e-6 or gap_start >= end - 1e-6:
                updated.append((start, end))
                continue
            if gap_start > start + 1e-6:
                updated.append((start, gap_start))
            if gap_end < end - 1e-6:
                updated.append((gap_end, end))
        result = updated
    return result


def build_walls(zones: list[dict], doors: list[dict]) -> list[list[float]]:
    """Collect every zone edge once, then cut the door openings out of it."""
    walls: dict[tuple[str, float], list[tuple[float, float]]] = {}
    for zone in zones:
        x0, y0, x1, y1 = _bbox(zone["polygon"])
        for axis, fixed, start, end in _zone_edges(x0, y0, x1, y1):
            walls.setdefault((axis, fixed), []).append((start, end))

    gaps: dict[tuple[str, float], list[tuple[float, float]]] = {}
    for door in doors:
        axis = door["axis"]
        point = door["point"]
        width = float(door["width_m"])
        if axis == "v":
            key = ("v", round(point[0], 3))
            center, span = point[1], (point[1] - width / 2, point[1] + width / 2)
        else:
            key = ("h", round(point[1], 3))
            center, span = point[0], (point[0] - width / 2, point[0] + width / 2)
        del center
        gaps.setdefault(key, []).append(span)

    segments: list[list[float]] = []
    for key, intervals in walls.items():
        axis, fixed = key
        merged = _merge_intervals(intervals)
        cut = _subtract(merged, gaps.get(key, []))
        for start, end in cut:
            if end - start < 0.02:
                continue
            if axis == "v":
                segments.append([fixed, start, fixed, end])
            else:
                segments.append([start, fixed, end, fixed])
    return segments


def build_pois(plan: dict) -> list[dict]:
    """Add a facing yaw and a display size to each furniture anchor."""
    centroid = {zone["id"]: zone["centroid"] for zone in plan["zones"]}
    pois = []
    for poi in plan["pois"]:
        x, y = poi["xy"]
        zone_center = centroid.get(poi["zone"], [x, y + 1.0])
        dx, dy = zone_center[0] - x, zone_center[1] - y
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            dx, dy = 1.0, 0.0
        yaw = math.atan2(dy, dx)
        size = POI_SIZES.get(poi["type"], [0.8, 0.8, 0.8])
        pois.append(
            {
                "id": poi["id"],
                "type": poi["type"],
                "zone": poi["zone"],
                "xy": [round(x, 3), round(y, 3)],
                "yaw": round(yaw, 4),
                "capacity": poi.get("capacity", 1),
                "label": poi.get("label", ""),
                "size": size,
                "seat": poi["type"] in SEAT_TYPES,
            }
        )
    return pois


def build_facility(plan_path: Path = DEFAULT_PLAN) -> dict:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    zones = []
    for zone in plan["zones"]:
        x0, y0, x1, y1 = _bbox(zone["polygon"])
        zones.append(
            {
                "id": zone["id"],
                "type": zone["type"],
                "name": zone["name"],
                "label": ZONE_LABELS.get(zone["type"], zone["type"]),
                "polygon": [[round(x, 3), round(y, 3)] for x, y in zone["polygon"]],
                "bbox": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
                "centroid": [round(c, 3) for c in zone["centroid"]],
                "area": zone["area_m2"],
                "walkable": bool(zone.get("walkable", True)),
                "color": ZONE_COLORS.get(zone["type"], "#e5e7eb"),
            }
        )

    doors = [
        {
            "id": door["id"],
            "a": door["zone_a"],
            "b": door["zone_b"],
            "point": [round(door["point"][0], 3), round(door["point"][1], 3)],
            "axis": door["axis"],
            "width": float(door["width_m"]),
        }
        for door in plan["doors"]
    ]

    return {
        "meta": {
            "name": plan["meta"].get("name", "facility"),
            "units": "meters",
            "bounds": plan["meta"]["bounds"],
            "source": "demo/assets/mock_plan.json",
            "note": plan["meta"].get("note", ""),
        },
        "zones": zones,
        "walls": build_walls(plan["zones"], plan["doors"]),
        "doors": doors,
        "pois": build_pois(plan),
        "zoneColors": ZONE_COLORS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    facility = build_facility(args.plan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(facility, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"zones={len(facility['zones'])} walls={len(facility['walls'])} "
        f"doors={len(facility['doors'])} pois={len(facility['pois'])} -> {args.out}"
    )


if __name__ == "__main__":
    main()
