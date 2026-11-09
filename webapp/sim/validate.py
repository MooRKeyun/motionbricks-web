"""Sanity checks for the generated scene data (facility.json + day.json)."""

from __future__ import annotations

import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def load() -> tuple[dict, dict]:
    return (
        json.loads((DATA / "facility.json").read_text(encoding="utf-8")),
        json.loads((DATA / "day.json").read_text(encoding="utf-8")),
    )


def segment_covers(segment: list[float], point: tuple[float, float], tol: float = 0.02) -> bool:
    x1, y1, x2, y2 = segment
    if abs(x1 - x2) < 1e-6:  # vertical
        return abs(point[0] - x1) < tol and min(y1, y2) - tol <= point[1] <= max(y1, y2) + tol
    if abs(y1 - y2) < 1e-6:  # horizontal
        return abs(point[1] - y1) < tol and min(x1, x2) - tol <= point[0] <= max(x1, x2) + tol
    return False


def main() -> None:
    facility, day = load()
    zones = {z["id"]: z for z in facility["zones"]}
    walls = facility["walls"]

    print(f"zones={len(zones)} walls={len(walls)} doors={len(facility['doors'])} pois={len(facility['pois'])}")

    diagonal = [w for w in walls if abs(w[0] - w[2]) > 1e-6 and abs(w[1] - w[3]) > 1e-6]
    print(f"walls that are not axis aligned: {len(diagonal)}")

    blocked = []
    for door in facility["doors"]:
        point = (door["point"][0], door["point"][1])
        if any(segment_covers(w, point) for w in walls):
            blocked.append(door["id"])
    print(f"doors blocked by a wall: {len(blocked)} {blocked[:5]}")

    # every zone's centre must be inside its own rectangle and not on a wall
    off_centre = []
    for zone in facility["zones"]:
        cx, cy = zone["centroid"]
        if any(segment_covers(w, (cx, cy), tol=0.05) for w in walls):
            off_centre.append(zone["id"])
    print(f"zone centres sitting on a wall: {len(off_centre)} {off_centre[:5]}")

    # agent samples inside the plan bounds and inside some zone rectangle
    outside = 0
    total = 0
    for agent in day["agents"]:
        for x, y in zip(agent["x"], agent["y"]):
            total += 1
            if not any(
                z["bbox"][0] <= x <= z["bbox"][2] and z["bbox"][1] <= y <= z["bbox"][3]
                for z in facility["zones"]
            ):
                outside += 1
    print(f"agent samples={total} outside every zone rectangle={outside}")

    # episode anchors
    episodes = day["episodes"]
    snapped = [e for e in episodes if e["poi"]]
    poi_by_id = {p["id"]: p for p in facility["pois"]}
    agents_by_id = {a["id"]: a for a in day["agents"]}
    offsets: list[float] = []
    for episode in snapped:
        agent = agents_by_id[episode["agent"]]
        dt = agent["dt"]
        i0 = max(0, int(round((episode["t0"] - agent["t0"]) / dt)))
        i1 = min(len(agent["x"]) - 1, int(round((episode["t1"] - agent["t0"]) / dt)))
        if i1 < i0:
            continue
        cx = sum(agent["x"][i0 : i1 + 1]) / (i1 - i0 + 1)
        cy = sum(agent["y"][i0 : i1 + 1]) / (i1 - i0 + 1)
        poi = poi_by_id[episode["poi"]]
        offsets.append(math.hypot(poi["xy"][0] - cx, poi["xy"][1] - cy))
    print(
        f"episodes={len(episodes)} withPoi={len(snapped)} checked={len(offsets)} "
        f"mean anchor offset={sum(offsets) / max(1, len(offsets)):.2f} m "
        f"max={max(offsets, default=0):.2f} m"
    )

    # furniture inside its zone
    bad_poi = [p["id"] for p in facility["pois"]
               if not (zones[p["zone"]]["bbox"][0] <= p["xy"][0] <= zones[p["zone"]]["bbox"][2]
                       and zones[p["zone"]]["bbox"][1] <= p["xy"][1] <= zones[p["zone"]]["bbox"][3])]
    print(f"pois outside their zone: {len(bad_poi)} {bad_poi[:5]}")

    # heading continuity
    jumps = 0
    for agent in day["agents"]:
        headings = agent["heading"]
        for a, b in zip(headings, headings[1:]):
            delta = abs((a - b + math.pi) % (2 * math.pi) - math.pi)
            if delta > math.radians(35):
                jumps += 1
    print(f"heading jumps > 35 deg between 5 s samples: {jumps}")


if __name__ == "__main__":
    main()
