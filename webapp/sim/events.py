"""Extract the interaction events that MotionBricks should fill in.

Two families of events, both derived from the recorded day (no simulation rerun):

* **seat transitions** – an episode that happened at a furniture anchor becomes a
  ``sit`` clip when the dwell starts and a ``stand`` clip when it ends;
* **meet transitions** – a proximity window (two agents within 1.5 m for at least
  15 s) becomes a ``meet_start`` clip when it begins and a ``meet_end`` clip when it
  ends, as long as somebody was actually walking into it.

Everything else in the day stays as a ground arrow. Each event carries the exact
start and end state (position, heading, pose key) that MotionBricks needs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

PROXIMITY_M = 1.5
MIN_WINDOW_S = 15.0
MOVE_SPEED = 0.45  # m/s, "was this agent walking?"
SIT_LEAD_S = 2.4
SIT_TAIL_S = 0.6
STAND_LEAD_S = 0.6
STAND_TAIL_S = 2.0
MIN_DWELL_MIN = 5.0


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def sample_at(agent: dict, t: float) -> tuple[float, float, float]:
    """Linear position and shortest-arc heading at time t."""
    dt = agent["dt"]
    n = len(agent["x"])
    f = (t - agent["t0"]) / dt
    i0 = max(0, min(n - 1, int(math.floor(f))))
    i1 = min(n - 1, i0 + 1)
    a = max(0.0, min(1.0, f - i0))
    x = agent["x"][i0] + (agent["x"][i1] - agent["x"][i0]) * a
    y = agent["y"][i0] + (agent["y"][i1] - agent["y"][i0]) * a
    h0, h1 = agent["heading"][i0], agent["heading"][i1]
    delta = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
    return x, y, h0 + delta * a


def speed_at(agent: dict, t: float) -> float:
    """Ground speed around time t, from the 5 s samples."""
    dt = agent["dt"]
    n = len(agent["x"])
    i = max(1, min(n - 2, int(round((t - agent["t0"]) / dt))))
    vx = (agent["x"][i + 1] - agent["x"][i - 1]) / (2 * dt)
    vy = (agent["y"][i + 1] - agent["y"][i - 1]) / (2 * dt)
    return math.hypot(vx, vy)


def facing(from_xy: tuple[float, float], to_xy: tuple[float, float]) -> float:
    return math.atan2(to_xy[1] - from_xy[1], to_xy[0] - from_xy[0])


def proximity_windows(day: dict) -> list[dict]:
    """Pairwise windows where two agents stayed within PROXIMITY_M."""
    agents = day["agents"]
    dt = agents[0]["dt"]
    count = len(agents[0]["x"])
    ids = [a["id"] for a in agents]
    xs = [a["x"] for a in agents]
    ys = [a["y"] for a in agents]

    open_windows: dict[tuple[str, str], dict] = {}
    windows: list[dict] = []

    def close(key: tuple[str, str], last_index: int) -> None:
        entry = open_windows.pop(key, None)
        if entry is None:
            return
        start, end = entry["start"], last_index * dt
        if end - start < MIN_WINDOW_S:
            return
        windows.append({
            "a": key[0],
            "b": key[1],
            "t0": start,
            "t1": end,
            "min_distance": entry["min_distance"],
            "a_moving": entry["a_moving"],
            "b_moving": entry["b_moving"],
        })

    for index in range(count):
        t = index * dt
        close_pairs: dict[tuple[str, str], float] = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                distance = math.hypot(xs[i][index] - xs[j][index], ys[i][index] - ys[j][index])
                if distance <= PROXIMITY_M:
                    close_pairs[(ids[i], ids[j])] = distance
        for key, distance in close_pairs.items():
            entry = open_windows.get(key)
            if entry is None:
                open_windows[key] = {
                    "start": t,
                    "last_index": index,
                    "min_distance": distance,
                    "a_moving": speed_at(agents[ids.index(key[0])], t) > MOVE_SPEED,
                    "b_moving": speed_at(agents[ids.index(key[1])], t) > MOVE_SPEED,
                }
            else:
                entry["last_index"] = index
                entry["min_distance"] = min(entry["min_distance"], distance)
        for key in [k for k in open_windows if k not in close_pairs]:
            if index - open_windows[key]["last_index"] > 1:
                close(key, open_windows[key]["last_index"])

    for key in list(open_windows):
        close(key, open_windows[key]["last_index"])
    windows.sort(key=lambda w: w["t0"])
    return windows


def seat_events(day: dict, facility: dict) -> list[dict]:
    pois = {p["id"]: p for p in facility["pois"]}
    agents = {a["id"]: a for a in day["agents"]}
    events: list[dict] = []
    for index, episode in enumerate(day["episodes"]):
        poi_id = episode.get("poi")
        if not poi_id:
            continue
        poi = pois.get(poi_id)
        if not poi or not poi.get("seat"):
            continue
        if episode["minutes"] < MIN_DWELL_MIN:
            continue
        agent = agents[episode["agent"]]
        seat_xy = (poi["xy"][0], poi["xy"][1])
        seat_yaw = poi["yaw"]

        t0 = episode["t0"]
        start = sample_at(agent, t0 - SIT_LEAD_S)
        moving = speed_at(agent, max(0.0, t0 - SIT_LEAD_S / 2)) > 0.4
        events.append({
            "id": f"sit-{episode['agent']}-{index:04d}",
            "agent": episode["agent"],
            "kind": "sit",
            "t0": t0 - SIT_LEAD_S,
            "t1": t0 + SIT_TAIL_S,
            "poi": poi_id,
            "partner": None,
            "episode": episode["activity"],
            "start": {"x": start[0], "y": start[1], "heading": start[2], "pose": "walk" if moving else "idle"},
            "end": {"x": seat_xy[0], "y": seat_xy[1], "heading": seat_yaw, "pose": "sit"},
        })

        t1 = episode["t1"]
        end = sample_at(agent, min(t1 + STAND_TAIL_S, agent["t0"] + (len(agent["x"]) - 1) * agent["dt"]))
        events.append({
            "id": f"stand-{episode['agent']}-{index:04d}",
            "agent": episode["agent"],
            "kind": "stand",
            "t0": t1 - STAND_LEAD_S,
            "t1": t1 + STAND_TAIL_S,
            "poi": poi_id,
            "partner": None,
            "episode": episode["activity"],
            "start": {"x": seat_xy[0], "y": seat_xy[1], "heading": seat_yaw, "pose": "sit"},
            "end": {"x": end[0], "y": end[1], "heading": end[2], "pose": "idle"},
        })
    return events


def meet_events(day: dict) -> list[dict]:
    agents = {a["id"]: a for a in day["agents"]}
    events: list[dict] = []
    for index, window in enumerate(proximity_windows(day)):
        a_id, b_id = window["a"], window["b"]
        a, b = agents[a_id], agents[b_id]
        t0, t1 = window["t0"], window["t1"]

        if window["a_moving"] or window["b_moving"]:
            for me, other in ((a_id, b_id), (b_id, a_id)):
                before = sample_at(agents[me], t0 - 2.0)
                at = sample_at(agents[me], t0)
                other_at = sample_at(agents[other], t0)
                events.append({
                    "id": f"meet-{me}-{index:04d}",
                    "agent": me,
                    "kind": "meet_start",
                    "t0": t0 - 2.0,
                    "t1": t0 + 0.8,
                    "poi": None,
                    "partner": other,
                    "episode": "相遇",
                    "start": {"x": before[0], "y": before[1], "heading": before[2], "pose": "walk"},
                    "end": {"x": at[0], "y": at[1], "heading": facing((at[0], at[1]), (other_at[0], other_at[1])),
                            "pose": "idle"},
                })

        for me, other in ((a_id, b_id), (b_id, a_id)):
            at = sample_at(agents[me], t1)
            other_at = sample_at(agents[other], t1)
            after_t = t1 + 2.0
            limit = agents[me]["t0"] + (len(agents[me]["x"]) - 1) * agents[me]["dt"]
            after = sample_at(agents[me], min(after_t, limit))
            events.append({
                "id": f"leave-{me}-{index:04d}",
                "agent": me,
                "kind": "meet_end",
                "t0": t1 - 0.8,
                "t1": min(after_t, limit),
                "poi": None,
                "partner": other,
                "episode": "告别",
                "start": {"x": at[0], "y": at[1],
                          "heading": facing((at[0], at[1]), (other_at[0], other_at[1])), "pose": "idle"},
                "end": {"x": after[0], "y": after[1], "heading": after[2], "pose": "walk"},
            })
    return events


def assign_seat_slots(events: list[dict], facility: dict) -> None:
    """Spread agents that share one piece of furniture along its long side.

    The plan has a single sofa in the activity room and one bench per garden spot, so
    several residents' episodes snap to the same anchor. Without an offset they would
    stand inside each other; here each gets a slot, and the slot is reused once its
    previous occupant has left.
    """
    pois = {p["id"]: p for p in facility["pois"]}
    stands: dict[tuple[str, str], list[float]] = {}
    for event in events:
        if event["kind"] == "stand" and event["poi"]:
            stands.setdefault((event["agent"], event["poi"]), []).append(event["t0"])
    for times in stands.values():
        times.sort()

    def dwell_end(event: dict) -> float:
        candidates = [t for t in stands.get((event["agent"], event["poi"]), []) if t > event["t0"]]
        return candidates[0] if candidates else event["t0"] + 600.0

    by_poi: dict[str, list[dict]] = {}
    for event in events:
        if event["kind"] == "sit" and event["poi"]:
            by_poi.setdefault(event["poi"], []).append(event)

    for poi_id, group in by_poi.items():
        poi = pois[poi_id]
        capacity = max(1, int(poi.get("capacity") or 1))
        if capacity == 1:
            continue
        group.sort(key=lambda e: e["t0"])
        slots: list[float] = []  # end time per slot
        spread = 0.55
        for event in group:
            end = dwell_end(event)
            free = None
            for index, until in enumerate(slots):
                if until <= event["t0"]:
                    free = index
                    break
            if free is None:
                if len(slots) < capacity:
                    free = len(slots)
                    slots.append(end)
                else:
                    free = min(range(len(slots)), key=lambda i: slots[i])
                    slots[free] = end
            else:
                slots[free] = end

            offset = (free - (capacity - 1) / 2.0) * spread
            lateral = (math.cos(poi["yaw"] + math.pi / 2), math.sin(poi["yaw"] + math.pi / 2))
            dx, dy = lateral[0] * offset, lateral[1] * offset
            event["slot"] = free
            event["end"]["x"] = round(event["end"]["x"] + dx, 3)
            event["end"]["y"] = round(event["end"]["y"] + dy, 3)
            if math.hypot(event["start"]["x"] - event["end"]["x"], event["start"]["y"] - event["end"]["y"]) < 1.2:
                event["start"]["x"] = round(event["start"]["x"] + dx, 3)
                event["start"]["y"] = round(event["start"]["y"] + dy, 3)
            stand = next((e for e in events if e["kind"] == "stand" and e["agent"] == event["agent"]
                          and e["poi"] == poi_id and e["t0"] > event["t0"]), None)
            if stand is not None:
                stand["start"]["x"] = event["end"]["x"]
                stand["start"]["y"] = event["end"]["y"]
                if math.hypot(stand["end"]["x"] - event["end"]["x"], stand["end"]["y"] - event["end"]["y"]) < 1.2:
                    stand["end"]["x"] = round(stand["end"]["x"] + dx, 3)
                    stand["end"]["y"] = round(stand["end"]["y"] + dy, 3)


def build_events(day: dict, facility: dict, include_meets: bool = True) -> list[dict]:
    limit = day["agents"][0]["t0"] + (len(day["agents"][0]["x"]) - 1) * day["agents"][0]["dt"]
    events = seat_events(day, facility)
    if include_meets:
        meets = meet_events(day)
        moved = {e["id"].split("-", 2)[-1] for e in meets if e["kind"] == "meet_start"}
        events += [e for e in meets if e["kind"] != "meet_end" or e["id"].split("-", 2)[-1] in moved]
    assign_seat_slots(events, facility)
    for event in events:
        event["t0"] = max(0.0, min(limit, event["t0"]))
        event["t1"] = max(event["t0"] + 0.2, min(limit, event["t1"]))
    events.sort(key=lambda e: (e["t0"], e["agent"]))
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DATA / "events.json")
    parser.add_argument("--no-meets", action="store_true")
    args = parser.parse_args()

    day = load("day.json")
    facility = load("facility.json")
    events = build_events(day, facility, include_meets=not args.no_meets)

    kinds: dict[str, int] = {}
    for event in events:
        kinds[event["kind"]] = kinds.get(event["kind"], 0) + 1

    args.out.write_text(json.dumps({"meta": {"count": len(events), "kinds": kinds}, "events": events},
                                   ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"events={len(events)} -> {args.out}")
    print("kinds:", kinds)
    print("first five:", [(e["id"], round(e["t0"])) for e in events[:5]])
    durations = [e["t1"] - e["t0"] for e in events]
    print(f"duration s: min={min(durations):.1f} max={max(durations):.1f} mean={sum(durations)/len(durations):.1f}")


if __name__ == "__main__":
    main()
