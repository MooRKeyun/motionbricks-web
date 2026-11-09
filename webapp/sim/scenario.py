"""The chest-pain scenario: one incident at 13:20 in the activity room.

Adds the beats as MotionBricks events (so the clip pipeline picks them up) and rewrites
the two nurses' timelines so they run to the resident and kneel beside her. Everything
else in the day keeps running.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

DATA = Path(__file__).resolve().parent / "data"
DEMO_ROOT = Path(r"C:\Users\13119\Desktop\工作\论文项目\调研\未来研究方向\社区\demo")

SUBJECT = "R01"
RESPONDERS = ("N01", "N02")
EVENT_T = 13 * 3600 + 20 * 60 - int(6.5 * 3600)  # 13:20 in seconds after 06:30
EVENT_END = EVENT_T + 600


def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _sample(agent: dict, t: float) -> tuple[float, float, float]:
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


def _floorplan():
    sys.path.insert(0, str(DEMO_ROOT / "src"))
    from eldercare.space.floorplan import FloorPlan  # noqa: PLC0415

    return FloorPlan.load(str(DEMO_ROOT / "assets" / "mock_plan.json"))


def _beat(event_id: str, agent: str, kind: str, t0: float, t1: float,
          start: dict, end: dict, note: str) -> dict:
    return {
        "id": event_id,
        "agent": agent,
        "kind": kind,
        "t0": t0,
        "t1": t1,
        "poi": None,
        "partner": None,
        "episode": note,
        "scenario": "chest_pain",
        "start": start,
        "end": end,
    }


def build_chest_pain(day: dict, staff: dict, facility: dict) -> tuple[list[dict], dict, dict]:
    plan = _floorplan()
    agents = {a["id"]: a for a in day["agents"]}
    staff_agents = {a["id"]: a for a in staff["agents"]}
    subject = agents[SUBJECT]
    sx, sy, sh = _sample(subject, EVENT_T)

    # Nudge the incident spot away from whoever else is standing there, so two
    # characters never occupy the same square metre.
    neighbours = []
    for other_id, other in agents.items():
        if other_id == SUBJECT:
            continue
        ox, oy, _ = _sample(other, EVENT_T)
        distance = math.hypot(ox - sx, oy - sy)
        if distance < 1.6:
            neighbours.append((distance, ox, oy))
    if neighbours:
        _, ox, oy = min(neighbours)
        away = math.atan2(sy - oy, sx - ox)
        sx += math.cos(away) * 1.3
        sy += math.sin(away) * 1.3

    # The nurse kneels 0.85 m to the resident's right.
    side_angle = sh + math.pi / 2
    kneel = (sx + math.cos(side_angle) * 0.85, sy + math.sin(side_angle) * 0.85)
    approach = (sx + math.cos(sh + math.pi) * 1.1, sy + math.sin(sh + math.pi) * 1.1)

    n01 = staff_agents["N01"]
    n02 = staff_agents["N02"]
    n01_from = _sample(n01, EVENT_T - 60)
    n02_from = _sample(n02, EVENT_T - 60)

    events = [
        _beat("chest-R01-collapse", SUBJECT, "chest_collapse", EVENT_T, EVENT_T + 20,
              {"x": round(sx, 3), "y": round(sy, 3), "heading": round(sh, 3), "pose": "idle"},
              {"x": round(sx, 3), "y": round(sy, 3), "heading": round(sh, 3), "pose": "grab_chest"},
              "突发胸痛"),
        _beat("chest-R01-sink", SUBJECT, "chest_sink", EVENT_T + 20, EVENT_T + 120,
              {"x": round(sx, 3), "y": round(sy, 3), "heading": round(sh, 3), "pose": "grab_chest"},
              {"x": round(sx, 3), "y": round(sy, 3), "heading": round(sh + 0.5, 3), "pose": "crouch"},
              "扶住胸口蹲下"),
        _beat("chest-N01-kneel", "N01", "chest_kneel", EVENT_T + 300, EVENT_T + 360,
              {"x": round(kneel[0], 3), "y": round(kneel[1], 3), "heading": round(sh + math.pi, 3), "pose": "idle"},
              {"x": round(kneel[0], 3), "y": round(kneel[1], 3), "heading": round(sh + math.pi, 3), "pose": "crouch"},
              "护理员蹲下查看"),
        _beat("chest-N02-kneel", "N02", "chest_kneel", EVENT_T + 420, EVENT_T + 480,
              {"x": round(approach[0], 3), "y": round(approach[1], 3), "heading": round(sh + math.pi, 3), "pose": "idle"},
              {"x": round(approach[0], 3), "y": round(approach[1], 3), "heading": round(sh + math.pi, 3), "pose": "crouch"},
              "第二名护理员到位"),
        _beat("chest-R01-recover", SUBJECT, "chest_recover", EVENT_T + 540, EVENT_T + 600,
              {"x": round(sx, 3), "y": round(sy, 3), "heading": round(sh + 0.5, 3), "pose": "crouch"},
              {"x": round(sx, 3), "y": round(sy, 3), "heading": round(sh, 3), "pose": "idle"},
              "搀扶起身"),
    ]

    # Walk the nurses over and hold them beside the resident.
    override_n01 = _walk_override(plan, n01, EVENT_T - 60, kneel, speed=1.5, hold_until=EVENT_END)
    override_n02 = _walk_override(plan, n02, EVENT_T - 60, approach, speed=1.35, hold_until=EVENT_END + 120)

    timeline = {
        "id": "chest_pain",
        "t": EVENT_T,
        "end": EVENT_END,
        "zone": "ACTIVITY",
        "subject": SUBJECT,
        "responders": list(RESPONDERS),
        "title": "13:20 活动室 · 胸痛事件",
        "detail": "R01 在看电视时突发胸痛，护理站 N01 赶到并蹲下查看，N02 随后到位协助，13:30 搀扶起身恢复。",
        "beats": [{"t": e["t0"], "agent": e["agent"], "label": e["episode"]} for e in events],
    }
    return events, timeline, {"N01": override_n01, "N02": override_n02, "start": EVENT_T - 60,
                              "n01_from": n01_from, "n02_from": n02_from}


def _walk_override(plan, agent: dict, t_start: float, target: tuple[float, float],
                   speed: float, hold_until: float) -> dict:
    start = _sample(agent, t_start)
    path = plan.astar((start[0], start[1]), target) or [(start[0], start[1]), target]
    return {"path": [[round(x, 3), round(y, 3)] for x, y in path], "t_start": t_start,
            "speed": speed, "hold_until": hold_until}


def apply_override(agent: dict, override: dict) -> None:
    """Rewrite a 5 s timeline: walk the path, hold, then blend back to the original."""
    dt = agent["dt"]
    t_start = override["t_start"]
    path = override["path"]
    speed = override["speed"]
    hold_until = override["hold_until"]
    xs, ys = list(agent["x"]), list(agent["y"])

    # walk schedule (t, x, y)
    steps: list[tuple[float, float, float]] = [(t_start, path[0][0], path[0][1])]
    t = t_start
    for index in range(1, len(path)):
        x0, y0 = path[index - 1]
        x1, y1 = path[index]
        distance = math.hypot(x1 - x0, y1 - y0)
        steps_count = max(1, int(math.ceil(distance / (speed * dt))))
        for step in range(1, steps_count + 1):
            ratio = step / steps_count
            t += dt
            steps.append((t, x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio))

    arrive = steps[-1][0]
    hold_point = steps[-1][1:]

    for i in range(len(xs)):
        time = agent["t0"] + i * dt
        if t_start <= time <= arrive:
            # interpolate along the walk
            for (ta, xa, ya), (tb, xb, yb) in zip(steps, steps[1:]):
                if ta <= time <= tb:
                    ratio = (time - ta) / max(1e-6, tb - ta)
                    xs[i], ys[i] = xa + (xb - xa) * ratio, ya + (yb - ya) * ratio
                    break
        elif arrive < time <= hold_until:
            xs[i], ys[i] = hold_point
        elif hold_until < time <= hold_until + 60:
            # blend back onto the recorded timeline over one minute
            ratio = (time - hold_until) / 60.0
            xs[i] = hold_point[0] + (agent["x"][i] - hold_point[0]) * ratio
            ys[i] = hold_point[1] + (agent["y"][i] - hold_point[1]) * ratio

    headings = [agent["heading"][0]] * len(xs)
    current = headings[0]
    for i in range(1, len(xs)):
        dx, dy = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
        if math.hypot(dx, dy) > 1e-3:
            current = math.atan2(dy, dx)
        headings[i] = current

    agent["x"] = [round(v, 2) for v in xs]
    agent["y"] = [round(v, 2) for v in ys]
    agent["heading"] = [round(v, 3) for v in headings]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild-events", action="store_true", help="start from the extracted events again")
    args = parser.parse_args()

    day = _load("day.json")
    facility = _load("facility.json")
    staff = _load("staff.json")
    events_doc = _load("events.json")

    if args.rebuild_events:
        import importlib

        from . import events as events_module

        importlib.reload(events_module)
        events_doc = {"meta": {}, "events": events_module.build_events(day, facility)}

    base_events = [e for e in events_doc["events"] if e.get("scenario") != "chest_pain"]
    chest_events, timeline, overrides = build_chest_pain(day, staff, facility)

    events_doc["events"] = base_events + chest_events
    events_doc["meta"]["count"] = len(events_doc["events"])
    events_doc["meta"]["chestPain"] = timeline
    (DATA / "events.json").write_text(json.dumps(events_doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    for staff_id, override in (("N01", overrides["N01"]), ("N02", overrides["N02"])):
        agent = next(a for a in staff["agents"] if a["id"] == staff_id)
        apply_override(agent, override)

    (DATA / "staff.json").write_text(json.dumps(staff, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (DATA / "scenario.json").write_text(json.dumps({"chestPain": timeline}, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"chest pain at t={timeline['t']} ({timeline['title']}), beats={len(chest_events)}")
    print(f"events total={len(events_doc['events'])}")
    for staff_id in ("N01", "N02"):
        agent = next(a for a in staff["agents"] if a["id"] == staff_id)
        print(f"  {staff_id}: x range {min(agent['x']):.1f}..{max(agent['x']):.1f}")


if __name__ == "__main__":
    main()
