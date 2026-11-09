"""Screenshot the day page at a given clock time."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from playwright.sync_api import sync_playwright

SHOTS = Path(__file__).resolve().parent / "shots"


def clock_to_seconds(clock: str) -> float:
    hours, minutes = (int(part) for part in clock.split(":"))
    return (hours - 6.5) * 3600 + minutes * 60


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/day")
    parser.add_argument("--clock", default="13:20")
    parser.add_argument("--out", default="day")
    parser.add_argument("--camera", default=None, help="x,y,z")
    parser.add_argument("--target", default=None, help="x,y,z")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()

    SHOTS.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader"])
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(f"[pageerror] {error}"))
        page.on("console", lambda message: errors.append(f"[{message.type}] {message.text}") if message.type == "error" else None)
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function("() => window.__day && window.__day.agents.length > 0", timeout=120_000)
        time.sleep(1.0)

        if args.camera:
            x, y, z = (float(v) for v in args.camera.split(","))
            page.evaluate(
                "([x, y, z]) => { window.__camera.position.set(x, y, z); }",
                [x, y, z],
            )
        if args.target:
            x, y, z = (float(v) for v in args.target.split(","))
            page.evaluate("([x, y, z]) => { window.__orbit.target.set(x, y, z); }", [x, y, z])

        page.evaluate("(seconds) => window.__setTime(seconds)", clock_to_seconds(args.clock))
        time.sleep(0.8)
        out = SHOTS / f"{args.out}.png"
        page.screenshot(path=str(out))
        print("screenshot:", out)

        info = page.evaluate(
            """() => ({
                agents: window.__day.agents.length,
                t: window.__day.t,
                clock: document.getElementById('clock').textContent,
                calls: window.__day ? null : null,
            })"""
        )
        print("state:", info)
        print("console:", errors[:6])
        browser.close()


if __name__ == "__main__":
    main()
