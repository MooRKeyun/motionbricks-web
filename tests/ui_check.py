"""Load the web app in Edge through Playwright, screenshot it, and report console errors."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

SHOTS = Path(__file__).resolve().parent / "shots"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/")
    parser.add_argument("--out", default="initial")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--wait", type=float, default=2.0)
    args = parser.parse_args()

    SHOTS.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        messages: list[str] = []
        page.on("console", lambda msg: messages.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: messages.append(f"[pageerror] {err}"))

        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function(
            "() => document.getElementById('generate') && !document.getElementById('generate').disabled",
            timeout=180_000,
        )
        time.sleep(args.wait)
        page.screenshot(path=str(SHOTS / f"{args.out}.png"))
        print(f"screenshot: {SHOTS / f'{args.out}.png'}")

        if args.generate:
            page.click("#generate")
            page.wait_for_function(
                "() => !document.getElementById('export').disabled",
                timeout=120_000,
            )
            time.sleep(0.4)
            page.screenshot(path=str(SHOTS / f"{args.out}_playing.png"))
            print(f"screenshot: {SHOTS / f'{args.out}_playing.png'}")
            stats = page.evaluate(
                """() => ({
                    segments: document.getElementById('stat-segments').textContent,
                    duration: document.getElementById('stat-duration').textContent,
                    frames: document.getElementById('stat-frames').textContent,
                    error: document.getElementById('stat-error').textContent,
                    hint: document.getElementById('hint').textContent,
                })"""
            )
            print("stats:", stats)

        print("--- console ---")
        for message in messages:
            print(message)
        browser.close()


if __name__ == "__main__":
    main()
