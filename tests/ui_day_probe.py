"""Inspect which agents are drawn as characters versus arrows at a given time."""

from __future__ import annotations

import json
import time

from playwright.sync_api import sync_playwright


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader"])
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on("pageerror", lambda error: print("[pageerror]", error))
        page.goto("http://127.0.0.1:8000/day", wait_until="networkidle")
        page.wait_for_function("() => window.__day && window.__day.agents.length > 0", timeout=180_000)
        time.sleep(0.5)

        for seconds in (2400, 24600, 31800, 46800):
            info = page.evaluate(
                """(seconds) => {
                    const s = window.__day;
                    window.__setTime(seconds);
                    return s.agents.map((a) => ({
                        id: a.meta.id, mode: a.mode,
                        clip: a.mode === 'arrow' ? null : (s.clipsByAgent.get(a.meta.id) || []).filter(c => c.t0 <= seconds).pop()?.kind,
                    }));
                }""",
                seconds,
            )
            chars = [row["id"] for row in info if row["mode"] != "arrow"]
            print(f"t={seconds:6d}: characters={len(chars)} {chars}")
            print(f"          modes: {[(r['id'], r['mode'], r['clip']) for r in info]}")
        browser.close()


if __name__ == "__main__":
    main()
