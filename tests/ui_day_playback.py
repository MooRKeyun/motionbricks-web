"""Check playback and scrubbing on the day page."""

from __future__ import annotations

import time

from playwright.sync_api import sync_playwright


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader"])
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on("pageerror", lambda error: print("[pageerror]", error))
        page.goto("http://127.0.0.1:8000/day", wait_until="networkidle")
        page.wait_for_function("() => window.__day && window.__day.agents.length > 0", timeout=120_000)
        time.sleep(0.5)

        # play for ~1.5 s at 60x -> the clock should advance about 90 s
        start = page.evaluate("() => window.__day.t")
        page.click("#play")
        time.sleep(1.5)
        playing = page.evaluate("() => window.__day.t")
        page.click("#play")
        print(f"playback: t {start:.0f} -> {playing:.0f} s (advanced {playing - start:.0f} s)")

        # scrub to a few times and measure the frame cost
        for clock, seconds in (("07:30", 3600), ("12:00", 19800), ("15:20", 31800), ("21:29", 53940)):
            began = time.perf_counter()
            page.evaluate("(s) => window.__setTime(s)", seconds)
            page.wait_for_function("(s) => Math.abs(window.__day.t - s) < 1", arg=seconds, timeout=5000)
            elapsed = (time.perf_counter() - began) * 1000
            label = page.evaluate("() => document.getElementById('clock').textContent")
            print(f"seek {clock}: {elapsed:.0f} ms, clock={label}")

        # agent positions must stay inside the building
        bounds = page.evaluate(
            """() => {
                const s = window.__day;
                let bad = 0;
                for (const a of s.agents) {
                    const i = Math.min(a.count - 1, Math.round((s.t - a.t0) / a.dt));
                    if (a.x[i] < 0 || a.x[i] > 72 || a.y[i] < 0 || a.y[i] > 40) bad += 1;
                }
                return bad;
            }"""
        )
        print("agents outside the plan bounds at the last seek:", bounds)
        browser.close()


if __name__ == "__main__":
    main()
