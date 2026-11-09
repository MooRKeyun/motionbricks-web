"""Sweep the whole day: check modes, errors, and frame rate."""

from __future__ import annotations

import time

from playwright.sync_api import sync_playwright


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader"])
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.goto("http://127.0.0.1:8000/day", wait_until="networkidle")
        page.wait_for_function("() => window.__day && window.__day.agents.length > 0", timeout=180_000)
        time.sleep(0.5)

        rows = page.evaluate(
            """() => {
                const s = window.__day;
                const out = [];
                for (let t = 0; t <= 54000; t += 1800) {
                    window.__setTime(t);
                    let chars = 0, arrows = 0;
                    for (const a of s.agents) {
                        if (a.mode === 'arrow') arrows += 1; else chars += 1;
                    }
                    out.push({t, chars, arrows});
                }
                return out;
            }"""
        )
        for row in rows:
            clock = row["t"] // 3600 + 6
            minute = (row["t"] % 3600) // 60
            print(f"{clock:02d}:{minute:02d}  characters={row['chars']:2d}  arrows={row['arrows']:2d}")

        # frame rate over 3 seconds while playing at 60x
        page.evaluate("() => window.__setTime(24000)")
        fps = page.evaluate(
            """() => new Promise((resolve) => {
                let frames = 0;
                const start = performance.now();
                document.getElementById('play').click();
                const loop = () => {
                    frames += 1;
                    if (performance.now() - start < 3000) requestAnimationFrame(loop);
                    else { document.getElementById('play').click(); resolve(frames / ((performance.now() - start) / 1000)); }
                };
                requestAnimationFrame(loop);
            })"""
        )
        print(f"fps (headless swiftshader, 15 agents): {fps:.1f}")
        print("errors:", errors[:5])
        browser.close()


if __name__ == "__main__":
    main()
