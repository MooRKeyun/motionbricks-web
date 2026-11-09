"""End-to-end browser test: change the states, generate, and export a GLB."""

from __future__ import annotations

import argparse
import time

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/")
    parser.add_argument("--out", default="tests/shots/e2e.png")
    args = parser.parse_args()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader"])
        context = browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)

        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function(
            "() => document.getElementById('generate') && !document.getElementById('generate').disabled",
            timeout=180_000,
        )

        # 1. change the end state: sit, moved to the side, turned 90 degrees
        page.select_option("#end-pose", "sit")
        page.fill("#end-x", "3.2")
        page.fill("#end-y", "1.6")
        page.eval_on_selector("#end-x", "el => el.dispatchEvent(new Event('input'))")
        page.eval_on_selector("#end-y", "el => el.dispatchEvent(new Event('input'))")
        page.eval_on_selector("#end-heading", """el => { el.value = '90'; el.dispatchEvent(new Event('input')); }""")
        time.sleep(0.5)

        # 2. generate
        page.click("#generate")
        page.wait_for_function("() => !document.getElementById('export').disabled", timeout=120_000)
        stats = page.evaluate(
            """() => ({
                segments: document.getElementById('stat-segments').textContent,
                duration: document.getElementById('stat-duration').textContent,
                frames: document.getElementById('stat-frames').textContent,
                inference: document.getElementById('stat-inference').textContent,
                error: document.getElementById('stat-error').textContent,
                hint: document.getElementById('hint').textContent,
            })"""
        )
        print("stats:", stats)

        # 3. the last frame lands on the end marker
        final = page.evaluate(
            """() => {
                const s = window.__state;
                const f = s.frames;
                const n = s.numFrames, j = s.numJoints;
                const o = ((n - 1) * j) * 7;
                return { x: f[o], y: f[o + 1], z: f[o + 2], marker: s.markers.end };
            }"""
        )
        print("final root:", {k: round(v, 3) if isinstance(v, float) else v for k, v in final.items()})

        # 4. export the animated GLB
        with page.expect_download(timeout=120_000) as download_info:
            page.click("#export")
        download = download_info.value
        path = "tests/shots/export_test.glb"
        download.save_as(path)
        print("exported:", path)

        page.screenshot(path=args.out)
        print("screenshot:", args.out)
        print("console errors:", errors)
        browser.close()


if __name__ == "__main__":
    main()
