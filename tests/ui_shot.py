"""Screenshot the web app at a chosen time of the generated motion."""

from __future__ import annotations

import argparse
import time

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="frame")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--seek", type=float, default=None, help="fraction of the motion [0, 1]")
    parser.add_argument("--camera", default=None, help="x,y,z of the camera")
    parser.add_argument("--target", default=None, help="x,y,z of the orbit target")
    parser.add_argument("--hide-ghosts", action="store_true")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader"])
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.on("pageerror", lambda error: print("[pageerror]", error))
        page.goto("http://127.0.0.1:8000/", wait_until="networkidle")
        page.wait_for_function(
            "() => document.getElementById('generate') && !document.getElementById('generate').disabled",
            timeout=180_000,
        )
        time.sleep(1.0)

        if args.camera:
            x, y, z = (float(v) for v in args.camera.split(","))
            page.evaluate(
                "([x, y, z]) => { window.__camera.position.set(x, y, z); window.__camera.lookAt(window.__state.rigs.actor.root.position); }",
                [x, y, z],
            )
        if args.target:
            x, y, z = (float(v) for v in args.target.split(","))
            page.evaluate(
                "([x, y, z]) => { window.__state ? null : null; }",
                [x, y, z],
            )

        if args.generate:
            page.click("#generate")
            page.wait_for_function("() => !document.getElementById('export').disabled", timeout=120_000)
            time.sleep(0.3)
        else:
            time.sleep(0.5)

        if args.seek is not None:
            page.evaluate(
                """(fraction) => {
                    const state = window.__state;
                    state.playing = false;
                    const scrub = document.getElementById('scrub');
                    scrub.value = String(fraction);
                    scrub.dispatchEvent(new Event('input'));
                }""",
                args.seek,
            )
            page.wait_for_timeout(400)

        if args.hide_ghosts:
            page.evaluate(
                """() => {
                    const state = window.__state;
                    ['start', 'end'].forEach((k) => { if (state.rigs[k]) state.rigs[k].root.visible = false; });
                }"""
            )
            page.wait_for_timeout(200)

        out = f"tests/shots/{args.out}.png"
        page.screenshot(path=out)
        print("screenshot:", out)
        browser.close()


if __name__ == "__main__":
    main()
