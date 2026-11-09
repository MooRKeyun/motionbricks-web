"""Compare the orientation of the actor and the two ghost previews."""

from __future__ import annotations

import json
import time

from playwright.sync_api import sync_playwright


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader"])
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.on("pageerror", lambda error: print("[pageerror]", error))
        page.goto("http://127.0.0.1:8000/", wait_until="networkidle")
        page.wait_for_function(
            "() => document.getElementById('generate') && !document.getElementById('generate').disabled",
            timeout=120_000,
        )
        time.sleep(1.2)
        info = page.evaluate(
            """() => {
                const three = window.__three;
                const state = window.__state;
                const v = new three.Vector3();
                const pos = (rig, name) => { rig.byName.get(name).getWorldPosition(v); return [ +v.x.toFixed(2), +v.y.toFixed(2), +v.z.toFixed(2) ]; };
                const out = {};
                for (const key of ['actor', 'start', 'end']) {
                    const rig = state.rigs[key];
                    if (!rig) continue;
                    out[key] = {
                        hips: pos(rig, 'Hips'),
                        head: pos(rig, 'Head'),
                        leftHand: pos(rig, 'LeftHand'),
                        rightHand: pos(rig, 'RightHand'),
                        leftToe: pos(rig, 'LeftToeBase'),
                        rightToe: pos(rig, 'RightToeBase'),
                    };
                }
                return out;
            }"""
        )
        print(json.dumps(info, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
