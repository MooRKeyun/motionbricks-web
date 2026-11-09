"""Report renderer statistics and project the character's head into screen space."""

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
        time.sleep(1.0)
        info = page.evaluate(
            """() => {
                const three = window.__three;
                const state = window.__state;
                const actor = state.rigs.actor;
                const v = new three.Vector3();
                actor.byName.get('Head').getWorldPosition(v);
                const ndc = v.clone().project(window.__camera);
                const mesh = actor.skinned;
                return {
                    headWorld: v.toArray().map(n => +n.toFixed(2)),
                    headNDC: ndc.toArray().map(n => +n.toFixed(2)),
                    camera: window.__camera.position.toArray().map(n => +n.toFixed(2)),
                    meshVisible: mesh.visible,
                    parentVisible: mesh.parent ? mesh.parent.visible : null,
                    render: window.__renderer.info.render,
                    memory: window.__renderer.info.memory,
                    actorRootVisible: actor.root.visible,
                    actorScale: actor.root.scale.toArray(),
                };
            }"""
        )
        print(json.dumps(info, indent=2))
        page.evaluate(
            """() => {
                const three = window.__three;
                const m = window.__state.rigs.actor.skinned;
                m.material = new three.MeshBasicMaterial({ color: 0xff3366, side: three.DoubleSide });
            }"""
        )
        time.sleep(0.6)
        page.screenshot(path="tests/shots/debug_red.png")
        print("saved tests/shots/debug_red.png")
        browser.close()


if __name__ == "__main__":
    main()
