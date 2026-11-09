"""Check where the loaded SOMA bones end up in the browser scene."""

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
                const state = window.__state;
                const actor = state.rigs.actor;
                const three = window.__three;
                const v = new three.Vector3();
                const sample = (name) => {
                    const bone = actor.byName.get(name);
                    if (!bone) return null;
                    bone.getWorldPosition(v);
                    return [name, +v.x.toFixed(3), +v.y.toFixed(3), +v.z.toFixed(3)];
                };
                const mesh = actor.skinned;
                mesh.updateMatrixWorld(true);
                const geo = mesh.geometry;
                geo.computeBoundingBox();
                const bb = geo.boundingBox;
                return {
                    bones: ['Hips','Head','LeftFoot','RightFoot','LeftHand','RightHand'].map(sample),
                    localRoot: actor.bones[0].position.toArray(),
                    worldRoot: (() => { actor.bones[0].getWorldPosition(v); return v.toArray(); })(),
                    rootQuat: actor.bones[0].quaternion.toArray(),
                    geometryBox: [bb.min.toArray(), bb.max.toArray()],
                    bindMatrix: mesh.bindMatrix.elements.slice(12, 15),
                    skeletonBone0: mesh.skeleton.bones[0].name,
                };
            }"""
        )
        print(json.dumps(info, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
