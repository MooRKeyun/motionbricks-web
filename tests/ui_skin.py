"""Evaluate CPU-side skinning of the loaded mesh to find why it is not visible."""

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
                const mesh = window.__state.rigs.actor.skinned;
                mesh.updateMatrixWorld(true);
                const pos = mesh.geometry.attributes.position;
                const v = new three.Vector3();
                const samples = [];
                let bad = 0, minZ = 1e9, maxZ = -1e9, minX=1e9, maxX=-1e9, minY=1e9, maxY=-1e9;
                for (let i = 0; i < pos.count; i++) {
                    v.fromBufferAttribute(pos, i);
                    mesh.applyBoneTransform(i, v);
                    v.applyMatrix4(mesh.matrixWorld);
                    if (!Number.isFinite(v.x) || !Number.isFinite(v.y) || !Number.isFinite(v.z)) { bad++; continue; }
                    minX = Math.min(minX, v.x); maxX = Math.max(maxX, v.x);
                    minY = Math.min(minY, v.y); maxY = Math.max(maxY, v.y);
                    minZ = Math.min(minZ, v.z); maxZ = Math.max(maxZ, v.z);
                    if (i % 4000 === 0) samples.push([i, +v.x.toFixed(2), +v.y.toFixed(2), +v.z.toFixed(2)]);
                }
                const jIdx = mesh.geometry.attributes.skinIndex;
                const jWei = mesh.geometry.attributes.skinWeight;
                return {
                    samples, bad,
                    skinBox: [[+minX.toFixed(2), +minY.toFixed(2), +minZ.toFixed(2)],
                              [+maxX.toFixed(2), +maxY.toFixed(2), +maxZ.toFixed(2)]],
                    jointSample: [Array.from(jIdx.array.slice(0, 8)), Array.from(jWei.array.slice(0, 8))],
                    boneMatrix0: Array.from(mesh.skeleton.boneMatrices.slice(0, 16)).map(v => +v.toFixed(3)),
                };
            }"""
        )
        print(json.dumps(info, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
