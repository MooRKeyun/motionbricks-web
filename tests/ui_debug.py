"""Inspect the live three.js scene in the browser for debugging."""

from __future__ import annotations

import json
import time

from playwright.sync_api import sync_playwright


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--use-angle=swiftshader"])
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        failures: list[str] = []
        page.on("requestfailed", lambda request: failures.append(f"FAILED {request.url}"))
        page.on("response", lambda response: failures.append(f"{response.status} {response.url}") if response.status >= 400 else None)
        page.on("console", lambda message: print("[console]", message.type, message.text))
        page.on("pageerror", lambda error: print("[pageerror]", error))

        page.goto("http://127.0.0.1:8000/", wait_until="networkidle")
        page.wait_for_function(
            "() => document.getElementById('generate') && !document.getElementById('generate').disabled",
            timeout=120_000,
        )
        time.sleep(1.5)

        info = page.evaluate(
            """() => {
                const out = {meshes: [], bones: 0, sceneChildren: 0};
                const walk = (node, depth) => {
                    if (!node) return;
                    if (node.isMesh || node.isSkinnedMesh) {
                        const geo = node.geometry;
                        const box = new (node.material.color.constructor === Object ? Object : Object)();
                        out.meshes.push({
                            name: node.name,
                            type: node.type,
                            visible: node.visible,
                            material: node.material.type,
                            color: node.material.color ? node.material.color.getHexString() : null,
                            opacity: node.material.opacity,
                            hasSkinIndex: !!(geo.attributes.skinIndex),
                            vertices: geo.attributes.position ? geo.attributes.position.count : 0,
                            boneCount: node.isSkinnedMesh ? node.skeleton.bones.length : 0,
                            parent: node.parent ? node.parent.name : null,
                        });
                    }
                    if (node.isSkinnedMesh) out.bones = node.skeleton.bones.length;
                    out.sceneChildren += 1;
                    node.children.forEach((child) => walk(child, depth + 1));
                };
                const canvas = document.getElementById('viewport');
                walk(window.__scene || null, 0);
                return out;
            }"""
        )
        print(json.dumps(info, indent=2, ensure_ascii=False))
        print("network problems:")
        for failure in failures:
            print(" ", failure)
        browser.close()


if __name__ == "__main__":
    main()
