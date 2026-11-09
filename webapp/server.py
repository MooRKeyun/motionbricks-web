# SPDX-License-Identifier: Apache-2.0
"""HTTP service that drives MotionBricks and serves the browser front end."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .motion import MotionService

LOGGER = logging.getLogger("webapp.server")
STATIC_DIR = Path(__file__).resolve().parent / "static"


class StatusHub:
    """Broadcast status events from the generation thread to websocket clients."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.last: dict = {"state": "starting", "message": "等待启动", "progress": 0.0}

    def publish(self, event: dict) -> None:
        self.last = event
        loop = self.loop
        if loop is None or not self.clients:
            return
        payload = {**event, "type": "status"}
        for client in list(self.clients):
            asyncio.run_coroutine_threadsafe(self._send(client, payload), loop)

    async def _send(self, client: WebSocket, payload: dict) -> None:
        try:
            await client.send_json(payload)
        except Exception:  # noqa: BLE001 - a dropped socket should not stop the broadcast
            self.clients.discard(client)


HUB = StatusHub()
SERVICE = MotionService(status_callback=HUB.publish)


@asynccontextmanager
async def lifespan(app: FastAPI):
    HUB.loop = asyncio.get_running_loop()
    threading.Thread(target=SERVICE.load, name="motionbricks-load", daemon=True).start()
    LOGGER.info("MotionBricks service loading in the background")
    yield


app = FastAPI(title="MotionBricks Web", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
async def favicon() -> JSONResponse:
    return JSONResponse(status_code=204, content=None)


@app.get("/api/status")
async def status() -> JSONResponse:
    return JSONResponse(SERVICE.status())


@app.get("/api/scene")
async def scene() -> JSONResponse:
    if SERVICE.state != "ready":
        raise HTTPException(status_code=503, detail=SERVICE.status())
    return JSONResponse(SERVICE.scene_info())


@app.get("/api/pose/{key}")
async def pose(key: str) -> JSONResponse:
    if SERVICE.state != "ready":
        raise HTTPException(status_code=503, detail=SERVICE.status())
    try:
        return JSONResponse(SERVICE.preview_pose(key))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/generate")
async def generate(request: dict) -> JSONResponse:
    if SERVICE.state != "ready":
        raise HTTPException(status_code=503, detail=SERVICE.status())
    try:
        result = await asyncio.to_thread(SERVICE.generate, request)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"缺少参数: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(result)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    HUB.clients.add(websocket)
    try:
        await websocket.send_json({**SERVICE.status(), "type": "status"})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        HUB.clients.discard(websocket)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
