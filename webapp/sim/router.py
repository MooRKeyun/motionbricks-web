"""Routes for the day simulation page: facility geometry, the timeline, and clips."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

DATA_DIR = Path(__file__).resolve().parent / "data"
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

router = APIRouter()

_cache: dict[str, object] = {}


def _load(name: str) -> dict:
    if name not in _cache:
        path = DATA_DIR / name
        if not path.is_file():
            raise HTTPException(status_code=503, detail=f"{name} 还没生成，先运行 python -m webapp.sim.day")
        _cache[name] = json.loads(path.read_text(encoding="utf-8"))
    return _cache[name]  # type: ignore[return-value]


@router.get("/day")
async def day_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "day.html")


@router.get("/api/day/facility")
async def facility() -> JSONResponse:
    return JSONResponse(_load("facility.json"))


@router.get("/api/day/day")
async def day() -> JSONResponse:
    return JSONResponse(_load("day.json"))


@router.get("/api/day/clips")
async def clips() -> JSONResponse:
    path = DATA_DIR / "clips.json"
    if not path.is_file():
        return JSONResponse({"clips": [], "built": False})
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@router.get("/api/day/clips.bin")
async def clips_bin() -> FileResponse:
    path = DATA_DIR / "clips.bin"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="clips.bin 还没生成")
    return FileResponse(path, media_type="application/octet-stream")


@router.get("/api/day/staff")
async def staff() -> JSONResponse:
    path = DATA_DIR / "staff.json"
    if not path.is_file():
        return JSONResponse({"agents": []})
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@router.get("/api/day/scenario")
async def scenario() -> JSONResponse:
    path = DATA_DIR / "scenario.json"
    if not path.is_file():
        return JSONResponse({"chestPain": None})
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
