@echo off
rem One-click launcher: start the MotionBricks web service and open the browser.
setlocal
cd /d "%~dp0"

set "PYTHON=C:\Users\13119\anaconda3\envs\motionbricks\python.exe"
if not exist "%PYTHON%" (
  echo [ERROR] conda env motionbricks not found at:
  echo         %PYTHON%
  echo         Create it as described in README.md, or edit PYTHON in this file.
  pause
  exit /b 1
)

if not exist "webapp\static\soma.glb" (
  echo [1/2] Exporting the SOMA character to GLB ...
  "%PYTHON%" -m webapp.glb_export
)

echo [2/2] Serving http://127.0.0.1:8000
start "" "http://127.0.0.1:8000/"
"%PYTHON%" -m uvicorn webapp.server:app --host 127.0.0.1 --port 8000
