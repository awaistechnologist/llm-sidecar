@echo off
REM Start the llm-sidecar daemon and open the dashboard (Windows).
REM Arguments are passed through to `llm-sidecar serve`, e.g. run.bat --no-ui
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\llm-sidecar.exe (
    echo Not installed yet. Run install.bat first.
    exit /b 1
)

set "PORT=%LLM_SIDECAR_PORT%"
if "%PORT%"=="" set "PORT=4001"

echo Starting llm-sidecar - http://localhost:%PORT%
start "" http://localhost:%PORT%
.venv\Scripts\llm-sidecar.exe serve %*
endlocal
