@echo off
REM Install llm-sidecar into a local virtualenv (Windows).
REM Safe to re-run: an existing venv is reused and the install refreshed.
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo llm-sidecar - install
echo.

REM ---- Python -----------------------------------------------------------
REM 3.11 is the floor: the codebase uses `X ^| Y` unions at runtime.
set "PYTHON="
for %%P in (python3.13 python3.12 python3.11 python) do (
    where %%P >nul 2>&1 && (
        %%P -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1 && (
            set "PYTHON=%%P"
            goto :found
        )
    )
)

echo   [x] Python 3.11 or newer is required, and I couldn't find one.
echo       Install it from https://python.org/downloads
exit /b 1

:found
for /f "delims=" %%V in ('!PYTHON! -c "import platform;print(platform.python_version())"') do set "PYVER=%%V"
echo   [ok] Python !PYVER! (!PYTHON!)

REM ---- virtualenv --------------------------------------------------------
if exist .venv (
    echo   [ok] Reusing existing .venv
) else (
    !PYTHON! -m venv .venv || exit /b 1
    echo   [ok] Created .venv
)

echo   ... installing dependencies ^(this takes a minute^)
.venv\Scripts\pip.exe install --quiet --upgrade pip
.venv\Scripts\pip.exe install --quiet -e ".[all]" || exit /b 1
for /f "delims=" %%V in ('.venv\Scripts\python.exe -c "import llm_sidecar;print(llm_sidecar.__version__)"') do set "SCVER=%%V"
echo   [ok] Installed llm-sidecar !SCVER!

REM ---- what it can reach -------------------------------------------------
echo.
echo Checking what's available

curl -sf --max-time 2 http://localhost:11434/api/tags >nul 2>&1 && (
    echo   [ok] Ollama is running
) || (
    where ollama >nul 2>&1 && (
        echo   [!] Ollama is installed but not running. Start it, or run: ollama serve
    ) || (
        echo   [!] No Ollama. Local models unavailable - install from https://ollama.com
        echo       or set OPENROUTER_API_KEY to use cloud models instead
    )
)

if defined OPENROUTER_API_KEY (
    echo   [ok] OPENROUTER_API_KEY is set - cloud models available
) else (
    echo   [!] No OpenRouter key - local models only
    echo       add one later in the dashboard, or put it in a .env file here
)

echo.
echo Done. Next:
echo.
echo   run.bat                              start the daemon + dashboard
echo   .venv\Scripts\llm-sidecar status     what it can see
echo.
echo To use it from an MCP client, add:
echo.
echo   {"mcpServers": {"llm-sidecar": {
echo     "command": "%CD%\.venv\Scripts\python.exe",
echo     "args": ["-m", "llm_sidecar.mcp_server"]}}}
echo.
endlocal
