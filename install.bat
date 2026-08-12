@echo off
REM Install llm-sidecar into a local virtualenv (Windows).
REM Safe to re-run: an existing venv is reused and the install refreshed.
REM
REM Note on style: no "!" appears in any echoed text. Delayed expansion is on
REM (needed for !PYTHON!), which makes cmd treat !...! as a variable reference
REM and silently swallow everything between two of them — that turned
REM "[!] No Ollama ... https://ollama.com" into "[//ollama.com".
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo llm-sidecar - install
echo.

REM ---- Python -----------------------------------------------------------
REM 3.11 is the floor: the codebase uses X ^| Y unions at runtime.
set "PYTHON="
for %%P in (python3.13 python3.12 python3.11 python) do (
    where %%P >nul 2>&1 && (
        %%P -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1 && (
            if not defined PYTHON set "PYTHON=%%P"
        )
    )
)

if not defined PYTHON (
    echo   [x] Python 3.11 or newer is required, and none was found.
    echo       Install it from https://python.org/downloads
    exit /b 1
)

for /f "delims=" %%V in ('!PYTHON! -c "import platform;print(platform.python_version())"') do set "PYVER=%%V"
echo   [ok] Python !PYVER! ^(!PYTHON!^)

REM ---- virtualenv --------------------------------------------------------
if exist .venv\Scripts\python.exe (
    echo   [ok] Reusing existing .venv
) else (
    !PYTHON! -m venv .venv || exit /b 1
    echo   [ok] Created .venv
)

set "VPY=.venv\Scripts\python.exe"

echo   ... installing dependencies ^(this takes a minute^)
REM Through python -m pip, not pip.exe: on Windows pip cannot replace its own
REM running executable, and pip.exe --upgrade pip fails outright.
"%VPY%" -m pip install --quiet --upgrade pip
"%VPY%" -m pip install --quiet -e ".[all]" || exit /b 1
for /f "delims=" %%V in ('"%VPY%" -c "import llm_sidecar;print(llm_sidecar.__version__)"') do set "SCVER=%%V"
echo   [ok] Installed llm-sidecar !SCVER!

REM ---- what it can reach -------------------------------------------------
echo.
echo Checking what's available

curl -sf --max-time 2 http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    where ollama >nul 2>&1
    if errorlevel 1 (
        echo   [--] No Ollama. Local models unavailable - install from https://ollama.com
        echo        or set OPENROUTER_API_KEY to use cloud models instead
    ) else (
        echo   [--] Ollama is installed but not running. Start it, or run: ollama serve
    )
) else (
    echo   [ok] Ollama is running
)

if defined OPENROUTER_API_KEY (
    echo   [ok] OPENROUTER_API_KEY is set - cloud models available
) else (
    echo   [--] No OpenRouter key - local models only
    echo        add one later in the dashboard, or put it in a .env file here
)

where docker >nul 2>&1
if errorlevel 1 (
    echo   [--] No Docker. Search falls back to DuckDuckGo
) else (
    echo   [ok] Docker found - "llm-sidecar searxng up" will work
)

REM ---- next steps --------------------------------------------------------
REM Forward slashes in the JSON on purpose: Windows accepts them in paths, and
REM a backslash would need escaping to be valid JSON for the user to paste.
set "JSONPATH=%CD:\=/%/.venv/Scripts/python.exe"

echo.
echo Done. Next:
echo.
echo   run.bat                              start the daemon + dashboard
echo   .venv\Scripts\llm-sidecar status     what it can see
echo.
echo To use it from an MCP client, add:
echo.
echo   {"mcpServers": {"llm-sidecar": {
echo     "command": "%JSONPATH%",
echo     "args": ["-m", "llm_sidecar.mcp_server"]}}}
echo.

REM Reaching here means the install worked. Without this, the last `where`
REM or `curl` check leaks its errorlevel and a successful run exits 1.
endlocal
exit /b 0
