@echo off
REM Start the llm-sidecar daemon and open the dashboard (Windows).
REM Arguments pass through to `llm-sidecar serve`, e.g. run.bat --no-ui
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\llm-sidecar.exe (
    echo Not installed yet. Run install.bat first.
    exit /b 1
)

REM A .env here is a convenience for keys you would rather not put in your
REM environment. Only NAME=value lines are read.
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        echo %%A | findstr /r "^[A-Za-z_][A-Za-z0-9_]*$" >nul 2>&1 && (
            if not defined %%A set "%%A=%%B"
        )
    )
)

set "PORT=%LLM_SIDECAR_PORT%"
if "%PORT%"=="" set "PORT=4001"

REM Don't open a browser when the dashboard is switched off — nothing to show.
echo %* | findstr /c:"--no-ui" >nul 2>&1
if errorlevel 1 (
    echo Starting llm-sidecar - http://localhost:%PORT%
    start "" http://localhost:%PORT%
) else (
    echo Starting llm-sidecar - API only on port %PORT%
)

.venv\Scripts\llm-sidecar.exe serve %*
endlocal
