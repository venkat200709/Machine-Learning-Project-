@echo off
REM ══════════════════════════════════════════════════════════════════
REM  RiskRadar launcher for Windows
REM  Double-click this file, or run  start.bat  from CMD.
REM
REM  Handles the two things that actually go wrong on Windows:
REM    1. `python` is not on PATH but the `py` launcher is
REM    2. an older server is still holding port 8000
REM ══════════════════════════════════════════════════════════════════
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title RiskRadar Server

echo.
echo  ================================================================
echo    RiskRadar - AI Women's Safety Intelligence Platform
echo  ================================================================
echo.

REM ---- 1. Find a working Python -----------------------------------
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)

if not defined PY (
    echo  [X] Python was not found.
    echo.
    echo      Install Python 3.10 or newer from https://python.org
    echo      and tick "Add Python to PATH" during setup.
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%v in ('%PY% -c "import sys;print(sys.version.split()[0])"') do set "PYVER=%%v"
echo  Python !PYVER! found  ^(using: !PY!^)

REM ---- 2. Dependencies --------------------------------------------
%PY% -c "import fastapi, uvicorn, sklearn, lightgbm, joblib" >nul 2>&1
if errorlevel 1 (
    echo  Installing dependencies, this may take a few minutes...
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo  [X] Dependency install failed. Try running this manually:
        echo      %PY% -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
) else (
    echo  Dependencies OK
)

REM ---- 3. Launch ---------------------------------------------------
echo.
echo  Starting the server. KEEP THIS WINDOW OPEN.
echo  Press Ctrl+C here to stop it.
echo.
%PY% run.py %*
set "RC=%errorlevel%"

echo.
if not "%RC%"=="0" (
    echo  ================================================================
    echo   The server stopped with an error ^(code %RC%^).
    echo   Run this in another window to find out why:
    echo       %PY% check.py
    echo  ================================================================
) else (
    echo  Server stopped.
)
echo.
pause
endlocal
