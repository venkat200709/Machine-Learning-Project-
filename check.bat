@echo off
REM Diagnose a running RiskRadar server. Double-click while start.bat is open.
setlocal
cd /d "%~dp0"
title RiskRadar Diagnostic

set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY python --version >nul 2>&1 && set "PY=python"

if not defined PY (
    echo  [X] Python was not found on this machine.
    pause
    exit /b 1
)

%PY% check.py %*
echo.
pause
endlocal
