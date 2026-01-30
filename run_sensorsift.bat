@echo off
setlocal

REM Determine Python interpreter
if "%PYTHON%"=="" (
    for %%P in (python python3 py) do (
        where %%P >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_CMD=%%P"
            goto :found_python
        )
    )
) else (
    set "PYTHON_CMD=%PYTHON%"
)

:found_python
if not defined PYTHON_CMD (
    echo Python executable not found on PATH.
    echo Install Python 3.9+ and rerun this script.
    pause
    exit /b 1
)

cd /d "%~dp0"
"%PYTHON_CMD%" "%~dp0snapsift_gui.py"

endlocal
