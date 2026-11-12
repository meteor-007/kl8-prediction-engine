@echo off
chcp 65001 >nul
setlocal
title KL8 Daily Auto Run
cd /d "%~dp0"
if not exist "output\logs" mkdir "output\logs"

rem Locate Python: python first, then Windows py launcher, then common install paths
set "PY_EXE="
set "PY_ARGS="

where python >nul 2>nul
if not errorlevel 1 set "PY_EXE=python"

if not defined PY_EXE (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PY_EXE=py"
        set "PY_ARGS=-3"
    )
)

if not defined PY_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY_EXE if exist "C:\Python311\python.exe" set "PY_EXE=C:\Python311\python.exe"
if not defined PY_EXE if exist "C:\Program Files\Python311\python.exe" set "PY_EXE=C:\Program Files\Python311\python.exe"

if not defined PY_EXE goto :nopython

rem Ensure git is on PATH for scheduled task environments
where git >nul 2>nul
if errorlevel 1 set "PATH=C:\Program Files\Git\cmd;%PATH%"

echo ============================================================
echo    KL8 Daily Auto Run: predict -^> commit -^> push
echo ============================================================
echo.
"%PY_EXE%" %PY_ARGS% scripts\daily_run.py %*
if errorlevel 1 goto :failed
exit /b 0

:nopython
echo [ERROR] Python not found. Install Python 3.11 or ensure the py launcher is available.
echo [%date% %time%] python not found >> "%~dp0output\logs\task_launcher.log"
exit /b 1

:failed
echo.
echo [ERROR] Daily run failed. See the latest log under output\logs.
echo [%date% %time%] daily_run failed, exit code %errorlevel% >> "%~dp0output\logs\task_launcher.log"
exit /b 1