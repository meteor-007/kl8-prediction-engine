@echo off
chcp 65001 >nul
setlocal
title 快乐8 每日自动执行
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 goto :nopython

echo ============================================================
echo    快乐8 每日自动执行：预测 -> 自动提交 -> 推送当前分支
echo ============================================================
echo.
python scripts\daily_run.py %*
if errorlevel 1 goto :failed
exit /b 0

:nopython
echo [错误] 未找到 python 命令，请安装 Python 3.11 并勾选 Add python.exe to PATH。
exit /b 1

:failed
echo.
echo [错误] 每日自动执行失败，请查看 output\logs 下的最新日志。
exit /b 1