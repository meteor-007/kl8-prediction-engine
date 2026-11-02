@echo off
chcp 65001 >nul
setlocal
title 快乐8 五层动力学预测系统
cd /d "%~dp0"

echo ============================================================
echo    快乐8 (Keno) 五层动力学预测系统  [日常预测]
echo ============================================================
echo.
where python >nul 2>nul
if errorlevel 1 goto :nopython

echo [1/2] 正在执行日常预测：跳过联网同步，使用本地历史数据，伴随 30 期回测...
echo.
python main.py --no-sync --backtest 30 --full-print
if errorlevel 1 goto :failed

echo.
echo [2/2] 完成。交付件位置：
echo     output\reports   TXT 精要报告
echo     output\wechat    微信发布文案
echo     output\excel     商业 Excel 报表
echo     output\state     自适应权重状态
echo.
echo 其他常用命令（复制到 PowerShell 手动执行）：
echo     python main.py --no-sync --mode backtest --backtest 100
echo     python main.py --no-sync --mode audit --backtest 30
echo     python main.py --no-sync --ml --backtest 0
echo.
pause
exit /b 0

:nopython
echo [错误] 未找到 python 命令，请安装 Python 3.11 并勾选 Add python.exe to PATH。
pause
exit /b 1

:failed
echo.
echo [错误] 预测流程返回非 0，请查看上方错误信息后重试。
pause
exit /b 1
