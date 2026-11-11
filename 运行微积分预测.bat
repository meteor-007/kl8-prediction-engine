@echo off
chcp 65001 >nul
setlocal
title 快乐8 集成预测系统
cd /d "%~dp0"

echo ============================================================
echo    快乐8 (Keno) 深度学习 + 图谱贝叶斯集成预测系统  [日常预测]
echo ============================================================
echo.
where python >nul 2>nul
if errorlevel 1 goto :nopython

echo [1/2] 自动同步最新开奖数据，执行完整预测（含复盘/点位/分模块明细）...
echo.
python main.py --full-print
if errorlevel 1 goto :failed

echo.
echo [2/2] 完成。交付件位置：
echo     output\reports     TXT 精要报告（含复盘/点位/分模块明细）
echo     output\wechat      微信发布文案
echo     output\excel       商业 Excel 报表
echo     output\shrinkage   30码缩水专报
echo     output\state       推荐日志（供次日复盘）
echo.
echo 其他常用命令（复制到 PowerShell 手动执行）：
echo     python main.py --no-sync --full-print
echo     python main.py --full-print --points "07 08 13 16 17 18 22 28 32 33 37 38 40 42 43 45 47 48 56 67"
echo     python main.py --full-print --points "07 08 13 16" --points-mode boost
echo     python scripts/backtest_evaluation.py --periods 30 --epochs 10
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