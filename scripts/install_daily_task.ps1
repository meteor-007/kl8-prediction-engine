param(
    [string]$Time = "18:00",
    [string]$TaskName = "KL8-Daily-Prediction"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$bat = Join-Path $root "每日自动执行.bat"
if (-not (Test-Path $bat)) {
    throw "未找到批处理文件: $bat"
}

$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$bat`"" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "KL8 每日预测：读取指令点位、运行双融合预测、自动提交并推送" `
    -Force | Out-Null

Write-Host "已注册计划任务: $TaskName"
Write-Host "执行时间: 每天 $Time（错过计划时间会在下次可用时补跑）"
Write-Host "执行内容: $bat"
Write-Host "卸载: powershell -ExecutionPolicy Bypass -File scripts\uninstall_daily_task.ps1"