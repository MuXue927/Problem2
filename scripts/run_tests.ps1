<#
PowerShell: 运行项目测试（只运行 ALNSCode/test 下的测试）
用法：
  powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\run_tests.ps1
#>

$venvPython = ".\.venv\Scripts\python.exe"
if (-Not (Test-Path $venvPython)) {
    Write-Host ".venv 未找到，请先运行 scripts\\setup_dev.ps1 或手动创建虚拟环境。"
    exit 1
}

Write-Host "运行 pytest ALNSCode/test"
& $venvPython -m pytest ALNSCode/test -q
