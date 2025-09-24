<#
PowerShell 脚本：创建虚拟环境并安装可编辑包
用法：在项目根目录下运行：
    powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\setup_dev.ps1
#>

param(
    [string]$venvName = ".venv"
)

Write-Host "创建虚拟环境: $venvName"
python -m venv $venvName

$python = Join-Path -Path $venvName -ChildPath "Scripts\python.exe"

Write-Host "升级 pip / setuptools / wheel"
& $python -m pip install --upgrade pip setuptools wheel

Write-Host "以可编辑模式安装当前包"
& $python -m pip install -e .

Write-Host "完成。你可以使用以下命令运行测试："
Write-Host "  & $python -m pytest ALNSCode/test -q"
