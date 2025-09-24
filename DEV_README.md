快速复现（Windows + PowerShell）

下面的步骤在 Windows + PowerShell 下测试通过，适用于本仓库的开发者快速创建虚拟环境、进行可编辑安装并运行测试。

一键式（推荐，使用 VS Code 任务或 PowerShell 脚本）：

- 使用 VS Code：打开命令面板 -> 运行任务 -> 选择 "Dev: Full setup then test"。

- 使用 PowerShell：在项目根执行：

```pwsh
# 创建虚拟环境并安装依赖、进行可编辑安装并运行测试
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_dev.ps1
# 或执行一键运行测试
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_tests.ps1
```

逐步命令（手动）：

```pwsh
# 在项目根
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest ALNSCode/test -q
```

常见问题与故障排查：

- 如果 `pip install -e .` 报错 `invalid command 'bdist_wheel'`，请先升级 wheel：
  ```pwsh
  .\.venv\Scripts\python.exe -m pip install wheel
  ```

- 如果 pytest 在收集时跑出了第三方包的测试（比如 matplotlib/Tk 的 TclError），请确认使用了项目根的 `pytest.ini` 或运行命令限定测试路径（如示例中的 `ALNSCode/test`）。

- 若你希望 CI 中运行测试：请确保在 CI 中也先执行可编辑安装（`pip install -e .`）或将包安装到环境路径中，并且只收集 `ALNSCode/test`。

其他说明：

- `ALNSCode/__init__.py` 对外导出 `DataALNS` 用于方便导入（`from ALNSCode import DataALNS`）。
- 若要运行某些测试脚本作为脚本（非 pytest），这些脚本仍保留 `main()` 并返回布尔值以支持直接运行退出码。

如果需要，我可以：
- 将这些脚本变更打包成 git 提交（分支名/提交信息按上一次回复建议），并生成 PR 文案。
- 添加 GitHub Actions CI 示例 workflow（安装 .venv、安装 -e .、运行 pytest）。
