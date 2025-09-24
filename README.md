# Problem2 — 本地开发与测试指南

简短说明：本仓库实现了多周期产品配送问题的 ALNS/CG/Monolithic 算法实现。下面说明如何在本地（Windows + PowerShell）创建虚拟环境、进行可编辑安装（editable install）并运行测试。

> 目标：使用项目内的 `.venv`、让 `ALNSCode` 包在该环境可导入、并用 `pytest` 运行本仓库的测试。

## 1. 系统与前提
- 操作系统：Windows（PowerShell）
- 建议 Python >= 3.10（与 `pyproject.toml` 要求一致）
- 需要按需安装 Gurobi（如果你要运行依赖 gurobipy 的模块/测试）

## 2. 在项目根创建并激活 `.venv`（PowerShell）
在仓库根（本 README 所在目录）运行：

```powershell
# 创建虚拟环境（只需执行一次）
python -m venv .venv

# 激活虚拟环境（每次打开终端时都要执行）
& .\.venv\Scripts\Activate.ps1
```

激活后，提示符通常会在前缀显示 `(Problem2)` 或 `(.venv)`。

## 3. 升级 pip / 安装 wheel（可选，但推荐）
```powershell
pip install --upgrade pip
pip install wheel
```

如果在可编辑安装时报错 `invalid command 'bdist_wheel'`，通常是因为缺少 `wheel`，按上面命令安装即可。

## 4. 在虚拟环境中以 editable 模式安装项目
在激活的 `.venv` 下运行：

```powershell
pip install -e .
```

这会把项目注册为可编辑包（在该环境中 `import ALNSCode` 将生效）。可编辑安装的包名由 `pyproject.toml` 的 `name` 字段决定（当前为 `problem2`）。

卸载 editable 包：
```powershell
pip uninstall problem2
```

## 5. 运行测试（pytest）
本仓库已包含 `pytest.ini`，默认只收集 `ALNSCode/test` 目录下的测试，以避免第三方包自带测试干扰。

运行全部测试：
```powershell
pytest -q
```

运行单个测试文件或测试用例：
```powershell
# 单个文件
pytest ALNSCode/test/test_initial_solution.py -q

# 运行单个测试函数
pytest ALNSCode/test/test_initial_solution.py::test_initial_solution -q
```

## 6. 常见问题与排查
- ModuleNotFoundError: No module named 'ALNSCode'
  - 原因：未在当前 Python 环境中安装项目，也没有把项目根加入 `PYTHONPATH`。
  - 解决：激活 `.venv` 并执行 `pip install -e .`，或在测试时确保 `conftest.py`/环境把项目根加入 `sys.path`。

- bdist_wheel / wheel 错误
  - 解决：在 venv 中 `pip install wheel` 后重试 `pip install -e .`。

- matplotlib / Tk / TclError（在无 GUI 的 CI 或 headless 环境）
  - 有些绘图测试会尝试打开图形后端（Tk），在没有系统 Tcl/Tk 的环境会报错。可用两种策略：
    1) 在测试前设置后端为 Agg（无头）：在测试入口（`conftest.py`）设置：
       ```python
       import matplotlib
       matplotlib.use('Agg')
       ```
    2) 或在系统层面安装 Tcl/Tk（Windows 可安装 tcl/tk 支持）以满足 tkinter 运行需求。

- pytest 警告：Test functions should return None
  - 一些测试以 `return True/False` 作为结果，这会触发 pytest 的样式警告。建议把这些测试改为使用 `assert`，例如 `assert myfunc(...) is True`。

## 7. 开发与贡献小贴士
- 修改代码后可直接在激活的 `.venv` 下运行 `pytest -q`，因为采用了可编辑安装，改动会即时生效。
- 若你在本地调试数据依赖（datasets），确保 `datasets/multiple-periods` 对应的数据集已经存在，部分测试在缺数据时会跳过或模拟处理。

## 8. 若需要我代劳
- 我可以帮你：
  - 自动把测试中返回 bool 的地方（return -> assert）替换并运行一次全套测试；
  - 把 matplotlib 后端安全地设置为 Agg（在 `ALNSCode/test/conftest.py` 中加入设置），以便在无 GUI 环境稳定运行；
  - 生成更完整的 CONTRIBUTING.md 或开发者文档。

## Developer Guide (开发者指南)

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

## 关于仓内 ALNS 包 (本地依赖说明)

本仓库包含一个名为 `ALNS/` 的本地包（历史上曾经作为 git 子模块），目前作为仓内可编辑的依赖保留在仓库中。该决策的目的与注意事项：

- 目的：在开发与回归测试时，我们需要一个确定且可修改的 ALNS 实现（例如对算法改动或 bug 修复），把它放在仓库内可以保证 CI 与本地开发环境使用一致的代码版本。
- CI 行为：CI 会优先尝试 `pip install -e ALNS`（可编辑安装），以便在需要时直接使用/调试仓内的 ALNS 源代码；如果仓库没有 `ALNS/` 目录，CI 会回退安装 PyPI 上的 `alns` 包。
- 开发者指南：若你在 `ALNS/` 中做了修改并希望在 CI 中生效，请确保同时更新 `ALNS/pyproject.toml`（或其他打包元数据），并在提交中说明对内嵌 ALNS 的改变。
- 兼容性：CI 保留对 PyPI 的回退安装以便外部贡献者不必总是携带仓内复制品；如果你希望完全依赖 PyPI，请移除仓内 `ALNS/` 并在根 `pyproject.toml` 中锁定合适的版本。

常见命令（本地开发）

```pwsh
# 在项目虚拟环境中安装仓内 ALNS（可编辑）
pip install -e ALNS

# 如果你只想使用 PyPI 版本
pip install alns
```

---
若你希望我现在把 README.md 提交（已写入仓库），或继续做上述任一项（如修复测试 return/警告 或 改后端为 Agg），告诉我具体选项即可，我会马上执行并报告结果。
