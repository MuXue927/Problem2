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

---
若你希望我现在把 README.md 提交（已写入仓库），或继续做上述任一项（如修复测试 return/警告 或 改后端为 Agg），告诉我具体选项即可，我会马上执行并报告结果。
