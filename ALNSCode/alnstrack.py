"""
模块: alnstrack
功能定位:
    提供 ALNS 求解过程中的统一跟踪(Tracking)与度量(Statistics)支持, 包含:
      1. 迭代级记录: 当前迭代号 / 当前目标值 / 历史最优值 / gap 序列 / 用时
      2. 最优解持久化: 自动保存迭代中发现的改进解 (深拷贝状态)
      3. 度量导出: 可选写入 CSV (便于后续分析/可视化/实验比较)
      4. ML 支持: 存储特征(feature)与标签(label)样本、训练后的模型与 scaler 缓存
      5. 外部交互: ParamAutoTuner 读取迭代统计 (total_iterations / max_iterations) 做阶段自适应

设计要点:
    - 轻量: 不介入搜索决策逻辑, 仅消费 state (SolutionState) 提供的接口
    - 解耦: ML/参数调优模块通过 get_statistics()/get_ml_cache_info() 获取所需信息
    - 容错: CSV 写入失败不应中断主流程 (可扩展 try/except)
    - 历史裁剪: (可选) 当前特征样本列表 features/labels 维持上限, 防止长期运行内存膨胀
    - 早停: 维护 no_improve_iters (自上次取得最优以来的迭代数) 与 early_stop_patience (容忍阈值)
            外部可在主循环中调用 tracker.should_stop() 判断是否提前终止

Gap 定义:
    gap = (current_obj - best_obj) / current_obj * 100%
    说明:
      - 目标为最小化场景; 当出现新最优时 current_obj == best_obj → gap = 0
      - 该实现假设 best_obj ≤ current_obj (若出现精度波动导致 best_obj > current_obj,
        gap 仍为正数; 如需严格单调可在更新前加 min/epsilon 保护)

使用流程参考:
    tracker = ALNSTracker(output_file='logs/alns_progress.csv')
    for iter in range(MAX):
        ... # 生成 / 接受新解
        tracker.on_iteration(state, rng)
    stats = tracker.get_statistics()

可改进点(未实现):
    - 引入滑动窗口/指数加权的 gap 平滑指标
    - 增加 JSON 行日志输出用于更丰富的后处理
"""

# =========================
# 标准库
# =========================
import os
import csv
import time
import copy
from typing import Optional, Any, Tuple, TYPE_CHECKING

# =========================
# 第三方库
# =========================
import numpy as np
import numpy.random as rnd

# =========================
# 项目内部依赖
# =========================
from .optutility import LogPrinter
from .alns_config import default_config as ALNSConfig

if TYPE_CHECKING:  # 仅类型检查时导入, 避免循环依赖与运行时开销
    from .alnsopt import SolutionState

# 全局日志打印器 (带启动时间基准)
log_printer = LogPrinter(time.time())


class ALNSTracker:
    """
    ALNS 过程跟踪器:
    负责记录优化搜索中的关键统计量, 支撑:
      - 结果分析 (目标值/Gap 曲线)
      - 阶段判断 (与 ParamAutoTuner 协调)
      - 机器学习算子特征/标签累积与模型缓存
    """

    def __init__(self, output_file: Optional[str] = None):
        # 迭代/目标相关
        self.iteration: int = 0
        self.max_iterations: int = getattr(ALNSConfig, 'MAX_ITERATIONS', 1000)
        self.current_obj: float = float("inf")
        self.best_obj: float = float("inf")
        self.best_solution: Optional["SolutionState"] = None

        # 时间 & 历史序列
        self.start_time: float = time.time()
        self.objectives: list[float] = []
        self.gaps: list[float] = []

        # 可选结果持久化
        self.output_file: Optional[str] = output_file
        if self.output_file:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Iteration", "Current_Obj", "Best_Obj", "Gap"])

        # ML 相关 (learning_based_repair / 未来学习机制)
        self.features: list[list[float]] = []  # 每条特征: 例如 [demand, size, day, inv, util, ...]
        self.labels: list[float] = []          # 对应改进值 improvement
        self._cached_model: Optional[Any] = None
        self._cached_scaler: Optional[Any] = None
        self._last_train_iteration: Optional[int] = None

        # 早停机制相关
        self.early_stop_patience: int = 500          # 可在外部根据问题规模/期望运行时间调节
        self.no_improve_iters: int = 0               # 自上次最优改进以来未改善的迭代计数
        self.best_update_iteration: int = 0          # 最近一次刷新最优解的迭代号 (用于分析改进分布)
        self.recent_improvement: float = 0.0         # 最近一次改进幅度 (prev_best - new_best); 未改进迭代置 0

    # -----------------------------------------------------------------
    # 主迭代回调
    # -----------------------------------------------------------------
    def on_iteration(self, state: "SolutionState", rng: rnd.Generator, **kwargs):
        """
        每次 ALNS 主循环结束后调用, 执行:
          1. 迭代号自增
          2. 获取当前解目标值 (内部会利用 objective 缓存)
          3. 若为可行且优于历史最优则更新 best_solution
          4. 计算并记录 gap
          5. 可选写入 CSV (频繁 IO 可根据需要做采样)
          6. 周期性打印进度 (当前: 每 100 次)

        参数:
            state : 当前解 (需提供 objective()/validate())
            rng   : 随机数生成器 (保留扩展接口; 当前未使用)
            kwargs: 预留扩展 (例如可传入附加日志/标记)
        """
        self.iteration += 1

        # 在每次迭代开始时同步更新 ParamAutoTuner 的迭代信息，
        # 以确保随后对算子参数的采样(get_operator_params)能够读取到最新的 exploration_rate。
        try:
            if hasattr(state, 'param_tuner') and state.param_tuner and hasattr(self, 'max_iterations'):
                current_iter = self.iteration
                max_iter = self.max_iterations
                state.param_tuner.set_iteration(current_iter, max_iter)
        except Exception as e:
            # 保护性容错：若更新失败仅打印警告，不影响主流程
            log_printer.print(f"[WARN] Failed to update param_tuner iteration: {e}", color='yellow')

        # 当前目标值 (若 objective 缓存有效则为 O(1))
        current_obj = state.objective()
        self.current_obj = current_obj
        self.objectives.append(current_obj)

        # 验证可行性 & 更新最优
        is_feasible, _ = state.validate()
        improved = False
        if current_obj < self.best_obj and is_feasible:
            prev_best = self.best_obj
            self.best_obj = current_obj
            # 深拷贝保存最优状态 (避免后续原地修改污染)
            try:
                # 优先使用 SolutionState.copy() 以获取轻量快照
                self.best_solution = state.copy()
            except Exception:
                # 兼容旧实现：若对象未实现 copy() 回退到 deepcopy
                self.best_solution = copy.deepcopy(state)
            improved = True
            # 记录本次实际改进幅度 (若之前 best 为 inf 则按 0 处理)
            if prev_best != float("inf"):
                self.recent_improvement = max(0.0, prev_best - current_obj)
            else:
                self.recent_improvement = 0.0

            gap_new_best = calculate_gap(current_obj, self.best_obj)  # 恒为 0
            elapsed_time = time.time() - self.start_time
            log_printer.print(
                f"Iteration {self.iteration}: New best feasible solution found!",
                color="bold green",
            )
            log_printer.print(
                f"Objective: {self.best_obj:.6f}, Gap: {gap_new_best:.2f}%, Time: {elapsed_time:.2f}s",
                color="bold green",
            )

        # 维护无改进计数 (早停)
        if improved:
            self.no_improve_iters = 0
            self.best_update_iteration = self.iteration
        else:
            self.no_improve_iters += 1
            # 未取得新最优时 recent_improvement 置 0，表示“最近一次迭代”无改进
            self.recent_improvement = 0.0

        # 计算当前 gap (使用更新后的 best_obj)
        gap = calculate_gap(self.current_obj, self.best_obj)
        self.gaps.append(gap)

        # 周期性打印概览
        if self.iteration % 100 == 0:
            log_printer.print(
                f"Iteration: {self.iteration}\tCurrent Obj: {self.current_obj:.6f}\t"
                f"Best Obj: {self.best_obj:.6f}\tGap: {gap:.2f}%"
            )

        # 可选持久化
        if self.output_file:
            try:
                with open(self.output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([self.iteration, self.current_obj, self.best_obj, gap])
            except Exception as e:
                log_printer.print(f"[WARN] Failed to append tracker CSV: {e}", color="yellow")

    # -----------------------------------------------------------------
    # ML 数据采集
    # -----------------------------------------------------------------
    def update_ml_data(self, feature: list, label: float) -> Tuple[int, int]:
        """
        存储一条 (feature, label) 样本。
        - feature: 自由结构 (约定维度在使用侧保持一致)
        - label:   通常为目标函数改进值 improvement (正数为好)

        返回:
            (当前特征样本数, 当前标签样本数)
        说明:
            这里设置一个硬上限 1000 (最近样本), 防止长时间运行内存膨胀。
            可扩展为基于时间/多级缓冲的采样策略。
        """
        self.features.append(feature)
        self.labels.append(label)
        if len(self.features) > 1000:  # 滑动窗口机制
            self.features = self.features[-1000:]
            self.labels = self.labels[-1000:]
        return len(self.features), len(self.labels)

    # -----------------------------------------------------------------
    # ML 模型缓存接口
    # -----------------------------------------------------------------
    def cache_ml_model(self, model: Any, scaler: Any, iteration: int):
        """
        缓存训练完成的 ML 模型及其标准化器, 并记录训练迭代号:
            model   : 已训练模型 (如 RandomForest / Ridge / 自定义)
            scaler  : 与 model 对应的特征预处理器
            iteration: 训练时对应的迭代编号
        用途:
            - 在后续迭代快速复用模型 (避免频繁重训)
            - 外部可据 _last_train_iteration 判断是否需要增量更新
        """
        self._cached_model = model
        self._cached_scaler = scaler
        self._last_train_iteration = iteration
        print(f"[Tracker] Cached ML model at iteration {iteration}")

    def get_cached_model(self) -> Tuple[Optional[Any], Optional[Any], Optional[int]]:
        """
        返回缓存 (model, scaler, last_train_iteration) 三元组。
        若任一为空, 表示尚未缓存或已被清除。
        """
        return self._cached_model, self._cached_scaler, self._last_train_iteration

    def has_cached_model(self) -> bool:
        """
        判断当前是否存在有效的模型缓存。
        条件: model, scaler, last_train_iteration 全部非 None。
        """
        return (
            self._cached_model is not None
            and self._cached_scaler is not None
            and self._last_train_iteration is not None
        )

    def clear_ml_cache(self):
        """
        清空模型缓存 (释放内存 / 触发重训)。
        """
        self._cached_model = None
        self._cached_scaler = None
        self._last_train_iteration = None
        print("[Tracker] Cleared ML model cache")

    def get_ml_cache_info(self) -> dict:
        """
        返回当前 ML 缓存状态摘要:
            has_model: 是否有缓存
            last_train_iteration: 上次训练迭代
            model_type / scaler_type: 类名 (便于调试)
            data_samples: 当前特征样本数量
        """
        return {
            "has_model": self.has_cached_model(),
            "last_train_iteration": self._last_train_iteration,
            "model_type": type(self._cached_model).__name__ if self._cached_model else None,
            "scaler_type": type(self._cached_scaler).__name__ if self._cached_scaler else None,
            "data_samples": len(self.features),
        }

    # -----------------------------------------------------------------
    # 早停判定
    # -----------------------------------------------------------------
    def should_stop(self) -> bool:
        """
        早停判定:
          若自上次取得最优解以来的未改进迭代数 >= early_stop_patience → 返回 True
        用法 (伪代码):
            while not tracker.should_stop():
                ...
        """
        return self.no_improve_iters >= self.early_stop_patience

    # -----------------------------------------------------------------
    # 汇总统计
    # -----------------------------------------------------------------
    def get_statistics(self) -> dict:
        """
        汇总当前追踪器的所有关键信息 (返回浅复制/值复制, 避免外部原地修改):
            total_iterations :     已完成迭代数
            max_iterations   :     规划的最大迭代数
            best_objective   :     历史最优目标值
            current_objective:     当前解目标值
            final_gap        :     当前 gap (若尚无历史则为 0)
            objectives_history:    目标值轨迹 (副本)
            gaps_history     :     gap 轨迹 (副本)
            elapsed_time     :     运行秒数
            best_solution    :     当前最优解 (对象引用; 深度使用需自行 deepcopy)
            features / labels:     ML 采样数据副本
            ml_cache_info    :     模型缓存摘要
            no_improve_iters :     自上次最优改进以来未改善的迭代计数
            early_stop_patience:   早停容忍阈值
            best_update_iteration: 最近一次刷新最优解的迭代号
            recent_improvement:    最近一次改进幅度 (prev_best - new_best); 未改进迭代置 0
        """
        return {
            "total_iterations": self.iteration,
            "max_iterations": self.max_iterations,
            "best_objective": self.best_obj,
            "current_objective": self.current_obj,
            "final_gap": calculate_gap(self.current_obj, self.best_obj) if self.objectives else 0.0,
            "objectives_history": self.objectives.copy(),
            "gaps_history": self.gaps.copy(),
            "elapsed_time": time.time() - self.start_time,
            "best_solution": self.best_solution,
            "features": self.features.copy(),
            "labels": self.labels.copy(),
            "ml_cache_info": self.get_ml_cache_info(),
            "no_improve_iters": self.no_improve_iters,
            "early_stop_patience": self.early_stop_patience,
            "best_update_iteration": self.best_update_iteration,
            "recent_improvement": self.recent_improvement,
        }


# ---------------------------------------------------------------------
# Gap 计算函数 (保持与 ALNSTracker 独立, 便于复用/测试)
# ---------------------------------------------------------------------
def _is_all_zero(x) -> bool:
    ax = np.asarray(x)
    if ax.size == 1:
        # 单元素数组或标量: 用 isclose 判定 0
        return bool(np.isclose(float(ax), 0.0))
    # 多元素数组: 全部接近 0 则认为为 0 向量
    return bool(np.allclose(ax, 0.0))

def _to_scalar_or_mean(x) -> float:
    ax = np.asarray(x)
    if ax.size == 1:
        return float(ax)
    # 向量化结果退化为均值以保证数值可比性（保守处理）
    return float(np.mean(ax))

def calculate_gap(current_obj: float, best_obj: float) -> float:
    """
    计算 gap:
        gap = |current_obj - best_obj| / |current_obj| * 100
    说明:
      - 兼容 scalar 与 numpy array（单元素或多元素）输入
      - 边界处理与原实现一致，但避免在 numpy 对象上使用 Python 布尔上下文
    """
    # 两者均为零向量/标量
    if _is_all_zero(current_obj) and _is_all_zero(best_obj):
        return 0.0
    # current == 0 且 best > 0 -> inf
    if _is_all_zero(current_obj):
        best_val = _to_scalar_or_mean(best_obj)
        if best_val > 0:
            return float("inf")
    # 将可能的数组退化为标量以做相对差计算（向量以均值衡量）
    cur_val = _to_scalar_or_mean(current_obj)
    best_val = _to_scalar_or_mean(best_obj)
    return abs(cur_val - best_val) / max(1e-12, abs(cur_val)) * 100
