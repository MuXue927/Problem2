"""
销毁(破坏)算子集合: 为自适应大邻域搜索(ALNS)提供多种“移除/释放”策略, 用于制造解的扰动并为后续修复算子创造改进空间。

设计要点说明
1. 统一后处理钩子 post_destroy:
   - 负责(若需要)批量回补被移除车辆携带货物对应的库存
   - 计算算子执行前后目标函数差值 improvement (稳健处理 inf/NaN)
   - 将效果反馈给参数调优器 ParamAutoTuner(用于自适应学习/权重更新)
   - 输出结构化日志 [OPJSON] 便于后续解析/可视化/ML特征抽取
   - 调整探索阶段标记 (adjust_exploration) 以支持阶段性策略 (exploration/exploitation/refinement)

2. 算子类别概览:
   - random_removal: 分层随机淘汰低效车辆, 引入多样性。
   - worst_removal: 按利用率+成本效益综合得分移除“劣”车辆, 偏向利用已知启发。
   - infeasible_removal: 针对当前解中产生的负库存 (不可行性) 做定向削减/部分回退。
   - surplus_inventory_removal: 面向高库存压力的工厂释放运输量, 缓解产地库存风险。
   - shaw_removal: 基于“相似/相关”车辆(工厂/经销商/天/货物)群体性移除, 产生结构性破坏。
   - periodic_shaw_removal: 引入聚类 + 特征工程(多维度) + 相关性评分, 适用于多期决策。
   - path_removal: 针对特定路径(工厂→经销商)整体评估并选择低效率子集移除。

3. 性能优化:
   - precompute_dealer_shipments 预计算 dealer 层面发运量与需求, 避免多重 O(V*S) 重复聚合。
   - 尽量降低在算子内反复扫描 state.vehicles 与 state.s_ikt 的频率。

4. 稳健性策略:
   - 目标值为 inf 或 NaN 时不计入正向学习, improvement 归零防止污染统计。
   - 早退分支(车辆不足/无候选/聚类失败等)统一通过 post_destroy(..., early=True) 合法登记调用频率。
   - 所有概率/除法运算使用保护 (分母>0 判定 / 归一化检查)。

5. 可扩展点:
   - post_destroy 可增补更多上下文字段 (如 removed 车辆平均载重、释放容量等)。
   - 可将 [OPJSON] 输出重定向到文件或统一 logger, 方便追踪。

后续维护指引
- 添加新销毁算子时：在算子内捕获 prev_obj = state.objective(), 执行破坏后调用 post_destroy。
- 若破坏逻辑可产生部分增量改变(如部分拆卸), 确保 removed 数组准确表示“整车”移除, 用于库存回补；对非整车调整请显式更新 state.s_ikt。
"""

# =========================
# 标准库
# =========================
import time
from collections import defaultdict

# =========================
# 第三方库
# =========================
import numpy as np
import numpy.random as rnd

# =========================
# 项目内部依赖
# =========================
from typing import TYPE_CHECKING, Optional
# 延迟导入 SolutionState 以避免与 alnsopt 的循环导入；运行时使用 object 占位
if TYPE_CHECKING:
    from .alnsopt import SolutionState
else:
    SolutionState = object
# batch_update_inventory 会在需要时局部导入，避免顶层循环依赖

from .alns_config import default_config as ALNSConfig
from .inventory_utils import precompute_plant_day_inventory, precompute_dealer_shipments

# 从 destroy_utils 中导入通用辅助函数以保持文件轻量
from .destroy_utils import (
    ensure_pyint,
    ensure_pylist,
    get_adaptive_parameters,
    post_destroy,
    resolve_degree,
)

# Short-lived cache for periodic_shaw_removal KMeans results keyed by
# (n_clusters, rounded_mean_features). This avoids re-running KMeans when
# feature summaries are effectively unchanged between successive calls.
_periodic_shaw_cache = {}


# Wrapper utilities for registering destroy operators with ALNS
class DestroyOperatorWrapper:
    """
    Lightweight wrapper that holds fixed keyword parameters for a destroy operator
    and exposes a callable compatible with ALNS (state, rng, **kwargs).
    """
    def __init__(self, func, name: str = None, **fixed_kwargs):
        self.func = func
        self.fixed_kwargs = fixed_kwargs or {}
        self.__name__ = name or getattr(func, "__name__", "destroy_op")

    def __call__(self, state, rng, **kwargs):
        # Merge fixed kwargs (lower priority) with runtime kwargs (higher priority)
        merged = {}
        merged.update(self.fixed_kwargs)
        if kwargs:
            merged.update(kwargs)
        return self.func(state, rng, **merged)

    def __repr__(self):
        return f"<DestroyOperatorWrapper {self.__name__} fixed={self.fixed_kwargs}>"


# Factory helpers for common destroy operators
def create_random_removal(degree: float):
    name = "random_removal"
    return DestroyOperatorWrapper(func=random_removal, name=name, degree=degree)

def create_shaw_removal(degree: float):
    name = "shaw_removal"
    return DestroyOperatorWrapper(func=shaw_removal, name=name, degree=degree)

def create_periodic_shaw_removal(**params):
    name = params.get("name", "periodic_shaw_removal")
    # 兼容旧配置或外部传参使用"k_clusters"的情况，映射到算子期望的"k_cluster"
    if "k_cluster" not in params and "k_clusters" in params:
        params["k_cluster"] = params.pop("k_clusters")
    return DestroyOperatorWrapper(func=periodic_shaw_removal, name=name, **params)

def wrap_operator_no_args(func):
    """Return a wrapper for operators that take no fixed parameters."""
    return DestroyOperatorWrapper(func=func, name=getattr(func, "__name__", None))

def create_worst_removal(degree: float):
    """Factory for worst_removal with a fixed degree parameter."""
    name = "worst_removal"
    return DestroyOperatorWrapper(func=worst_removal, name=name, degree=degree)

def create_surplus_inventory_removal(degree: float):
    """Factory for surplus_inventory_removal with a fixed degree parameter."""
    name = "surplus_inventory_removal"
    return DestroyOperatorWrapper(func=surplus_inventory_removal, name=name, degree=degree)

def create_path_removal(degree: float):
    """Factory for path_removal with a fixed degree parameter."""
    name = "path_removal"
    return DestroyOperatorWrapper(func=path_removal, name=name, degree=degree)

def random_removal(current: SolutionState, rng: rnd.Generator, degree: Optional[float] = None):
    """
    简洁且可导入的 random_removal 实现：
      - 随机选择若干车辆移除（保证至少保留 1 辆车）
      - 计算 prev_obj / 调用 post_destroy / 标记 objective dirty
      - 避免使用未定义的辅助函数，类型转换至 Python 原生类型以保持兼容
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 random_removal 算子")
    state = current.copy()
    prev_obj = state.objective()

    if len(state.vehicles) <= 1:
        post_destroy(state, [], 'random_removal', early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] random_removal: 车辆数不足 ({time.time() - t0:.4f}s)")
        return state

    degree = resolve_degree('random_removal', state, degree)

    # 计算移除数量，至少移除1辆但保留至少1辆车
    num_remove = int(len(state.vehicles) * degree)
    num_remove = max(1, num_remove)
    num_remove = min(num_remove, max(0, len(state.vehicles) - 1))

    if num_remove <= 0:
        post_destroy(state, [], 'random_removal', early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] random_removal: num_remove=0 ({time.time() - t0:.4f}s)")
        return state

    try:
        sel = rng.choice(range(len(state.vehicles)), size=num_remove, replace=False)
        sel = np.atleast_1d(sel)
        remove_indices = [int(x) for x in sel]
    except Exception:
        # 兜底：顺序取前 num_remove 个索引
        remove_indices = list(range(min(num_remove, len(state.vehicles))))

    remove_indices = sorted(remove_indices, reverse=True)
    removed = [state.vehicles[i] for i in remove_indices]
    state.vehicles = [veh for i, veh in enumerate(state.vehicles) if i not in remove_indices]

    try:
        state.mark_objective_dirty()
    except AttributeError:
        pass

    post_destroy(state, removed, 'random_removal', prev_obj=prev_obj, t0=t0)
    print(f"[OPLOG] random_removal: 移除 {len(removed)}/{len(current.vehicles)} 辆车 ({time.time() - t0:.4f}s)")
    return state

def worst_removal(current: SolutionState, rng: rnd.Generator, degree: Optional[float] = None, value_bias: Optional[float] = None):
    """
    简洁且可导入的 worst_removal 实现（保留原有逻辑要点）：
      - 计算每辆车的综合得分（空间利用率 + 成本/价值效率）
      - 选取低分车辆进行移除
      - 调用 post_destroy 并标记目标缓存失效
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 worst_removal 算子")
    state = current.copy()
    prev_obj = state.objective()
    if len(state.vehicles) <= 1:
        post_destroy(state, [], 'worst_removal', early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] worst_removal: 车辆数不足 ({time.time() - t0:.4f}s)")
        return state

    shipments_by_dealer_sku_day, _, _ = precompute_dealer_shipments(state)
    # 使用 (dealer, sku) 跨天累计发运量来计算满足度, 避免当日口径失真
    shipments_by_dealer_sku_total = {}
    for (dlr, sku, _day), q in shipments_by_dealer_sku_day.items():
        shipments_by_dealer_sku_total[(dlr, sku)] = shipments_by_dealer_sku_total.get((dlr, sku), 0) + q

    # 货值偏置系数可调(默认 2.0), 用于控制高满足度时的价值递减幅度
    defaults = ALNSConfig.get_operator_default('worst_removal').get('params', {})
    if value_bias is None:
        params = get_adaptive_parameters(op_name='worst_removal', param_tuner=state.param_tuner)
        try:
            value_bias = float(params.get('value_bias', defaults.get('value_bias', 2.0)))
        except Exception:
            value_bias = defaults.get('value_bias', 2.0)
    try:
        value_bias = float(value_bias)
    except Exception:
        value_bias = defaults.get('value_bias', 2.0)

    vehicle_metrics = []
    for i, veh in enumerate(state.vehicles):
        capacity = state.data.veh_type_cap[veh.type]
        load = state.compute_veh_load(veh)
        space_util = load / capacity if capacity > 0 else 0
        veh_cost = state.data.veh_type_cost[veh.type]

        cargo_value = 0
        for (sku_id, day), qty in veh.cargo.items():
            dealer = veh.dealer_id
            demand = state.data.demands.get((dealer, sku_id), 0)
            if demand > 0:
                shipped_total = shipments_by_dealer_sku_total.get((dealer, sku_id), 0)
                satisfaction = min(1.0, shipped_total / demand)
                # 使用可配置的价值偏置系数, 并用 hinge 保证非负
                sku_value = qty * state.data.sku_sizes[sku_id] * max(0.0, value_bias - satisfaction)
                cargo_value += sku_value

        cost_efficiency = cargo_value / veh_cost if veh_cost > 0 else 0
        score = 0.4 * space_util + 0.6 * cost_efficiency
        vehicle_metrics.append((i, score))

    vehicle_metrics.sort(key=lambda x: x[1])

    degree = resolve_degree('worst_removal', state, degree)
    num_remove = int(len(state.vehicles) * degree)
    if num_remove <= 0:
        post_destroy(state, [], 'worst_removal', early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] worst_removal: degree={degree:.3f} 未移除车辆 ({time.time() - t0:.4f}s)")
        return state

    selection_pool = vehicle_metrics[:min(len(vehicle_metrics), int(num_remove * 2))]
    if not selection_pool:
        post_destroy(state, [], 'worst_removal', early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] worst_removal: 无可选车辆 ({time.time() - t0:.4f}s)")
        return state

    scores = np.array([1.0 / (1.0 + item[1]) for item in selection_pool])
    scores_sum = scores.sum()
    if scores_sum <= 0:
        probabilities = None
    else:
        probabilities = scores / scores_sum

    try:
        selected_indices = rng.choice(
            [idx for idx, _ in selection_pool],
            size=min(num_remove, len(selection_pool)),
            replace=False,
            p=probabilities
        )
    except Exception:
        selected_indices = rng.choice(
            [idx for idx, _ in selection_pool],
            size=min(num_remove, len(selection_pool)),
            replace=False
        )

    # 统一将可能为 numpy scalar/array 的选中索引转换为 Python int 列表, 避免 np.int64/ndarray 导致后续索引或序列化问题
    try:
        selected_indices = np.atleast_1d(selected_indices)
        selected_indices = [int(x) for x in selected_indices]
    except Exception:
        if isinstance(selected_indices, (list, tuple)):
            selected_indices = [int(x) for x in selected_indices]
        else:
            selected_indices = [int(selected_indices)]

    removed = [state.vehicles[i] for i in sorted(selected_indices, reverse=True)]
    state.vehicles = [veh for i, veh in enumerate(state.vehicles) if i not in selected_indices]

    try:
        state.mark_objective_dirty()
    except AttributeError:
        pass

    post_destroy(state, removed, 'worst_removal', prev_obj=prev_obj, t0=t0)
    print(f"[OPLOG] worst_removal: 移除了 {len(removed)}/{len(current.vehicles)} 辆车 ({time.time() - t0:.4f}s)")
    return state

# ---------------------------------------------------------------------
# 3. 不可行性修复导向移除 (负库存缓解)
# ---------------------------------------------------------------------
def infeasible_removal(current: SolutionState, rng: rnd.Generator):
    """
    针对 state.s_ikt 中出现的负库存(plant, sku, day) 进行定向“回拉”:
      过程:
        1. 收集所有 inv < 0 条目, 按严重程度排序
        2. 对关联车辆(同工厂/天/SKU) 进行部分或整车回退
        3. 部分调整时: 减少 veh.cargo, 释放容量并从 day..horizons 回补库存
    适用:
      - 修复阶段也可作为 destroy 以快速使解重新可行
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 infeasible_removal 算子")
    state = current.copy()
    prev_obj = state.objective()

    if len(state.vehicles) <= 1:
        post_destroy(state, [], 'infeasible_removal', early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] infeasible_removal: 车辆数不足 ({time.time() - t0:.4f}s)")
        return state

    neg_inv_items = [(plant, sku_id, day, inv)
                     for (plant, sku_id, day), inv in state.s_ikt.items()
                     if inv < 0]

    if not neg_inv_items:
        print(f"[OPLOG] infeasible_removal: 无负库存, 不需要修复 ({time.time() - t0:.4f}s)")
        post_destroy(state, [], 'infeasible_removal', early=True, prev_obj=prev_obj, t0=t0)
        return state

    neg_inv_items.sort(key=lambda x: x[3])  # 更负的(更小)优先

    remove_indices = []       # 整车移除
    partial_adjustments = []  # (vehicle_index, sku_id, day, reduce_qty)

    for plant, sku_id, day, neg_amount in neg_inv_items:
        related_vehicles = []
        for idx, veh in enumerate(state.vehicles):
            if veh.fact_id == plant and veh.day == day and (sku_id, day) in veh.cargo:
                shipped_qty = veh.cargo[(sku_id, day)]
                related_vehicles.append((idx, veh, shipped_qty))
        if not related_vehicles:
            continue

        related_vehicles.sort(key=lambda x: x[2], reverse=True)

        amount_to_reduce = abs(neg_amount)
        reduced_so_far = 0
        for idx, veh, shipped_qty in related_vehicles:
            if reduced_so_far + shipped_qty <= amount_to_reduce:
                reduced_so_far += shipped_qty
                # 将 numpy 返回的聚合结果与 shipped_qty 统一转换为 Python int 再比较，避免 np.int64/np.generic 导致隐蔽类型问题
                try:
                    total_cargo_qty = int(np.sum(list(veh.cargo.values())))
                except Exception:
                    total_cargo_qty = sum(veh.cargo.values()) if veh.cargo else 0
                if total_cargo_qty == int(shipped_qty):  # 如果整车货物量等于要回退的量, 则整车移除
                    if idx not in remove_indices:
                        remove_indices.append(idx)
                else:
                    partial_adjustments.append((idx, sku_id, day, shipped_qty))  # 记录部分调整
            else:
                adjust_qty = amount_to_reduce - reduced_so_far
                partial_adjustments.append((idx, sku_id, day, adjust_qty))
                reduced_so_far = amount_to_reduce

    # 记录 partial_adjustments 为 partial_updates(不在算子内修改 veh.cargo 或 state.s_ikt)
    partial_updates = defaultdict(int)
    for idx, sku_id, day, reduce_qty in partial_adjustments:
        if idx >= len(state.vehicles):
            continue
        try:
            veh = state.vehicles[idx]
            key = (veh.fact_id, sku_id, day)
            partial_updates[key] += reduce_qty
        except Exception:
            # 保持鲁棒性, 若索引或 vehicle 异常则跳过
            continue

    # 不在此算子中进行部分回退或清理空车；仅收集 partial_updates, 由 post_destroy 统一执行回补与清理
    empty_after_partial = []
    # 若有 partial_updates, 将其作为 extra 传入 post_destroy 以便统一回补校验
    # 注意：算子内部不修改 veh.cargo/state.s_ikt, post_destroy 将执行 batch_update_inventory
    # partial_updates 已在上方构建
    pass

    # 在这里, 已经将所有需要移除的车辆索引收集完毕, 包括部分回退后变为空的车辆
    # 对于最终要从 vehicles 中删除的空车,
    # 可将其加入 removed, 但应保证其 cargo 在 removed 中为空或已反映在 state.s_ikt 中, 并在 post_destroy 中传 skip_batch_update=True

    remove_indices.sort(reverse=True)
    removed = [veh for i, veh in enumerate(state.vehicles) if i in remove_indices]
    state.vehicles = [veh for i, veh in enumerate(state.vehicles) if i not in remove_indices]

    try:
        state.mark_objective_dirty()
    except AttributeError:
        pass

    # 将 partial_updates 通过 extra 传递给 post_destroy, 由 post_destroy 统一执行批量回补与校验
    # 选择策略：算子不在内部回补 state.s_ikt, 由 post_destroy 执行 batch_update_inventory
    try:
        extra_payload = {'partial_updates': dict(partial_updates)} if partial_updates else None
    except Exception:
        extra_payload = None
    post_destroy(state, removed, 'infeasible_removal', prev_obj=prev_obj, t0=t0, skip_batch_update=False, extra=extra_payload)
    print(f"[OPLOG] infeasible_removal: 移除 {len(removed)} 辆车, 部分调整 {len(partial_adjustments)} 个SKU ({time.time() - t0:.4f}s)")
    return state

# ---------------------------------------------------------------------
# 4. 高库存压力导向移除
# ---------------------------------------------------------------------
def surplus_inventory_removal(current: SolutionState, rng: rnd.Generator, degree: Optional[float] = None):
    """
    聚焦库存高风险工厂:
      - 统计 plant_sku_inventory, 用最大期利用率衡量风险 (max_util - 0.8 的放大系数)
      - 在高风险工厂中挑选发往“需求已较高满足率”经销商的车辆释放运力
      - 优先级 = 风险系数 * 目的地满足率
    适用:
      - 避免工厂库存溢出风险, 通过回撤发运量为后续重排提供空间

    单位说明:
      - plant_inv_limit 为生产基地可存储的 SKU 数量上限(非体积), 与 s_ikt 单位一致
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 surplus_inventory_removal 算子")
    state = current.copy()
    prev_obj = state.objective()
    data = state.data

    if len(state.vehicles) <= 1:
        post_destroy(state, [], 'surplus_inventory_removal', early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] surplus_inventory_removal: 车辆数不足 ({time.time() - t0:.4f}s)")
        return state

    plant_sku_inventory = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for (plant, sku_id, day), inv in state.s_ikt.items():
        plant_sku_inventory[plant][day][sku_id] += inv

    plant_risk = {}
    for plant, day_dict in plant_sku_inventory.items():
        max_capacity = data.plant_inv_limit[plant]  # SKU 数量上限(非体积, 与 s_ikt 一致)
        if max_capacity <= 0:
            plant_risk[plant] = 0.0
            continue
        utilizations = []
        for day, sku_invs in day_dict.items():
            total_inv_day = np.sum(list(sku_invs.values()))
            utilizations.append(total_inv_day / max_capacity)
        max_util = max(utilizations) if utilizations else 0.0  # 最大期利用率, 可以反映库存峰值
        risk = max(0.0, max_util - 0.8) * 5  # 超过80%后线性放大
        plant_risk[plant] = risk

    high_risk_plants = sorted(plant_risk.items(), key=lambda x: x[1], reverse=True)

    candidates = []
    _, shipments_by_dealer_total, dealer_total_demand = precompute_dealer_shipments(state)

    for plant, risk in high_risk_plants:
        if risk <= 0:
            continue
        for i, veh in enumerate(state.vehicles):
            if veh.fact_id == plant:
                total_qty = sum(qty for (_, _), qty in veh.cargo.items())
                dealer = veh.dealer_id
                dealer_demand = dealer_total_demand.get(dealer, 0)
                dealer_shipped = shipments_by_dealer_total.get(dealer, 0)
                satisfaction = dealer_shipped / dealer_demand if dealer_demand > 0 else 1.0
                priority = risk * satisfaction
                candidates.append((i, veh, priority, total_qty))

    if not candidates:
        print(f"[OPLOG] surplus_inventory_removal: 无高风险工厂 ({time.time() - t0:.4f}s)")
        post_destroy(state, [], 'surplus_inventory_removal', early=True, prev_obj=prev_obj, t0=t0)
        return state

    candidates.sort(key=lambda x: x[2], reverse=True)

    degree = resolve_degree('surplus_inventory_removal', state, degree)

    num_remove = min(int(len(state.vehicles) * degree), len(candidates))
    if num_remove <= 0:
        print(f"[OPLOG] surplus_inventory_removal: degree={degree:.3f} 未移除车辆 ({time.time() - t0:.4f}s)")
        post_destroy(state, [], 'surplus_inventory_removal', early=True, prev_obj=prev_obj, t0=t0)
        return state

    selected_indices = [idx for idx, _, _, _ in candidates[:num_remove]]
    selected_indices.sort(reverse=True)

    removed = [veh for i, veh in enumerate(state.vehicles) if i in selected_indices]
    state.vehicles = [veh for i, veh in enumerate(state.vehicles) if i not in selected_indices]

    try:
        state.mark_objective_dirty()
    except AttributeError:
        pass

    post_destroy(state, removed, 'surplus_inventory_removal', prev_obj=prev_obj, t0=t0)
    print(f"[OPLOG] surplus_inventory_removal: 移除 {len(removed)}/{len(current.vehicles)} 辆车 ({time.time() - t0:.4f}s)")
    return state

# ---------------------------------------------------------------------
# 5. Shaw 相似性破坏
# ---------------------------------------------------------------------
def shaw_removal(current: SolutionState, rng: rnd.Generator, degree: Optional[float] = None):
    """
    通过“相似性”聚焦一组相关车辆进行集群性移除:
      相似度维度: 工厂/经销商/天数接近性 + 货物 SKU 交集比例 + 随机噪声
      权重: fact(3) dealer(2) day(1) cargo(2)
      种子选取: 利用率低/后期/工厂库存高 → 得分高 → 轮盘赌选为种子
    目的:
      - 制造结构性缺口 (例如集中移除某工厂某时间段车辆), 为修复算子重构装载提供机会

    单位说明:
      - plant_inv_limit 为工厂可存储的 SKU 数量上限(非体积), 与 s_ikt 单位一致
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 shaw_removal 算子")
    state = current.copy()
    prev_obj = state.objective()

    if len(state.vehicles) <= 1:
        post_destroy(state, [], 'shaw_removal', early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] shaw_removal: 车辆数不足 ({time.time() - t0:.4f}s)")
        return state

    degree = resolve_degree('shaw_removal', state, degree)

    seed_candidates = []
    plant_inv_limit = state.data.plant_inv_limit
    # 预聚合 (plant, day) 维度的库存, 避免在车辆循环中重复扫描 state.s_ikt(性能优化)
    plant_day_inventory = precompute_plant_day_inventory(state)
    for i, veh in enumerate(state.vehicles):
        load = state.compute_veh_load(veh)
        capacity = state.data.veh_type_cap[veh.type]
        utilization = load / capacity if capacity > 0 else 0
        
        plant_inventory = plant_day_inventory.get((veh.fact_id, veh.day), 0)
        seed_score = (1 - utilization) * 0.4 + (veh.day / state.data.horizons) * 0.3 + \
                     (plant_inventory / (plant_inv_limit[veh.fact_id] + 1)) * 0.3
        seed_candidates.append((i, seed_score))

    seed_scores = np.array([score for _, score in seed_candidates])
    if seed_scores.sum() > 0:
        probabilities = seed_scores / seed_scores.sum()
        try:
            # 使用 ensure_pyint 统一将可能的 numpy 返回值转换为 Python int
            sel = rng.choice(len(seed_candidates), p=probabilities)
            pos = ensure_pyint(sel)
        except Exception:
            pos = ensure_pyint(rng.integers(0, len(seed_candidates)))
        # pos 已由 ensure_pyint 返回 Python int
        seed_idx = int(seed_candidates[pos][0])
    else:
        pos = ensure_pyint(rng.integers(0, len(seed_candidates)))
        seed_idx = int(seed_candidates[pos][0])

    seed_veh = state.vehicles[seed_idx]
    seed_skus = set(sku_id for (sku_id, _) in seed_veh.cargo.keys())

    relatedness = []
    weight_fact, weight_dealer, weight_day, weight_cargo = 3.0, 2.0, 1.0, 2.0
    for i, veh in enumerate(state.vehicles):
        if i == seed_idx:
            continue
        fact_sim = 1.0 if veh.fact_id == seed_veh.fact_id else 0.0
        dealer_sim = 1.0 if veh.dealer_id == seed_veh.dealer_id else 0.0
        day_diff = abs(veh.day - seed_veh.day)
        day_sim = max(0.0, 1.0 - day_diff / state.data.horizons)
        veh_skus = set(sku_id for (sku_id, _) in veh.cargo.keys())
        if seed_skus and veh_skus:
            common_skus = len(seed_skus & veh_skus)
            all_skus = len(seed_skus | veh_skus)
            cargo_sim = common_skus / all_skus if all_skus > 0 else 0.0
        else:
            cargo_sim = 0.0
        total_sim = (weight_fact * fact_sim + weight_dealer * dealer_sim +
                     weight_day * day_sim + weight_cargo * cargo_sim)
        noise = rng.uniform(-0.1, 0.1)
        total_sim = max(0.0, total_sim + noise)
        relatedness.append((i, total_sim))

    relatedness.sort(key=lambda x: x[1], reverse=True)

    num_remove = min(int(len(state.vehicles) * degree), len(relatedness))
    if num_remove <= 0:
        post_destroy(state, [], 'shaw_removal', early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] shaw_removal: degree={degree:.3f} 未移除车辆 ({time.time() - t0:.4f}s)")
        return state

    remove_indices = [idx for idx, _ in relatedness[:num_remove]]
    remove_indices.append(seed_idx)
    remove_indices.sort(reverse=True)

    removed = [veh for i, veh in enumerate(state.vehicles) if i in remove_indices]
    state.vehicles = [veh for i, veh in enumerate(state.vehicles) if i not in remove_indices]

    try:
        state.mark_objective_dirty()
    except AttributeError:
        pass

    # 为便于归因与调试，在日志中加入算子上下文（seed_idx, params, degree）
    try:
        extra_payload = {
            "seed_idx": int(seed_idx) if 'seed_idx' in locals() else None,
            "params": {"degree": float(degree) if degree is not None else None},
            "degree": float(degree) if degree is not None else None
        }
    except Exception:
        extra_payload = None

    post_destroy(state, removed, 'shaw_removal', prev_obj=prev_obj, t0=t0, extra=extra_payload)
    print(f"[OPLOG] shaw_removal: 移除 {len(removed)}/{len(current.vehicles)} 辆车 ({time.time() - t0:.4f}s)")
    return state

# ---------------------------------------------------------------------
# 6. 多期/聚类增强版 Shaw 破坏
# ---------------------------------------------------------------------
def periodic_shaw_removal(current: SolutionState, rng: rnd.Generator, degree: Optional[float] = None,
                          alpha: Optional[float] = None, beta: Optional[float] = None, gamma: Optional[float] = None,
                          k_cluster: Optional[int] = None) -> SolutionState:
    """
    多维特征聚类 + 相关性破坏:
      特征: [归一化天数, 经销商需求满足率, 前日库存(归一), 车辆利用率, 工厂库存压力]
      流程:
        1. 为每个 (veh_idx, veh, sku, 当期day) 构造特征行
        2. KMeans 聚类 (自适应簇数, IQR 标准化防极值)
        3. 选取较大簇之一 → 内部随机选种子 → 计算成员相似度 (组合 alpha/beta/gamma 各维差异)
        4. 选相似度低 (或距离近) 的前若干比例进行移除 (移除粒度: SKU分配级别)
      回退:
        聚类失败 → 简单随机整车移除
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 periodic_shaw_removal 算子")
    state = current.copy()
    prev_obj = state.objective()
    data = state.data

    if len(state.vehicles) <= 1:
        post_destroy(state, [], 'periodic_shaw_removal', early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] periodic_shaw_removal: 车辆数不足 ({time.time() - t0:.4f}s)")
        return state

    defaults = ALNSConfig.get_operator_default('periodic_shaw_removal').get('params', {})
    degree = resolve_degree('periodic_shaw_removal', state, degree)
    # other params: prefer explicit kwargs, then tuner (only for missing), then defaults
    params_other = {}
    if alpha is None or beta is None or gamma is None or k_cluster is None:
        params_other = get_adaptive_parameters(op_name='periodic_shaw_removal', param_tuner=state.param_tuner)
    try:
        alpha = float(alpha if alpha is not None else params_other.get('alpha', defaults.get('alpha', 0.4)))
    except Exception:
        alpha = defaults.get('alpha', 0.4)
    try:
        beta = float(beta if beta is not None else params_other.get('beta', defaults.get('beta', 0.3)))
    except Exception:
        beta = defaults.get('beta', 0.3)
    try:
        gamma = float(gamma if gamma is not None else params_other.get('gamma', defaults.get('gamma', 0.3)))
    except Exception:
        gamma = defaults.get('gamma', 0.3)
    try:
        k_cluster = int(k_cluster if k_cluster is not None else params_other.get('k_cluster', params_other.get('k_clusters', defaults.get('k_clusters', 3))))
    except Exception:
        k_cluster = defaults.get('k_clusters', 3)

    allocations = []  # (veh_idx, veh, sku, day, feature_vector)
    
    # 预计算辅助数据, 避免在车辆循环中重复扫描 state.s_ikt等数据 (性能优化)
    _, shipments_by_dealer_total, dealer_total_demand = precompute_dealer_shipments(state)
    plant_day_inventory = precompute_plant_day_inventory(state)

    for veh_idx, veh in enumerate(state.vehicles):
        plant = veh.fact_id
        dealer = veh.dealer_id
        day = veh.day
        load = state.compute_veh_load(veh)
        cap = data.veh_type_cap[veh.type]
        
        veh_util = load / cap if cap > 0 else 0
        total_demand = dealer_total_demand.get(dealer, 0)
        dealer_shipped = shipments_by_dealer_total.get(dealer, 0)
        satisfaction = dealer_shipped / total_demand if total_demand > 0 else 1.0

        plant_inv = plant_day_inventory.get((plant, day), 0)
        
        plant_capacity = data.plant_inv_limit[plant]
        plant_pressure = plant_inv / plant_capacity if plant_capacity > 0 else 1.0
        for (sku, d), qty in list(veh.cargo.items()):
            if d != day:
                continue
            # 注意, 这里对 “前日库存”特征进行了修正,
            # 将由 day>0 改为 day>=2 时, 取 day-1 库存, 否则取 day 日库存 (day=1 时无前日)
            # 目的是避免 day=1 时, 取到 day=0 时候的库存(对应的是期初库存, 永远不会被修改), 导致该特征失效
            inv_transfer = state.s_ikt.get((plant, sku, day - 1), 0) if day >= 2 else state.s_ikt.get((plant, sku, day), 0)
            norm_day = day / data.horizons
            norm_inv = min(1.0, inv_transfer / (plant_capacity * 0.1)) if plant_capacity > 0 else 0.0
            feature = np.array([norm_day, satisfaction, norm_inv, veh_util, plant_pressure])
            allocations.append((veh_idx, veh, sku, day, feature))

    if not allocations:
        print(f"[OPLOG] periodic_shaw_removal: 无可处理的分配 ({time.time() - t0:.4f}s)")
        post_destroy(state, [], 'periodic_shaw_removal', early=True, prev_obj=prev_obj, t0=t0)
        return state

    try:
        features = np.array([a[4] for a in allocations])
        features_median = np.median(features, axis=0)
        features_iqr = np.percentile(features, 75, axis=0) - np.percentile(features, 25, axis=0)
        features_iqr[features_iqr == 0] = 1.0
        features_normalized = (features - features_median) / features_iqr
        optimal_clusters = min(k_cluster, len(allocations), max(1, len(allocations) // 3))

        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=optimal_clusters,
            # 使用 ensure_pyint 统一处理 numpy 返回值，防止 np.int64/0-d array 泄漏
            random_state=ensure_pyint(rng.integers(0, 1000)),
            n_init=3,
            max_iter=50,
            tol=1e-3
        )  # 根据官方文档, KMeans会默认使用 "lloyd" 算法
        labels = kmeans.fit_predict(features_normalized)
        del kmeans
    except Exception as e:
        print(f"[OPLOG] periodic_shaw_removal: 聚类失败 ({type(e).__name__}), 回退到简单随机移除")
        num_remove = int(len(state.vehicles) * degree)
        if num_remove > 0:
            # 兼容 numpy 返回的 scalar/ndarray：确保返回为至少一维数组然后强制转为 Python int 列表
            # 兼容 numpy 返回的 scalar/ndarray：使用 ensure_pylist 统一转换为 Python int 列表
            remove_indices = ensure_pylist(rng.choice(range(len(state.vehicles)), size=num_remove, replace=False))
            remove_indices = sorted(remove_indices, reverse=True)
            removed = [veh for i, veh in enumerate(state.vehicles) if i in remove_indices]
            state.vehicles = [veh for i, veh in enumerate(state.vehicles) if i not in remove_indices]
            try:
                state.mark_objective_dirty()
            except AttributeError:
                pass
            post_destroy(state, removed, 'periodic_shaw_removal', prev_obj=prev_obj, t0=t0)
            print(f"[OPLOG] periodic_shaw_removal: 回退模式 - 移除了 {num_remove} 辆车 ({time.time() - t0:.4f}s)")
        else:
            post_destroy(state, [], 'periodic_shaw_removal', early=True, prev_obj=prev_obj, t0=t0)
            print(f"[OPLOG] periodic_shaw_removal: 回退模式 - 未移除车辆 ({time.time() - t0:.4f}s)")
        return state

    unique_labels = np.unique(labels)
    if len(unique_labels) == 0:
        print(f"[OPLOG] periodic_shaw_removal: 无有效聚类标签 ({time.time() - t0:.4f}s)")
        post_destroy(state, [], 'periodic_shaw_removal', early=True, prev_obj=prev_obj, t0=t0)
        return state

    # 将聚类标签与计数强制为 Python 原生类型 (int)，避免后续使用时出现 numpy 类型泄漏
    cluster_sizes = [(int(label), int(np.sum(labels == label))) for label in unique_labels]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    eligible_clusters = cluster_sizes[:max(1, len(cluster_sizes) // 2)]  # 取较大的一半簇
    if not eligible_clusters:
        print(f"[OPLOG] periodic_shaw_removal: 无合格簇 ({time.time() - t0:.4f}s)")
        post_destroy(state, [], 'periodic_shaw_removal', early=True, prev_obj=prev_obj, t0=t0)
        return state

    # 兼容 numpy 返回的 scalar/ndarray：优先用 atleast_1d 取第0项再转为 Python int；若失败则回退到 integers，再回退到可用的簇标签
    try:
        # 优先使用 ensure_pyint 统一将可能的 numpy 返回值转换为 Python int
        selected_cluster = ensure_pyint(rng.choice([label for label, _ in eligible_clusters]))
    except Exception:
        try:
            selected_cluster = ensure_pyint(rng.integers(0, len(eligible_clusters)))
        except Exception:
            # 最后回退：取第一个合格簇的标签并强制为 Python int
            try:
                selected_cluster = int(eligible_clusters[0][0])
            except Exception:
                selected_cluster = ensure_pyint([eligible_clusters[0][0]])
    # 将 numpy 返回的索引数组转换为 Python int 列表，避免 np.ndarray / np.int64 在后续索引和序列化中引发类型不一致
    cluster_indices = ensure_pylist(np.where(labels == selected_cluster)[0])
    cluster_allocs = [allocations[i] for i in cluster_indices]
    if len(cluster_allocs) <= 1:
        print(f"[OPLOG] periodic_shaw_removal: 簇内元素过少 ({time.time() - t0:.4f}s)")
        post_destroy(state, [], 'periodic_shaw_removal', early=True, prev_obj=prev_obj, t0=t0)
        return state

    # 使用 ensure_pyint 统一处理随机整数返回值
    seed_idx = ensure_pyint(rng.integers(0, len(cluster_allocs)))
    seed_feature = cluster_allocs[int(seed_idx)][4]

    similarities = []
    for i, alloc in enumerate(cluster_allocs):
        if i == seed_idx:
            continue
        feat = alloc[4]
        time_diff = abs(feat[0] - seed_feature[0])
        demand_cos = max(0, np.dot(feat[1:3], seed_feature[1:3])) / (
                np.linalg.norm(feat[1:3]) * np.linalg.norm(seed_feature[1:3]) + 1e-8)
        op_diff = np.linalg.norm(feat[3:5] - seed_feature[3:5])
        sim = alpha * time_diff + beta * (1 - demand_cos) + gamma * op_diff
        noise = rng.uniform(-0.1, 0.1) * sim
        sim += noise
        similarities.append((alloc, sim))

    similarities.sort(key=lambda x: x[1])
    num_remove = int(len(cluster_allocs) * degree)
    num_remove = min(num_remove, len(similarities))

    # 统一在循环后集中清理空车, 避免在循环中频繁修改列表
    empties_idx = set()

    # 收集部分回退更新(不在算子内修改 state.s_ikt 或 veh.cargo)
    partial_updates = defaultdict(int)
    for i in range(num_remove):
        alloc, _sim = similarities[i]
        veh_idx, veh, sku_id, day, _ = alloc
        qty = veh.cargo.get((sku_id, day), 0)
        if qty > 0:
            # 为保持与原逻辑等价, 按原来在算子中回补的语义：
            # 对于每个 d in [day, horizons], 将 qty 累加到 partial_updates 中, 
            # 由 post_destroy 统一执行批量回补(batch_update_inventory)
            for d in range(day, data.horizons + 1):
                key = (veh.fact_id, sku_id, d)
                # 强制将 numpy 类型转换为 Python 原生类型以避免序列化问题
                partial_updates[key] += int(qty) if isinstance(qty, (int, np.integer)) else qty
            # 检查如果该 SKU 是车辆上唯一的货物条目, 则该车辆在移除该 SKU 后会变为空
            total_cargo = sum(veh.cargo.values()) if veh.cargo else 0
            if total_cargo == qty:
                empties_idx.add(veh_idx)

    # 循环结束后统一按索引倒序删除车辆
    if empties_idx:
        remove_list = sorted(list(empties_idx), reverse=True)
        removed = [state.vehicles[i] for i in remove_list if i < len(state.vehicles)]
        state.vehicles = [veh for idx, veh in enumerate(state.vehicles) if idx not in empties_idx]
    else:
        removed = []

    try:
        state.mark_objective_dirty()
    except AttributeError:
        pass
    # 将 partial_updates 交由 post_destroy 批量回补；算子内部不再直接修改 state.s_ikt 或 veh.cargo
    try:
        extra_payload = {'partial_updates': dict(partial_updates)} if partial_updates else None
    except Exception:
        extra_payload = None
    post_destroy(state, removed, 'periodic_shaw_removal', prev_obj=prev_obj, t0=t0, skip_batch_update=False, extra=extra_payload)
    print(f"[OPLOG] periodic_shaw_removal: 处理了簇 {selected_cluster}, 移除了 {len(removed)} 辆空车/分配 ({time.time() - t0:.4f}s)")
    return state

# ---------------------------------------------------------------------
# 7. 路径整体移除 (工厂→经销商)
# ---------------------------------------------------------------------
def path_removal(current: SolutionState, rng: rnd.Generator, degree: Optional[float] = None):
    """
    针对 (plant, dealer) 路径进行整体评估:
      - 路径指标: 需求满足率 / 平均车辆成本 / 车辆数量占比
      - 优先移除高满足率 + 成本高 + 车辆多的路径中效率较低的车辆
      - 若路径车数占全局过半, 降低破坏比例 (避免一次性清空造成结构崩塌)
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 path_removal 算子")
    state = current.copy()
    prev_obj = state.objective()
    data = state.data

    if len(state.vehicles) <= 1:
        post_destroy(state, [], 'path_removal', early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] path_removal: 车辆数不足 ({time.time() - t0:.4f}s)")
        return state

    path_info = {}
    for veh in state.vehicles:
        path = (veh.fact_id, veh.dealer_id)
        if path not in path_info:
            path_info[path] = {'vehicles': [], 'total_demand': 0, 'shipped': 0, 'total_cost': 0}
        path_info[path]['vehicles'].append(veh)
        path_info[path]['total_cost'] += data.veh_type_cost[veh.type]
        for (sku_id, _), qty in veh.cargo.items():
            path_info[path]['shipped'] += qty

    for (plant, dealer), info in path_info.items():
        dealer_demand = sum(data.demands.get((dealer, sku), 0) for sku in data.all_skus)
        info['total_demand'] = dealer_demand

    if not path_info:
        print(f"[OPLOG] path_removal: 无有效路径 ({time.time() - t0:.4f}s)")
        post_destroy(state, [], 'path_removal', early=True, prev_obj=prev_obj, t0=t0)
        return state

    path_scores = []
    for path, info in path_info.items():
        if not info['vehicles']:
            continue
        satisfaction = info['shipped'] / info['total_demand'] if info['total_demand'] > 0 else 1.0
        avg_cost = info['total_cost'] / len(info['vehicles']) if info['vehicles'] else 0
        path_score = (
                satisfaction * 0.4 +
                (avg_cost / max(data.veh_type_cost.values())) * 0.3 +
                (len(info['vehicles']) / len(state.vehicles)) * 0.3
        )
        path_scores.append((path, path_score))

    if not path_scores:
        print(f"[OPLOG] path_removal: 无有效路径评分 ({time.time() - t0:.4f}s)")
        post_destroy(state, [], 'path_removal', early=True, prev_obj=prev_obj, t0=t0)
        return state

    path_scores.sort(key=lambda x: x[1], reverse=True)
    top_paths = path_scores[:max(1, len(path_scores) // 3)]  # 取评分最高的三分之一路径
    weights = np.array([score for _, score in top_paths])
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else None

    try:
        # 兼容 numpy 返回的 scalar/ndarray：优先使用 ensure_pyint 将可能的 numpy 返回值转换为 Python int
        try:
            sel = rng.choice(len(top_paths), p=weights)
            selected_idx = ensure_pyint(sel)
        except Exception:
            selected_idx = ensure_pyint(rng.integers(0, len(top_paths)))
        target_path = top_paths[int(selected_idx)][0]
    except Exception:
        # 回退到稳健的 integers 方式, 再次确保为 Python int；最终再以随机 path 回退
        try:
            selected_idx = int(np.atleast_1d(rng.integers(0, len(top_paths)))[0])
            target_path = top_paths[selected_idx][0]
        except Exception:
            # 随机选择一个 path 并确保返回值为纯 Python tuple 或原始 path 类型, 避免 numpy.object_ 导致序列化/比较问题
            candidate = rng.choice([path for path, _ in path_scores])
            # 强制将可能为 numpy 类型的 candidate 转为 Python 原生类型（兼容 np.ndarray / np.generic / np.str_ 等）
            try:
                if isinstance(candidate, np.ndarray):
                    # ndarray -> tuple of native elements
                    try:
                        candidate = tuple(candidate.tolist())
                    except Exception:
                        candidate = tuple(candidate)
                elif isinstance(candidate, (np.generic, np.integer, np.floating, np.bool_)):
                    # numpy scalar -> python scalar via item()
                    try:
                        candidate = candidate.item()
                    except Exception:
                        try:
                            candidate = int(candidate)
                        except Exception:
                            try:
                                candidate = float(candidate)
                            except Exception:
                                candidate = str(candidate)
                # list -> tuple
                if isinstance(candidate, list):
                    candidate = tuple(candidate)
                # If candidate is bytes-like or numpy.str_, convert to str
                if isinstance(candidate, (bytes,)):
                    try:
                        candidate = candidate.decode()
                    except Exception:
                        candidate = str(candidate)
            except Exception:
                # 最后兜底，保证 candidate 为可序列化的 Python 原生类型
                try:
                    candidate = tuple(candidate) if isinstance(candidate, (list, tuple)) else str(candidate)
                except Exception:
                    candidate = str(candidate)

            # 最终确保 target_path 为纯 Python tuple 或原始标量
            if isinstance(candidate, (list, tuple)):
                target_path = tuple(candidate)
            else:
                target_path = candidate

    path_vehicles = [(i, veh) for i, veh in enumerate(state.vehicles)
                     if (veh.fact_id, veh.dealer_id) == target_path]

    if not path_vehicles or len(path_vehicles) <= 1:
        print(f"[OPLOG] path_removal: 选定路径上车辆不足 ({time.time() - t0:.4f}s)")
        post_destroy(state, [], 'path_removal', early=True, prev_obj=prev_obj, t0=t0)
        return state

    degree = resolve_degree('path_removal', state, degree)

    path_ratio = len(path_vehicles) / len(state.vehicles)
    if path_ratio > 0.5:
        degree *= 0.5

    num_remove = int(len(path_vehicles) * degree)
    if num_remove <= 0:
        post_destroy(state, [], 'path_removal', early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] path_removal: degree={degree:.3f} 未移除车辆 ({time.time() - t0:.4f}s)")
        return state

    num_remove = min(num_remove, len(path_vehicles) - 1) if len(path_vehicles) > 1 else 0
    if num_remove <= 0:
        post_destroy(state, [], 'path_removal', early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] path_removal: 计算后 num_remove=0, 早退 ({time.time() - t0:.4f}s)")
        return state

    vehicle_metrics = []
    max_cost = max(data.veh_type_cost.values())
    for i, veh in path_vehicles:
        load = state.compute_veh_load(veh)
        capacity = data.veh_type_cap[veh.type]
        utilization = load / capacity if capacity > 0 else 0
        
        cost = data.veh_type_cost[veh.type]
        unit_cost = cost / load if load > 0 else max_cost
        score = 0.5 * utilization + 0.5 * (1 - unit_cost / (max_cost + 1))
        
        vehicle_metrics.append((i, score))

    vehicle_metrics.sort(key=lambda x: x[1])
    remove_indices = [idx for idx, _ in vehicle_metrics[:num_remove]]
    remove_indices.sort(reverse=True)

    removed = [veh for i, veh in enumerate(state.vehicles) if i in remove_indices]
    state.vehicles = [veh for i, veh in enumerate(state.vehicles) if i not in remove_indices]

    try:
        state.mark_objective_dirty()
    except AttributeError:
        pass

    post_destroy(state, removed, 'path_removal', prev_obj=prev_obj, t0=t0)
    print(f"[OPLOG] path_removal: 从路径 {target_path} 移除了 {len(removed)}/{len(path_vehicles)} 辆车 ({time.time() - t0:.4f}s)")
    return state
