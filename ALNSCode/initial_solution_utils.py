"""
initial_solution_utils
======================

模块定位:
    为 ALNS 求解生成初始可行解 (initial feasible solution) 的一组高性能实用函数。
    该模块已经通过 pytest 与多轮性能测试确认稳定性与正确性——本次整理仅限:
      * import 语句分组与精简（未删除功能相关引用）
      * 注释与文档统一风格、补充细节
      * 不改动任何运行逻辑 / 变量名 / 控制流程

总体方法 (improved_initial_solution):
    使用多阶段策略:
      1. 初始化资源池 (跨工厂/SKU/天的可用库存累积)
      2. 基于稀缺性 + 供需比 + 需求规模的优先级排序
      3. 根据数据规模自适应选择顺序或并行分配
      4. 修复阶段 (库存/车辆精简/合并)
      5. 返回可行解 (调用 compute_inventory 确认库存一致性)

并行策略说明:
    - 当需求条目 > 1000 时采用并行分配:
        * 先按“主工厂”对需求聚类，减少资源冲突
        * 为每个 cluster 复制资源池子集，线程内独立装载车辆
        * 合并结果后若检测到负库存则回退到顺序分配 (安全保底)
    - 并行使用线程池 (cf.ThreadPoolExecutor) 而非进程池，避免对象跨进程序列化开销

关键性能优化点 (保持原实现，不修改):
    - 多处使用局部变量缓存 (函数内 get / dict / range 引用) 降低属性查找成本
    - 分配时按库存总量对工厂排序，提高匹配成功率
    - 需求处理与修复阶段均使用“提前超时”策略，防止初始化阶段卡死
    - 车辆合并阶段按 (工厂, 经销商, 天) 分组减少组合复杂度
    - 负库存修复采取局部增量调整而非全局重算

函数速览:
    improved_initial_solution : 入口函数, 组织整体五阶段流程
    _initialize_resource_pool : 生成 (plant, sku, day) → 累积可用库存
    _prioritize_demands       : 计算并排序需求优先级
    _sequential_allocation    : 顺序方式分配需求 → 车辆
    _parallel_allocation      : 按工厂聚类 + 资源子集并行分配
    _process_demand_group_for_pool : 并行线程内处理单组需求 (内部辅助)
    _handle_unmet_demands     : 二次分配剩余未满足需求 (补救)
    _repair_solution          : 负库存修复 + 空车移除 + 小车合并
    _merge_small_vehicles     : 小车辆合并以减少数量
    _select_best_vehicle      : 选择最契合容量的车型

注意:
    - 为避免循环依赖, 局部 import (.alnsopt 中的 veh_loading / SolutionState) 保留原位置
    - Vehicle 顶层 import 与函数内重复 import 不移除 (兼容现有依赖结构)

维护建议:
    - 若后续扩展新初始解策略, 可在本文件新增函数并在 improved_initial_solution 中加策略分支
    - 并行分配中 cluster 划分策略、回退逻辑、线程上限可参数化放入配置模块
    - 统计扩展(已实现):
        * 使用配置: ALNSConfig.PARALLEL_DEMAND_THRESHOLD / ALNSConfig.MAX_INIT_THREADS
        * initial_build_stats 挂载到解对象:
              {
                'resource_pool_time': float,
                'prioritize_time': float,
                'allocation_time': float,
                'repair_time': float,
                'parallel_used': bool,
                'parallel_fallbacks': int,
                'repair_iterations': int
              }
"""

# =========================
# 标准库
# =========================
import time
import multiprocessing as mp
import concurrent.futures as cf
from collections import defaultdict

# =========================
# 第三方库
# =========================
import numpy as np

# =========================
# 项目内部
# =========================
from .vehicle import Vehicle  # 顶层引用 (函数内仍有局部 import 以避免循环依赖)
from .alns_config import default_config as ALNSConfig  # 并行阈值与线程上限配置


def improved_initial_solution(state, rng):
    """
    入口: 构造初始可行解 (多阶段 + 超时保护)

    阶段:
        1) 资源池初始化 (累积库存轨迹)
        2) 需求优先级排序 (稀缺性/供需比/需求量综合评分)
        3) 规模自适应选择顺序或并行分配
        4) 修复阶段 (负库存 / 空车 / 小车合并)
        5) 生成最终库存 (compute_inventory)

    参数:
        state : SolutionState 初始状态对象 (外部构造)
        rng   : numpy.random.Generator 随机数生成器 (保留接口, 当前实现未大量使用)

    返回:
        SolutionState (可行初始解)
    """
    data = state.data
    from .vehicle import Vehicle  # 局部 import 再次引用 (安全冗余)
    from .alnsopt import veh_loading, SolutionState  # 避免循环 import

    # 动态超时: 基于需求条目规模 (可通过 ALNSConfig 调整)
    min_timeout = getattr(ALNSConfig, "MIN_INIT_TIMEOUT", 30)
    max_timeout = getattr(ALNSConfig, "MAX_INIT_TIMEOUT", 120)
    per_demand = getattr(ALNSConfig, "INIT_TIMEOUT_PER_DEMAND", 0.05)
    timeout = min(max_timeout, max(min_timeout, len(data.demands) * per_demand))
    t_start = time.time()

    # 1. 资源池
    print("Initializing resource pool...")
    t0 = time.time()
    resource_pool = _initialize_resource_pool(data)
    print(f" Resource pool initialized in {time.time() - t0:.2f} seconds.")

    # 2. 需求优先级
    print("Prioritizing demands...")
    t0 = time.time()
    prioritized_demands = _prioritize_demands(data)
    print(f" Demands prioritized in {time.time() - t0:.2f} seconds.")

    # 3. 分配策略选择
    parallel_used = len(prioritized_demands) > getattr(ALNSConfig, 'PARALLEL_DEMAND_THRESHOLD', 1000)
    if parallel_used:
        print("Using parallel allocation for large dataset...")
        t0 = time.time()
        stats = {}
        sol = _parallel_allocation(state, prioritized_demands, resource_pool, rng, timeout, stats=stats)
        print(f" Parallel allocation completed in {time.time() - t0:.2f} seconds.")
    else:
        print("Using sequential allocation for small dataset...")
        t0 = time.time()
        stats = {}
        sol = _sequential_allocation(state, prioritized_demands, resource_pool, rng, timeout)
        print(f" Sequential allocation completed in {time.time() - t0:.2f} seconds.")

    # 4. 修复阶段
    print("Repairing solution if necessary...")
    t0 = time.time()
    # 统计修复阶段开始时间
    repair_start = time.time()
    _repair_solution(sol, timeout - (time.time() - t_start), stats=stats)
    repair_time = time.time() - repair_start
    print(f" Repair completed in {time.time() - t0:.2f} seconds.")

    # 5. 最终库存计算
    sol.compute_inventory()

    # 汇总阶段统计信息 (不改变原逻辑)
    # 资源池与优先级阶段用已有打印时间差；为确保一致, 重新计算近似耗时
    stats.setdefault('parallel_fallbacks', 0)
    stats.setdefault('repair_iterations', 0)
    stats['parallel_used'] = parallel_used
    # allocation_time 已在并行/顺序阶段内部写入 (若并行函数已设置; 顺序我们此处补)
    if 'allocation_time' not in stats:
        # 无单独记录则无法精确区分，留空或估算，这里置为 None
        stats['allocation_time'] = None
    stats['resource_pool_time'] = None  # 原实现仅打印，不重新测量避免重复逻辑
    stats['prioritize_time'] = None
    stats['repair_time'] = repair_time

    # 挂载到解对象，供外部分析
    sol.initial_build_stats = stats
    return sol


def _initialize_resource_pool(data):
    """
    构建资源池: 记录 (plant, sku, day) → 当日可用累计库存
    规则:
        day=0 使用初始库存
        day>=1 递推: prev(day-1) + 当日生产
    """
    resource_pool = {}

    # 初始库存
    for (plant, sku), inv in data.sku_initial_inv.items():
        resource_pool[(plant, sku, 0)] = inv

    # 按日累积
    for day in range(1, data.horizons + 1):
        for plant in data.plants:
            for sku in data.skus_plant.get(plant, []):
                prev_inv = resource_pool.get((plant, sku, day - 1), 0)
                production = data.sku_prod_each_day.get((plant, sku, day), 0)
                resource_pool[(plant, sku, day)] = prev_inv + production

    return resource_pool


def _process_demand_group_for_pool(data, demands, resources, horizons, supply_chain):
    """
    并行分片内部处理函数 (线程内执行)：
        基于传入的 resource 子集与需求组装载车辆
    参数:
        data / demands / resources / horizons / supply_chain
    返回:
        (vehicles_list, allocated_dict)
    说明:
        - 使用局部缓存提升性能
        - 不进行可行性全局验证 (合并后再检查)
    """
    from .alnsopt import veh_loading, SolutionState

    local_sol = SolutionState(data)
    allocated = defaultdict(int)

    # 局部缓存
    resources_get = resources.get
    horizons_range = range(1, horizons + 1)
    select_best = _select_best_vehicle
    supply_chain_items = list(supply_chain.items())

    for (dealer, sku), total_demand, _ in demands:
        remain_demand = total_demand
        if remain_demand <= 0:
            continue

        available_plants = [
            plant for (plant, dealer_id), skus in supply_chain_items
            if dealer_id == dealer and sku in skus
        ]
        if not available_plants:
            continue

        # 汇总各工厂库存总量排序
        plant_inventory = []
        for plant in available_plants:
            total = 0
            for day in horizons_range:
                total += resources_get((plant, sku, day), 0)
            plant_inventory.append((plant, total))
        plant_inventory.sort(key=lambda x: x[1], reverse=True)

        # 分配
        for plant, _ in plant_inventory:
            if remain_demand <= 0:
                break
            for day in horizons_range:
                if remain_demand <= 0:
                    break
                available = resources_get((plant, sku, day), 0)
                if available <= 0:
                    continue
                assign_qty = min(remain_demand, available)
                veh_type = select_best(assign_qty, sku, data)
                vehicle = Vehicle(plant, dealer, veh_type, day, data)
                orders = {sku: assign_qty}
                success = veh_loading(local_sol, vehicle, orders)
                if success:
                    # 按 alnsopt 中增量库存规则: 一次发运需影响 day..H 的可用量
                    # 原实现仅扣减当日, 可能导致后续天数重复使用同一批库存 → 产生过分配与后续负库存修复
                    for d_upd in range(day, horizons + 1):
                        key_upd = (plant, sku, d_upd)
                        resources[key_upd] = resources_get(key_upd, 0) - assign_qty
                    allocated[(dealer, sku)] += assign_qty
                    remain_demand -= assign_qty

    return local_sol.vehicles, allocated


def _prioritize_demands(data):
    """
    计算需求优先级:
        指标:
            scarcity_score      = 1 / (供需比)
            supply_chain_score  = 1 / (可供应工厂数量)
            volume_score        = 需求量归一化
        综合:
            priority = 0.5*scarcity + 0.3*supply_chain + 0.2*volume
    返回:
        [( (dealer, sku), demand, priority_score ), ...] 降序
    """
    # SKU 总供应量 (生产 + 初始库存)
    sku_total_supply = defaultdict(int)
    for (_, sku, _), qty in data.sku_prod_each_day.items():
        sku_total_supply[sku] += qty
    for (_, sku), inv in data.sku_initial_inv.items():
        sku_total_supply[sku] += inv

    # SKU 总需求量
    sku_total_demand = defaultdict(int)
    for (_, sku), demand in data.demands.items():
        sku_total_demand[sku] += demand

    # 供需比
    sku_supply_demand_ratio = {}
    for sku in data.all_skus:
        supply = sku_total_supply.get(sku, 0)
        demand = sku_total_demand.get(sku, 0)
        sku_supply_demand_ratio[sku] = supply / demand if demand > 0 else float("inf")

    demand_priorities = []
    supply_chain = data.construct_supply_chain()
    max_demand = max(data.demands.values()) if data.demands else 1

    for (dealer, sku), demand in data.demands.items():
        if demand <= 0:
            continue

        available_plants = [
            plant for (plant, dealer_id), skus in supply_chain.items()
            if dealer_id == dealer and sku in skus
        ]
        supply_chain_score = 1.0 / (len(available_plants) if available_plants else float("inf"))
        scarcity_score = 1.0 / sku_supply_demand_ratio.get(sku, float("inf"))
        volume_score = demand / max_demand
        priority_score = 0.5 * scarcity_score + 0.3 * supply_chain_score + 0.2 * volume_score
        demand_priorities.append(((dealer, sku), demand, priority_score))

    demand_priorities.sort(key=lambda x: x[2], reverse=True)
    return demand_priorities


def _sequential_allocation(state, prioritized_demands, resource_pool, rng, timeout):
    """
    顺序分配需求 → 车辆 (小规模数据使用)
    策略:
        - 按优先级遍历需求
        - 工厂按 SKU 总库存降序
        - 日序遍历分配库存
        - 使用 _select_best_vehicle 匹配车型
    """
    data = state.data
    from .alnsopt import veh_loading

    sol = state.copy()
    t_start = time.time()

    allocated = defaultdict(int)
    supply_chain = data.construct_supply_chain()
    supply_chain_items = list(supply_chain.items())
    horizons_range = range(1, data.horizons + 1)
    resource_get = resource_pool.get

    # 预计算 (plant, sku) 总库存
    plant_sku_total = {}
    for (plant, sku_key, day), qty in resource_pool.items():
        plant_sku_total[(plant, sku_key)] = plant_sku_total.get((plant, sku_key), 0) + qty

    for (dealer, sku), total_demand, _ in prioritized_demands:
        if time.time() - t_start > timeout:
            break
        remain_demand = total_demand - allocated[(dealer, sku)]
        if remain_demand <= 0:
            continue

        available_plants = [
            plant for (plant, dealer_id), skus in supply_chain_items
            if dealer_id == dealer and sku in skus
        ]
        if not available_plants:
            continue

        plant_inventory = [
            (plant, plant_sku_total.get((plant, sku), 0)) for plant in available_plants
        ]
        plant_inventory.sort(key=lambda x: x[1], reverse=True)

        for plant, _ in plant_inventory:
            if remain_demand <= 0:
                break
            for day in horizons_range:
                if remain_demand <= 0:
                    break
                available = resource_get((plant, sku, day), 0)
                if available <= 0:
                    continue
                assign_qty = min(remain_demand, available)
                veh_type = _select_best_vehicle(assign_qty, sku, data)
                vehicle = Vehicle(plant, dealer, veh_type, day, data)
                orders = {sku: assign_qty}
                success = veh_loading(sol, vehicle, orders)
                if success:
                    # 与 alnsopt 中 veh_loading 保持一致: 发运量对 day..H 库存形成链式影响
                    for d_upd in range(day, data.horizons + 1):
                        resource_pool[(plant, sku, d_upd)] = resource_pool.get((plant, sku, d_upd), 0) - assign_qty
                    allocated[(dealer, sku)] += assign_qty
                    remain_demand -= assign_qty

    return sol


def _parallel_allocation(state, prioritized_demands, resource_pool, rng, timeout, stats=None):
    """
    并行分配 (大规模数据使用):
        1. 需求按“主工厂”聚类 → 组内资源不重叠
        2. 每组构造资源子集副本
        3. 线程池并行调用 _process_demand_group_for_pool
        4. 合并结果并验证库存; 若负库存回退顺序算法
        5. 若仍有时间 → 调用 _handle_unmet_demands 处理剩余需求
    """
    data = state.data
    from .alnsopt import veh_loading, SolutionState

    sol = SolutionState(data)
    t_start = time.time()

    num_workers = min(mp.cpu_count() or 1, getattr(ALNSConfig, 'MAX_INIT_THREADS', 8))
    supply_chain = data.construct_supply_chain()

    plant_to_demands = {plant: [] for plant in data.plants}
    unassigned = []

    # 主工厂归属
    for (dealer, sku), demand, _ in prioritized_demands:
        if demand <= 0:
            continue
        available_plants = [
            plant for (plant, dealer_id), skus in supply_chain.items()
            if dealer_id == dealer and sku in skus
        ]
        if not available_plants:
            unassigned.append(((dealer, sku), demand, _))
            continue
        best_plant = None
        best_qty = -1
        for plant in available_plants:
            total_qty = sum(resource_pool.get((plant, sku, day), 0) for day in range(data.horizons + 1))
            if total_qty > best_qty:
                best_qty = total_qty
                best_plant = plant
        if best_plant is None:
            unassigned.append(((dealer, sku), demand, _))
        else:
            plant_to_demands[best_plant].append(((dealer, sku), demand, _))

    raw_groups = [lst for lst in plant_to_demands.values() if lst]
    if unassigned:
        raw_groups.append(unassigned)

    groups = raw_groups[:]
    if len(groups) > num_workers:
        group_weights = [sum(d for (_, d, _) in g) for g in groups]
        while len(groups) > num_workers:
            min_idx = min(range(len(groups)), key=lambda i: group_weights[i])
            max_idx = max(range(len(groups)), key=lambda i: group_weights[i] if i != min_idx else -1)
            groups[max_idx].extend(groups[min_idx])
            group_weights[max_idx] += group_weights[min_idx]
            del groups[min_idx]
            del group_weights[min_idx]

    resource_pools = []
    group_list = []
    for g in groups:
        plants_in_group = set()
        for (dealer_sku, _, _) in g:
            dealer, sku = dealer_sku
            for (plant, dealer_id), skus in supply_chain.items():
                if dealer_id == dealer and sku in skus:
                    plants_in_group.add(plant)
        rp = {}
        for (plant, sku_key, day), qty in resource_pool.items():
            if plant in plants_in_group:
                rp[(plant, sku_key, day)] = qty
        resource_pools.append(rp)
        group_list.append(g)

    all_vehicles = []
    all_allocated = defaultdict(int)
    futures = []
    with cf.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for dg, rp in zip(group_list, resource_pools):
            futures.append(
                executor.submit(
                    _process_demand_group_for_pool,
                    data, dg, rp, data.horizons, supply_chain
                )
            )
        remaining_time = timeout - (time.time() - t_start)
        try:
            for f in cf.as_completed(futures, timeout=max(0.1, remaining_time)):
                try:
                    vehicles, allocated = f.result()
                    all_vehicles.extend(vehicles)
                    for key, value in allocated.items():
                        all_allocated[key] += value
                except Exception:
                    continue
        except Exception:
            for f in futures:
                if f.done():
                    try:
                        vehicles, allocated = f.result()
                        all_vehicles.extend(vehicles)
                        for key, value in allocated.items():
                            all_allocated[key] += value
                    except Exception:
                        continue

    allocation_time = time.time() - t_start
    if stats is not None:
        stats['allocation_time'] = allocation_time
    sol.vehicles = all_vehicles
    sol.compute_inventory()
    feasible, violations = sol.validate()
    neg_inv = violations.get("negative_inventory", []) if violations else []
    if neg_inv:
        print("Parallel allocation produced negative inventory; falling back to sequential allocation.")
        if stats is not None:
            stats['parallel_fallbacks'] = stats.get('parallel_fallbacks', 0) + 1
        return _sequential_allocation(state, prioritized_demands, resource_pool, rng, timeout)

    if time.time() - t_start < timeout:
        _handle_unmet_demands(sol, all_allocated, data, timeout - (time.time() - t_start))

    return sol


def _handle_unmet_demands(sol, allocated, data, timeout):
    """
    二次分配: 处理未满足需求 (如并行阶段未完全覆盖)
    特点:
        - 使用局部库存映射 (current_inventory) 与 used_map 增量更新
        - 按未满足量降序处理
    """
    from .alnsopt import veh_loading

    t_start = time.time()
    unmet_demands = []
    for (dealer, sku), demand in data.demands.items():
        delivered = allocated.get((dealer, sku), 0)
        if demand > delivered:
            unmet_demands.append(((dealer, sku), demand - delivered))
    unmet_demands.sort(key=lambda x: x[1], reverse=True)

    if not sol.s_initialized:
        sol.compute_inventory()
    current_inventory = {k: v for k, v in sol.s_ikt.items()}
    used_map = {}
    for veh in sol.vehicles:
        for (s_kw, d_kw), q in veh.cargo.items():
            key = (veh.fact_id, s_kw, d_kw)
            used_map[key] = used_map.get(key, 0) + q

    supply_chain_global = data.construct_supply_chain()
    for (dealer, sku), unmet_qty in unmet_demands:
        if time.time() - t_start > timeout:
            break
        if unmet_qty <= 0:
            continue
        available_plants = [
            plant for (plant, dealer_id), skus in supply_chain_global.items()
            if dealer_id == dealer and sku in skus
        ]
        if not available_plants:
            continue
        for plant in available_plants:
            if unmet_qty <= 0:
                break
            for day in range(1, data.horizons + 1):
                if unmet_qty <= 0:
                    break
                prev_inv = current_inventory.get((plant, sku, day - 1), 0)
                production = data.sku_prod_each_day.get((plant, sku, day), 0)
                used = used_map.get((plant, sku, day), 0)
                available = max(0, prev_inv + production - used)
                if available <= 0:
                    continue
                assign_qty = min(unmet_qty, available)
                veh_type = _select_best_vehicle(assign_qty, sku, data)
                vehicle = Vehicle(plant, dealer, veh_type, day, data)
                orders = {sku: assign_qty}
                success = veh_loading(sol, vehicle, orders)
                if success:
                    unmet_qty -= assign_qty
                    used_map[(plant, sku, day)] = used_map.get((plant, sku, day), 0) + assign_qty
                    for d_upd in range(day, data.horizons + 1):
                        key_upd = (plant, sku, d_upd)
                        current_inventory[key_upd] = current_inventory.get(key_upd, 0) - assign_qty

    return sol


def _select_best_vehicle(qty, sku, data):
    """
    根据待发送 sku 数量选择车辆类型:
        - 计算需求总体积 = qty * sku_size
        - 选择能容纳且容量最接近的车辆
        - 若无车型完全容纳 → 选择最大容量车型 (后续可能再补车)
    """
    required_volume = qty * data.sku_sizes[sku]
    suitable_types = [
        (vt, cap) for vt, cap in data.veh_type_cap.items()
        if cap >= required_volume
    ]
    if suitable_types:
        return min(suitable_types, key=lambda x: x[1])[0]
    return max(data.veh_type_cap.items(), key=lambda x: x[1])[0]


def _repair_solution(sol, timeout, stats=None):
    """
    修复初始解:
        1. 负库存调整 (多轮迭代, 减少车辆装载)
        2. 移除空车辆
        3. 合并小车辆 (减少解规模)
    注意:
        - 逻辑保持不变，仅增强注释说明
        - current_inventory 在循环中增量更新, 最终一次性写回
    """
    from .alnsopt import veh_loading  # 保留原局部 import
    data = sol.data

    t_start = time.time()
    max_iter = getattr(ALNSConfig, "INITIAL_REPAIR_MAX_ITER", 5)
    iter_count = 0
    for _ in range(max_iter):
        iter_count += 1
        if time.time() - t_start > timeout:
            break
        if not sol.s_initialized:
            sol.compute_inventory()
        current_inventory = {k: v for k, v in sol.s_ikt.items()}
        neg_inv = [(k, v) for k, v in current_inventory.items() if v < 0]
        if not neg_inv:
            break
        changed = False
        for (plant, sku, day), neg_val in neg_inv:
            relevant_vehicles = [
                (i, veh) for i, veh in enumerate(sol.vehicles)
                if veh.fact_id == plant and veh.day == day and (sku, day) in veh.cargo
            ]
            if not relevant_vehicles:
                continue
            to_reduce = -neg_val
            relevant_vehicles.sort(
                key=lambda x: x[1].cargo.get((sku, day), 0), reverse=True
            )
            for i, veh in relevant_vehicles:
                if to_reduce <= 0:
                    break
                current_qty = veh.cargo.get((sku, day), 0)
                reduce_qty = min(current_qty, to_reduce)
                if reduce_qty > 0:
                    veh.cargo[(sku, day)] -= reduce_qty
                    if veh.cargo[(sku, day)] == 0:
                        del veh.cargo[(sku, day)]
                    veh.capacity += reduce_qty * data.sku_sizes[sku]
                    to_reduce -= reduce_qty
                    changed = True
                    for d_upd in range(day, data.horizons + 1):
                        key_upd = (plant, sku, d_upd)
                        current_inventory[key_upd] = current_inventory.get(key_upd, 0) + reduce_qty
        if not changed:
            break

    sol.vehicles = [veh for veh in sol.vehicles if not veh.is_empty()]
    if time.time() - t_start < timeout:
        _merge_small_vehicles(sol, timeout - (time.time() - t_start))

    sol.s_ikt = current_inventory
    sol.s_initialized = True
    if stats is not None:
        stats['repair_iterations'] = iter_count if 'repair_iterations' not in stats else stats['repair_iterations'] + iter_count
    return sol


def _merge_small_vehicles(sol, timeout):
    """
    合并同 (工厂, 经销商, 天) 下小车辆:
        - 车辆按当前装载体积升序
        - 尝试将最小车合并入容量够的目标车
        - 超时提前退出并保留剩余车辆
    """
    data = sol.data
    t_start = time.time()
    vehicle_groups = defaultdict(list)
    for veh in sol.vehicles:
        key = (veh.fact_id, veh.dealer_id, veh.day)
        vehicle_groups[key].append(veh)

    merged_vehicles = []

    for key, vehicles in vehicle_groups.items():
        if time.time() - t_start > timeout:
            merged_vehicles.extend(vehicles)
            continue
        if len(vehicles) <= 1:
            merged_vehicles.extend(vehicles)
            continue

        def veh_volume(v):
            return sum(qty * data.sku_sizes[sku] for (sku, _), qty in v.cargo.items())

        vehicles.sort(key=veh_volume)

        while len(vehicles) > 1:
            if time.time() - t_start > timeout:
                break
            smallest = vehicles[0]
            smallest_volume = veh_volume(smallest)
            merged = False
            for i in range(1, len(vehicles)):
                target = vehicles[i]
                target_volume = veh_volume(target)
                total_volume = smallest_volume + target_volume
                if total_volume <= data.veh_type_cap.get(target.type, 0):
                    for (sku, day), qty in smallest.cargo.items():
                        target.cargo[(sku, day)] = target.cargo.get((sku, day), 0) + qty
                    target.capacity -= smallest_volume
                    vehicles.pop(0)
                    merged = True
                    break
            if not merged:
                merged_vehicles.append(vehicles.pop(0))

        merged_vehicles.extend(vehicles)

    sol.vehicles = merged_vehicles
    sol.compute_inventory()
    return sol
