"""
initial_solution_utils.py
All functions related to initial solution generation for ALNS, for clear and maintainable workflow.
"""
import time
import multiprocessing as mp
from collections import defaultdict
import numpy as np
from ALNSCode.vehicle import Vehicle


def improved_initial_solution(state, rng):
    """
    Generate an initial feasible solution using a multi-stage approach optimized for large datasets.
    
    Key improvements:
    1. Proper inventory tracking across periods
    2. Resource-aware demand allocation
    3. Parallel processing for large datasets
    4. Efficient vehicle utilization
    5. Adaptive timeout protection
    
    Args:
        state: SolutionState, the initial solution state object.
        rng: numpy random generator.
    Returns:
        SolutionState, a feasible initial solution.
    """
    data = state.data
    from ALNSCode.vehicle import Vehicle
    from ALNSCode.alnsopt import veh_loading, SolutionState  # avoid circular import
    
    # 设置超时保护
    timeout = min(120, max(30, len(data.demands) * 0.05))  # 根据问题规模自适应超时时间
    t_start = time.time()
    
    # 1. 初始化资源池 - 跟踪每个(工厂,SKU,天)的可用库存
    print("Initializing resource pool...")
    t0 = time.time()
    resource_pool = _initialize_resource_pool(data)
    print(f" Resource pool initialized in {time.time() - t0:.2f} seconds.")
    
    # 2. 优先级排序需求
    print("Prioritizing demands...")
    t0 = time.time()
    prioritized_demands = _prioritize_demands(data)
    print(f" Demands prioritized in {time.time() - t0:.2f} seconds.")
    
    # 3. 根据数据集大小决定是否使用并行处理
    if len(prioritized_demands) > 1000:  # 大型数据集
        print("Using parallel allocation for large dataset...")
        t0 = time.time()
        sol = _parallel_allocation(state, prioritized_demands, resource_pool, rng, timeout)
        print(f" Parallel allocation completed in {time.time() - t0:.2f} seconds.")
    else:  # 小型数据集
        print("Using sequential allocation for small dataset...")
        t0 = time.time()
        sol = _sequential_allocation(state, prioritized_demands, resource_pool, rng, timeout)
        print(f" Sequential allocation completed in {time.time() - t0:.2f} seconds.")
    
    # 4. 验证解并修复任何剩余问题
    print("Repairing solution if necessary...")
    t0 = time.time()
    _repair_solution(sol, timeout - (time.time() - t_start))
    print(f" Repair completed in {time.time() - t0:.2f} seconds.")
    
    # 5. 最终验证
    sol.compute_inventory()
    return sol


def _initialize_resource_pool(data):
    """
    初始化资源池，跟踪每个(工厂,SKU,天)的可用库存
    考虑期初库存和每日生产量
    """
    resource_pool = {}
    
    # 首先初始化期初库存(day=0)
    for (plant, sku), inv in data.sku_initial_inv.items():
        resource_pool[(plant, sku, 0)] = inv
    
    # 然后计算每天的累积可用库存
    for day in range(1, data.horizons + 1):
        for plant in data.plants:
            for sku in data.skus_plant.get(plant, []):
                # 前一天的库存
                prev_inv = resource_pool.get((plant, sku, day-1), 0)
                # 当天的生产量
                production = data.sku_prod_each_day.get((plant, sku, day), 0)
                # 当天的可用库存 = 前一天库存 + 当天生产
                resource_pool[(plant, sku, day)] = prev_inv + production
    
    return resource_pool


def _prioritize_demands(data):
    """
    对需求进行优先级排序
    考虑需求量、SKU稀缺性和供应链约束
    """
    # 计算每个SKU的总供应量
    sku_total_supply = defaultdict(int)
    for (plant, sku, day), qty in data.sku_prod_each_day.items():
        sku_total_supply[sku] += qty
    
    # 加上期初库存
    for (plant, sku), inv in data.sku_initial_inv.items():
        sku_total_supply[sku] += inv
    
    # 计算每个SKU的总需求量
    sku_total_demand = defaultdict(int)
    for (dealer, sku), demand in data.demands.items():
        sku_total_demand[sku] += demand
    
    # 计算每个SKU的供需比
    sku_supply_demand_ratio = {}
    for sku in data.all_skus:
        supply = sku_total_supply.get(sku, 0)
        demand = sku_total_demand.get(sku, 0)
        if demand > 0:
            sku_supply_demand_ratio[sku] = supply / demand
        else:
            sku_supply_demand_ratio[sku] = float('inf')
    
    # 计算每个需求的优先级分数
    demand_priorities = []
    for (dealer, sku), demand in data.demands.items():
        if demand <= 0:
            continue
            
        # 计算供应链约束分数 - 该经销商可从多少工厂获取该SKU
        supply_chain = data.construct_supply_chain()
        available_plants = [plant for (plant, dealer_id), skus in supply_chain.items() 
                           if dealer_id == dealer and sku in skus]
        supply_chain_score = 1.0 / (len(available_plants) if available_plants else float('inf'))
        
        # 计算稀缺性分数 - 供需比的倒数
        scarcity_score = 1.0 / sku_supply_demand_ratio.get(sku, float('inf'))
        
        # 计算需求量分数 - 归一化需求量
        volume_score = demand / max(data.demands.values())
        
        # 综合优先级分数 (权重可调整)
        priority_score = (0.5 * scarcity_score + 0.3 * supply_chain_score + 0.2 * volume_score)
        
        demand_priorities.append(((dealer, sku), demand, priority_score))
    
    # 按优先级降序排序
    demand_priorities.sort(key=lambda x: x[2], reverse=True)
    return demand_priorities


def _sequential_allocation(state, prioritized_demands, resource_pool, rng, timeout):
    """
    顺序分配需求到车辆
    适用于小型数据集
    """
    data = state.data
    from ALNSCode.alnsopt import veh_loading
    
    sol = state.copy()
    t_start = time.time()
    
    # 跟踪已分配的需求量
    allocated = defaultdict(int)
    
    # 按优先级处理需求
    for (dealer, sku), total_demand, _ in prioritized_demands:
        if time.time() - t_start > timeout:
            break
            
        remain_demand = total_demand - allocated[(dealer, sku)]
        if remain_demand <= 0:
            continue
        
        # 获取可供应的工厂
        supply_chain = data.construct_supply_chain()
        available_plants = [plant for (plant, dealer_id), skus in supply_chain.items() 
                           if dealer_id == dealer and sku in skus]
        
        if not available_plants:
            continue
        
        # 按库存量降序排序工厂
        plant_inventory = [(plant, sum(resource_pool.get((plant, sku, day), 0) 
                                      for day in range(data.horizons + 1)))
                          for plant in available_plants]
        plant_inventory.sort(key=lambda x: x[1], reverse=True)
        
        # 尝试从每个工厂分配
        for plant, _ in plant_inventory:
            if remain_demand <= 0:
                break
                
            # 按天数顺序分配
            for day in range(1, data.horizons + 1):
                if remain_demand <= 0:
                    break
                    
                # 计算当天可用库存
                available = resource_pool.get((plant, sku, day), 0)
                if available <= 0:
                    continue
                
                # 计算本次分配量
                assign_qty = min(remain_demand, available)
                
                # 选择合适的车型
                veh_type = _select_best_vehicle(assign_qty, sku, data)
                
                # 创建车辆并装载
                vehicle = Vehicle(plant, dealer, veh_type, day, data)
                orders = {sku: assign_qty}
                
                # 装载车辆
                success = veh_loading(sol, vehicle, orders)
                
                if success:
                    # 更新资源池
                    resource_pool[(plant, sku, day)] -= assign_qty
                    # 更新已分配量
                    allocated[(dealer, sku)] += assign_qty
                    # 更新剩余需求
                    remain_demand -= assign_qty
    
    return sol


def _parallel_allocation(state, prioritized_demands, resource_pool, rng, timeout):
    """
    并行分配需求到车辆
    适用于大型数据集
    """
    data = state.data
    from ALNSCode.alnsopt import veh_loading, SolutionState
    
    # 创建一个新的解对象
    sol = SolutionState(data)
    t_start = time.time()
    
    # 确定处理器数量
    num_processors = min(mp.cpu_count(), 8)  # 最多使用8个核心
    
    # 将需求分组
    demand_groups = []
    group_size = max(1, len(prioritized_demands) // num_processors)
    for i in range(0, len(prioritized_demands), group_size):
        demand_groups.append(prioritized_demands[i:i+group_size])
    
    # 为每个进程准备资源池的副本
    resource_pools = [resource_pool.copy() for _ in range(len(demand_groups))]
    
    # 定义并行处理函数
    def process_demand_group(demands, resources):
        local_sol = SolutionState(data)
        allocated = defaultdict(int)
        
        for (dealer, sku), total_demand, _ in demands:
            remain_demand = total_demand
            if remain_demand <= 0:
                continue
            
            # 获取可供应的工厂
            supply_chain = data.construct_supply_chain()
            available_plants = [plant for (plant, dealer_id), skus in supply_chain.items() 
                               if dealer_id == dealer and sku in skus]
            
            if not available_plants:
                continue
            
            # 按库存量降序排序工厂
            plant_inventory = [(plant, sum(resources.get((plant, sku, day), 0) 
                                          for day in range(data.horizons + 1)))
                              for plant in available_plants]
            plant_inventory.sort(key=lambda x: x[1], reverse=True)
            
            # 尝试从每个工厂分配
            for plant, _ in plant_inventory:
                if remain_demand <= 0:
                    break
                    
                # 按天数顺序分配
                for day in range(1, data.horizons + 1):
                    if remain_demand <= 0:
                        break
                        
                    # 计算当天可用库存
                    available = resources.get((plant, sku, day), 0)
                    if available <= 0:
                        continue
                    
                    # 计算本次分配量
                    assign_qty = min(remain_demand, available)
                    
                    # 选择合适的车型
                    veh_type = _select_best_vehicle(assign_qty, sku, data)
                    
                    # 创建车辆并装载
                    vehicle = Vehicle(plant, dealer, veh_type, day, data)
                    orders = {sku: assign_qty}
                    
                    # 装载车辆
                    success = veh_loading(local_sol, vehicle, orders)
                    
                    if success:
                        # 更新资源池
                        resources[(plant, sku, day)] -= assign_qty
                        # 更新已分配量
                        allocated[(dealer, sku)] += assign_qty
                        # 更新剩余需求
                        remain_demand -= assign_qty
        
        return local_sol.vehicles, allocated
    
    # 使用线程池并行处理
    results = []
    with mp.Pool(processes=num_processors) as pool:
        for i in range(len(demand_groups)):
            results.append(pool.apply_async(process_demand_group, 
                                          (demand_groups[i], resource_pools[i])))
        
        # 等待所有进程完成或超时
        remaining_time = timeout - (time.time() - t_start)
        for result in results:
            try:
                result.wait(timeout=max(1, remaining_time / len(results)))
            except:
                pass
    
    # 合并结果
    all_vehicles = []
    all_allocated = defaultdict(int)
    
    for result in results:
        if result.ready():
            try:
                vehicles, allocated = result.get(timeout=0.1)
                all_vehicles.extend(vehicles)
                for key, value in allocated.items():
                    all_allocated[key] += value
            except:
                pass
    
    # 将合并的车辆添加到解中
    sol.vehicles = all_vehicles
    
    # 处理未满足的需求
    if time.time() - t_start < timeout:
        _handle_unmet_demands(sol, all_allocated, data, timeout - (time.time() - t_start))
    
    return sol


def _handle_unmet_demands(sol, allocated, data, timeout):
    """
    处理未满足的需求
    """
    from ALNSCode.alnsopt import veh_loading
    
    t_start = time.time()
    
    # 找出未满足的需求
    unmet_demands = []
    for (dealer, sku), demand in data.demands.items():
        if demand > allocated.get((dealer, sku), 0):
            unmet_qty = demand - allocated.get((dealer, sku), 0)
            unmet_demands.append(((dealer, sku), unmet_qty))
    
    # 按未满足量降序排序
    unmet_demands.sort(key=lambda x: x[1], reverse=True)
    
    # 计算当前库存状态
    sol.compute_inventory()
    current_inventory = {}
    for (plant, sku, day), inv in sol.s_ikt.items():
        current_inventory[(plant, sku, day)] = inv
    
    # 尝试满足未满足的需求
    for (dealer, sku), unmet_qty in unmet_demands:
        if time.time() - t_start > timeout:
            break
            
        if unmet_qty <= 0:
            continue
        
        # 获取可供应的工厂
        supply_chain = data.construct_supply_chain()
        available_plants = [plant for (plant, dealer_id), skus in supply_chain.items() 
                           if dealer_id == dealer and sku in skus]
        
        if not available_plants:
            continue
        
        # 尝试从每个工厂分配
        for plant in available_plants:
            if unmet_qty <= 0:
                break
                
            # 按天数顺序分配
            for day in range(1, data.horizons + 1):
                if unmet_qty <= 0:
                    break
                    
                # 计算当天可用库存 (前一天库存 + 当天生产)
                prev_inv = current_inventory.get((plant, sku, day-1), 0)
                production = data.sku_prod_each_day.get((plant, sku, day), 0)
                
                # 计算已使用的库存
                used = sum(veh.cargo.get((sku, day), 0) for veh in sol.vehicles 
                          if veh.fact_id == plant and veh.day == day)
                
                available = max(0, prev_inv + production - used)
                
                if available <= 0:
                    continue
                
                # 计算本次分配量
                assign_qty = min(unmet_qty, available)
                
                # 选择合适的车型
                veh_type = _select_best_vehicle(assign_qty, sku, data)
                
                # 创建车辆并装载
                vehicle = Vehicle(plant, dealer, veh_type, day, data)
                orders = {sku: assign_qty}
                
                # 装载车辆
                success = veh_loading(sol, vehicle, orders)
                
                if success:
                    # 更新未满足量
                    unmet_qty -= assign_qty
                    # 更新库存状态
                    sol.compute_inventory()
                    for (p, s, d), inv in sol.s_ikt.items():
                        current_inventory[(p, s, d)] = inv
    
    return sol


def _select_best_vehicle(qty, sku, data):
    """
    根据需求量选择最合适的车型
    """
    # 计算所需容量
    required_volume = qty * data.sku_sizes[sku]
    
    # 找到能满足需求的最小车型
    suitable_types = [(vt, cap) for vt, cap in data.veh_type_cap.items() 
                     if cap >= required_volume]
    
    if suitable_types:
        # 选择容量刚好够用的车型（避免浪费）
        return min(suitable_types, key=lambda x: x[1])[0]
    else:
        # 如果没有车型能完全满足, 选择最大的车型
        return max(data.veh_type_cap.items(), key=lambda x: x[1])[0]


def _repair_solution(sol, timeout):
    """
    修复解中的问题
    - 处理负库存
    - 移除空车辆
    - 合并小车辆
    """
    from ALNSCode.alnsopt import veh_loading
    data = sol.data
    
    t_start = time.time()
    
    # 1. 处理负库存
    max_iter = 5
    for _ in range(max_iter):
        if time.time() - t_start > timeout:
            break
            
        sol.compute_inventory()
        neg_inv = [(k, v) for k, v in sol.s_ikt.items() if v < 0]
        
        if not neg_inv:
            break
            
        changed = False
        for (plant, sku, day), neg_val in neg_inv:
            # 找出从该工厂在该天发出该SKU的车辆
            relevant_vehicles = [(i, veh) for i, veh in enumerate(sol.vehicles) 
                               if veh.fact_id == plant and veh.day == day 
                               and (sku, day) in veh.cargo]
            
            if not relevant_vehicles:
                continue
                
            # 计算需要减少的总量
            to_reduce = -neg_val
            
            # 按装载量降序排序车辆
            relevant_vehicles.sort(key=lambda x: x[1].cargo.get((sku, day), 0), reverse=True)
            
            # 从车辆中减少装载量
            for i, veh in relevant_vehicles:
                if to_reduce <= 0:
                    break
                    
                current_qty = veh.cargo.get((sku, day), 0)
                reduce_qty = min(current_qty, to_reduce)
                
                if reduce_qty > 0:
                    veh.cargo[(sku, day)] -= reduce_qty
                    if veh.cargo[(sku, day)] == 0:
                        del veh.cargo[(sku, day)]
                    
                    # 更新车辆容量
                    veh.capacity += reduce_qty * data.sku_sizes[sku]
                    
                    # 更新待减少量
                    to_reduce -= reduce_qty
                    changed = True
            
        # 如果没有变化，跳出循环
        if not changed:
            break
    
    # 2. 移除空车辆
    sol.vehicles = [veh for veh in sol.vehicles if not veh.is_empty()]
    
    # 3. 尝试合并小车辆以减少车辆数量
    if time.time() - t_start < timeout:
        _merge_small_vehicles(sol, timeout - (time.time() - t_start))
    
    sol.compute_inventory()
    return sol


def _merge_small_vehicles(sol, timeout):
    """
    合并小车辆以减少车辆数量
    """
    data = sol.data
    t_start = time.time()
    
    # 按(工厂,经销商,天)分组车辆
    vehicle_groups = defaultdict(list)
    for veh in sol.vehicles:
        key = (veh.fact_id, veh.dealer_id, veh.day)
        vehicle_groups[key].append(veh)
    
    # 处理每个分组
    merged_vehicles = []
    for key, vehicles in vehicle_groups.items():
        if time.time() - t_start > timeout:
            # 超时时添加剩余车辆
            merged_vehicles.extend(vehicles)
            continue
        
        if len(vehicles) <= 1:
            merged_vehicles.extend(vehicles)
            continue
        
        # 按装载量升序排序
        vehicles.sort(key=lambda v: sum(qty * data.sku_sizes[sku] for (sku, _), qty in v.cargo.items()))
        
        # 尝试合并
        while len(vehicles) > 1:
            if time.time() - t_start > timeout:
                break
                
            smallest = vehicles[0]
            
            # 找到可以合并的目标车辆
            merged = False
            for i in range(1, len(vehicles)):
                target = vehicles[i]
                
                # 检查是否可以合并
                total_volume = sum(qty * data.sku_sizes[sku] for (sku, _), qty in smallest.cargo.items())
                target_volume = sum(qty * data.sku_sizes[sku] for (sku, _), qty in target.cargo.items())
                
                if total_volume + target_volume <= data.veh_type_cap[target.type]:
                    # 可以合并
                    for (sku, day), qty in smallest.cargo.items():
                        target.cargo[(sku, day)] = target.cargo.get((sku, day), 0) + qty
                    
                    # 更新目标车辆容量
                    target.capacity -= total_volume
                    
                    # 移除已合并的车辆
                    vehicles.pop(0)
                    merged = True
                    break
            
            if not merged:
                # 无法合并，保留当前车辆
                merged_vehicles.append(vehicles.pop(0))
        
        # 添加剩余车辆
        merged_vehicles.extend(vehicles)
    
    # 更新解的车辆列表
    sol.vehicles = merged_vehicles
    sol.compute_inventory()
    
    return sol
