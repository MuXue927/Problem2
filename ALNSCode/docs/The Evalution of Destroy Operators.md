# The question that I asked

Please help me evaluate the validity of each destruction operator in `alnsopt.py`, examining potential flaws in the underlying design philosophy. If such issues are identified, please report them and analyze the causes of these potential defects, ultimately attempting to fix them.

# Answered By Claude Sonnet 3.7 in VSCode Copilot

## 随机移除算子 (random_removal)

分析：
- 设计理念：随机移除一定比例的车辆，引入随机性和多样性。
- 实现方式：使用 rng.choice 随机选择一定比例的车辆索引进行移除。
- 潜在问题：
    1. 缺乏针对性：纯随机移除没有任何启发式指导，可能移除掉好的解决方案部分。
    2. s_ikt 更新方式：在移除车辆后，使用循环增量更新库存，效率较低。
    3. 缺乏参数自适应：移除比例 degree 是固定的，没有根据迭代状态动态调整。
改进建议：
```python
def random_removal(current: SolutionState, rng: rnd.Generator, degree: float = 0.25):
    t0 = time.time()
    print(f"[OPLOG] 开始执行 random_removal 算子")
    state = current.copy()
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] random_removal: {time.time() - t0:.4f}s")
        return state
    
    # 自适应移除比例：在探索初期使用较大的移除比例，后期减小
    if state.tracker:
        iteration = state.tracker.get_statistics().get('total_iterations', 0)
        max_iterations = state.tracker.get_statistics().get('max_iterations', 100)
        if max_iterations > 0 and iteration > 0:
            # 动态调整移除比例：从 degree 逐渐降低到 degree/3
            adaptive_degree = degree * (1 - 0.67 * min(1.0, iteration / (0.7 * max_iterations)))
            degree = max(0.05, adaptive_degree)
            print(f"[OPLOG] 自适应移除比例: {degree:.4f}")
    
    num_remove = max(1, int(len(state.vehicles) * degree))
    
    # 分层随机策略：按成本效益比分组，优先从低效益车辆中移除
    efficiency_groups = []
    for i, veh in enumerate(state.vehicles):
        # 计算车辆效益比：总装载量/车辆成本
        load = state.compute_veh_load(veh)
        cost = state.data.veh_type_cost[veh.type]
        efficiency = load / cost if cost > 0 else 0
        efficiency_groups.append((i, efficiency))
    
    # 按效益排序，分成三组
    efficiency_groups.sort(key=lambda x: x[1])
    group_size = max(1, len(efficiency_groups) // 3)
    
    # 从低效益组中选择60%，中等效益组选择30%，高效益组选择10%
    low_indices = [idx for idx, _ in efficiency_groups[:group_size]]
    mid_indices = [idx for idx, _ in efficiency_groups[group_size:2*group_size]]
    high_indices = [idx for idx, _ in efficiency_groups[2*group_size:]]
    
    # 按比例从各组选择
    low_count = int(num_remove * 0.6)
    mid_count = int(num_remove * 0.3)
    high_count = num_remove - low_count - mid_count
    
    remove_indices = []
    if low_indices and low_count > 0:
        remove_indices.extend(rng.choice(low_indices, min(low_count, len(low_indices)), replace=False))
    if mid_indices and mid_count > 0:
        remove_indices.extend(rng.choice(mid_indices, min(mid_count, len(mid_indices)), replace=False))
    if high_indices and high_count > 0:
        remove_indices.extend(rng.choice(high_indices, min(high_count, len(high_indices)), replace=False))
    
    # 如果选择的数量不够，从剩余车辆中补充
    if len(remove_indices) < num_remove:
        remaining = [i for i in range(len(state.vehicles)) if i not in remove_indices]
        if remaining:
            additional = min(num_remove - len(remove_indices), len(remaining))
            remove_indices.extend(rng.choice(remaining, additional, replace=False))
    
    # 获取要移除的车辆
    removed = [state.vehicles[i] for i in sorted(remove_indices, reverse=True)]
    
    # 更新车辆列表
    state.vehicles = [veh for i, veh in enumerate(state.vehicles) if i not in remove_indices]
    
    # 批量更新库存
    if removed:
        # 确保s_ikt已初始化
        if not state.s_initialized:
            state.compute_inventory()
            
        # 优化的库存更新：按(plant,sku,day)分组批量更新
        inventory_updates = defaultdict(int)
        for veh in removed:
            for (sku_id, d_shipped), qty in veh.cargo.items():
                # 收集所有需要更新的库存变化
                for d in range(d_shipped, state.data.horizons + 1):
                    key = (veh.fact_id, sku_id, d)
                    inventory_updates[key] += qty
        
        # 一次性应用所有库存更新
        for key, qty in inventory_updates.items():
            state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
            
    print(f"[OPLOG] random_removal: 移除了 {len(removed)}/{len(current.vehicles)} 辆车 ({time.time() - t0:.4f}s)")
    return state
```

## 最差移除算子 (worst_removal)

分析:
- 设计理念：移除剩余空间最大的车辆，避免车辆空间浪费。
- 实现方式：计算每个车辆的剩余容量，选择剩余容量最大的车辆进行移除。
- 潜在问题：
    1. 目标不一致：该算子关注的是空间利用率，但问题的目标函数是最小化车辆成本。空间最大的车辆不一定是成本效益最差的。
    2. 键值冲突：使用 (veh.fact_id, veh.dealer_id, veh.type, veh.day) 作为键可能导致不同车辆共享同一个键，造成数据混乱。
    3. 车辆索引计算：在移除车辆时，采用索引反向遍历的方式，当多个车辆共享同一键值时会出现问题。
改进建议：
```python
def worst_removal(current: SolutionState, rng: rnd.Generator, degree: float = 0.25):
    t0 = time.time()
    print(f"[OPLOG] 开始执行 worst_removal 算子")
    state = current.copy()
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] worst_removal: {time.time() - t0:.4f}s")
        return state
    
    # 计算每辆车的空间利用率和成本效益比
    vehicle_metrics = []
    for i, veh in enumerate(state.vehicles):
        # 计算空间利用率
        capacity = state.data.veh_type_cap[veh.type]
        load = state.compute_veh_load(veh)
        space_util = load / capacity if capacity > 0 else 0
        
        # 计算成本效益比（运载价值/成本）
        veh_cost = state.data.veh_type_cost[veh.type]
        
        # 计算车辆运载的货物价值（可以根据需求、紧急程度等加权）
        cargo_value = 0
        for (sku_id, day), qty in veh.cargo.items():
            # 将SKU价值与需求满足程度关联
            dealer = veh.dealer_id
            demand = state.data.demands.get((dealer, sku_id), 0)
            if demand > 0:
                # 计算该SKU对应的需求满足比例
                shipped = sum(v.cargo.get((sku_id, v.day), 0) for v in state.vehicles 
                              if v.dealer_id == dealer)
                satisfaction = min(1.0, shipped / demand)
                # SKU价值考虑：数量、尺寸和需求满足程度
                sku_value = qty * state.data.sku_sizes[sku_id] * (2 - satisfaction)
                cargo_value += sku_value
        
        # 计算综合评分：空间利用率(40%) + 成本效益(60%)
        cost_efficiency = cargo_value / veh_cost if veh_cost > 0 else 0
        score = 0.4 * space_util + 0.6 * cost_efficiency
        
        vehicle_metrics.append((i, score))
    
    # 按综合评分排序，选择最差的一部分
    vehicle_metrics.sort(key=lambda x: x[1])
    
    # 动态确定移除数量
    num_remove = max(1, int(len(state.vehicles) * degree))
    
    # 选择评分最低的车辆，但引入随机性避免确定性行为
    # 使用轮盘赌选择法，概率与评分成反比
    selection_pool = vehicle_metrics[:min(len(vehicle_metrics), int(num_remove * 2))]
    if not selection_pool:
        return state
        
    # 计算选择概率（评分越低，概率越高）
    scores = np.array([1.0 / (1.0 + item[1]) for item in selection_pool])
    probabilities = scores / scores.sum()
    
    # 随机选择车辆索引
    try:
        selected_indices = rng.choice(
            [idx for idx, _ in selection_pool], 
            size=min(num_remove, len(selection_pool)), 
            replace=False, 
            p=probabilities
        )
    except ValueError:
        # 如果概率计算出错，退回到简单随机选择
        selected_indices = rng.choice(
            [idx for idx, _ in selection_pool], 
            size=min(num_remove, len(selection_pool)),
            replace=False
        )
    
    # 获取要移除的车辆
    removed = [state.vehicles[i] for i in sorted(selected_indices, reverse=True)]
    
    # 更新车辆列表
    state.vehicles = [veh for i, veh in enumerate(state.vehicles) if i not in selected_indices]
    
    # 批量更新库存
    if not state.s_initialized:
        state.compute_inventory()
        
    # 优化的库存更新
    inventory_updates = defaultdict(int)
    for veh in removed:
        for (sku_id, d_shipped), qty in veh.cargo.items():
            for d in range(d_shipped, state.data.horizons + 1):
                key = (veh.fact_id, sku_id, d)
                inventory_updates[key] += qty
    
    # 一次性应用所有库存更新
    for key, qty in inventory_updates.items():
        state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
        
    print(f"[OPLOG] worst_removal: 移除了 {len(removed)}/{len(current.vehicles)} 辆车 ({time.time() - t0:.4f}s)")
    return state
```

## 不可行解移除算子 (infeasible_removal)
分析：
- 设计理念：移除导致负库存的车辆，以恢复解的可行性。
- 实现方式：找出当前库存为负的所有 (plant, sku_id, day) 组合，移除对应的车辆。
- 潜在问题：
    1. 过度移除：当多个车辆共同导致一个库存为负时，该算法会移除所有相关车辆，而不是只移除足够恢复可行性的车辆。
    2. 不考虑影响大小：没有考虑每个车辆对负库存的贡献大小，可能移除影响较小的车辆。
    3. 缺乏精确性：简单地移除整个车辆，而不是仅减少导致负库存的特定SKU的装载量。
```python
def infeasible_removal(current: SolutionState, rng: rnd.Generator):
    t0 = time.time()
    print(f"[OPLOG] 开始执行 infeasible_removal 算子")
    state = current.copy()
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] infeasible_removal: {time.time() - t0:.4f}s")
        return state
    
    # 确保库存计算是最新的
    state.compute_inventory()
    
    # 查找负库存
    neg_inv_items = []
    for (plant, sku_id, day), inv in state.s_ikt.items():
        if inv < 0:
            neg_inv_items.append((plant, sku_id, day, inv))
    
    # 如果没有负库存，直接返回
    if not neg_inv_items:
        print(f"[OPLOG] infeasible_removal: 无负库存，不需要修复 ({time.time() - t0:.4f}s)")
        return state
    
    # 按负库存量排序（从最严重到最轻微）
    neg_inv_items.sort(key=lambda x: x[3])
    
    # 针对每个负库存问题，寻找并逐步调整相关车辆
    vehicles_to_remove = []
    partial_adjustments = []  # 记录部分调整：(vehicle_index, sku_id, day, reduce_qty)
    
    for plant, sku_id, day, neg_amount in neg_inv_items:
        # 寻找相关车辆：同一工厂、同一天、同一SKU
        related_vehicles = []
        for idx, veh in enumerate(state.vehicles):
            if veh.fact_id == plant and veh.day == day and (sku_id, day) in veh.cargo:
                shipped_qty = veh.cargo[(sku_id, day)]
                related_vehicles.append((idx, veh, shipped_qty))
        
        if not related_vehicles:
            continue
            
        # 按装载量排序（从大到小）
        related_vehicles.sort(key=lambda x: x[2], reverse=True)
        
        # 计算需要减少的总量（取绝对值）
        amount_to_reduce = abs(neg_amount)
        reduced_so_far = 0
        
        # 从装载量最大的车辆开始调整
        for idx, veh, shipped_qty in related_vehicles:
            # 判断是否需要完全移除还是部分调整
            if reduced_so_far + shipped_qty <= amount_to_reduce:
                # 需要完全移除该车辆的这部分货物
                reduced_so_far += shipped_qty
                if sum(veh.cargo.values()) == shipped_qty:
                    # 如果这是车辆唯一的货物，直接移除整个车辆
                    if idx not in vehicles_to_remove:
                        vehicles_to_remove.append(idx)
                else:
                    # 否则记录部分调整
                    partial_adjustments.append((idx, sku_id, day, shipped_qty))
            else:
                # 部分调整即可
                adjust_qty = amount_to_reduce - reduced_so_far
                partial_adjustments.append((idx, sku_id, day, adjust_qty))
                reduced_so_far = amount_to_reduce
                
            # 检查是否已经解决了负库存问题
            if reduced_so_far >= amount_to_reduce:
                break
    
    # 执行部分调整（必须在移除整车之前执行）
    for idx, sku_id, day, reduce_qty in partial_adjustments:
        if idx >= len(state.vehicles):  # 安全检查
            continue
            
        veh = state.vehicles[idx]
        if (sku_id, day) in veh.cargo:
            current_qty = veh.cargo[(sku_id, day)]
            new_qty = max(0, current_qty - reduce_qty)
            
            if new_qty > 0:
                veh.cargo[(sku_id, day)] = new_qty
            else:
                del veh.cargo[(sku_id, day)]
                
            # 更新库存（仅增加被减少的量）
            diff = current_qty - new_qty
            for d in range(day, state.data.horizons + 1):
                key = (veh.fact_id, sku_id, d)
                state.s_ikt[key] = state.s_ikt.get(key, 0) + diff
    
    # 移除整车
    vehicles_to_remove.sort(reverse=True)  # 从高到低排序确保索引有效
    removed_vehicles = []
    
    for idx in vehicles_to_remove:
        if 0 <= idx < len(state.vehicles):
            veh = state.vehicles[idx]
            removed_vehicles.append(veh)
            
            # 更新库存
            for (sku_id, d_shipped), qty in veh.cargo.items():
                for d in range(d_shipped, state.data.horizons + 1):
                    key = (veh.fact_id, sku_id, d)
                    state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
                    
            # 移除车辆
            state.vehicles.pop(idx)
    
    # 最后全表重新计算一次以确保一致性
    state.compute_inventory()
    
    print(f"[OPLOG] infeasible_removal: 移除 {len(removed_vehicles)} 辆车, 部分调整 {len(partial_adjustments)} 个SKU ({time.time() - t0:.4f}s)")
    return state
```

## 剩余库存移除算子 (surplus_inventory_removal)

分析：
- 设计理念：移除对应库存过高的SKU的车辆，以降低库存风险。
- 实现方式：找出每个工厂库存最高的SKU，然后移除相关车辆。
- 潜在问题：
    1. 逻辑错误：该算子试图通过移除装载某SKU的车辆来降低该SKU的库存水平，这在逻辑上是矛盾的。移除运输车辆实际上会增加工厂库存，而非降低。
    2. 代码不一致：先找出工厂库存最高的SKU，然后移除相应车辆，但这会导致库存更高。
    3. 效果评估：没有评估移除这些车辆后对库存水平的实际影响。
改进建议：
```python
def surplus_inventory_removal(current: SolutionState, rng: rnd.Generator, degree: float = 0.25):
    t0 = time.time()
    print(f"[OPLOG] 开始执行 surplus_inventory_removal 算子")
    state = current.copy()
    data = state.data
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] surplus_inventory_removal: {time.time() - t0:.4f}s")
        return state
    
    # 确保库存计算是最新的
    state.compute_inventory()
    
    # 计算各工厂各SKU的库存水平
    plant_sku_inventory = defaultdict(lambda: defaultdict(int))
    for (plant, sku_id, day), inv in state.s_ikt.items():
        plant_sku_inventory[plant][sku_id] += inv
    
    # 计算各工厂的库存风险系数
    plant_risk = {}
    for plant, sku_invs in plant_sku_inventory.items():
        # 计算该工厂的总库存
        total_inv = sum(sku_invs.values())
        # 计算库存上限
        max_capacity = data.plant_inv_limit[plant]
        # 计算库存利用率
        utilization = total_inv / max_capacity if max_capacity > 0 else 1.0
        # 库存风险系数：接近上限时风险高
        risk = max(0, utilization - 0.8) * 5  # 超过80%容量时开始考虑风险
        plant_risk[plant] = risk
    
    # 按风险系数降序排列工厂
    high_risk_plants = sorted(plant_risk.items(), key=lambda x: x[1], reverse=True)
    
    # 选择高风险工厂，寻找可以调整的车辆
    candidates = []
    
    for plant, risk in high_risk_plants:
        if risk <= 0:
            continue  # 忽略低风险工厂
            
        # 查找从该工厂发出的车辆，优先选择当天发车并且发往库存较低的经销商的车辆
        for i, veh in enumerate(state.vehicles):
            if veh.fact_id == plant:
                # 计算该车运输的总量
                total_qty = sum(qty for (sku_id, day), qty in veh.cargo.items())
                
                # 计算目的地经销商的需求满足率
                dealer = veh.dealer_id
                dealer_demand = sum(data.demands.get((dealer, sku), 0) for sku in data.all_skus)
                dealer_shipped = sum(
                    v.cargo.get((sku, v.day), 0) 
                    for v in state.vehicles if v.dealer_id == dealer
                    for sku in data.all_skus
                )
                satisfaction = dealer_shipped / dealer_demand if dealer_demand > 0 else 1.0
                
                # 计算调整优先级：高风险工厂+高满足率经销商的车辆优先调整
                priority = risk * satisfaction
                candidates.append((i, veh, priority, total_qty))
    
    if not candidates:
        print(f"[OPLOG] surplus_inventory_removal: 无高风险工厂 ({time.time() - t0:.4f}s)")
        return state
    
    # 按优先级排序
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    # 计算需要移除的车辆数量
    num_remove = min(int(len(state.vehicles) * degree), len(candidates))
    
    # 按优先级选择车辆
    selected_indices = [idx for idx, _, _, _ in candidates[:num_remove]]
    selected_indices.sort(reverse=True)  # 从高到低排序确保索引有效
    
    # 执行移除
    removed_vehicles = []
    for idx in selected_indices:
        if 0 <= idx < len(state.vehicles):
            veh = state.vehicles[idx]
            removed_vehicles.append(veh)
            
            # 更新库存
            for (sku_id, d_shipped), qty in veh.cargo.items():
                for d in range(d_shipped, state.data.horizons + 1):
                    key = (veh.fact_id, sku_id, d)
                    state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
                    
            # 移除车辆
            state.vehicles.pop(idx)
    
    print(f"[OPLOG] surplus_inventory_removal: 移除 {len(removed_vehicles)}/{len(current.vehicles)} 辆车 ({time.time() - t0:.4f}s)")
    return state
```

## Shaw移除算子 (shaw_removal)

分析：
- 设计理念：移除相似的车辆，以期在重建时获得更好的组合。
- 实现方式：选择一个种子车辆，计算其他车辆与种子车辆的相似度，移除相似度高的车辆。
- 潜在问题：
    1. 相似度定义简单：仅考虑了工厂、经销商和天数三个维度，且权重固定，没有考虑货物类型、数量等更重要的因素。
    2. 固定移除比例：移除比例固定，不会随着搜索进展自适应调整。
    3. 种子选择随机：种子车辆完全随机选择，没有考虑选择更有代表性或问题更严重的车辆作为种子。
改进建议：
```python
def shaw_removal(current: SolutionState, rng: rnd.Generator, degree: float = 0.3):
    t0 = time.time()
    print(f"[OPLOG] 开始执行 shaw_removal 算子")
    state = current.copy()
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] shaw_removal: {time.time() - t0:.4f}s")
        return state
    
    # 自适应移除比例：迭代开始时使用较高的破坏程度，后期降低
    if state.tracker:
        iteration = state.tracker.get_statistics().get('total_iterations', 0)
        max_iterations = state.tracker.get_statistics().get('max_iterations', 100)
        if max_iterations > 0 and iteration > 0:
            # 动态调整移除比例
            adaptive_degree = degree * (1 - 0.5 * min(1.0, iteration / max_iterations))
            degree = max(0.1, adaptive_degree)
            print(f"[OPLOG] 自适应移除比例: {degree:.4f}")
    
    # 改进种子选择：根据问题状态智能选择种子车辆
    seed_candidates = []
    for i, veh in enumerate(state.vehicles):
        # 计算车辆的各项指标
        load = state.compute_veh_load(veh)
        capacity = state.data.veh_type_cap[veh.type]
        utilization = load / capacity if capacity > 0 else 0
        
        # 获取该车相关的库存状态
        plant_inventory = 0
        for (plant, sku_id, day), inv in state.s_ikt.items():
            if plant == veh.fact_id:
                plant_inventory += inv
        
        # 计算种子分数：低利用率、高成本、关键时间窗口的车辆优先
        seed_score = (1 - utilization) * 0.4 + (day / state.data.horizons) * 0.3 + (plant_inventory / (state.data.plant_inv_limit[veh.fact_id] + 1)) * 0.3
        seed_candidates.append((i, seed_score))
    
    # 轮盘赌选择种子：分数越高，被选中概率越大
    seed_scores = np.array([score for _, score in seed_candidates])
    if np.sum(seed_scores) > 0:
        probabilities = seed_scores / np.sum(seed_scores)
        try:
            seed_idx = rng.choice(len(seed_candidates), p=probabilities)
        except ValueError:
            # 如果概率计算出错，退回到简单随机选择
            seed_idx = rng.integers(0, len(state.vehicles))
    else:
        seed_idx = rng.integers(0, len(state.vehicles))
    
    seed_veh = state.vehicles[seed_idx]
    
    # 改进相似度计算：考虑更多维度和动态权重
    relatedness = []
    
    # 动态权重：根据当前解的特点调整各维度权重
    weight_fact = 3.0  # 工厂权重
    weight_dealer = 2.0  # 经销商权重
    weight_day = 1.0  # 天数权重
    weight_cargo = 2.0  # 货物相似度权重
    
    # 预处理种子车辆的货物信息
    seed_skus = set(sku_id for (sku_id, _) in seed_veh.cargo.keys())
    
    for i, veh in enumerate(state.vehicles):
        if i == seed_idx:
            continue
            
        # 基础相似度：工厂、经销商、天数
        fact_sim = 1.0 if veh.fact_id == seed_veh.fact_id else 0.0
        dealer_sim = 1.0 if veh.dealer_id == seed_veh.dealer_id else 0.0
        day_diff = abs(veh.day - seed_veh.day)
        day_sim = max(0.0, 1.0 - day_diff / state.data.horizons)
        
        # 货物相似度：共同SKU比例
        veh_skus = set(sku_id for (sku_id, _) in veh.cargo.keys())
        if seed_skus and veh_skus:
            common_skus = len(seed_skus.intersection(veh_skus))
            all_skus = len(seed_skus.union(veh_skus))
            cargo_sim = common_skus / all_skus if all_skus > 0 else 0.0
        else:
            cargo_sim = 0.0
        
        # 综合相似度
        total_sim = (
            weight_fact * fact_sim +
            weight_dealer * dealer_sim +
            weight_day * day_sim +
            weight_cargo * cargo_sim
        )
        
        # 添加噪声避免确定性行为
        noise = rng.uniform(-0.1, 0.1)
        total_sim += noise
        
        relatedness.append((i, total_sim))
    
    # 按相似度排序（从高到低）
    relatedness.sort(key=lambda x: x[1], reverse=True)
    
    # 计算移除数量
    num_remove = min(int(len(state.vehicles) * degree), len(relatedness))
    
    # 选择最相似的车辆移除
    remove_indices = [idx for idx, _ in relatedness[:num_remove]]
    remove_indices.append(seed_idx)
    remove_indices.sort(reverse=True)
    
    # 移除车辆并更新库存
    removed_vehicles = []
    for idx in remove_indices:
        if 0 <= idx < len(state.vehicles):
            veh = state.vehicles[idx]
            removed_vehicles.append(veh)
            
            # 移除车辆前先更新库存
            for (sku_id, d_shipped), qty in veh.cargo.items():
                for d in range(d_shipped, state.data.horizons + 1):
                    key = (veh.fact_id, sku_id, d)
                    state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
            
            # 移除车辆
            state.vehicles.pop(idx)
    
    print(f"[OPLOG] shaw_removal: 移除 {len(removed_vehicles)}/{len(current.vehicles)} 辆车 ({time.time() - t0:.4f}s)")
    return state
```

## 周期Shaw移除算子 (periodic_shaw_removal)

分析：
- 设计理念：使用聚类方法识别具有相似特征的分配，进行更系统的相关移除。
- 实现方式：使用KMeans算法按特征（周期、需求、库存）聚类，然后在选定簇中移除相关性高的元素。
- 潜在问题：
    1. 内存泄漏风险：KMeans 在Windows上有内存泄漏风险，虽然代码中有做环境变量配置，但仍可能导致问题。
    2. 特征设计局限：仅使用周期、平均需求和库存转移作为特征，未考虑更广泛的上下文。
    3. 超参数固定：alpha, beta, gamma等参数固定，没有根据不同实例或搜索阶段动态调整。
    4. 相似度计算不完善：相似度计算中使用需求余弦相似度等指标，但未经充分验证。
改进建议：
```python
def periodic_shaw_removal(current: SolutionState, rng: rnd.Generator, degree: float=0.3, 
                          alpha: float=0.4, beta: float=0.3, gamma: float=0.3,
                          k_clusters: int=3) -> SolutionState:
    t0 = time.time()
    print(f"[OPLOG] 开始执行 periodic_shaw_removal 算子")
    state = current.copy()
    data = state.data
    
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] periodic_shaw_removal: {time.time() - t0:.4f}s")
        return state
    
    # 自适应参数：根据迭代进度调整
    if state.tracker:
        iteration = state.tracker.get_statistics().get('total_iterations', 0)
        max_iterations = state.tracker.get_statistics().get('max_iterations', 100)
        if max_iterations > 0 and iteration > 0:
            progress = min(1.0, iteration / max_iterations)
            # 后期更注重时间差异和库存水平
            alpha = 0.4 + 0.2 * progress  # 增加时间权重
            beta = 0.3 - 0.1 * progress   # 减少需求权重
            gamma = 0.3 + 0.1 * progress  # 增加库存权重
    
    # 动态聚类数：根据车辆数自适应
    vehicle_count = len(state.vehicles)
    optimal_k = min(max(2, vehicle_count // 10), 10)  # 保持在2-10之间
    k_clusters = optimal_k
    
    # Step 1: 增强的特征提取
    allocations = []  # list of (veh, sku_id, day, feature_vector)
    for veh in state.vehicles:
        plant = veh.fact_id
        dealer = veh.dealer_id
        day = veh.day
        
        # 计算车辆利用率
        veh_util = state.compute_veh_load(veh) / data.veh_type_cap[veh.type] if data.veh_type_cap[veh.type] > 0 else 0
        
        # 计算该经销商的总需求和满足率
        dealer_demands = [data.demands.get((dealer, sku), 0) for sku in data.all_skus]
        total_demand = sum(dealer_demands)
        dealer_shipped = sum(
            v.cargo.get((sku, v.day), 0) 
            for v in state.vehicles if v.dealer_id == dealer
            for sku in data.all_skus
        )
        satisfaction = dealer_shipped / total_demand if total_demand > 0 else 1.0
        
        # 计算工厂库存压力
        plant_inv = sum(inv for (p, _, d), inv in state.s_ikt.items() if p == plant and d == day)
        plant_capacity = data.plant_inv_limit[plant]
        plant_pressure = plant_inv / plant_capacity if plant_capacity > 0 else 1.0
        
        for (sku, d), qty in list(veh.cargo.items()):
            if d != day:
                continue
                
            # 增强特征向量：[天数, 需求满足率, 库存水平, 车辆利用率, 工厂压力]
            inv_transfer = state.s_ikt.get((plant, sku, day - 1), 0)
            
            # 归一化特征
            norm_day = day / data.horizons
            norm_inv = min(1.0, inv_transfer / (data.plant_inv_limit[plant] * 0.1)) if data.plant_inv_limit[plant] > 0 else 0
            
            feature = np.array([norm_day, satisfaction, norm_inv, veh_util, plant_pressure])
            allocations.append((veh, sku, day, feature))
    
    if not allocations:
        print(f"[OPLOG] periodic_shaw_removal: 无可处理的分配 ({time.time() - t0:.4f}s)")
        return state
    
    # Step 2: 改进的聚类过程
    try:
        features = np.array([a[3] for a in allocations])
        
        # 数据预处理：使用鲁棒的标准化
        features_median = np.median(features, axis=0)
        features_iqr = np.percentile(features, 75, axis=0) - np.percentile(features, 25, axis=0)
        features_iqr[features_iqr == 0] = 1  # 避免除零
        features_normalized = (features - features_median) / features_iqr
        
        # 安全聚类：使用更稳健的参数
        optimal_clusters = min(k_clusters, len(allocations), max(1, len(allocations) // 3))
        
        kmeans = KMeans(
            n_clusters=optimal_clusters,
            random_state=rng.integers(0, 1000),
            n_init=3,
            max_iter=50,
            tol=1e-3,
            algorithm='lloyd'
        )
        
        labels = kmeans.fit_predict(features_normalized)
        
        # 清理KMeans对象
        del kmeans
        
    except Exception as e:
        print(f"[OPLOG] periodic_shaw_removal: 聚类失败 ({type(e).__name__}), 回退到简单随机移除")
        # 回退到简单随机移除策略
        num_remove = int(len(state.vehicles) * degree)
        remove_indices = rng.choice(range(len(state.vehicles)), num_remove, replace=False)
        remove_indices = sorted(remove_indices, reverse=True)
        
        for idx in remove_indices:
            if 0 <= idx < len(state.vehicles):
                veh = state.vehicles[idx]
                for (sku_id, d_shipped), qty in veh.cargo.items():
                    for d in range(d_shipped, data.horizons + 1):
                        key = (veh.fact_id, sku_id, d)
                        state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
                state.vehicles.pop(idx)
        
        print(f"[OPLOG] periodic_shaw_removal: 回退模式 - 移除了 {num_remove} 辆车 ({time.time() - t0:.4f}s)")
        return state
    
    # Step 3: 智能簇选择与相似度计算
    unique_labels = np.unique(labels)
    if len(unique_labels) == 0:
        print(f"[OPLOG] periodic_shaw_removal: 无有效聚类标签 ({time.time() - t0:.4f}s)")
        return state
        
    # 选择最大的簇，提高移除效果
    cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # 随机选择一个较大的簇（前50%中随机选择）
    eligible_clusters = cluster_sizes[:max(1, len(cluster_sizes) // 2)]
    if not eligible_clusters:
        print(f"[OPLOG] periodic_shaw_removal: 无合格簇 ({time.time() - t0:.4f}s)")
        return state
        
    selected_cluster = rng.choice([label for label, _ in eligible_clusters])
    cluster_indices = np.where(labels == selected_cluster)[0]
    cluster_allocs = [allocations[i] for i in cluster_indices]
    
    if len(cluster_allocs) <= 1:
        print(f"[OPLOG] periodic_shaw_removal: 簇内元素过少 ({time.time() - t0:.4f}s)")
        return state
    
    # 改进相似度计算：随机选择种子
    seed_idx = rng.integers(0, len(cluster_allocs))
    seed_alloc = cluster_allocs[seed_idx]
    seed_feature = seed_alloc[3]
    
    similarities = []
    for i, alloc in enumerate(cluster_allocs):
        if i == seed_idx:
            continue
            
        feat = alloc[3]
        
        # 计算各维度相似度
        time_diff = abs(feat[0] - seed_feature[0])  # 天数差异
        
        # 使用余弦相似度计算需求满足率和库存相似性
        demand_cos = max(0, np.dot(feat[1:3], seed_feature[1:3])) / (np.linalg.norm(feat[1:3]) * np.linalg.norm(seed_feature[1:3]) + 1e-8)
        
        # 计算车辆和工厂相似度
        op_diff = np.linalg.norm(feat[3:5] - seed_feature[3:5])
        
        # 综合相似度：加权和 + 随机噪声
        sim = alpha * time_diff + beta * (1 - demand_cos) + gamma * op_diff
        noise = rng.uniform(-0.1, 0.1) * sim
        sim += noise
        
        similarities.append((alloc, sim))
    
    # 排序并选择移除对象
    similarities.sort(key=lambda x: x[1])  # 按相似度升序（相似度低的排前面）
    
    # 计算移除数量
    num_remove = int(len(cluster_allocs) * degree)
    num_remove = min(num_remove, len(similarities))
    
    # 执行移除
    for i in range(num_remove):
        veh, sku_id, day, _ = similarities[i][0]
        
        # 从车辆中移除该SKU
        qty = veh.cargo.pop((sku_id, day), 0)
        if qty > 0:
            sku_size = data.sku_sizes[sku_id]
            veh.capacity += sku_size * qty
            
            # 更新库存
            for d in range(day, data.horizons + 1):
                key = (veh.fact_id, sku_id, d)
                state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
        
        # 如果车辆已空，移除车辆
        if veh.is_empty() and veh in state.vehicles:
            state.vehicles.remove(veh)
    
    print(f"[OPLOG] periodic_shaw_removal: 处理了簇 {selected_cluster}，移除了 {num_remove} 个分配 ({time.time() - t0:.4f}s)")
    return state
```

## 路径移除算子 (path_removal)

分析：
- 设计理念：移除特定工厂-经销商对之间的车辆，以重新优化这条路径上的分配。
- 实现方式：随机选择一个工厂-经销商对，然后移除该对上的部分车辆。
- 潜在问题：
    1. 随机性太高：完全随机选择路径和车辆数量，没有考虑路径的重要性或问题程度。
    2. 缺乏分析：没有对所选路径的当前状态进行分析，如满足率、效率等。
    3. 车辆选择无策略：从选定路径上随机选择车辆移除，没有基于车辆特性（如满载率、成本效益等）进行选择。
改进建议：
```python
def path_removal(current: SolutionState, rng: rnd.Generator, degree: float = 0.5):
    t0 = time.time()
    print(f"[OPLOG] 开始执行 path_removal 算子")
    state = current.copy()
    data = state.data
    
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] path_removal: {time.time() - t0:.4f}s")
        return state
    
    # 统计各路径上的车辆和相关信息
    path_info = {}
    for veh in state.vehicles:
        path = (veh.fact_id, veh.dealer_id)
        if path not in path_info:
            path_info[path] = {
                'vehicles': [],
                'total_demand': 0,
                'shipped': 0,
                'total_cost': 0
            }
        
        # 记录车辆
        path_info[path]['vehicles'].append(veh)
        
        # 累计成本
        path_info[path]['total_cost'] += data.veh_type_cost[veh.type]
        
        # 累计发货量
        for (sku_id, _), qty in veh.cargo.items():
            path_info[path]['shipped'] += qty
    
    # 计算各路径的需求满足情况
    for (plant, dealer), info in path_info.items():
        # 计算该经销商的总需求
        dealer_demand = sum(data.demands.get((dealer, sku), 0) for sku in data.all_skus)
        path_info[(plant, dealer)]['total_demand'] = dealer_demand
    
    # 如果没有有效路径，返回原解
    if not path_info:
        print(f"[OPLOG] path_removal: 无有效路径 ({time.time() - t0:.4f}s)")
        return state
    
    # 计算各路径的评分：考虑需求满足率、成本、车辆数量等
    path_scores = []
    for path, info in path_info.items():
        if not info['vehicles']:
            continue
            
        # 需求满足率
        satisfaction = info['shipped'] / info['total_demand'] if info['total_demand'] > 0 else 1.0
        
        # 平均车辆成本
        avg_cost = info['total_cost'] / len(info['vehicles']) if info['vehicles'] else 0
        
        # 计算路径分数：高满足率、高成本、多车辆的路径优先考虑
        path_score = (
            satisfaction * 0.4 +
            (avg_cost / max(data.veh_type_cost.values())) * 0.3 +
            (len(info['vehicles']) / len(state.vehicles)) * 0.3
        )
        
        path_scores.append((path, path_score))
    
    if not path_scores:
        print(f"[OPLOG] path_removal: 无有效路径评分 ({time.time() - t0:.4f}s)")
        return state
    
    # 按评分排序（从高到低）
    path_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 倾向于选择高分路径，但保持一定随机性
    top_paths = path_scores[:max(1, len(path_scores) // 3)]
    weights = np.array([score for _, score in top_paths])
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else None
    
    try:
        selected_idx = rng.choice(len(top_paths), p=weights)
        target_path = top_paths[selected_idx][0]
    except ValueError:
        # 随机选择一条路径
        target_path = rng.choice([path for path, _ in path_scores])
    
    # 获取所选路径上的所有车辆
    path_vehicles = [(i, veh) for i, veh in enumerate(state.vehicles) 
                     if (veh.fact_id, veh.dealer_id) == target_path]
    
    if not path_vehicles or len(path_vehicles) <= 1:
        print(f"[OPLOG] path_removal: 选定路径上车辆不足 ({time.time() - t0:.4f}s)")
        return state
    
    # 智能确定移除数量：根据路径车辆数量和总车辆数的比例自适应
    path_ratio = len(path_vehicles) / len(state.vehicles)
    if path_ratio > 0.5:  # 如果该路径占比超过50%，减少移除比例
        degree = degree * 0.5
    
    num_remove = max(1, int(len(path_vehicles) * degree))
    num_remove = min(num_remove, len(path_vehicles) - 1)  # 至少保留一辆车
    
    # 对路径上的车辆按效率排序
    vehicle_metrics = []
    for i, veh in path_vehicles:
        # 计算车辆利用率
        load = state.compute_veh_load(veh)
        capacity = data.veh_type_cap[veh.type]
        utilization = load / capacity if capacity > 0 else 0
        
        # 计算单位成本
        cost = data.veh_type_cost[veh.type]
        unit_cost = cost / load if load > 0 else float('inf')
        
        # 综合指标
        score = 0.5 * utilization + 0.5 * (1 - unit_cost / (max(data.veh_type_cost.values()) + 1))
        
        vehicle_metrics.append((i, score))
    
    # 按效率排序（从低到高）
    vehicle_metrics.sort(key=lambda x: x[1])
    
    # 选择效率最低的车辆移除
    remove_indices = [idx for idx, _ in vehicle_metrics[:num_remove]]
    remove_indices.sort(reverse=True)  # 从高到低排序确保索引有效
    
    # 执行移除
    removed_vehicles = []
    for idx in remove_indices:
        if 0 <= idx < len(state.vehicles):
            veh = state.vehicles[idx]
            removed_vehicles.append(veh)
            
            # 更新库存
            for (sku_id, d_shipped), qty in veh.cargo.items():
                for d in range(d_shipped, data.horizons + 1):
                    key = (veh.fact_id, sku_id, d)
                    state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
            
            # 移除车辆
            state.vehicles.pop(idx)
    
    print(f"[OPLOG] path_removal: 从路径 {target_path} 移除了 {len(removed_vehicles)}/{len(path_vehicles)} 辆车 ({time.time() - t0:.4f}s)")
    return state
```

# 总结和建议

通过对 alnsopt.py 中各个破坏算子的评估，我发现了以下普遍存在的问题：
    1. 缺乏自适应机制：大多数算子使用固定的破坏程度，没有根据搜索阶段动态调整。
    2. 库存更新效率低：多数算子采用循环增量更新库存的方式，没有使用批量更新策略。
    3. 随机性过高：某些算子（如random_removal和path_removal）过度依赖随机性，缺乏足够的启发式引导。
    4. 缺乏问题特性利用：未充分利用问题特性进行针对性优化，如经销商需求满足率、工厂库存状况等。
    5. 代码结构优化空间：大量重复代码模式，如车辆移除和库存更新逻辑。

全局改进建议：
    1. 统一的库存更新机制：设计一个通用的库存批量更新函数，减少重复代码。
```python
def batch_update_inventory(state: SolutionState, vehicles_to_remove: List[Vehicle]):
    """批量更新因移除车辆而释放的库存"""
    if not vehicles_to_remove:
        return
        
    # 确保库存已初始化
    if not state.s_initialized:
        state.compute_inventory()
        
    # 按(plant,sku,day)分组批量更新
    inventory_updates = defaultdict(int)
    for veh in vehicles_to_remove:
        for (sku_id, d_shipped), qty in veh.cargo.items():
            # 收集所有需要更新的库存变化
            for d in range(d_shipped, state.data.horizons + 1):
                key = (veh.fact_id, sku_id, d)
                inventory_updates[key] += qty
    
    # 一次性应用所有库存更新
    for key, qty in inventory_updates.items():
        state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
```
    2. 自适应破坏程度框架：设计一个通用的自适应破坏程度调整机制。
```python
def adaptive_degree(state: SolutionState, base_degree: float, 
                   min_degree: float = 0.05, max_degree: float = 0.5,
                   decay_rate: float = 0.67):
    """根据搜索阶段动态调整破坏程度"""
    if not state.tracker:
        return base_degree
        
    iteration = state.tracker.get_statistics().get('total_iterations', 0)
    max_iterations = state.tracker.get_statistics().get('max_iterations', 100)
    
    if max_iterations <= 0 or iteration <= 0:
        return base_degree
        
    # 从base_degree开始，逐渐降低到min_degree
    progress = min(1.0, iteration / (0.7 * max_iterations))
    adaptive_value = base_degree * (1 - decay_rate * progress)
    
    return max(min_degree, min(max_degree, adaptive_value))
```
    3. 智能车辆评估机制：设计通用的车辆评估函数，用于各算子中选择要移除的车辆。
```python
def evaluate_vehicle_efficiency(state: SolutionState, veh: Vehicle):
    """评估车辆效率，返回综合评分（越高越好）"""
    data = state.data
    
    # 计算空间利用率
    capacity = data.veh_type_cap[veh.type]
    load = state.compute_veh_load(veh)
    space_util = load / capacity if capacity > 0 else 0
    
    # 计算成本效益
    cost = data.veh_type_cost[veh.type]
    cost_efficiency = load / cost if cost > 0 else 0
    
    # 计算需求满足贡献
    demand_contribution = 0
    dealer = veh.dealer_id
    for (sku_id, _), qty in veh.cargo.items():
        total_demand = data.demands.get((dealer, sku_id), 0)
        if total_demand > 0:
            contribution = qty / total_demand
            demand_contribution += contribution
    
    # 标准化需求贡献
    if veh.cargo:
        demand_contribution /= len(veh.cargo)
    
    # 综合评分：空间利用率(40%) + 成本效益(40%) + 需求满足贡献(20%)
    return 0.4 * space_util + 0.4 * cost_efficiency + 0.2 * demand_contribution
```
    4. 参数自动调整：设计一个参数自动调整机制，根据问题实例特性和搜索历史动态调整各算子的参数。
    5. 机器学习增强：扩展现有的learning_based_repair思路，为破坏算子也引入机器学习能力，实现自动选择最适合当前解状态的破坏算子和参数。

通过以上改进，可以提高ALNS算法的效率和解的质量，使得算法在面对不同规模和特性的多周期产品配送问题时都能表现良好。