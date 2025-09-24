import copy
import time
import random
import numpy.random as rnd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from InputDataALNS import DataALNS
from vehicle import Vehicle

@dataclass
class SolutionState:
    data: DataALNS
    vehicles: List[Vehicle] = field(default_factory=list)
    s_ikt: Dict[Tuple[str, str, int], int] = field(default_factory=dict)
    s_indices: Set[Tuple[str, str, int]] = field(default_factory=set)

    def __post_init__(self):
        self.vehicles = []
        self.s_ikt = {}
        self.s_indices = self.construct_indices()
        # 对s_ikt进行初始化, s_ik0表示期初库存
        for (plant, sku_id, day), inv in self.data.historical_s_ikt.items():
            if day == 0:
                self.s_ikt[plant, sku_id, day] = inv
    
    
    def validate(self):
        """验证解的一致性"""
        # 检查库存计算是否正确
        self.compute_inventory()
        
        # 检查是否有负库存
        has_negative = any(inv < 0 for inv in self.s_ikt.values())
        
        # 检查车辆容量是否超限
        veh_overload = False
        for veh in self.vehicles:
            if self.compute_veh_load(veh) > self.data.veh_type_cap[veh.type]:
                veh_overload = True
                break
        
        # 检查是否满足所有需求
        shipped = self.compute_shipped()
        demands_not_met = False
        for (dealer, sku_id), demand in self.data.demands.items():
            if shipped.get((dealer, sku_id), 0) < demand:
                demands_not_met = True
                break
        
        return not (has_negative or veh_overload or demands_not_met)
    

    def objective(self):
        # 添加缩放因子以控制目标函数值的尺度
        # 避免目标函数值过大, 影响模拟退火中计算接收概率时, 出现数值溢出
        scale_factor = 1e-3  # 缩放因子，可以根据实际情况调整
        
        self.compute_inventory()  # 计算并更新库存水平
        
        # 检查解的可行性
        if not self.validate():
            return float('inf')  # 返回一个非常大的值，表示解不可行
        
        total_cost = sum(self.data.veh_type_cost[veh.type] for veh in self.vehicles)
        
      
        total_cost = self.punish_non_fulfill_demand(total_cost)
        
        total_cost = self.punish_exceeded_inventory_limit(total_cost)
        
        total_cost = self.punish_deficient_veh_min_load(total_cost)
        
        total_cost = self.punish_negative_inventory(total_cost)
        
        return total_cost * scale_factor
    
    def calculate_cost(self):
        """计算当前解的运输成本"""
        total_cost = sum(self.data.veh_type_cost[veh.type] for veh in self.vehicles)
        return total_cost
    
    
    def compute_violations(self):
        """计算在当前解中, 每种惩罚项的总违反量"""
        violations = {
            "non_fulfill_demand": 0,
            "exceeded_inventory_limit": 0,
            "deficient_veh_min_load": 0,
            "negative_inventory": 0
        }
        
        # 计算违反需求量的惩罚项
        shipped = self.compute_shipped()
        for (dealer, sku_id), demand in self.data.demands.items():
            shipped_qty = shipped.get((dealer, sku_id), 0)
            if shipped_qty < demand:
                violations["non_fulfill_demand"] += demand - shipped_qty
                
        # 计算违反库存限制的惩罚项
        for plant, max_cap in self.data.plant_inv_limit.items():
            for day_id in range(1, self.data.horizons + 1):
                total_inv = sum(inv for (fact_id, sku_id, day), inv in self.s_ikt.items()
                                if fact_id == plant and day == day_id)
                if total_inv > max_cap:
                    violations["exceeded_inventory_limit"] += total_inv - max_cap
        
        # 计算违反最小起运量的惩罚项
        for veh in self.vehicles:
            total_volume = self.compute_veh_load(veh)
            min_load = self.data.veh_type_min_load[veh.type]
            if total_volume < min_load:
                violations["deficient_veh_min_load"] += min_load - total_volume
        
        # 计算违反非负库存的惩罚项
        for (plant, sku_id, day), inv in self.s_ikt.items():
            if inv < 0:
                violations["negative_inventory"] += abs(inv)
                
        return violations
    
    
    
    def punish_non_fulfill_demand(self, obj: float):
        """检查是否满足经销商的需求, 如果不满足, 需要再目标函数中添加惩罚项"""
        shipped = self.compute_shipped()
        for (dealer, sku_id), demand in self.data.demands.items():
            shipped_qty = shipped.get((dealer, sku_id), 0)
            if shipped_qty < demand:
                obj += self.data.param_pun_factor1 * (demand - shipped_qty)
        return obj
    
    
    def punish_exceeded_inventory_limit(self, obj: float):
        """检查是否满足生产基地的库存限制, 如果不满足, 需要再目标函数中添加惩罚项"""
        for plant, max_cap in self.data.plant_inv_limit.items():
            for day_id in range(1, self.data.horizons + 1):
                total_inv = sum(inv for (fact_id, sku_id, day), inv in self.s_ikt.items() 
                                if fact_id == plant and day == day_id)
                if total_inv > max_cap:
                    obj += self.data.param_pun_factor2 * (total_inv - max_cap)
        return obj
    
    def compute_veh_load(self, veh: Vehicle):
        """计算当前车辆装载量"""
        total_volume = 0
        for (sku_id, day), qty in veh.cargo.items():
            total_volume += self.data.sku_sizes[sku_id] * qty
        return total_volume
    
    def punish_deficient_veh_min_load(self, obj: float):
        """判断当前车辆是否满足最小起运量, 如果不满足, 需要再目标函数中添加惩罚项"""
        for veh in self.vehicles:
            total_volume = self.compute_veh_load(veh)
            min_load = self.data.veh_type_min_load[veh.type]
            if total_volume < min_load:
                obj += self.data.param_pun_factor3 * (min_load - total_volume)
        return obj
    
    
    def punish_negative_inventory(self, obj: float):
        """检查库存是否存在负值, 如果存在, 需要再目标函数中添加惩罚项"""
        for (plant, sku_id, day), inv in self.s_ikt.items():
            if inv < 0:
                obj += self.data.param_pun_factor4 * abs(inv)
        return obj
    
    def compute_shipped(self):
        """
        计算在所有周期内, 满足经销商 j 对 SKU k 的需求量
        """
        shipped = {}
        for veh in self.vehicles:
            dealer = veh.dealer_id
            for (sku_id, day), qty in veh.cargo.items():
                shipped[dealer, sku_id] = shipped.get((dealer, sku_id), 0) + qty
        return shipped
    
    def construct_indices(self):
        triple_plant_dealer_sku = {
            (plant, dealer, sku_id) for (plant, dealer), skus in self.data.construct_supply_chain().items()
            for sku_id in skus
        }
        s_indices = {(plant, sku_id, day) for (plant, dealer, sku_id) in triple_plant_dealer_sku
                    for day in range(self.data.horizons + 1)}
        return s_indices
    
    def compute_inventory(self):
        """计算并更新库存水平"""
        for (plant, sku_id, day) in self.s_indices:
            if day > 0:  # 只处理day>0的情况，因为day=0是期初库存，不应该被修改
                # 计算从该生产基地运出的SKU数量
                shipped_from_plant = sum(veh.cargo.get((sku_id, day), 0) 
                                        for veh in self.vehicles 
                                        if veh.fact_id == plant and veh.day == day)
                
                # 获取前一天的库存，确保期初库存被正确考虑
                prev_inventory = self.s_ikt.get((plant, sku_id, day - 1), 0)
                
                # 获取当天的生产量
                production = self.data.sku_prod_each_day.get((plant, sku_id, day), 0)
                
                # 限制发货数量不可超过可用库存
                shipped_from_plant = min(shipped_from_plant, prev_inventory + production)
                
                # 计算当前库存：前一天库存 + 当天生产 - 当天发出
                current_inventory = prev_inventory + production - shipped_from_plant
                
                # 更新库存
                self.s_ikt[plant, sku_id, day] = current_inventory
                
                # 更新历史库存记录
                self.data.historical_s_ikt[plant, sku_id, day] = current_inventory
    
    def copy(self):
        """复制当前解：确保每个解都是不可变的"""
        new_state = SolutionState(self.data)
        new_state.vehicles = copy.deepcopy(self.vehicles)
        new_state.s_ikt = copy.deepcopy(self.s_ikt)
        new_state.s_indices = copy.deepcopy(self.s_indices)
        return new_state

 
def veh_loading(state: SolutionState, veh: Vehicle, orders: Dict[str, int]):
    """
    车辆装载函数
    考虑车辆容量约束
    """
    data = state.data
    
    # 计算车辆中已经装载的SKU总体积
    current_load = sum(data.sku_sizes[sku_id] * qty for (sku_id, day), qty in veh.cargo.items())

    # 创建临时车辆进行装载尝试
    temp_vehicle = Vehicle(veh.fact_id, veh.dealer_id, veh.type, veh.day, data)
    
    # 计算当前可用的库存数量, 需要考虑已有车辆装载的SKU
    used_inv = {}
    for vehicle in state.vehicles:
        if vehicle.fact_id == veh.fact_id and vehicle.day == veh.day:
            for (sku_id, day), qty in vehicle.cargo.items():
                used_inv[sku_id, day] = used_inv.get((sku_id, day), 0) + qty
    
    for sku_id, qty in orders.items():
        if sku_id in data.skus_plant[veh.fact_id]:
            # 计算当前生产基地可以提供的SKU数量
            # 确保正确考虑期初库存
            prev_inventory = state.s_ikt.get((veh.fact_id, sku_id, veh.day - 1), 0)
            production = data.sku_prod_each_day.get((veh.fact_id, sku_id, veh.day), 0)
            supply = prev_inventory + production
            
            used = used_inv.get((sku_id, veh.day), 0)
            available = max(0, supply - used)
            # 确定可装载数量
            loadable_qty = min(qty, available)
            
            # 尝试装载
            while loadable_qty > 0:
                # 计算此次装载后的总体积
                sku_volume = data.sku_sizes[sku_id] * loadable_qty
                if current_load + sku_volume <= data.veh_type_cap[veh.type]:
                    # 可以全部装载
                    temp_vehicle.load(sku_id, loadable_qty)
                    current_load += sku_volume
                    used_inv[sku_id, veh.day] = used_inv.get((sku_id, veh.day), 0) + loadable_qty
                    loadable_qty = 0
                else:
                    # 计算还能装载多少
                    remaining_space = data.veh_type_cap[veh.type] - current_load
                    partial_qty = remaining_space // data.sku_sizes[sku_id]
                    if partial_qty > 0:
                        temp_vehicle.load(sku_id, partial_qty)
                        current_load += partial_qty * data.sku_sizes[sku_id]
                        used_inv[sku_id, veh.day] = used_inv.get((sku_id, veh.day), 0) + partial_qty
                        loadable_qty -= partial_qty
                    break
    
    # 不论是否满足最小起运量, 都记录在当前解中
    if not temp_vehicle.is_empty():
        veh.cargo = temp_vehicle.cargo.copy()
        veh.capacity = temp_vehicle.capacity
        state.vehicles.append(veh)
    
    # 装载完毕后, 重新计算库存, 确保库存计算的准确性
    state.compute_inventory()
    
    # 当前经销商的订单考虑完毕, 返回True
    return True


def initial_solution(state: SolutionState, rng: rnd.Generator):
    """
    使用贪心算法生成初始解, 确保满足所有经销商的订单需求
    """
    data = state.data
    demands = data.demands.copy()
    
    # 按需求量从大到小排序
    sorted_demands = sorted(demands.items(), key=lambda x: x[1], reverse=True)
    
    # 首先尝试满足大需求
    for (dealer_id, sku_id), demand in sorted_demands:
        if demand <= 0:
            continue
            
        # 找到可以供应该经销商的所有生产基地
        for plant in [p for p, d in data.construct_supply_chain() 
                     if d == dealer_id and sku_id in data.skus_plant[p]]:
            for day in range(1, data.horizons + 1):
                # 检查库存是否充足
                state.compute_inventory()
                current_stock = state.s_ikt.get((plant, sku_id, day - 1), 0) + \
                    data.sku_prod_each_day.get((plant, sku_id, day), 0)
                
                if current_stock <= 0:
                    continue
                
                # 选择合适的车型，优先选择较大容量的车型
                veh_types = sorted(list(data.all_veh_types), 
                                 key=lambda x: data.veh_type_cap[x],
                                 reverse=True)
                
                for veh_type in veh_types:
                    # 如果需求已满足，跳出循环
                    if demands.get((dealer_id, sku_id), 0) <= 0:
                        break
                        
                    # 创建新车辆
                    vehicle = Vehicle(plant, dealer_id, veh_type, day, data)
                    orders = {sku_id: demands.get((dealer_id, sku_id), 0)}
                    
                    # 尝试装载
                    value = veh_loading(state, vehicle, orders)
                    
                    if value:
                        # 更新需求
                        loaded_qty = vehicle.cargo.get((sku_id, day), 0)
                        demands[(dealer_id, sku_id)] -= loaded_qty
    
    # 第二阶段：检查是否所有需求都已满足，如果没有，使用需求优先修复算子来满足剩余需求
    unsatisfied = {k: v for k, v in demands.items() if v > 0}
    if unsatisfied:
        # 使用需求优先修复算子来满足剩余需求
        state = demand_first_repair(state, rng)
    
    # 最终检查是否所有需求都已满足
    shipped = state.compute_shipped()
    all_demands_fulfilled = True
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if shipped_qty < demand:
            all_demands_fulfilled = False
            break
    
    if not all_demands_fulfilled:
        print("Warning: Initial solution failed to satisfy all demands!")
    
    return state


def random_removal(current: SolutionState, rng: rnd.Generator):
    """
    随机移除解中25%的车辆
    目的是引入随机性, 使搜索更加多样化
    """
    state = current.copy()
    
    degree = 0.25  # 移除比例
    
    if len(state.vehicles) <= 1:
        return state

    num_remove = int(len(state.vehicles) * degree)
    remove_indices = rng.choice(range(len(state.vehicles)), num_remove, replace=False)
    new_vehicles = [veh for i, veh in enumerate(state.vehicles) if i not in remove_indices]
    state.vehicles = new_vehicles
    
    # 移除车辆后, 重新计算库存
    state.compute_inventory()

    return state


def worst_removal(current: SolutionState, rng: rnd.Generator):
    """
    移除解中剩余体积最大的 q 辆车
    目的是避免车辆空间的浪费
    """
    state = current.copy()
    if len(state.vehicles) <= 1:
        return state
    
    # 计算每个车辆的剩余体积
    free_volumes = {}
    for veh in state.vehicles:
        key = (veh.fact_id, veh.dealer_id, veh.type, veh.day)
        free_volumes[key] = state.data.veh_type_cap[veh.type] - state.compute_veh_load(veh)
    
    # 随机选择要移除的数量 q, q 的取值范围为 [1, len(state.vehicles) // 2]
    ub = max(1, len(state.vehicles) // 2)
    num_remove = random.randint(1, ub)
    
    # 找出 num_remove 辆剩余体积最大的车辆
    target_keys = []
    for i in range(num_remove):
        if len(free_volumes) > 0:
            target_keys.append(max(free_volumes, key=free_volumes.get))
            free_volumes.pop(target_keys[-1])
    
    # 找出目标车辆在state.vehicles中的位置
    removal_candidates = set()
    for i, veh in enumerate(state.vehicles):
        key = (veh.fact_id, veh.dealer_id, veh.type, veh.day)
        if key in target_keys:
            removal_candidates.add(i)
            
    removal_candidates = sorted(list(removal_candidates), reverse=True)
    
    # 从state.vehicles中移除这些车辆
    for idx in removal_candidates:
        if 0 <= idx < len(state.vehicles):
            state.vehicles.pop(idx)
    
    # 移除车辆后, 重新计算库存
    state.compute_inventory()
    
    return state

def infeasible_removal(current: SolutionState, rng: rnd.Generator):
    """
    应用在不可行解上, 移除当前周期 t 结束时, 库存 < 0 的SKU对应的车辆
    目的是防止运出去的SKU数量超过生产基地的供给量, 保证解的可行性
    """
    state = current.copy()
    if len(state.vehicles) <= 1:
        return state
    
    neg_inv = {}
    for (plant, sku_id, day), inv in state.s_ikt.items():
        if inv < 0:
            neg_inv[plant, sku_id, day] = inv
    
    # 找出这些库存数量为负值的SKU装载在哪些车辆中
    removal_vehicles = set()
    for veh in state.vehicles:
        for (plant, sku_id, day) in neg_inv:
            if veh.fact_id == plant and veh.day == day:
                removal_vehicles.add(veh)
    
    # 从当前解中全部移除这些车辆
    state.vehicles = [veh for veh in state.vehicles if veh not in removal_vehicles]
    
    # 移除车辆后, 重新计算库存
    state.compute_inventory()
    
    return state
    

def demand_removal(current: SolutionState, rng: rnd.Generator):
    """
    移除当前周期 t 结束时, 剩余库存数量最多的SKU对应的 q 辆车
    目的是降低当前周期结束时, 库存数量过多的风险
    """
    state = current.copy()
    data = state.data
    
    if len(state.vehicles) <= 1:
        return state
    
    # 按生产基地分组，找出每个生产基地库存最多的SKU
    plant_max_inv = {}  # 存储每个生产基地的最大库存信息
    for (plant, sku_id, day), inv in state.s_ikt.items():
        # 如果这个生产基地还没有记录过，或者当前SKU的库存更大
        if plant not in plant_max_inv or inv > plant_max_inv[plant][1]:
            plant_max_inv[plant] = (sku_id, inv)
    
    # 构建需要考虑的(plant, sku_id, day)组合
    highest_inv = {(plant, sku_info[0], day): sku_info[1] 
                  for plant, sku_info in plant_max_inv.items() 
                  for day in range(1, data.horizons + 1)}
    
    # 找出这些SKU对应的车辆
    removal_candidates = set()  # 使用集合去除重复索引
    for i, veh in enumerate(state.vehicles):
        for (plant, sku_id, day) in highest_inv:
            if veh.fact_id == plant and veh.day == day:
                removal_candidates.add(i)
    
    # 转换为列表并排序
    removal_candidates = sorted(list(removal_candidates), reverse=True)
    
    # 随机选择要移除的数量 q
    if removal_candidates:
        # 确保不会选择超过可用数量的索引
        ub = max(1, len(removal_candidates) // 2)
        num_remove = random.randint(1, ub)
        
        # 随机选择要移除的索引
        selected_indices = rng.choice(removal_candidates, size=num_remove, replace=False)
        
        # 确保索引是从大到小排序的
        selected_indices = sorted(selected_indices, reverse=True)
        
        # 从后向前移除选中的车辆
        for idx in selected_indices:
            if 0 <= idx < len(state.vehicles):  # 添加索引范围检查
                state.vehicles.pop(idx)

    # 移除车辆后, 重新计算库存
    state.compute_inventory()
    
    return state
    

# 实现Shaw移除算子
def shaw_removal(current: SolutionState, rng: rnd.Generator):
    """Shaw移除算子: 移除相关联的车辆"""
    state =  current.copy()
    if len(state.vehicles) <= 1:
        return state
    
    # 随机选择一个种子车辆
    seed_idx = rng.integers(0, len(state.vehicles))
    seed_veh = state.vehicles[seed_idx]
    
    # 计算其他车辆与种子车辆的相关性
    relatedness = []
    for i, veh in enumerate(state.vehicles):
        if i == seed_idx:
            continue
        
        # 计算相关性（相同生产基地或经销商、相同日期等）
        score = 0
        if veh.fact_id == seed_veh.fact_id:
            score += 3
        if veh.dealer_id == seed_veh.dealer_id:
            score += 2
        if veh.day == seed_veh.day:
            score += 1
        
        relatedness.append((i, score))
        
    # 按相关性排序
    relatedness.sort(key=lambda x: x[1], reverse=True)
    
    # 移除最相关的q个车辆
    num_remove = min(int(len(state.vehicles) * 0.3), len(relatedness))
    remove_indices = [idx for idx, _ in relatedness[:num_remove]]
    remove_indices.append(seed_idx)  # 也移除种子车辆
    
    # 按索引从大到小排序，以便从后向前移除
    remove_indices.sort(reverse=True)
    
    # 从state.vehicles中移除这些车辆
    for idx in remove_indices:
        if 0 <= idx < len(state.vehicles):
            state.vehicles.pop(idx)
    
    # 移除车辆后, 重新计算库存
    state.compute_inventory()
    
    return state
    
    
# 路径移除算子
def path_removal(current: SolutionState, rng: rnd.Generator):
    """路径移除算子: 移除特定生产基地-经销商对上的车辆"""
    state = current.copy()
    if len(state.vehicles) <= 1:
        return state
    
    # 随机选择一条路径（工厂-经销商对）
    paths = set((veh.fact_id, veh.dealer_id) for veh in state.vehicles)
    if not paths:
        return state
    
    target_path = rng.choice(list(paths))
    target_path = tuple(target_path)
    
    # 找出该路径上的所有车辆
    path_vehicles = []
    for i, veh in enumerate(state.vehicles):
        if (veh.fact_id, veh.dealer_id) == target_path:
            path_vehicles.append(i)
    
    # 随机移除该路径上的一部分车辆
    if path_vehicles and len(path_vehicles) > 1:
        num_remove = rng.integers(1, len(path_vehicles))
        remove_indices = rng.choice(path_vehicles, size=num_remove, replace=False)
        
        # 按索引从大到小排序，以便从后向前移除
        remove_indices = sorted(remove_indices, reverse=True)
        
        for idx in remove_indices:
            if 0 <= idx < len(state.vehicles):
                state.vehicles.pop(idx)
    
    # 移除车辆后, 重新计算库存
    state.compute_inventory()
    
    return state
    
    
# 局部搜索修复
def local_search_repair(partial: SolutionState, rng: rnd.Generator):
    """使用局部搜索改进解"""
    state = greedy_repair(partial, rng)  # 先使用贪心修复获得可行解
    data = state.data
    
    # 局部搜索操作：尝试替换车型以减少成本
    improved = True
    while improved:
        improved = False
        # 记录改进之前的目标函数值
        old_obj = state.objective()
        for i, veh in enumerate(state.vehicles):
            # 获取当前装载量
            current_load = state.compute_veh_load(veh)
            # 获取当前车型容量
            capacity = data.veh_type_cap[veh.type]
            
            
            # 尝试更小的车型
            for veh_type in state.data.all_veh_types:
                # 如果新车型容量足够且成本更低
                condtion1 = data.veh_type_cap[veh_type] >= current_load
                condtion2 = current_load >= data.veh_type_min_load[veh_type]
                condtion3 = data.veh_type_cap[veh_type] < capacity
                
                if condtion1 and condtion2 and condtion3:
                    # 创建新车辆并复制货物
                    new_veh = Vehicle(veh.fact_id, veh.dealer_id, veh_type, veh.day, data)
                    new_veh.cargo = veh.cargo.copy()
                    
                    # 临时替换并评估
                    old_veh = state.vehicles[i]
                    state.vehicles[i] = new_veh
                    
                    new_obj = state.objective()
                    
                    # 如果改进，接受变更
                    if new_obj < old_obj:
                        improved = True
                    else:
                        # 还原变更
                        state.vehicles[i] = old_veh

    return state
    
    
def greedy_repair(partial: SolutionState, rng: rnd.Generator):
    """
    使用改进的贪心算法修复不可行解
    1. 考虑订单优先级
    2. 优化车型选择
    3. 考虑生产基地库存状况
    """
    state = partial.copy()
    data = state.data
    shipped = state.compute_shipped()
    unsatisfied = {}
    
    # 计算未满足的需求并评估优先级
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if shipped_qty < demand and demand > 0:
            unmet_demand = demand - shipped_qty
            
            # 计算订单优先级指标
            demand_ratio = unmet_demand / demand  # 未满足比例
            
            # 计算可用库存, 所有生产基地在所有周期内, 可以提供的总库存数量
            total_available = 0
            # 计算每个生产基地在所有周期内, 可以提供的总库存数量
            available_plants = []
            for plant in [p for p, d in data.construct_supply_chain() 
                        if d == dealer and sku_id in data.skus_plant[p]]:
                for day in range(1, data.horizons + 1):
                    current_stock = state.s_ikt.get((plant, sku_id, day - 1), 0) + \
                        data.sku_prod_each_day.get((plant, sku_id, day), 0)
                    total_available += current_stock
                    available_plants.append((plant, current_stock))
            
            # 计算库存紧急度
            stock_urgency = 1.0
            if total_available > 0:
                stock_urgency = min(1.0, unmet_demand / total_available)
            
            # 综合优先级计算（可调整权重）
            priority = 0.8 * demand_ratio + 0.2 * stock_urgency
            
            unsatisfied[(dealer, sku_id)] = {
                'demand': unmet_demand,
                'priority': priority,
                'available_plants': available_plants
            }
    
    # 按优先级排序订单
    sorted_demands = sorted(unsatisfied.items(), 
                          key=lambda x: x[1]['priority'], 
                          reverse=True)
    
    # 从最高优先级的订单开始修复
    for (dealer, sku_id), order_info in sorted_demands:
        unmet_demand = order_info['demand']
        available_plants = order_info['available_plants']
        
        # 按库存量排序生产基地
        available_plants.sort(key=lambda x: x[1], reverse=True)
        
        if not available_plants:
            continue
        
        # 选择库存最多的生产基地
        plant = available_plants[0][0]
        
        # 选择最合适的车型（优先选择较大容量的车型）
        veh_types = sorted(list(data.all_veh_types), 
                         key=lambda x: data.veh_type_cap[x],
                         reverse=True)
        
        for veh_type in veh_types:
            day = rng.choice(range(1, data.horizons + 1))  # 随机选择一个周期
            vehicle = Vehicle(plant, dealer, veh_type, day, data)
            orders = {sku_id: unmet_demand}
            value = veh_loading(state, vehicle, orders)
        
            if value:
                # 更新未满足需求
                actual_loaded = sum(qty for (s, d), qty in vehicle.cargo.items() 
                                if s == sku_id and d == day)
                unmet_demand -= actual_loaded
                
                # 如果还有未满足的需求，继续尝试装载
                if unmet_demand > 0:
                    continue
                else:
                    break
    
    return state


def inventory_balance_repair(partial: SolutionState, rng: rnd.Generator):
    """
    基于库存平衡的修复算子
    平衡各生产基地库存水平，避免库存积压
    """
    state = partial.copy()
    data = state.data
    shipped = state.compute_shipped()
    unsatisfied = {}
    
    # 计算未满足的需求
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if shipped_qty < demand and demand > 0:
            unsatisfied[(dealer, sku_id)] = demand - shipped_qty
    
    # 计算在周期 t 结束时, 各生产基地的库存水平
    plant_inventory = {}
    for (plant, sku_id, day), inv in state.s_ikt.items():
        plant_inventory[plant, day] = plant_inventory.get((plant, day), 0) + inv
    
    # 按库存水平从高到低排序生产基地
    sorted_plants = sorted(plant_inventory.items(), key=lambda x: x[1], reverse=True)
    
    # 从库存较高的生产基地开始修复
    for (plant, day), _ in sorted_plants:
        # 获取该生产基地可以供应的经销商
        dealers = [dealer for (p, dealer) in data.construct_supply_chain() if p == plant]
        
        for dealer in dealers:
            # 选择合适的车型
            veh_type = rng.choice(list(data.all_veh_types))
            vehicle = Vehicle(plant, dealer, veh_type, day, data)
            
            # 获取该经销商的未满足订单
            orders = {sku_id: qty for (dealer_id, sku_id), qty in unsatisfied.items()
                     if dealer_id == dealer}
            
            if not orders:
                continue
                
            # 尝试装载车辆
            value = veh_loading(state, vehicle, orders)
            
            # 更新未满足订单
            if value:
                for sku_id in orders:
                    unsatisfied[(dealer, sku_id)] -= vehicle.cargo.get((sku_id, day), 0)
    
    return state

def urgency_repair(partial: SolutionState, rng: rnd.Generator):
    """
    基于紧急程度的修复算子
    综合考虑多个因素计算订单紧急程度
    """
    state = partial.copy()
    data = state.data
    shipped = state.compute_shipped()
    unsatisfied = {}
    
    # 计算未满足的需求并评估紧急程度
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if shipped_qty < demand and demand > 0:
            # 计算多维度的紧急程度指标
            unmet_ratio = (demand - shipped_qty) / demand  # 未满足需求比例
            
            # 计算所有所有生产基地在所有周期内, 可以提供的库存
            total_available = 0
            for plant in [p for p, d in data.construct_supply_chain() 
                          if d == dealer and sku_id in data.skus_plant[p]]:
                for day in range(1, data.horizons + 1):
                    total_available += state.s_ikt.get((plant, sku_id, day - 1), 0) + \
                        data.sku_prod_each_day.get((plant, sku_id, day), 0)
            
            inventory_urgency = 1.0
            if total_available > 0:
                # 计算库存紧急度, 未满足的需求占可以提供的库存的比例
                inventory_urgency = min(1.0, (demand - shipped_qty) / total_available)
            
            # 综合紧急度计算（可以调整各个因素的权重）
            urgency = 0.8 * unmet_ratio + 0.2 * inventory_urgency
            
            unsatisfied[(dealer, sku_id)] = (demand - shipped_qty, urgency)
    
    # 按紧急程度排序
    sorted_demands = sorted(unsatisfied.items(), key=lambda x: x[1][1], reverse=True)
    
    # 从最紧急的订单开始修复
    for (dealer, sku_id), (unmet_demand, _) in sorted_demands:
        # 找到可以供应该经销商的生产基地，并按库存量排序
        available_plants = []
        for plant in [p for p, d in data.construct_supply_chain() 
                     if d == dealer and sku_id in data.skus_plant[p]]:
            for day in range(1, data.horizons + 1):
                current_stock = state.s_ikt.get((plant, sku_id, day - 1), 0) + \
                    data.sku_prod_each_day.get((plant, sku_id, day), 0)
                available_plants.append((plant, current_stock))
        
        # 按库存量排序
        available_plants.sort(key=lambda x: x[1], reverse=True)
        
        if not available_plants:
            continue
        
        # 选择库存最多的生产基地
        plant = available_plants[0][0]
        
        # 选择最合适的车型（优先选择较大容量的车型）
        veh_types = sorted(list(data.all_veh_types), 
                         key=lambda x: data.veh_type_cap[x],
                         reverse=True)
        
        for veh_type in veh_types:
            day = rng.choice(range(1, data.horizons + 1))  # 随机选择一个周期
            vehicle = Vehicle(plant, dealer, veh_type, day, data)
            orders = {sku_id: unmet_demand}
            value = veh_loading(state, vehicle, orders)
            
            if value:
                break  # 成功装载，不需要尝试其他车型
    
    return state


def infeasible_repair(partial: SolutionState, rng: rnd.Generator):
    """
    修复不可行解, 找出当前周期 t 结束时, 库存数量为负的SKU对应的车辆, 进行修复
    """
    state = partial.copy()
    data = state.data
    neg_inv = {k: v for k, v in state.s_ikt.items() if v < 0}
    
    # 处理负库存
    while neg_inv:
        for veh in state.vehicles[:]:  # 复制列表避免修改时出错
            for (plant, sku_id, day), inv in list(neg_inv.items()):
                if veh.fact_id == plant and veh.day == day and (sku_id, day) in veh.cargo:
                    qty = veh.cargo[(sku_id, day)]
                    decrease_qty = min(qty, -inv)  # 减少到库存非负
                    veh.cargo[(sku_id, day)] -= decrease_qty
                    if veh.cargo[(sku_id, day)] == 0:
                        del veh.cargo[(sku_id, day)]
                    if veh.is_empty():
                        state.vehicles.remove(veh)
                    
                    # 同步更新实际库存
                    state.s_ikt[plant, sku_id, day] += decrease_qty
                    
                    # 更新临时字典并移除已修复项
                    neg_inv[plant, sku_id, day] += decrease_qty
                    if neg_inv[plant, sku_id, day] >= 0:
                        del neg_inv[plant, sku_id, day]
                    break
            if not neg_inv:
                break
    
    # 重新计算库存，确保一致性
    state.compute_inventory()
    
    return state

def demand_first_repair(partial: SolutionState, rng: rnd.Generator):
    """
    需求优先修复算子
    将需求满足作为最高优先级的硬约束，确保所有需求都得到满足
    """
    state = partial.copy()
    data = state.data
    shipped = state.compute_shipped()
    unsatisfied = {}
    
    # 计算未满足的需求
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if shipped_qty < demand and demand > 0:
            unsatisfied[(dealer, sku_id)] = demand - shipped_qty
    
    # 如果没有未满足的需求，直接返回
    if not unsatisfied:
        return state
    
    # 按需求量从大到小排序
    sorted_demands = sorted(unsatisfied.items(), key=lambda x: x[1], reverse=True)
    
    # 为每个未满足的需求执行更加激进的尝试
    for (dealer, sku_id), unmet_demand in sorted_demands:
        # 找到所有可能的生产基地，包括考虑期初库存
        available_plants = []
        for plant in [p for p, d in data.construct_supply_chain() 
                     if d == dealer and sku_id in data.skus_plant[p]]:
            # 计算期初库存+所有日期的生产总量
            total_supply = state.s_ikt.get((plant, sku_id, 0), 0)  # 确保考虑期初库存
            for day in range(1, data.horizons + 1):
                total_supply += data.sku_prod_each_day.get((plant, sku_id, day), 0)
            if total_supply > 0:
                available_plants.append((plant, total_supply))
        
        # 按总供应量排序
        available_plants.sort(key=lambda x: x[1], reverse=True)
        
        # 对每个可用生产基地尝试满足需求
        remaining_demand = unmet_demand
        for plant, _ in available_plants:
            if remaining_demand <= 0:
                break
                
            # 从第1天开始尝试，优先使用早期生产的产品
            for day in range(1, data.horizons + 1):
                if remaining_demand <= 0:
                    break
                
                # 计算当天可用库存
                prev_inventory = state.s_ikt.get((plant, sku_id, day - 1), 0)
                production = data.sku_prod_each_day.get((plant, sku_id, day), 0)
                available = prev_inventory + production
                
                if available > 0:
                    # 选择合适的车型，优先选择较大容量的车型
                    veh_types = sorted(list(data.all_veh_types), 
                                    key=lambda x: data.veh_type_cap[x],
                                    reverse=True)
                    
                    for veh_type in veh_types:
                        if remaining_demand <= 0:
                            break
                            
                        # 创建新车辆并尝试装载
                        vehicle = Vehicle(plant, dealer, veh_type, day, data)
                        orders = {sku_id: remaining_demand}
                        value = veh_loading(state, vehicle, orders)
                        
                        if value:
                            # 更新剩余需求
                            loaded_qty = vehicle.cargo.get((sku_id, day), 0)
                            remaining_demand -= loaded_qty
    
    # 如果仍有未满足的需求，使用最小车型尝试满足
    shipped = state.compute_shipped()
    still_unsatisfied = {}
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if shipped_qty < demand:
            still_unsatisfied[(dealer, sku_id)] = demand - shipped_qty
    
    # 如果仍有未满足的需求，进行更激进的尝试
    if still_unsatisfied:
        # 按需求量从大到小排序
        sorted_demands = sorted(still_unsatisfied.items(), key=lambda x: x[1], reverse=True)
        
        for (dealer, sku_id), unmet_demand in sorted_demands:
            # 尝试所有可能的生产基地
            for plant in [p for p, d in data.construct_supply_chain() 
                         if d == dealer and sku_id in data.skus_plant[p]]:
                for day in range(1, data.horizons + 1):
                    # 计算当天可用库存，确保考虑期初库存的影响
                    prev_inventory = state.s_ikt.get((plant, sku_id, day - 1), 0)
                    production = data.sku_prod_each_day.get((plant, sku_id, day), 0)
                    available = prev_inventory + production
                    
                    if available > 0:
                        # 使用最小车型，强制装载
                        veh_type = min(data.all_veh_types, key=lambda x: data.veh_type_cap[x])
                        vehicle = Vehicle(plant, dealer, veh_type, day, data)
                        orders = {sku_id: min(unmet_demand, available)}
                        
                        # 忽略最小起运量约束强制装载
                        value = veh_loading(state, vehicle, orders)
                        
                        if value:
                            loaded_qty = vehicle.cargo.get((sku_id, day), 0)
                            unmet_demand -= loaded_qty
                            
                            if unmet_demand <= 0:
                                break
                
                if unmet_demand <= 0:
                    break
    
    # 最终检查是否所有需求都已满足
    shipped = state.compute_shipped()
    all_demands_fulfilled = True
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if shipped_qty < demand:
            all_demands_fulfilled = False
            print(f"Warning: Failed to satisfy demand for dealer {dealer}, SKU {sku_id}: {demand - shipped_qty} units remaining")

    if all_demands_fulfilled:
        # print("All demands successfully satisfied!")
        pass
    
    return state
