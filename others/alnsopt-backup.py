import copy
import time
import random
import numpy.random as rnd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from InputDataALNS import DataALNS
from vehicle import Vehicle
from optutility import LogPrinter


log_printer = LogPrinter(time.time())

@dataclass
class SolutionState:
    # day: int  # 不能将每个周期的约束条件割裂开来, 作为整体考虑
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
    

    def objective(self):
        # # 添加缩放因子以控制目标函数值的尺度
        # # 避免目标函数值过大, 影响模拟退火中计算接收概率时, 出现数值溢出
        # scale_factor = 1e-2  # 缩放因子，可以根据实际情况调整
        
        total_cost = sum(self.data.veh_type_cap[veh.type] for veh in self.vehicles)
        
        self.compute_inventory()  
        
        return total_cost
    
    
    def check_fulfill_demand(self):
        """检查是否满足经销商的需求, 如果满足, 返回True, 否则返回False"""
        shipped = self.compute_shipped()
        for (dealer, sku_id), demand in self.data.demands.items():
            shipped_qty = shipped.get((dealer, sku_id), 0)
            if shipped_qty < demand:
                log_printer.print(f"Demand for {dealer} {sku_id} is not satisfied.\n", color="bold yellow")
                log_printer.print(f"Shipped quantity: {shipped_qty}, Demand: {demand}.\n", color='bold yellow')
                return False
        return True
    
    
    def check_inventory_limit(self, day_id: int):
        """检查是否满足生产基地的库存限制, 如果满足, 返回True, 否则返回False"""
        for plant, max_cap in self.data.plant_inv_limit.items():
            total_inv = sum(inv for (fact_id, sku_id, day), inv in self.s_ikt.items() 
                            if fact_id == plant and day == day_id)
            if total_inv > max_cap:
                log_printer.print(f"Inventory limit for {plant} in period {day_id} is exceeded.\n", color="bold yellow")
                log_printer.print(f"Total inventory: {total_inv}, Max capacity: {max_cap}.\n", color='bold yellow')
                return False
        return True
    
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
        if len(self.vehicles) > 0:
            for (plant, sku_id, day) in self.s_indices:
                # 计算从该生产基地运出的SKU数量
                shipped_from_plant = sum(veh.cargo.get((sku_id, day), 0) 
                                        for veh in self.vehicles 
                                        if veh.fact_id == plant and veh.day == day)
                
                # 获取前一天的库存
                prev_inventory = self.s_ikt.get((plant, sku_id, day - 1), 0)
                
                # 获取当天的生产量
                production = self.data.sku_prod_each_day.get((plant, sku_id, day), 0)
                
                # 计算当前库存：前一天库存 + 当天生产 - 当天发出
                current_inventory = prev_inventory + production - shipped_from_plant
                
                # 更新库存
                self.s_ikt[plant, sku_id, day] = current_inventory
                
                # 更新历史库存记录
                self.data.historical_s_ikt[plant, sku_id, day] = current_inventory
    
    def compute_veh_load(self, veh: Vehicle):
        """计算当前车辆装载量"""
        total_volume = 0
        for (sku_id, day), qty in veh.cargo.items():
            total_volume += self.data.sku_sizes[sku_id] * qty
        return total_volume
    
    def is_veh_min_load(self, veh: Vehicle):
        """判断当前车辆是否满足最小起运量, 如果不满足, 返回TRUE, 否则返回FALSE"""
        total_volume = self.compute_veh_load(veh)
        if total_volume < self.data.veh_type_min_load[veh.type]:
            log_printer.print(f"Vehicle {veh.fact_id} {veh.dealer_id} {veh.type} {veh.day} does not satisfy the minimum load requirement.\n", color="bold yellow")
            log_printer.print(f"Total volume: {total_volume}, Minimum load: {self.data.veh_type_min_load[veh.type]}.\n", color='bold yellow')
            return True
        return False
    
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
    current_load = 0  # 当前装载量,指的是装载的SKU的总体积
    
    # 创建临时车辆进行装载尝试
    temp_vehicle = Vehicle(veh.fact_id, veh.dealer_id, veh.type, veh.day, data)
    
    for sku_id, qty in orders.items():
        if sku_id in data.skus_plant[veh.fact_id]:
            # 计算当前生产基地可以提供的SKU数量
            supply = state.s_ikt.get((veh.fact_id, sku_id, veh.day - 1), 0) + \
                data.sku_prod_each_day.get((veh.fact_id, sku_id, veh.day), 0)
            
            # 确定可装载数量
            loadable_qty = min(qty, supply)
            
            # 尝试装载
            while loadable_qty > 0:
                # 计算此次装载后的总体积
                sku_volume = data.sku_sizes[sku_id] * loadable_qty
                if current_load + sku_volume <= data.veh_type_cap[veh.type]:
                    # 可以全部装载
                    temp_vehicle.load(sku_id, loadable_qty)
                    current_load += sku_volume
                    loadable_qty = 0
                else:
                    # 计算还能装载多少
                    remaining_space = data.veh_type_cap[veh.type] - current_load
                    partial_qty = remaining_space // data.sku_sizes[sku_id]
                    if partial_qty > 0:
                        temp_vehicle.load(sku_id, partial_qty)
                        current_load += partial_qty * data.sku_sizes[sku_id]
                        loadable_qty -= partial_qty
                    break
    
    # 不论是否满足最小起运量, 都记录在当前解中
    veh.cargo = temp_vehicle.cargo.copy()
    veh.capacity = temp_vehicle.capacity
    state.vehicles.append(veh)
    
    # 当前经销商的订单考虑完毕, 返回True
    return True


def initial_solution(state: SolutionState, rng: rnd.Generator):
    """
    使用贪心算法生成初始解, 尽可能满足经销商的订单需求
    """
    data = state.data
    demands = data.demands.copy()
    veh_type = rng.choice(list(data.all_veh_types))
    triples = {(plant, dealer, day) for (plant, dealer) in data.construct_supply_chain()
               for day in range(1, data.horizons + 1)}
    
    for (plant, dealer, day) in triples:
        vehicle = Vehicle(plant, dealer, veh_type, day, data)
        orders = {sku_id: qty for (dealer_id, sku_id), qty in demands.items()
                  if dealer_id == dealer and qty > 0}
        
        # 如果没有订单，跳过当前循环
        if not orders:
            continue
            
        # 尝试装载车辆
        value = veh_loading(state, vehicle, orders)
        
        # 如果装载成功，更新demands
        if value:
            for sku_id in orders:
                demands[(dealer, sku_id)] -= vehicle.cargo.get((sku_id, day), 0)
    
    return state


def random_removal(current: SolutionState, rng: rnd.Generator):
    """
    随机移除解中25%的车辆
    目的是引入随机性, 使搜索更加多样化
    """
    state = current.copy()
    degree = 0.25
    num_remove = int(len(state.vehicles) * degree)
    remove_indices = rng.choice(range(len(state.vehicles)), num_remove, replace=False)
    new_vehicles = [veh for i, veh in enumerate(state.vehicles) if i not in remove_indices]
    state.vehicles = new_vehicles

    return state


def worst_removal(current: SolutionState, rng: rnd.Generator):
    """
    移除解中剩余体积最大的 q 辆车
    目的是避免车辆空间的浪费
    """
    state = current.copy()
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
    
    return state

def infeasible_removal(current: SolutionState, rng: rnd.Generator):
    """
    应用在不可行解上, 移除当前周期 t 结束时, 库存 < 0 的SKU对应的车辆
    目的是防止运出去的SKU数量超过生产基地的供给量, 保证解的可行性
    """
    state = current.copy()
    neg_inv = {}
    for (plant, sku_id, day), inv in state.s_ikt.items():
        if inv < 0:
            neg_inv[plant, sku_id, day] = inv
    
    # 找出这些库存数量为负值的SKU装载在哪些车辆中
    removal_vehicles = []
    for veh in state.vehicles:
        for (plant, sku_id, day) in neg_inv:
            if veh.fact_id == plant and veh.day == day:
                removal_vehicles.append(veh)
    
    # 从当前解中全部移除这些车辆
    state.vehicles = [veh for veh in state.vehicles if veh not in removal_vehicles]
    
    return state
    

def demand_removal(current: SolutionState, rng: rnd.Generator):
    """
    移除当前周期 t 结束时, 剩余库存数量最多的SKU对应的 q 辆车
    目的是降低当前周期结束时, 库存数量过多的风险
    """
    state = current.copy()
    data = state.data
    
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
            priority = 0.7 * demand_ratio + 0.3 * stock_urgency
            
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
        
        # 尝试不同车型
        for veh_type in veh_types:
            # 检查是否满足最小装载量约束
            if unmet_demand * data.sku_sizes[sku_id] >= data.veh_type_min_load[veh_type]:
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
            
            # 只有在装载成功且车辆为空时才需要移除
            if value and vehicle.is_empty():
                state.vehicles.remove(vehicle)
            
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
            urgency = 0.7 * unmet_ratio + 0.3 * inventory_urgency
            
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
            elif value and vehicle.is_empty():
                state.vehicles.remove(vehicle)
    
    return state


def infeasible_repair(partial: SolutionState, rng: rnd.Generator):
    """
    修复不可行解, 找出当前周期 t 结束时, 库存数量为负的SKU对应的车辆, 进行修复
    """
    state = partial.copy()
    data = state.data
    neg_inv = {}
    for (plant, sku_id, day), inv in state.s_ikt.items():
        if inv < 0:
            neg_inv[plant, sku_id, day] = inv
    
    # 找出这些库存数量为负值的SKU装载在哪些车辆中
    neg_vehicles = []
    for veh in state.vehicles:
        for (plant, sku_id, day) in neg_inv:
            if veh.fact_id == plant and veh.day == day:
                neg_vehicles.append(veh)
    
    # 减少这些车辆中装载的SKU数量, 直到库存数量为非负
    for veh in neg_vehicles:
        plant = veh.fact_id
        for (sku_id, day), qty in veh.cargo.items():
            if (plant, sku_id, day) in neg_inv:
                while neg_inv[plant, sku_id, day] < 0:
                    # 确保减少的数量不会超过当前装载的数量
                    if qty <= 0:
                        break   
                    decrease_qty = min(random.randint(1, int(qty)), qty)
                    veh.cargo[sku_id, day] -= decrease_qty
                    neg_inv[plant, sku_id, day] += decrease_qty
                    # 如果库存恢复到0，退出循环
                    if neg_inv[plant, sku_id, day] == 0:
                        break

    return state
