# 标准库导入
import copy
import os
import random
import time
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Tuple, Set, Optional
from alnstrack import ALNSTracker

# 在导入科学计算库之前配置环境以避免 KMeans 内存泄漏
if os.name == 'nt':  # Windows系统
    if 'OMP_NUM_THREADS' not in os.environ:
        # 根据 sklearn 警告建议，设置为 1 以完全避免内存泄漏
        os.environ['OMP_NUM_THREADS'] = '1'
    # 抑制 KMeans 内存泄漏警告
    warnings.filterwarnings('ignore', 
                          message='KMeans is known to have a memory leak on Windows with MKL*',
                          category=UserWarning)

# 第三方库导入
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import numpy.random as rnd

# 本地模块导入
from InputDataALNS import DataALNS
from vehicle import Vehicle
from optutility import POSITIVEEPS

@dataclass
class SolutionState:
    data: DataALNS
    vehicles: List[Vehicle] = field(default_factory=list)
    s_ikt: Dict[Tuple[str, str, int], int] = field(default_factory=dict)
    s_indices: Set[Tuple[str, str, int]] = field(default_factory=set)
    tracker: Optional['ALNSTracker'] = None  # 引入追踪器, 便于访问迭代历史数据,用于ML训练

    def __post_init__(self):
        self.vehicles = []
        self.s_ikt = {}
        self.s_indices = self.construct_indices()
        self._iteration_count = 0  # 迭代计数器, 用于ALNSTracker
        
        # 对s_ikt进行初始化, s_ik0表示期初库存
        for (plant, sku_id, day), inv in self.data.historical_s_ikt.items():
            if day == 0:
                self.s_ikt[plant, sku_id, day] = inv
                
    def set_tracker(self, tracker: 'ALNSTracker'):
        """
        设置tracker引用, 用于ML-based operators访问迭代历史数据
        """
        self.tracker = tracker
    
    
    def validate(self):
        """
        验证解的一致性, 无副作用。
        返回 (is_feasible, violations) 元组: 
        - is_feasible: bool, 是否可行
        - violations: dict, 包含各类违反项的详细信息
        """
        self.compute_inventory()
        violations = {
            'negative_inventory': [],
            'veh_overload': [],
            'unmet_demand': []
        }
        # 检查是否有负库存
        for key, inv in self.s_ikt.items():
            if inv < 0:
                violations['negative_inventory'].append((key, inv))
        
        # 检查车辆容量是否超限
        for veh in self.vehicles:
            loaded = self.compute_veh_load(veh)
            cap = self.data.veh_type_cap[veh.type]
            if loaded - cap > POSITIVEEPS:
                violations['veh_overload'].append({
                    'veh': veh,
                    'loaded': loaded,
                    'cap': cap
                })
        
        # 检查是否满足所有需求
        shipped = self.compute_shipped()
        for (dealer, sku_id), demand in self.data.demands.items():
            shipped_qty = shipped.get((dealer, sku_id), 0)
            if shipped_qty < demand:
                violations['unmet_demand'].append({
                    'dealer': dealer,
                    'sku_id': sku_id,
                    'demand': demand,
                    'shipped': shipped_qty
                })
        is_feasible = not (violations['negative_inventory'] or violations['veh_overload'] or violations['unmet_demand'])
        
        return is_feasible, violations
    

    def objective(self):
        """这个目标函数的作用是引导搜索过程, 并不是当前解的实际成本"""
        # 添加缩放因子以控制目标函数值的尺度
        # 避免目标函数值过大, 影响模拟退火中计算接收概率时, 出现数值溢出
        scale_factor = 1e-3  # 缩放因子, 可以根据实际情况调整
        
        self.compute_inventory()  # 计算并更新库存水平
        
        # 检查解的可行性
        feasibility, violations = self.validate()
        if not feasibility:
            return float('inf')  # 返回一个非常大的值, 表示解不可行
        
        total_cost = sum(self.data.veh_type_cost[veh.type] for veh in self.vehicles)
      
        total_cost = self.punish_non_fulfill_demand(total_cost)
        
        total_cost = self.punish_exceeded_inventory_limit(total_cost)
        
        total_cost = self.punish_deficient_veh_min_load(total_cost)
        
        total_cost = self.punish_negative_inventory(total_cost)
        
        return total_cost * scale_factor
    
    def calculate_cost(self):
        """计算当前解的实际成本"""
        # 使用车辆的固定成本
        total_cost = sum(self.data.veh_type_cost[veh.type] for veh in self.vehicles)
        # 计算解中不满足最小运量约束的惩罚成本
        veh_nums_punished = sum(1 for veh in self.vehicles if self.compute_veh_load(veh) < self.data.veh_type_min_load[veh.type])
        punishment_cost = veh_nums_punished * self.data.param_pun_objective
        
        return total_cost + punishment_cost

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
            if day > 0:  # 只处理day>0的情况, 因为day=0是期初库存, 不应该被修改
                # 计算从该生产基地运出的SKU数量
                shipped_from_plant = sum(veh.cargo.get((sku_id, day), 0) 
                                        for veh in self.vehicles 
                                        if veh.fact_id == plant and veh.day == day)
                
                # 获取前一天的库存, 确保期初库存被正确考虑
                prev_inventory = self.s_ikt.get((plant, sku_id, day - 1), 0)
                
                # 获取当天的生产量
                production = self.data.sku_prod_each_day.get((plant, sku_id, day), 0)
                
                # 计算当前库存: 前一天库存 + 当天生产 - 当天发出
                current_inventory = prev_inventory + production - shipped_from_plant
                
                # 更新库存
                self.s_ikt[plant, sku_id, day] = current_inventory
    
    def copy(self):
        """复制当前解: 确保每个解都是不可变的"""
        new_state = SolutionState(self.data)
        new_state.vehicles = copy.deepcopy(self.vehicles)
        new_state.s_ikt = copy.deepcopy(self.s_ikt)
        new_state.s_indices = copy.deepcopy(self.s_indices)
        new_state.tracker = self.tracker  # 共享同一个tracker引用
        return new_state

def veh_loading(state: SolutionState, veh: Vehicle, orders: Dict[str, int], commit: bool = True):
    """
    车辆装载函数, 考虑车辆容量约束和可用库存, 
    根据车辆可用容量和订单数量, 自动分配多辆车辆

    参数:
    - commit: 如果为 False, 则函数不向 state.vehicles 添加新 vehicle
      便于在模拟时评估插入效果 (caller 应在临时 state 上模拟或使用 copy)
    """
    data = state.data
    fact_id, dealer_id, veh_type, day = veh.fact_id, veh.dealer_id, veh.type, veh.day

    # 统计同基地同天所有车辆已装载的SKU数量
    used_inv = {}
    for vehicle in state.vehicles:
        if vehicle.fact_id == fact_id and vehicle.day == day:
            for (sku_id, d), qty in vehicle.cargo.items():
                if d == day:
                    used_inv[sku_id] = used_inv.get(sku_id, 0) + qty
    
    for sku_id, order_qty in orders.items():
        if sku_id not in data.skus_plant.get(fact_id, []):
            continue
        remain_qty = order_qty
        
        while remain_qty > 0:
            # 计算可用库存
            prev_inv = state.s_ikt.get((fact_id, sku_id, day-1), 0)
            production = data.sku_prod_each_day.get((fact_id, sku_id, day), 0)
            available = max(0, prev_inv + production - used_inv.get(sku_id, 0))
            if available <= 0:
                break
            
            # 计算当前车辆已经装载的SKU体积
            current_load = sum(data.sku_sizes[sku_id] * q for (s, d), q in veh.cargo.items() if d == day)
            cap = data.veh_type_cap[veh_type]
            
            # 计算当前车辆中可以装载该SKU的最大数量
            max_qty_by_cap = int((cap - current_load) // data.sku_sizes[sku_id]) if data.sku_sizes[sku_id] > 0 else 0
            load_qty = min(remain_qty, available, max_qty_by_cap)
            
            if load_qty <= 0:
                # 当前车辆已满或无法装载更多，若当前veh已有货且commit=True则加入state.vehicles；否则换新车
                if veh.cargo and commit:
                    state.vehicles.append(veh)
                veh = Vehicle(fact_id, dealer_id, veh_type, day, data)
                continue
            veh.load(sku_id, load_qty)
            used_inv[sku_id] = used_inv.get(sku_id, 0) + load_qty
            remain_qty -= load_qty
        
        # 最后一次循环后, 若veh有货且commit且未加入, 则加入
        if veh.cargo and commit and veh not in state.vehicles:
            state.vehicles.append(veh)
    
    # 装载完毕后, 重新计算库存（仅在commit=True时更新主state库存）
    if commit:
        state.compute_inventory()
    
    return True
                

def initial_solution(state: SolutionState, rng: rnd.Generator):
    """
    使用贪心算法生成初始解, 确保满足所有经销商的订单需求
    """
    data = state.data
    demands = data.demands.copy()
    # 1. 按需求量从大到小排序, 优先满足大需求
    sorted_demands = sorted(demands.items(), key=lambda x: x[1], reverse=True)
    for (dealer_id, sku_id), demand in sorted_demands:
        if demand <= 0:
            continue  # 跳过无需求
        remain_demand = demand
        
        # 2. 每次分配前, 实时更新已发货量, 避免重复分配
        shipped = state.compute_shipped()
        remain_demand -= shipped.get((dealer_id, sku_id), 0)
        if remain_demand <= 0:
            continue  # 已满足
        
        # 3. 遍历所有可用生产基地和天, 最大化满足需求
        supply_chain = data.construct_supply_chain()
        for plant in [plant for (plant, dealer_key), skus in supply_chain.items() if dealer_key == dealer_id and sku_id in skus]:
            for day in range(1, data.horizons + 1):
                state.compute_inventory()  # 保证库存最新
                # 计算该天该基地可用库存
                current_stock = state.s_ikt.get((plant, sku_id, day - 1), 0) + data.sku_prod_each_day.get((plant, sku_id, day), 0)
                if current_stock <= 0 or remain_demand <= 0:
                    continue
                # 4. 优先用大车型, 尽量减少车辆数
                veh_types = sorted(list(data.all_veh_types), key=lambda x: data.veh_type_cap[x], reverse=True)
                for veh_type in veh_types:
                    if remain_demand <= 0:
                        break
                    # 5. 创建车辆并尝试装载
                    vehicle = Vehicle(plant, dealer_id, veh_type, day, data)
                    orders = {sku_id: remain_demand}
                    veh_loading(state, vehicle, orders)
                    
                    # 6. 装载后, 实时更新已发货量和剩余需求, 避免重复分配
                    shipped = state.compute_shipped()
                    # 本次实际装载的SKU数量, demand - remain_demand 是本次循环之前的已发货量
                    loaded_qty = shipped.get((dealer_id, sku_id), 0) - (demand - remain_demand)
                    loaded_qty = max(0, loaded_qty)
                    loaded_qty = min(loaded_qty, remain_demand)
                    
                    remain_demand -= loaded_qty
        # 7. 记录剩余未满足需求, 供后续修复算子使用
        demands[(dealer_id, sku_id)] = remain_demand
    # 8. 若仍有未满足需求, 调用智能分批修复算子补足
    unsatisfied = {k: v for k, v in demands.items() if v > 0}
    if unsatisfied:
        state = smart_batch_repair(state, rng)
    return state

def random_removal(current: SolutionState, rng: rnd.Generator, degree: float = 0.25):
    """
    随机移除解中指定比例的车辆
    目的是引入随机性, 使搜索更加多样化
    
    Parameters:
    ----------
    current : SolutionState
        当前解状态
    rng : rnd.Generator
        随机数生成器
    degree : float
        移除比例, 默认为0.25（25%）
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 random_removal 算子")
    state = current.copy()
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] random_removal: {time.time() - t0:.4f}s")
        return state
    num_remove = int(len(state.vehicles) * degree)
    remove_indices = rng.choice(range(len(state.vehicles)), num_remove, replace=False)
    new_vehicles = [veh for i, veh in enumerate(state.vehicles) if i not in remove_indices]
    state.vehicles = new_vehicles
    state.compute_inventory()
    print(f"[OPLOG] random_removal: {time.time() - t0:.4f}s")
    return state

class RandomRemovalOperator:
    """
    带参数的随机移除算子包装类
    """
    def __init__(self, degree: float = 0.25):
        self.degree = degree
        self.__name__ = "random_removal"
    
    def __call__(self, current: SolutionState, rng: rnd.Generator):
        return random_removal(current, rng, self.degree)

def worst_removal(current: SolutionState, rng: rnd.Generator):
    """
    移除解中剩余体积最大的 q 辆车
    目的是避免车辆空间的浪费
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 worst_removal 算子")
    state = current.copy()
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] worst_removal: {time.time() - t0:.4f}s")
        return state
    free_volumes = {}
    for veh in state.vehicles:
        key = (veh.fact_id, veh.dealer_id, veh.type, veh.day)
        free_volumes[key] = state.data.veh_type_cap[veh.type] - state.compute_veh_load(veh)
    ub = max(1, len(state.vehicles) // 2)
    num_remove = random.randint(1, ub)
    target_keys = []
    for i in range(num_remove):
        if len(free_volumes) > 0:
            target_keys.append(max(free_volumes, key=free_volumes.get))
            free_volumes.pop(target_keys[-1])
    removal_candidates = set()
    for i, veh in enumerate(state.vehicles):
        key = (veh.fact_id, veh.dealer_id, veh.type, veh.day)
        if key in target_keys:
            removal_candidates.add(i)
    removal_candidates = sorted(list(removal_candidates), reverse=True)
    for idx in removal_candidates:
        if 0 <= idx < len(state.vehicles):
            state.vehicles.pop(idx)
    state.compute_inventory()
    print(f"[OPLOG] worst_removal: {time.time() - t0:.4f}s")
    return state

def infeasible_removal(current: SolutionState, rng: rnd.Generator):
    """
    应用在不可行解上, 移除当前周期 t 结束时, 库存 < 0 的SKU对应的车辆
    目的是防止运出去的SKU数量超过生产基地的供给量, 保证解的可行性
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 infeasible_removal 算子")
    state = current.copy()
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] infeasible_removal: {time.time() - t0:.4f}s")
        return state
    neg_inv = {}
    for (plant, sku_id, day), inv in state.s_ikt.items():
        if inv < 0:
            neg_inv[plant, sku_id, day] = inv
    removal_vehicles = set()
    for veh in state.vehicles:
        for (plant, sku_id, day) in neg_inv:
            if veh.fact_id == plant and veh.day == day:
                removal_vehicles.add(veh)
    state.vehicles = [veh for veh in state.vehicles if veh not in removal_vehicles]
    state.compute_inventory()
    print(f"[OPLOG] infeasible_removal: {time.time() - t0:.4f}s")
    return state
    

def surplus_inventory_removal(current: SolutionState, rng: rnd.Generator):
    """
    移除当前周期 t 结束时, 剩余库存数量最多的SKU对应的 q 辆车
    目的是降低当前周期结束时, 库存数量过多的风险
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 surplus_inventory_removal 算子")
    state = current.copy()
    data = state.data
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] surplus_inventory_removal: {time.time() - t0:.4f}s")
        return state

    # 计算每个 (plant, day) 的总库存
    plant_day_inv = {}
    for (plant, sku_id, day), inv in state.s_ikt.items():
        if day >= 1:
            plant_day_inv[(plant, day)] = plant_day_inv.get((plant, day), 0) + inv

    if not plant_day_inv:
        print(f"[OPLOG] surplus_inventory_removal: 无库存记录")
        return state

    # 找到库存最多的若干 (plant, day)
    sorted_plant_days = sorted(plant_day_inv.items(), key=lambda x: x[1], reverse=True)
    # 选取候选的 (plant, day) 集合
    candidate_plant_days = [pd for pd, _ in sorted_plant_days[:max(1, len(sorted_plant_days)//2)]]

    removal_candidates = set()
    for i, veh in enumerate(state.vehicles):
        if (veh.fact_id, veh.day) in candidate_plant_days:
            removal_candidates.add(i)
    removal_candidates = sorted(list(removal_candidates), reverse=True)
    if removal_candidates:
        ub = max(1, len(removal_candidates) // 2)
        num_remove = random.randint(1, ub)
        selected_indices = rnd.choice(list(removal_candidates), size=num_remove, replace=False)
        selected_indices = sorted(selected_indices, reverse=True)
        for idx in selected_indices:
            if 0 <= idx < len(state.vehicles):
                state.vehicles.pop(idx)
    state.compute_inventory()
    print(f"[OPLOG] surplus_inventory_removal: {time.time() - t0:.4f}s")
    return state
    

# 实现Shaw移除算子
def shaw_removal(current: SolutionState, rng: rnd.Generator, degree: float = 0.3):
    """Shaw移除算子: 移除相关联的车辆, degree为移除比例"""
    t0 = time.time()
    print(f"[OPLOG] 开始执行 shaw_removal 算子")
    state = current.copy()
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] shaw_removal: {time.time() - t0:.4f}s")
        return state
    seed_idx = rng.integers(0, len(state.vehicles))
    seed_veh = state.vehicles[seed_idx]
    relatedness = []
    for i, veh in enumerate(state.vehicles):
        if i == seed_idx:
            continue
        score = 0
        if veh.fact_id == seed_veh.fact_id:
            score += 3
        if veh.dealer_id == seed_veh.dealer_id:
            score += 2
        if veh.day == seed_veh.day:
            score += 1
        relatedness.append((i, score))
    relatedness.sort(key=lambda x: x[1], reverse=True)
    num_remove = min(int(len(state.vehicles) * degree), len(relatedness))
    remove_indices = [idx for idx, _ in relatedness[:num_remove]]
    remove_indices.append(seed_idx)
    remove_indices.sort(reverse=True)
    for idx in remove_indices:
        if 0 <= idx < len(state.vehicles):
            state.vehicles.pop(idx)
    state.compute_inventory()
    print(f"[OPLOG] shaw_removal: {time.time() - t0:.4f}s")
    return state

class ShawRemovalOperator:
    """
    带参数的Shaw移除算子包装类
    """
    def __init__(self, degree: float = 0.3):
        self.degree = degree
        self.__name__ = "shaw_removal"

    def __call__(self, current: SolutionState, rng: rnd.Generator):
        return shaw_removal(current, rng, self.degree)
    

def periodic_shaw_removal(current: SolutionState, rng: rnd.Generator, degree: float=0.3, 
                          alpha: float=0.4, beta: float=0.3, gamma: float=0.3,
                          k_clusters: int=3) -> SolutionState:
    """
    周期性Shaw移除算子: 
    是一种用于优化问题, 如车辆路径或库存分配的“周期性Shaw移除算子”。
    其主要思想是: 通过聚类分析, 将具有相似特征（如同一周期、需求、库存转移等）的分配项分为若干簇, 
    然后在某个簇内根据相似度指标, 移除一部分相关性高的分配项, 从而为后续的重构或优化创造空间。
    这种方法有助于跳出局部最优, 提高整体解的多样性和质量。
    - current: 当前解状态
    - rng: 随机数生成器
    - degree: 移除比例
    - alpha: 相似度的参数
    - beta: 相似度的参数
    - gamma: 相似度的参数
    - k_clusters: 聚类的数量
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 periodic_shaw_removal 算子")
    state = current.copy()
    data = state.data
    
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] periodic_shaw_removal: {time.time() - t0:.4f}s")
        return state
    
    # step 1: 预先计算所有分配元素的特征向量
    allocations = []  # list of (veh, sku_id, day, feature_vector)
    for veh in state.vehicles:
        plant = veh.fact_id
        dealer = veh.dealer_id
        day = veh.day
        demand_vector = np.array([data.demands.get((dealer, sku), 0) for sku in data.all_skus])
        for (sku, d), qty in list(veh.cargo.items()):
            if d != day:
                continue
            inv_transfer = state.s_ikt.get((plant, sku, day - 1), 0)  # Only considers stock carried over from the previous period. It ignores current-day production, so the feature is backward-looking (historical supply).
            feature = np.array([day, demand_vector.mean(), inv_transfer])
            allocations.append((veh, sku, day, feature))
    
    if not allocations:
        return state
    
    # step 2: 使用K-means聚类分组周期相关簇（优化版本）
    features = np.array([a[3] for a in allocations])  # 使用列表推导式而不是生成器
    
    # 数据预处理：标准化特征以提高聚类效果
    features_std = np.std(features, axis=0)
    features_std[features_std == 0] = 1  # 避免除零
    features_normalized = (features - np.mean(features, axis=0)) / features_std
    
    try:
        # 优化的 KMeans 配置：减少内存使用和提高稳定性
        optimal_clusters = min(k_clusters, len(allocations), max(1, len(allocations) // 3))
        
        kmeans = KMeans(
            n_clusters=optimal_clusters,
            random_state=rng.integers(0, 1000),
            n_init=3,  # 减少初始化次数以降低内存使用
            max_iter=50,  # 限制迭代次数
            tol=1e-3,  # 较宽松的收敛容忍度
            algorithm='lloyd'  # 使用传统算法避免某些内存问题
        )
        
        labels = kmeans.fit_predict(features_normalized)
        
        # 主动清理 KMeans 对象以释放内存
        del kmeans
        
    except (ValueError, MemoryError) as e:
        print(f"[OPLOG] periodic_shaw_removal: KMeans聚类失败 ({type(e).__name__}), 不执行移除")
        return state
    
    # step 3: 选择种子簇, 计算相似度并移除
    unique_labels = np.unique(labels)
    if len(unique_labels) == 0:
        print(f"[OPLOG] periodic_shaw_removal: 无有效聚类标签, 不执行移除")
        return state
    seed_label = rng.choice(unique_labels)
    cluster_indices = np.where(labels == seed_label)[0]
    cluster_allocs = [allocations[i] for i in cluster_indices]
    
    if len(cluster_allocs) <= 1:
        print(f"[OPLOG] periodic_shaw_removal: 聚类内元素过少, 不执行移除")
        return state
    
    similarities = []
    seed_feature = cluster_allocs[0][3]
    for alloc in cluster_allocs[1:]:  # 遍历每个簇内除了种子簇之外的所有元素
        feat = alloc[3]
        time_diff = abs(feat[0] - seed_feature[0])  # 周期差异
        # 需求余弦相似度, 避免除零
        # feat[1:] 是特征向量的后两部分 [demand_vector.mean(), inv_transfer]，表示需求平均值和库存转移
        # demand_cos 越接近1表示需求越相似, 1 - demand_cos 越接近0表示需求越相似
        demand_cos = np.dot(feat[1:], seed_feature[1:]) / (np.linalg.norm(feat[1:]) * np.linalg.norm(seed_feature[1:]) + 1e-8)
        # 库存差异
        inv_diff = abs(feat[2] - seed_feature[2])
        sim = alpha * time_diff + beta * (1 - demand_cos) + gamma * inv_diff
        noise = rng.uniform(-0.1, 0.1) * sim  # 添加少量噪声
        sim += noise
        similarities.append((alloc, sim))
        
    # step 4: 排序并移除相关性高 (sim低) 的元素
    similarities.sort(key=lambda x: x[1])  # 升序
    num_remove = int(len(cluster_allocs) * degree)
    for i in range(min(num_remove, len(similarities))):
        veh, sku_id, day, _ = similarities[i][0]
        qty = veh.cargo.pop((sku_id, day), 0)
        sku_size = data.sku_sizes[sku_id]
        veh.capacity += sku_size * qty
        if veh.is_empty() and veh in state.vehicles:
            state.vehicles.remove(veh)
    
    state.compute_inventory()
    print(f"[OPLOG] periodic_shaw_removal: {time.time() - t0:.4f}s")
    return state

class PeriodicShawRemovalOperator:
    """
    带参数的周期性Shaw移除算子包装类
    """
    def __init__(self, degree: float = 0.3, alpha: float=0.4, beta: float=0.3, gamma: float=0.3, k_clusters: int=3):
        self.degree = degree
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k_clusters = k_clusters
        self.__name__ = "periodic_shaw_removal"

    def __call__(self, current: SolutionState, rng: rnd.Generator):
        return periodic_shaw_removal(current, rng, self.degree, self.alpha, self.beta, self.gamma, self.k_clusters)
        
# 路径移除算子
def path_removal(current: SolutionState, rng: rnd.Generator):
    """路径移除算子: 移除特定生产基地-经销商对上的车辆"""
    t0 = time.time()
    print(f"[OPLOG] 开始执行 path_removal 算子")
    state = current.copy()
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] path_removal: {time.time() - t0:.4f}s")
        return state
    paths = set((veh.fact_id, veh.dealer_id) for veh in state.vehicles)
    if not paths:
        print(f"[OPLOG] path_removal: {time.time() - t0:.4f}s")
        return state
    target_path = rng.choice(list(paths))
    target_path = tuple(target_path)
    path_vehicles = []
    for i, veh in enumerate(state.vehicles):
        if (veh.fact_id, veh.dealer_id) == target_path:
            path_vehicles.append(i)
    if path_vehicles and len(path_vehicles) > 1:
        num_remove = rng.integers(1, len(path_vehicles))
        remove_indices = rng.choice(path_vehicles, size=num_remove, replace=False)
        remove_indices = sorted(remove_indices, reverse=True)
        for idx in remove_indices:
            if 0 <= idx < len(state.vehicles):
                state.vehicles.pop(idx)
    state.compute_inventory()
    print(f"[OPLOG] path_removal: {time.time() - t0:.4f}s")
    return state
    
    
# 局部搜索修复
def local_search_repair(partial: SolutionState, rng: rnd.Generator, max_iter: int = 10, time_limit: float = 5.0):
    """使用局部搜索改进解"""
    t0 = time.time()
    print(f"[OPLOG] 开始执行 local_search_repair 算子")
    # 先基于贪心修复得到一个稳定解的副本
    state = partial.copy()
    state = greedy_repair(state, rng)
    data = state.data
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter and (time.time() - t0) < time_limit:
        iter_count += 1
        improved = False
        old_obj = state.objective()
        for i, veh in enumerate(list(state.vehicles)):
            current_load = state.compute_veh_load(veh)
            capacity = data.veh_type_cap[veh.type]
            for veh_type in state.data.all_veh_types:
                condtion1 = data.veh_type_cap[veh_type] >= current_load
                condtion2 = current_load >= data.veh_type_min_load[veh_type]
                condtion3 = data.veh_type_cap[veh_type] < capacity
                if condtion1 and condtion2 and condtion3:
                    new_veh = Vehicle(veh.fact_id, veh.dealer_id, veh_type, veh.day, data)
                    new_veh.cargo = veh.cargo.copy()
                    old_veh = state.vehicles[i]
                    state.vehicles[i] = new_veh
                    new_obj = state.objective()
                    if new_obj < old_obj - 1e-12:
                        improved = True
                        old_obj = new_obj
                    else:
                        state.vehicles[i] = old_veh
        # 若无改进则退出
    print(f"[OPLOG] local_search_repair: {time.time() - t0:.4f}s (iters={iter_count})")
    return state
    
    
def greedy_repair(partial: SolutionState, rng: rnd.Generator):
    """
    贪心修复算子: 根据需求优先级, 依次遍历所有生产基地和周期, 根据最新库存水平计算实际需要装载的SKU数量
    将其装载到车辆上, 直到满足所有需求, 是批量修复
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 greedy_repair 算子")
    state = partial
    data = state.data
    shipped = state.compute_shipped()
    unsatisfied = []
    supply_chain = data.construct_supply_chain()
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if demand > 0 and shipped_qty < demand:
            unmet = demand - shipped_qty
            demand_ratio = unmet / demand
            total_available = 0
            # 统一使用 supply_chain dict 的 items() 访问模式
            for (plant, dealer_key), skus in supply_chain.items():
                if dealer_key == dealer and sku_id in skus:
                    for day in range(1, data.horizons + 1):
                        total_available += state.s_ikt.get((plant, sku_id, day - 1), 0) + data.sku_prod_each_day.get((plant, sku_id, day), 0)
            stock_urgency = 1.0 if total_available == 0 else min(1.0, unmet / total_available)
            priority = 0.8 * demand_ratio + 0.2 * stock_urgency
            unsatisfied.append(((dealer, sku_id), unmet, priority))
    unsatisfied.sort(key=lambda x: x[2], reverse=True)
    for (dealer, sku_id), remain_demand, _ in unsatisfied:
        if remain_demand <= 0:
            continue
        supply_chain = data.construct_supply_chain()
        plants = [plant for (plant, dealer_key), skus in supply_chain.items() if dealer_key == dealer and sku_id in skus]
        for plant in plants:
            for day in range(1, data.horizons + 1):
                state.compute_inventory()
                available = state.s_ikt.get((plant, sku_id, day - 1), 0) + data.sku_prod_each_day.get((plant, sku_id, day), 0)
                if available <= 0 or remain_demand <= 0:
                    continue
                veh_types = sorted(list(data.all_veh_types), key=lambda x: data.veh_type_cap[x], reverse=True)
                for veh_type in veh_types:
                    cap = data.veh_type_cap[veh_type]
                    max_qty = int(cap // data.sku_sizes[sku_id]) if data.sku_sizes[sku_id] > 0 else 0
                    load_qty = min(remain_demand, available, max_qty)
                    if load_qty <= 0:
                        continue
                    vehicle = Vehicle(plant, dealer, veh_type, day, data)
                    orders = {sku_id: load_qty}
                    veh_loading(state, vehicle, orders)
                    remain_demand -= load_qty
                    state.compute_inventory()
                    if remain_demand <= 0:
                        break
                if remain_demand <= 0:
                    break
            if remain_demand <= 0:
                break
    state.compute_inventory()
    print(f"[OPLOG] greedy_repair: {time.time() - t0:.4f}s")
    return state

def inventory_balance_repair(partial: SolutionState, rng: rnd.Generator):
    """
    基于库存平衡的修复算子
    平衡各生产基地库存水平, 避免库存积压
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 inventory_balance_repair 算子")
    state = partial
    data = state.data
    shipped = state.compute_shipped()
    unsatisfied = {}
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if shipped_qty < demand and demand > 0:
            unsatisfied[(dealer, sku_id)] = demand - shipped_qty

    # 计算每个 (plant, day) 的总库存
    plant_inventory = {}
    for (plant, sku_id, day), inv in state.s_ikt.items():
        # 只关注 day >=1 的周期库存
        if day >= 1:
            plant_inventory[(plant, day)] = plant_inventory.get((plant, day), 0) + inv

    # 将 (plant,day) 按库存降序排列
    sorted_plant_days = sorted(plant_inventory.items(), key=lambda x: x[1], reverse=True)
    supply_chain = data.construct_supply_chain()
    for (plant, day), _ in sorted_plant_days:
        # 获取该 plant 对应的经销商列表
        dealers = [dealer for (p, dealer), skus in supply_chain.items() if p == plant]
        for dealer in dealers:
            veh_type = rng.choice(list(data.all_veh_types))
            vehicle = Vehicle(plant, dealer, veh_type, day, data)
            orders = {sku_id: qty for (dealer_id, sku_id), qty in unsatisfied.items()
                     if dealer_id == dealer}
            if not orders:
                continue
            value = veh_loading(state, vehicle, orders)
            if value:
                for sku_id in list(orders.keys()):
                    unsatisfied[(dealer, sku_id)] = max(0, unsatisfied[(dealer, sku_id)] - vehicle.cargo.get((sku_id, day), 0))
    print(f"[OPLOG] inventory_balance_repair: {time.time() - t0:.4f}s")
    return state

# urgency_repair 已与 greedy_repair 合并并移除，避免冗余实现

def infeasible_repair(partial: SolutionState, rng: rnd.Generator):
    """
    旧的 infeasible_repair 已弃用，现使用 destroy_operators.infeasible_removal。
    保留该占位函数以兼容调用，但直接调用 infeasible_removal。
    """
    print("[OPLOG] 使用外部 infeasible_removal 修复不可行解")
    return infeasible_removal(partial, rng)

def _construct_resource_availability(state: SolutionState, plant: str, sku_id: str, day: int):
    """返回在指定 (plant,sku,day) 的可用库存数量 (不修改 state)"""
    prev_inv = state.s_ikt.get((plant, sku_id, day - 1), 0)
    production = state.data.sku_prod_each_day.get((plant, sku_id, day), 0)
    used_inv = sum(veh.cargo.get((sku_id, day), 0) for veh in state.vehicles if veh.fact_id == plant and veh.day == day)
    return max(0, prev_inv + production - used_inv)

def get_removal_list(state: SolutionState) -> Dict[Tuple[str, str], int]:
    """
    获取当前解中未满足的需求列表
    
    Returns:
    --------
    Dict[Tuple[str, str], int]
        {(dealer, sku_id): unmet_quantity} 字典
    """
    shipped = state.compute_shipped()
    removal_list = {}
    
    for (dealer, sku_id), total_demand in state.data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        unmet_qty = total_demand - shipped_qty
        
        if unmet_qty > 0:
            removal_list[(dealer, sku_id)] = unmet_qty
    
    return removal_list

def smart_batch_repair(partial: SolutionState, rng: rnd.Generator):
    """
    智能分批修复算法 (Smart Batch Repair Algorithm - SBRA)
    
    核心改进: 
    1. 资源池预计算: 避免重复计算库存和可用性
    2. 分批聚类处理: 进行简单分批处理
    3. 多目标优化: 平衡需求满足率、车辆数量、资源利用率
    4. 智能车型选择: 根据需求量和可用容量动态选择
    5. 自适应超时: 根据问题规模调整时间限制
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 smart_batch_repair 算子")
    
    state = partial
    data = state.data
    
    # 自适应参数设置
    num_demands = len(data.demands)
    timeout = min(60.0, max(10.0, num_demands * 0.1))  # 根据问题规模调整超时
    batch_size = min(20, max(5, num_demands // 10))   # 动态批次大小
    
    print(f"[OPLOG] 问题规模: {num_demands} 需求, 超时: {timeout}s, 批次大小: {batch_size}")
    
    # Step 1: 预计算资源池 (一次性计算, 避免重复)
    resource_pool = _compute_resource_pool(state, data)
    
    # Step 2: 获取未满足需求并分批处理
    iteration = 0
    while iteration < 10 and time.time() - t0 < timeout:
        iteration += 1
        
        # 重新计算当前未满足需求
        unsatisfied_demands = _get_unsatisfied_demands(state, data)
        if not unsatisfied_demands:
            print(f"[OPLOG] 所有需求已满足, 算法提前退出")
            break
            
        print(f"[OPLOG] 迭代 {iteration}: 剩余 {len(unsatisfied_demands)} 个未满足需求")
        
        # Step 3: 智能分批
        demand_batches = _create_smart_batches(unsatisfied_demands, data, batch_size)
        
        batch_progress = False
        for batch_idx, batch in enumerate(demand_batches):
            if time.time() - t0 > timeout:
                break
                
            # Step 4: 批次内优化分配
            batch_result = _process_demand_batch(state, batch, resource_pool, data, rng)
            if batch_result:
                batch_progress = True
                print(f"[OPLOG] 批次 {batch_idx + 1}/{len(demand_batches)} 处理成功")
        
        if not batch_progress:
            print(f"[OPLOG] 无法继续改进, 准备最终强制装载")
            break
    
    # Step 5: 最终强制装载（确保所有需求得到满足）
    if time.time() - t0 < timeout:
        _force_load_remaining_demands(state, data)
    
    state.compute_inventory()
    elapsed = time.time() - t0
    print(f"[OPLOG] Smart Batch Repair 完成: {elapsed:.4f}s")
    return state

def _compute_resource_pool(state: SolutionState, data: DataALNS):
    """预计算所有可用资源, 避免重复计算"""
    state.compute_inventory()
    resource_pool = {}
    
    for plant in data.plants:
        for sku_id in data.skus_plant.get(plant, []):
            for day in range(1, data.horizons + 1):
                prev_inv = state.s_ikt.get((plant, sku_id, day - 1), 0)
                production = data.sku_prod_each_day.get((plant, sku_id, day), 0)
                
                # 计算已被使用的库存
                used_inv = sum(veh.cargo.get((sku_id, day), 0) 
                             for veh in state.vehicles 
                             if veh.fact_id == plant and veh.day == day)
                
                available = max(0, prev_inv + production - used_inv)
                resource_pool[(plant, sku_id, day)] = available
    
    return resource_pool

def _get_unsatisfied_demands(state: SolutionState, data: DataALNS):
    """获取当前未满足的需求列表"""
    shipped = state.compute_shipped()
    unsatisfied = []
    
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if demand > shipped_qty:
            unsatisfied.append({
                'dealer': dealer,
                'sku_id': sku_id,
                'remain_demand': demand - shipped_qty,
                'total_demand': demand,
                'priority': demand - shipped_qty  # 可以根据业务需求调整优先级
            })
    
    return unsatisfied

def _create_smart_batches(unsatisfied_demands, data: DataALNS, batch_size: int):
    """
    智能分批: 简化版本, 根据传入的batch_size参数将需求分批,
    如果需要, 可以扩展为基于地域和SKU特征的聚类, 至于怎么定义地域和SKU特征, 需要重新考虑
    当前简化版本, 能够满足基本的分批需求
    """
    if not unsatisfied_demands:
        return []
    
    # 简化版: 按优先级排序后分批
    sorted_demands = sorted(unsatisfied_demands, key=lambda x: x['priority'], reverse=True)
    
    batches = []
    for i in range(0, len(sorted_demands), batch_size):
        batch = sorted_demands[i:i + batch_size]
        batches.append(batch)
    
    return batches

def _process_demand_batch(state: SolutionState, batch, resource_pool, data: DataALNS, rng: rnd.Generator):
    """处理单个需求批次"""
    batch_progress = False
    
    for demand_info in batch:
        dealer = demand_info['dealer']
        sku_id = demand_info['sku_id']
        remain_demand = demand_info['remain_demand']
        
        if remain_demand <= 0:
            continue
            
        # 获取可供应的工厂列表
        supply_chain = data.construct_supply_chain()
        available_plants = [plant for (plant, dealer_key), skus in supply_chain.items() 
                          if dealer_key == dealer and sku_id in skus]
        
        if not available_plants:
            continue
            
        # 寻找最优的 (工厂, 天数) 组合
        best_allocation = _find_best_allocation(
            remain_demand, sku_id, available_plants, resource_pool, data
        )
        
        if best_allocation:
            plant, day, available_qty = best_allocation
            
            # 选择最合适的车型
            optimal_veh_type = _select_optimal_vehicle_type(
                remain_demand, sku_id, data
            )
            
            # 计算实际装载量
            load_qty = min(remain_demand, available_qty)
            if load_qty > 0:
                vehicle = Vehicle(plant, dealer, optimal_veh_type, day, data)
                orders = {sku_id: load_qty}
                
                try:
                    veh_loading(state, vehicle, orders)
                    # 更新资源池
                    resource_pool[(plant, sku_id, day)] -= load_qty
                    batch_progress = True
                except Exception as e:
                    print(f"[OPLOG] 批次装载失败: {e}")
                    continue
    
    return batch_progress

def _find_best_allocation(remain_demand, sku_id, available_plants, resource_pool, data: DataALNS):
    """找到最优的资源分配方案"""
    best_score = -1
    best_allocation = None
    
    for plant in available_plants:
        for day in range(1, data.horizons + 1):
            available_qty = resource_pool.get((plant, sku_id, day), 0)
            if available_qty <= 0:
                continue
                
            # 计算分配得分（可满足量 + 资源利用效率）
            satisfaction_ratio = min(1.0, available_qty / remain_demand)
            efficiency_score = available_qty / max(1, available_qty + remain_demand)
            score = satisfaction_ratio * 0.7 + efficiency_score * 0.3
            
            if score > best_score:
                best_score = score
                best_allocation = (plant, day, available_qty)
    
    return best_allocation

def _select_optimal_vehicle_type(demand_qty, sku_id, data: DataALNS):
    """根据需求量智能选择车型"""
    required_volume = demand_qty * data.sku_sizes[sku_id]
    
    # 找到能满足需求的最小车型
    suitable_types = [(vt, cap) for vt, cap in data.veh_type_cap.items() 
                     if cap >= required_volume]
    
    if suitable_types:
        # 选择容量刚好够用的车型（避免浪费）
        return min(suitable_types, key=lambda x: x[1])[0]
    else:
        # 如果没有车型能完全满足, 选择最大的车型
        return max(data.veh_type_cap.keys(), key=lambda x: data.veh_type_cap[x])

def _force_load_remaining_demands(state: SolutionState, data: DataALNS):
    """最终强制装载, 确保所有需求得到满足"""
    print(f"[OPLOG] 开始最终强制装载")
    
    shipped = state.compute_shipped()
    force_loaded = 0
    
    for (dealer, sku_id), demand in data.demands.items():
        remain_demand = demand - shipped.get((dealer, sku_id), 0)
        if remain_demand <= 0:
            continue
            
        # 寻找任意可用工厂
        supply_chain = data.construct_supply_chain()
        available_plants = [plant for (plant, dealer_key), skus in supply_chain.items() 
                          if dealer_key == dealer and sku_id in skus]
        
        if not available_plants:
            print(f"[OPLOG] 警告: 无法为 {dealer}-{sku_id} 找到供应工厂")
            continue
            
        plant = available_plants[0]  # 使用第一个可用工厂
        day = 1  # 使用第一天
        
        # 强制装载, 忽略库存限制
        veh_type = min(data.veh_type_cap.keys(), key=lambda x: data.veh_type_cap[x])
        vehicle = Vehicle(plant, dealer, veh_type, day, data)
        orders = {sku_id: remain_demand}
        
        try:
            veh_loading(state, vehicle, orders)
            force_loaded += 1
        except Exception as e:
            print(f"[OPLOG] 强制装载失败 {dealer}-{sku_id}: {e}")
    
    print(f"[OPLOG] 强制装载完成, 处理 {force_loaded} 个需求")

def regret_based_repair(partial: SolutionState, rng: rnd.Generator, k: int = 2, topN: int = 6, time_limit: float = 10.0):
    """
    后悔值修复算子（regret-k style）
    核心思想：对于每个未满足需求，评估若干候选插入位置的改进值，
    计算 top1 与 topk 之间的差值（regret），优先修复后悔值最大的需求。
    实现要点：
    - 在临时副本（state.copy()）上模拟插入以评估改进，不修改主 state，直到确认 commit。
    - 限制每个需求评估的候选数 topN 以控制开销。
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 regret_based_repair 算子")
    state = partial
    data = state.data
    start = time.time()

    while time.time() - start < time_limit:
        removal = get_removal_list(state)
        if not removal:
            break
        prev_obj = state.objective()
        regret_list = []

        # 遍历每个未满足需求，评估若干候选
        for (dealer, sku_id), remain_qty in list(removal.items()):
            candidates_scores = []
            # 获取可用工厂
            supply_chain = data.construct_supply_chain()
            available_plants = [plant for (plant, dealer_key), skus in supply_chain.items() if dealer_key == dealer and sku_id in skus]
            # 限制候选数
            for plant in available_plants:
                for day in range(1, data.horizons + 1):
                    for veh_type in data.all_veh_types:
                        # 估算该候选最大可装量
                        available = _construct_resource_availability(state, plant, sku_id, day)
                        if available <= 0:
                            continue
                        cap = data.veh_type_cap[veh_type]
                        max_qty_by_cap = int(cap // data.sku_sizes[sku_id]) if data.sku_sizes[sku_id] > 0 else 0
                        est_qty = min(remain_qty, available, max_qty_by_cap)
                        if est_qty <= 0:
                            continue
                        # 在临时副本上模拟插入并评估改进
                        tmp = state.copy()
                        veh = Vehicle(plant, dealer, veh_type, day, data)
                        veh_loading(tmp, veh, {sku_id: est_qty})
                        new_obj = tmp.objective()
                        improvement = prev_obj - new_obj
                        candidates_scores.append((improvement, (plant, veh_type, day, est_qty)))
                        # 限制总候选数 per demand
                        if len(candidates_scores) >= topN:
                            break
                    if len(candidates_scores) >= topN:
                        break
                if len(candidates_scores) >= topN:
                    break

            if not candidates_scores:
                continue
            candidates_scores.sort(key=lambda x: x[0], reverse=True)
            top1 = candidates_scores[0][0]
            topk = candidates_scores[min(k-1, len(candidates_scores)-1)][0]
            regret = top1 - topk
            regret_list.append(((dealer, sku_id), regret, candidates_scores[0][1]))  # 保存 top1 的候选方案

        if not regret_list:
            break

        # 选择具有最大 regret 的需求并在主 state 上应用 top1 插入
        regret_list.sort(key=lambda x: x[1], reverse=True)
        (dealer_sel, sku_sel), _, best_candidate = regret_list[0]
        plant_sel, veh_type_sel, day_sel, qty_sel = best_candidate

        # 在主 state 上执行插入
        vehicle = Vehicle(plant_sel, dealer_sel, veh_type_sel, day_sel, data)
        orders = {sku_sel: qty_sel}
        veh_loading(state, vehicle, orders)
        # veh_loading 已在 commit 模式下将 vehicle append 到 state.vehicles 并更新库存

    state.compute_inventory()
    elapsed = time.time() - t0
    print(f"[OPLOG] regret_based_repair 完成: {elapsed:.4f}s")
    return state

class RegretBasedRepairOperator:
    def __init__(self, k: int = 2, topN: int = 6, time_limit: float = 10.0):
        self.k = k
        self.topN = topN
        self.time_limit = time_limit
        self.__name__ = "regret_based_repair"

    def __call__(self, current: SolutionState, rng: rnd.Generator):
        return regret_based_repair(current, rng, self.k, self.topN, self.time_limit)

# 以下为原始文件中与 learning_based_repair 关联的内容已移除
# 移除：_construct_training_data, _call_greedy_repair, _random_select_repair_operator, _compute_feature,
# learning_based_repair 以及 LearningBasedRepairOperator
# 这些逻辑已迁移到外部 ML operator selector (ALNSCode/ml_operator_selector.py) 中

def _compute_feature(state: SolutionState) -> List[float]:
    """
    当需要计算当前解的特征向量时使用的实用函数（保留用于外部 tracker）
    特征向量: [avg_demand, avg_sku_size, avg_day, avg_inventory, avg_capacity_util]
    """
    data = state.data
    
    # 获取未满足需求
    removal_list = get_removal_list(state)
    
    if not removal_list:
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    
    # 初始化累加器
    total_demand = 0.0
    total_weighted_size = 0.0
    total_quantity = 0
    inventory_values = []
    
    # 对每个未满足需求，计算对特征的贡献
    for (dealer, sku_id), unmet_qty in removal_list.items():
        demand = data.demands.get((dealer, sku_id), 0)
        sku_size = data.sku_sizes[sku_id]

        total_demand += demand
        total_weighted_size += sku_size * unmet_qty
        total_quantity += unmet_qty
        
        # 获取相关工厂/SKU组合的库存水平
        supply_chain = state.data.construct_supply_chain()
        available_plants = [plant for (plant, dealer_key), skus in supply_chain.items() 
                           if dealer_key == dealer and sku_id in skus]
        
        for plant in available_plants:
            for day in range(1, data.horizons + 1):
                inv = state.s_ikt.get((plant, sku_id, day-1), 0)
                inventory_values.append(inv)
    
    # 计算平均特征
    num_demands = len(removal_list)
    avg_demand = total_demand / num_demands if num_demands > 0 else 0.0
    
    avg_size = total_weighted_size / total_quantity if total_quantity > 0 else 0.0
    
    periods = list(range(1, data.horizons + 1))
    # avg_day 是数组periods中位于中间位置的元素
    avg_day = float(periods[len(periods) // 2]) if periods else 1.0
    
    avg_inventory = np.mean(inventory_values) if inventory_values else 0.0
    
    # 计算当前解中所有车辆的平均容量利用率
    capacity_utils = []
    for veh in state.vehicles:
        used_capacity = state.compute_veh_load(veh)
        total_capacity = data.veh_type_cap[veh.type]
        if total_capacity > 0:
            capacity_util = used_capacity / total_capacity
            capacity_utils.append(capacity_util)

    avg_capacity_util = np.mean(capacity_utils) if capacity_utils else 0.0
    
    feature = [float(avg_demand), float(avg_size), float(avg_day), float(avg_inventory), float(avg_capacity_util)]
    
    print(f"[OPLOG] 计算的实际特征: {feature}")
    return feature

# 其余辅助函数保持不变（如 _compute_resource_pool, _find_best_allocation 等），在上方已保留
