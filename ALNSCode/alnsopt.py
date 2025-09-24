# 标准库导入
import copy
import os
import random
import time
import math
import warnings
from copy import deepcopy
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Tuple, Set, Optional
from .alnstrack import ALNSTracker
from .vehicle import Vehicle


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

from .InputDataALNS import DataALNS
from .vehicle import Vehicle
from .optutility import POSITIVEEPS
# Import initial solution utilities
from .initial_solution_utils import improved_initial_solution

@dataclass
class SolutionState:
    data: DataALNS
    vehicles: List[Vehicle] = field(default_factory=list)
    s_ikt: Dict[Tuple[str, str, int], int] = field(default_factory=dict)
    s_indices: Set[Tuple[str, str, int]] = field(default_factory=set)
    tracker: Optional['ALNSTracker'] = None  # 引入追踪器, 便于访问迭代历史数据,用于ML训练
    s_initialized: bool = field(default=False)

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
        验证解是否为可行解, 无副作用。
        返回 (is_feasible, violations) 元组: 
        - is_feasible: bool, 是否可行
        - violations: dict, 包含各类违反项的详细信息
        """
        self.compute_inventory()
        violations = {
            'negative_inventory': [],
            'veh_over_load': [],
            'plant_inv_exceed': []
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
                violations['veh_over_load'].append({
                    'veh': veh,
                    'loaded': loaded,
                    'cap': cap
                })
        
        # 检查在每个周期内, 生产基地中的库存是否超过限制
        # 计算 {(plant, day): total_inventory}
        plant_day_inventory = defaultdict(int)
        for (plant, sku, day), inv in self.s_ikt.items():
            plant_day_inventory[(plant, day)] += inv
        
        for (plant, day), total_inv in plant_day_inventory.items():
            max_cap = self.data.plant_inv_limit[plant]
            if total_inv - max_cap > POSITIVEEPS:
                violations['plant_inv_exceed'].append({
                    'plant': plant,
                    'day': day,
                    'total_inv': total_inv,
                    'max_cap': max_cap
                })
        
        # 获得的解不必满足所有需求, 但最终会进行统一修正, 意识到这一点很关键
        # 原因如下:
        #     1. ALNS的目标是探索解空间, 通过移除和修复算子引入多样性, 在迭代过程中，破坏算子可能会移除满足需求的车辆
        #     2. 目标函数中已经包含了未满足需求的惩罚项, 因此不需要在validate中强制要求满足所有需求
        #     3. 通过允许部分需求未满足, ALNS可以更灵活地探索解空间, 寻找潜在的更优解
        #     4. 最终的修正步骤确保了解的可行性, 但在ALNS迭代过程中, 允许部分需求未满足有助于算法跳出局部最优
        #     5. 强制满足所有经销商的需求可能会限制解空间, 导致算法过早收敛, 且会增加计算复杂度
        #     6. 如果在初始解生产算法中也要求满足所有需求, 看似合理, 但实际上会大大增加初始解生成的难度, 影响算法效率
        #     7. 通过在目标函数中添加未满足需求的惩罚项, 可以引导ALNS逐步改进解, 最终达到满足需求的目标
        # 因此, 在validate中不检查需求满足情况, 只关注库存和车辆容量等硬约束
        
        # 综合判断解的可行性
        is_feasible = not (violations['negative_inventory'] or violations['veh_over_load'] or violations['plant_inv_exceed'])
        
        return is_feasible, violations
    

    def objective(self):
        """这个目标函数的作用是引导搜索过程, 并不是当前解的实际成本"""
        # 添加缩放因子以控制目标函数值的尺度
        # 避免目标函数值过大, 影响模拟退火中计算接收概率时, 出现数值溢出
        scale_factor = 1e-3  # 缩放因子, 可以根据实际情况调整
        
    # NOTE: validate() will call compute_inventory(); avoid double full recompute here
    # 检查解的可行性, 已经在validate中验证了解是否满足以下约束:
        # 1. 库存非负
        # 2. 车辆容量不超限
        # 3. 在每个周期内, 生产基地中的库存不超过限制
        # 所以, 在目标函数中不需要再对其进行惩罚, 只需要检查以下两个约束:
        # 1. 是否满足经销商的需求
        # 2. 是否满足车辆的最小起运量
        
        feasibility, violations = self.validate()
        if not feasibility:
            return float('inf')  # 返回一个非常大的值, 表示解不可行
        
        total_cost = sum(self.data.veh_type_cost[veh.type] for veh in self.vehicles)
      
        total_cost = self.punish_non_fulfill_demand(total_cost)
        
        # total_cost = self.punish_exceeded_inventory_limit(total_cost)
        
        total_cost = self.punish_deficient_veh_min_load(total_cost)
        
        # total_cost = self.punish_negative_inventory(total_cost)
        
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
        # 预计算每个 (plant, sku, day) 的发运量，避免在每个索引上扫描车辆列表
        shipped_by_plant_sku_day = defaultdict(int)
        for veh in self.vehicles:
            fact = veh.fact_id
            d = veh.day
            for (sku_id, day_k), q in veh.cargo.items():
                # 只计当日发运
                if day_k == d:
                    shipped_by_plant_sku_day[(fact, sku_id, day_k)] += q

        for (plant, sku_id, day) in self.s_indices:
            if day > 0:  # 只处理day>0的情况, 因为day=0是期初库存, 不应该被修改
                # 直接从预计算表查询当日发运量
                shipped_from_plant = shipped_by_plant_sku_day.get((plant, sku_id, day), 0)

                # 获取前一天的库存, 确保期初库存被正确考虑
                prev_inventory = self.s_ikt.get((plant, sku_id, day - 1), 0)

                # 获取当天的生产量
                production = self.data.sku_prod_each_day.get((plant, sku_id, day), 0)

                # 计算当前库存: 前一天库存 + 当天生产 - 当天发出
                current_inventory = prev_inventory + production - shipped_from_plant

                # 更新库存
                self.s_ikt[plant, sku_id, day] = current_inventory

        # 完成全表重算后标记 s_ikt 已初始化 (future periods 已计算完)
        self.s_initialized = True
    
    def copy(self):
        """复制当前解: 确保每个解都是不可变的"""
        new_state = SolutionState(self.data)
        new_state.vehicles = copy.deepcopy(self.vehicles)
        new_state.s_ikt = copy.deepcopy(self.s_ikt)
        new_state.s_indices = copy.deepcopy(self.s_indices)
        new_state.tracker = self.tracker  # 共享同一个tracker引用
        # 保持 s_initialized 标志的一致性
        new_state.s_initialized = bool(self.s_initialized)
        return new_state


def veh_loading(state: SolutionState, veh: Vehicle, orders: Dict[str, int]):
    """
    车辆装载函数, 考虑车辆容量约束和可用库存, 
    根据车辆可用容量和订单数量, 自动分配多辆车辆
    """
    data = state.data
    fact_id, dealer_id, veh_type, day = veh.fact_id, veh.dealer_id, veh.type, veh.day

    # 统计同基地同天所有车辆已装载的SKU数量
    used_inv = {}
    vehicles_list = state.vehicles
    vehicle_ids = {v.id for v in vehicles_list}
    for vehicle in vehicles_list:
        if vehicle.fact_id == fact_id and vehicle.day == day:
            for (sku_id, d), qty in vehicle.cargo.items():
                if d == day:
                    used_inv[sku_id] = used_inv.get(sku_id, 0) + qty
    
    # prepare locals for hot loop
    s_ikt = state.s_ikt
    sku_sizes = data.sku_sizes
    prod_get = data.sku_prod_each_day.get
    horizons = data.horizons

    for sku_id, order_qty in orders.items():
        if sku_id not in data.skus_plant.get(fact_id, []):
            continue
        remain_qty = order_qty

        while remain_qty > 0:
            # 计算可用库存
            prev_inv = s_ikt.get((fact_id, sku_id, day-1), 0)
            production = prod_get((fact_id, sku_id, day), 0)
            used_qty = used_inv.get(sku_id, 0)
            available = prev_inv + production - used_qty
            if available <= 0:
                break

            # 直接使用 veh.capacity 作为剩余可用体积，避免重复计算 current_load
            sku_size = sku_sizes[sku_id]
            max_qty_by_cap = veh.capacity // sku_size
            load_qty = min(remain_qty, available, max_qty_by_cap)

            if load_qty <= 0:
                # 当前车辆已满, 加入队列, 换新车
                if not veh.is_empty():
                    if veh.id not in vehicle_ids:
                        state.vehicles.append(veh)
                        vehicle_ids.add(veh.id)
                veh = Vehicle(fact_id, dealer_id, veh_type, day, data)
                continue

            veh.load(sku_id, load_qty)
            used_inv[sku_id] = used_inv.get(sku_id, 0) + load_qty
            remain_qty -= load_qty

            # 确保 s_ikt 已初始化
            if not state.s_initialized:
                state.compute_inventory()

            # 将该次装载量从装运当天及之后的库存中扣减
            for d in range(day, horizons + 1):
                key = (fact_id, sku_id, d)
                s_ikt[key] = s_ikt.get(key, 0) - load_qty
        
        # 最后一次循环后, 若veh有货且未加入, 则加入（用 id 集合避免昂贵的 eq 比较）
        if not veh.is_empty() and veh.id not in vehicle_ids:
            state.vehicles.append(veh)
    
    # 增量更新已在装载时完成，不再进行全表 recompute 以减少开销
    
    return True


def initial_solution(state: SolutionState, rng: rnd.Generator):
    solution = improved_initial_solution(state, rng)
    return solution


def random_removal(current: SolutionState, rng: rnd.Generator, degree: float = 0.25):
    # Randomly remove a specified proportion of vehicles from the solution to introduce randomness and diversity.

    t0 = time.time()
    print(f"[OPLOG] 开始执行 random_removal 算子")
    state = current.copy()
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] random_removal: {time.time() - t0:.4f}s")
        return state
    num_remove = int(len(state.vehicles) * degree)
    remove_indices = rng.choice(range(len(state.vehicles)), num_remove, replace=False)
    new_vehicles = [veh for i, veh in enumerate(state.vehicles) if i not in remove_indices]
    # Identify removed vehicles and incrementally restore their shipped quantities to s_ikt
    removed = [veh for i, veh in enumerate(state.vehicles) if i in remove_indices]
    state.vehicles = new_vehicles
    # Ensure s_ikt is initialized (use flag to avoid expensive key scans)
    if not state.s_initialized:
        state.compute_inventory()
    for veh in removed:
        for (sku_id, d_shipped), qty in veh.cargo.items():
            # add back the shipped qty to inventory for shipped day and afterwards
            for d in range(d_shipped, state.data.horizons + 1):
                key = (veh.fact_id, sku_id, d)
                state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
    print(f"[OPLOG] random_removal: {time.time() - t0:.4f}s")
    return state

class RandomRemovalOperator:
    def __init__(self, degree: float = 0.25):
        self.degree = degree
        self.__name__ = "random_removal"
    def __call__(self, current: SolutionState, rng: rnd.Generator):
        return random_removal(current, rng, self.degree)


def worst_removal(current: SolutionState, rng: rnd.Generator):
    # Remove the q vehicles with the largest remaining volume to avoid vehicle space waste.
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
            # pop and recover inventory incrementally
            veh = state.vehicles.pop(idx)
            if not state.s_initialized:
                state.compute_inventory()
            for (sku_id, d_shipped), qty in veh.cargo.items():
                for d in range(d_shipped, state.data.horizons + 1):
                    key = (veh.fact_id, sku_id, d)
                    state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
    # inventory updated incrementally for removed vehicles; no full recompute
    print(f"[OPLOG] worst_removal: {time.time() - t0:.4f}s")
    return state

def infeasible_removal(current: SolutionState, rng: rnd.Generator):
    # Remove vehicles corresponding to SKUs with negative inventory at the end of the current period to ensure feasibility.
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
    # collect vehicles to remove
    removal_vehicles = []
    for veh in state.vehicles:
        for (plant, sku_id, day) in neg_inv:
            if veh.fact_id == plant and veh.day == day:
                removal_vehicles.append(veh)
                break

    if removal_vehicles:
        # ensure inventory baseline is initialized once
        if not state.s_initialized:
            state.compute_inventory()

        # remove vehicles and incrementally restore their shipped quantities
        state.vehicles = [veh for veh in state.vehicles if veh not in removal_vehicles]
        for veh in removal_vehicles:
            for (sku_id, d_shipped), qty in veh.cargo.items():
                for d in range(d_shipped, state.data.horizons + 1):
                    key = (veh.fact_id, sku_id, d)
                    state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
    print(f"[OPLOG] infeasible_removal: {time.time() - t0:.4f}s")
    return state
    

def surplus_inventory_removal(current: SolutionState, rng: rnd.Generator):
    # Remove q vehicles corresponding to SKUs with the highest remaining inventory at the end of the current period to reduce inventory risk.
    t0 = time.time()
    print(f"[OPLOG] 开始执行 surplus_inventory_removal 算子")
    state = current.copy()
    data = state.data
    if len(state.vehicles) <= 1:
        print(f"[OPLOG] surplus_inventory_removal: {time.time() - t0:.4f}s")
        return state
    plant_max_inv = {}
    for (plant, sku_id, day), inv in state.s_ikt.items():
        if plant not in plant_max_inv or inv > plant_max_inv[plant][1]:
            plant_max_inv[plant] = (sku_id, inv)
    highest_inv = {(plant, sku_info[0], day): sku_info[1] 
                  for plant, sku_info in plant_max_inv.items() 
                  for day in range(1, data.horizons + 1)}
    removal_candidates = set()
    for i, veh in enumerate(state.vehicles):
        for (plant, sku_id, day) in highest_inv:
            if veh.fact_id == plant and veh.day == day:
                removal_candidates.add(i)
    removal_candidates = sorted(list(removal_candidates), reverse=True)
    if removal_candidates:
        ub = max(1, len(removal_candidates) // 2)
        num_remove = random.randint(1, ub)
        selected_indices = rng.choice(removal_candidates, size=num_remove, replace=False)
        selected_indices = sorted(selected_indices, reverse=True)
        # collect removed vehicles then remove and restore inventory incrementally
        removed = []
        for idx in selected_indices:
            if 0 <= idx < len(state.vehicles):
                removed.append(state.vehicles[idx])
        for idx in selected_indices:
            if 0 <= idx < len(state.vehicles):
                state.vehicles.pop(idx)
        if not state.s_initialized:
            state.compute_inventory()
        for veh in removed:
            for (sku_id, d_shipped), qty in veh.cargo.items():
                for d in range(d_shipped, state.data.horizons + 1):
                    key = (veh.fact_id, sku_id, d)
                    state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
    print(f"[OPLOG] surplus_inventory_removal: {time.time() - t0:.4f}s")
    return state
    

# 实现Shaw移除算子
def shaw_removal(current: SolutionState, rng: rnd.Generator, degree: float = 0.3):
    # Shaw removal operator: removes related vehicles, degree is the removal ratio.
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
    # collect removed vehicles then remove and restore inventory incrementally
    removed = []
    for idx in remove_indices:
        if 0 <= idx < len(state.vehicles):
            removed.append(state.vehicles[idx])
    for idx in remove_indices:
        if 0 <= idx < len(state.vehicles):
            state.vehicles.pop(idx)
    if not state.s_initialized:
        state.compute_inventory()
    for veh in removed:
        for (sku_id, d_shipped), qty in veh.cargo.items():
            for d in range(d_shipped, state.data.horizons + 1):
                key = (veh.fact_id, sku_id, d)
                state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
    print(f"[OPLOG] shaw_removal: {time.time() - t0:.4f}s")
    return state


class ShawRemovalOperator:
    # Shaw removal operator wrapper class
    def __init__(self, degree: float = 0.3):
        self.degree = degree
        self.__name__ = "shaw_removal"

    def __call__(self, current: SolutionState, rng: rnd.Generator):
        return shaw_removal(current, rng, self.degree)
    

def periodic_shaw_removal(current: SolutionState, rng: rnd.Generator, degree: float=0.3, 
                          alpha: float=0.4, beta: float=0.3, gamma: float=0.3,
                          k_clusters: int=3) -> SolutionState:
    # Periodic Shaw removal operator: For optimization problems such as vehicle routing or inventory allocation, this operator clusters assignments with similar features (e.g., same period, demand, inventory transfer) and removes a proportion of highly related assignments within a cluster to create space for reconstruction or optimization. This helps escape local optima and improves solution diversity and quality.
    # Args:
    #     current: current solution state
    #     rng: random number generator
    #     degree: removal ratio
    #     alpha, beta, gamma: similarity parameters
    #     k_clusters: number of clusters
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
        # restore qty to plant inventory for day and afterwards
        if qty > 0:
            if not state.s_ikt:
                state.compute_inventory()
            for d in range(day, data.horizons + 1):
                key = (veh.fact_id, sku_id, d)
                state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
        if veh.is_empty() and veh in state.vehicles:
            state.vehicles.remove(veh)
    print(f"[OPLOG] periodic_shaw_removal: {time.time() - t0:.4f}s")
    return state

class PeriodicShawRemovalOperator:
    # Periodic Shaw removal operator wrapper class
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
    # Path removal operator: removes vehicles on a specific plant-dealer pair.
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
        # collect removed vehicles then remove and restore inventory incrementally
        removed = []
        for idx in remove_indices:
            if 0 <= idx < len(state.vehicles):
                removed.append(state.vehicles[idx])
        # actually remove
        for idx in remove_indices:
            if 0 <= idx < len(state.vehicles):
                state.vehicles.pop(idx)

        if not state.s_initialized:
            state.compute_inventory()

        for veh in removed:
            for (sku_id, d_shipped), qty in veh.cargo.items():
                for d in range(d_shipped, state.data.horizons + 1):
                    key = (veh.fact_id, sku_id, d)
                    state.s_ikt[key] = state.s_ikt.get(key, 0) + qty
    print(f"[OPLOG] path_removal: {time.time() - t0:.4f}s")
    return state
    
    
# 局部搜索修复
def local_search_repair(partial: SolutionState, rng: rnd.Generator):
    # Improve solution using local search.
    t0 = time.time()
    print(f"[OPLOG] 开始执行 local_search_repair 算子")
    state = greedy_repair(partial, rng)  # 先使用贪心修复获得可行解
    data = state.data
    improved = True
    while improved:
        improved = False
        old_obj = state.objective()
        for i, veh in enumerate(state.vehicles):
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
                    if new_obj < old_obj:
                        improved = True
                    else:
                        state.vehicles[i] = old_veh
    print(f"[OPLOG] local_search_repair: {time.time() - t0:.4f}s")
    return state
    
    
def greedy_repair(partial: SolutionState, rng: rnd.Generator):
    # Greedy repair operator: Traverse all plants and periods by demand priority, calculate the actual SKU quantity to be loaded based on the latest inventory, and load it onto vehicles until all demands are met. Batch repair.
    t0 = time.time()
    print(f"[OPLOG] 开始执行 greedy_repair 算子")
    state = partial
    data = state.data
    shipped = state.compute_shipped()
    unsatisfied = []
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if demand > 0 and shipped_qty < demand:
            unmet = demand - shipped_qty
            demand_ratio = unmet / demand
            total_available = 0
            supply_chain = data.construct_supply_chain()
            for plant in [plant for (plant, dealer_key), skus in supply_chain.items() if dealer_key == dealer and sku_id in skus]:
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
        # Ensure s_ikt is initialized once before attempting greedy loading
        if not state.s_ikt:
            state.compute_inventory()
        for plant in plants:
            for day in range(1, data.horizons + 1):
                # available is determined from previous day's inventory + production
                available = state.s_ikt.get((plant, sku_id, day - 1), 0) + data.sku_prod_each_day.get((plant, sku_id, day), 0)
                if available <= 0 or remain_demand <= 0:
                    continue
                veh_types = sorted(list(data.all_veh_types), key=lambda x: data.veh_type_cap[x], reverse=True)
                for veh_type in veh_types:
                    cap = data.veh_type_cap[veh_type]
                    max_qty = int(cap // data.sku_sizes[sku_id])
                    load_qty = min(remain_demand, available, max_qty)
                    if load_qty <= 0:
                        continue
                    vehicle = Vehicle(plant, dealer, veh_type, day, data)
                    orders = {sku_id: load_qty}
                    veh_loading(state, vehicle, orders)
                    remain_demand -= load_qty
                    if remain_demand <= 0:
                        break
                if remain_demand <= 0:
                    break
            if remain_demand <= 0:
                break
    print(f"[OPLOG] greedy_repair: {time.time() - t0:.4f}s")
    return state


def inventory_balance_repair(partial: SolutionState, rng: rnd.Generator):
    # Inventory balance repair operator
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
    plant_inventory = {}
    for (plant, sku_id, day), inv in state.s_ikt.items():
        plant_inventory[plant, day] = plant_inventory.get((plant, day), 0) + inv
    sorted_plants = sorted(plant_inventory.items(), key=lambda x: x[1], reverse=True)
    for (plant, day), _ in sorted_plants:
        dealers = [dealer for (p, dealer) in data.construct_supply_chain() if p == plant]
        for dealer in dealers:
            veh_type = rng.choice(list(data.all_veh_types))
            vehicle = Vehicle(plant, dealer, veh_type, day, data)
            orders = {sku_id: qty for (dealer_id, sku_id), qty in unsatisfied.items()
                     if dealer_id == dealer}
            if not orders:
                continue
            value = veh_loading(state, vehicle, orders)
            if value:
                for sku_id in orders:
                    unsatisfied[(dealer, sku_id)] -= vehicle.cargo.get((sku_id, day), 0)
    print(f"[OPLOG] inventory_balance_repair: {time.time() - t0:.4f}s")
    return state

def urgency_repair(partial: SolutionState, rng: rnd.Generator):
    # Urgency-based repair operator
    t0 = time.time()
    print(f"[OPLOG] 开始执行 urgency_repair 算子")
    state = partial
    data = state.data
    shipped = state.compute_shipped()
    unsatisfied = {}
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if shipped_qty < demand and demand > 0:
            unmet_ratio = (demand - shipped_qty) / demand
            total_available = 0
            supply_chain = data.construct_supply_chain()
            for plant in [plant for (plant, dealer_key), skus in supply_chain.items() 
                          if dealer_key == dealer and sku_id in skus]:
                for day in range(1, data.horizons + 1):
                    total_available += state.s_ikt.get((plant, sku_id, day - 1), 0) + \
                        data.sku_prod_each_day.get((plant, sku_id, day), 0)
            inventory_urgency = 1.0
            if total_available > 0:
                inventory_urgency = min(1.0, (demand - shipped_qty) / total_available)
            urgency = 0.8 * unmet_ratio + 0.2 * inventory_urgency
            unsatisfied[(dealer, sku_id)] = (demand - shipped_qty, urgency)
    sorted_demands = sorted(unsatisfied.items(), key=lambda x: x[1][1], reverse=True)
    for (dealer, sku_id), (unmet_demand, _) in sorted_demands:
        available_plants = []
        supply_chain = data.construct_supply_chain()
        for plant in [plant for (plant, dealer_key), skus in supply_chain.items() 
                     if dealer_key == dealer and sku_id in skus]:
            for day in range(1, data.horizons + 1):
                current_stock = state.s_ikt.get((plant, sku_id, day - 1), 0) + \
                    data.sku_prod_each_day.get((plant, sku_id, day), 0)
                available_plants.append((plant, current_stock))
        available_plants.sort(key=lambda x: x[1], reverse=True)
        if not available_plants:
            continue
        plant = available_plants[0][0]
        veh_types = sorted(list(data.all_veh_types), 
                         key=lambda x: data.veh_type_cap[x],
                         reverse=True)
        for veh_type in veh_types:
            day = rng.choice(range(1, data.horizons + 1))
            vehicle = Vehicle(plant, dealer, veh_type, day, data)
            orders = {sku_id: unmet_demand}
            value = veh_loading(state, vehicle, orders)
            if value:
                break
    print(f"[OPLOG] urgency_repair: {time.time() - t0:.4f}s")
    return state


def infeasible_repair(partial: SolutionState, rng: rnd.Generator):
    # Repair infeasible solution: find vehicles corresponding to SKUs with negative inventory at the end of the current period and repair. Add max iteration protection to prevent infinite loops.
    t0 = time.time()
    print(f"[OPLOG] 开始执行 infeasible_repair 算子")
    state = partial
    data = state.data
    neg_inv = {k: v for k, v in state.s_ikt.items() if v < 0}
    # Iteratively reduce shipments from vehicles that cause negative inventory.
    # For each reduction we incrementally add the removed quantity back to s_ikt
    # for the shipment day and afterwards, and update neg_inv accordingly.
    while neg_inv:
        progress_made = False
        # iterate over a snapshot of vehicles because we may modify the list
        for veh in state.vehicles[:]:
            for (plant, sku_id, day), inv in list(neg_inv.items()):
                if veh.fact_id == plant and veh.day == day and (sku_id, day) in veh.cargo:
                    qty = veh.cargo[(sku_id, day)]
                    decrease_qty = min(qty, -inv)
                    # remove from vehicle
                    veh.cargo[(sku_id, day)] -= decrease_qty
                    if veh.cargo[(sku_id, day)] == 0:
                        del veh.cargo[(sku_id, day)]
                    if veh.is_empty():
                        # remove empty vehicle from solution
                        try:
                            state.vehicles.remove(veh)
                        except ValueError:
                            pass
                    # incrementally restore inventory for shipment day and onwards
                    for d in range(day, state.data.horizons + 1):
                        key = (plant, sku_id, d)
                        state.s_ikt[key] = state.s_ikt.get(key, 0) + decrease_qty
                    # update negative inventory tracker
                    neg_inv[plant, sku_id, day] += decrease_qty
                    if neg_inv[plant, sku_id, day] >= 0:
                        del neg_inv[plant, sku_id, day]
                    progress_made = True
                    break
            if not neg_inv:
                break
        # safety: if no progress in an iteration, break to avoid infinite loop
        if not progress_made:
            print("[OPLOG] infeasible_repair: no progress can be made, breaking to avoid infinite loop")
            break
    # inventory has been updated incrementally during vehicle adjustments; full recompute omitted
    print(f"[OPLOG] infeasible_repair: {time.time() - t0:.4f}s")
    return state


def _construct_training_data(partial: SolutionState, improvement: float):
    # Manually construct feature and label data when there is no data in the tracker. The learning_based_repair operator will randomly select a repair operator in this case, so this function is called to construct data to help the model learn in the initial stage.
    # Note: This function is only applicable when there is no data in the tracker. If there is data, use actual data instead.
    # Args:
    #     state: current solution state, contains tracker reference
    #     improvement: objective improvement compared to previous solution
    state = partial
    data = state.data
    
    avg_demand = np.mean(list(data.demands.values())) if data.demands else 1.0
    avg_sku_size = np.mean([data.sku_sizes[sku] for sku in data.all_skus]) if data.all_skus else 1.0
    
    periods = list(range(1, data.horizons + 1))
    # avg_day 是数组periods中位于中间位置的元素
    avg_day = float(periods[len(periods) // 2]) if periods else 1.0
    
    avg_inventory = np.mean([inv for (plant, sku_id, day), inv in state.s_ikt.items() if day == avg_day-1]) if state.s_ikt else 1.0
    avg_capacity_util = float(rnd.uniform(0, 1))
    
    feature = [
        avg_demand,
        avg_sku_size,
        avg_day,
        avg_inventory,
        avg_capacity_util
    ]
    
    # 这里不需要提前对特征进行标准化处理, 因为可能会破坏数据的一致性

    feat_length, label_length = state.tracker.update_ml_data(feature, improvement)
    print(f"构建的特征数据: {feature}")
    print(f"构建训练数据: 特征长度 {feat_length}, 标签长度 {label_length}, 目标函数改进 {improvement:.4f}")


def _call_greedy_repair(partial: SolutionState, rng: rnd.Generator):
    # Call greedy repair operator and compute objective improvement
    state = partial
    
    prev_obj = state.objective()
    new_state = greedy_repair(state, rng)
    new_obj = new_state.objective()
    
    # 计算目标函数改进值 (prev - new, 正值表示改进)
    improvement = prev_obj - new_obj
    
    # 处理无穷大情况，避免nan
    if math.isnan(improvement):
        # 如果两者都是inf，改进为0
        improvement = 0.0
    elif math.isinf(improvement):
        # 如果是inf，可能是prev是inf且new是有限值，设为一个大的正值
        improvement = 1e6 if improvement > 0 else -1e6
    
    return new_state, improvement


def _random_select_repair_operator(partial: SolutionState, rng: rnd.Generator):
    # Randomly select a repair operator for ML-based repair
    state = partial
    
    # 定义可用的修复算子列表
    REPAIR_OPERATORS = [
        greedy_repair,
        local_search_repair,
        inventory_balance_repair,
        urgency_repair,
        infeasible_repair,
        smart_batch_repair
    ]
    
    operator = rng.choice(REPAIR_OPERATORS)  # 随机选择一种修复算子
    print(f"[OPLOG] 随机选择修复算子: {operator.__name__}")
    
    prev_obj = state.objective()            # 计算调用前的目标函数值
    new_state = operator(state, rng)
    new_obj = new_state.objective()        # 计算调用后的目标函数值
    
    improvement = prev_obj - new_obj        # 计算目标函数改进值 (prev - new, 正值表示改进)
    
    # 处理无穷大情况，避免nan
    if math.isnan(improvement):
        # 如果两者都是inf，改进为0
        improvement = 0.0
    elif math.isinf(improvement):
        # 如果是inf，可能是prev是inf且new是有限值，设为一个大的正值
        improvement = 1e6 if improvement > 0 else -1e6
    
    return new_state, improvement


def _compute_feature(state: SolutionState) -> List[float]:
    # Compute feature vector for ML-based repair
    data = state.data
    
    # 获取未满足需求
    removal_list = get_removal_list(state)
    
    if not removal_list:
        # 如果没有未满足需求，返回默认特征
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
    
    # 将特征值显式转换为 Python 的原生 float 类型, 避免潜在的类型不一致问题
    # 这里不需要提前对特征进行标准化处理, 因为可能会破坏数据的一致性
    feature = [float(avg_demand), float(avg_size), float(avg_day), float(avg_inventory), float(avg_capacity_util)]
    
    print(f"[OPLOG] 计算的实际特征: {feature}")
    return feature

def learning_based_repair(partial: SolutionState, rng: rnd.Generator,
                          model_type: str='adaptive', min_score: float=0.4,
                          initial_sample_size: int=20, adaptive_sample_size: int=100,
                          retrain_interval: int=80) -> SolutionState:
    # Learning-based repair operator: Uses ML to learn the best insertion rules from iteration history. The core is to train a model to predict the quality of insertion positions (based on objective improvement). When the sample size is insufficient, randomly select one of the other 6 repair operators and record the feature and objective improvement as training data. This gradually accumulates training data and enters the ML prediction phase. This also avoids a single repair strategy and improves data diversity.
    # Args:
    #     state: current solution state, contains tracker reference
    #     rng: random number generator
    #     model_type: 'linear' (Ridge), 'random_forest' (RandomForest), 'adaptive' (auto choose)
    #     min_score: minimum acceptable prediction score threshold
    #     initial_sample_size: initial training sample size threshold
    #     adaptive_sample_size: adaptive training sample size threshold
    #     retrain_interval: model retrain interval (iterations)
    t0 = time.time()
    print(f"[OPLOG] 开始执行 learning_based_repair 算子")
    state = partial
    data = state.data
    
    
    # step 1: 检查tracker可用性，无tracker时fallback到greedy
    if state.tracker is None:  # 不存在tracker引用, 只调用greedy修复, 不需要保存训练数据
        print(f"[OPLOG] 无tracker引用, fallback到greedy修复")
        new_state, improvement = _call_greedy_repair(state, rng)
        return new_state
    
    tracker_stats = state.tracker.get_statistics()
    features_data = state.tracker.features
    labels_data = state.tracker.labels
    current_iteration = tracker_stats['total_iterations']
    
    # 当tracker中没有保存任何训练数据时, 该算子会随机选择一种修复算子
    # 但是由于缺少样本数据, 无法进入ML预测阶段
    # 因此, 需要人为构建特征和标签数据
    if not features_data or not labels_data:
        print(f"[OPLOG] 正在构建特征和标签数据 ...")
        print(f"[OPLOG] tracker中无数据, 随机选择一种修复算子 ...")
        new_state, improvement = _random_select_repair_operator(state, rng)
        _construct_training_data(new_state, improvement)
        return new_state
        
    
    # 初始阶段数据不足时, 随机选择一种修复算子
    if len(labels_data) < initial_sample_size:
        print(f"[OPLOG] 数据量不足 {len(labels_data)} < {initial_sample_size}, 随机选择一种修复算子 ...")
        new_state, improvement = _random_select_repair_operator(state, rng)
        
        # 使用实际数据作为特征值, 而不是人为构建
        # 注意, 这里需要使用new_state而不是state
        feature = _compute_feature(new_state)
        
        feat_length, label_length = state.tracker.update_ml_data(feature, improvement)
        print(f"[OPLOG] 已保存实际训练数据: 特征值长度={feat_length}, 标签长度={label_length}, 改进={improvement:.4f}")
        
        return new_state
    
    # step 2: 检查是否需要重训练模型
    # 使用tracker存储上次训练信息，避免频繁重新训练
    need_retrain = True
    model = None
    scaler = None
    
    # 检查tracker中是否有缓存的模型和上次训练迭代数
    if state.tracker.has_cached_model():
        cached_model, cached_scaler, last_train_iter = state.tracker.get_cached_model()
        if current_iteration - last_train_iter < retrain_interval:
            # 距离上次训练未达到间隔, 使用缓存模型
            need_retrain = False
            model = cached_model
            scaler = cached_scaler
            print(f"[OPLOG] 使用缓存模型, 距离上次训练 {current_iteration - last_train_iter} 迭代")
        else:
            print(f"[OPLOG] 缓存模型过期 ({current_iteration - last_train_iter} > {retrain_interval}), 需要重新训练")
    else:
        print("[OPLOG] 无缓存模型，开始首次训练")
    
    # step 3: 根据 model_type 参数选择模型进行训练   
    if need_retrain:
        try:
            X = np.array(features_data)
            y = np.array(labels_data)
            
            # 特征标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 根据model_type参数选择模型
            if model_type == 'linear':
                # 岭回归模型
                model = Ridge(alpha=1.0, random_state=rng.integers(0, 1000))
                print(f"[OPLOG] 使用指定的Ridge模型")
            
            elif model_type == 'random_forest':
                # 大数据集使用随机森林
                model = RandomForestRegressor(
                    n_estimators=50,  # 减少树数量提升速度
                    max_depth=10,     # 限制树深防止过拟合
                    random_state=rng.integers(0, 1000),
                    n_jobs=1          # 单线程避免并发问题
                )
                print(f"[OPLOG] 使用指定的随机森林模型")
            
            elif model_type == 'adaptive':
                # 自动根据数据量选择模型
                if len(X) < adaptive_sample_size:
                    model = Ridge(alpha=1.0, random_state=rng.integers(0, 1000))
                    print(f"[OPLOG] 自适应选择Ridge模型 (数据量: {len(X)})")
                else:
                    model = RandomForestRegressor(
                        n_estimators=50,
                        max_depth=10,
                        random_state=rng.integers(0, 1000),
                        n_jobs=1
                    )
                    print(f"[OPLOG] 自适应选择随机森林模型 (数据量: {len(X)})")
            
            else:
                # 未知模型类型，fallback到自适应
                print(f"[OPLOG] 未知模型类型 '{model_type}'，使用自适应模式")
                model = Ridge(alpha=1.0, random_state=rng.integers(0, 1000)) if len(X) < adaptive_sample_size else RandomForestRegressor(n_estimators=50, max_depth=10, random_state=rng.integers(0, 1000), n_jobs=1)
                
            # 训练模型
            model.fit(X_scaled, y)
            
            # 缓存模型和相关信息
            state.tracker.cache_ml_model(model, scaler, current_iteration)
            print(f"[OPLOG] 模型训练完成, 已缓存到tracker")
        
        except Exception as e:  # 捕获所有异常
            print(f"[OPLOG] 模型训练失败 ({type(e).__name__}), fallback到greedy修复 ...")
            new_state, improvement = _call_greedy_repair(state, rng)
            feature = _compute_feature(new_state)
            feat_length, label_length = state.tracker.update_ml_data(feature, improvement)
            print(f"[OPLOG] 已保存实际训练数据: 特征值长度={feat_length}, 标签长度={label_length}, 改进={improvement:.4f}")
            return new_state
    
    # step 4: 获取待插入的未满足需求
    removal_list = get_removal_list(state)
    if not removal_list:
        print(f"[OPLOG] 无未满足需求, 直接返回当前解")
        return state
    
    # step 5: 为每个待插入需求生成候选并预测分数
    total_predictions = 0
    successful_insertions = 0
    failed_demands = []  # 记录ML插入失败的需求
    
    for (dealer, sku_id), remain_qty in removal_list.items():
        candidates = []
        
        # 生成候选插入位置
        supply_chain = data.construct_supply_chain()
        available_plants = [plant for (plant, dealer_key), skus in supply_chain.items() 
                           if dealer_key == dealer and sku_id in skus]
        
        for plant in available_plants:
            for day in range(1, data.horizons + 1):
                for veh_type in data.all_veh_types:
                    try:
                        # 提取特征向量
                        demand = data.demands.get((dealer, sku_id), 0)
                        sku_size = data.sku_sizes[sku_id] * remain_qty
                        inventory = state.s_ikt.get((plant, sku_id, day-1), 0)
                        veh_capacity = data.veh_type_cap[veh_type]
                        capacity_util = sku_size / veh_capacity if veh_capacity > 0 else 0
                        
                        # 构建特征向量[demand, sku_size, day, inventory, capacity_util]
                        feature_vector = np.array([[demand, sku_size, day, inventory, capacity_util]])
                        feature_scaled = scaler.transform(feature_vector)
                        
                        # 预测插入质量分数
                        pred_score = model.predict(feature_scaled)[0]
                        total_predictions += 1
                        
                        # 只保留高质量候选
                        if pred_score >= min_score:
                            candidates.append((plant, veh_type, day, pred_score, feature_vector[0]))

                    except Exception as e:
                        continue  # 跳过异常候选
        
        # step 6: 选择最佳候选进行插入
        if candidates:
            # 按预测分数降序排序
            candidates.sort(key=lambda x: x[3], reverse=True)
            
            for plant, veh_type, day, pred_score, raw_feature in candidates:
                try:
                    # 尝试插入
                    veh = Vehicle(plant, dealer, veh_type, day, data)
                    orders = {sku_id: remain_qty}
                    
                    # 记录插入前的目标函数值
                    prev_obj = state.objective()
                    
                    # 执行插入
                    success = veh_loading(state, veh, orders)
                    if success and veh.cargo:
                        state.vehicles.append(veh)
                        # veh_loading already updates s_ikt incrementally; defer full recompute to function end
                        
                        # 计算实际改进并更新tracker
                        new_obj = state.objective()
                        
                        # 计算实际改进值, 正值表示改进
                        actual_improvement = prev_obj - new_obj
                        
                        if math.isnan(actual_improvement):
                            # 如果两者都是inf，实际改进为0
                            actual_improvement = 0.0
                        elif math.isinf(actual_improvement):
                            # 如果是inf，可能是prev_obj和new_obj这两者中, 一个是inf, 另一个是有限值
                            actual_improvement = 1e6 if actual_improvement > 0 else -1e6
                        
                        # 更新ML训练数据
                        state.tracker.update_ml_data(raw_feature.tolist(), actual_improvement)
                        successful_insertions += 1
                        
                        break  # 成功插入, 处理下一个需求
                
                except Exception as e:
                    continue  # 插入失败, 尝试下一个候选
            
        # 如果所有候选都失败, 记录失败需求
        if not candidates or remain_qty > 0:
            failed_demands.append((dealer, sku_id))
    
    # step 7: 如果有ML插入失败的需求，使用greedy一次性修复所有
    # 因为greedy修复会尝试插入所有未满足需求
    if failed_demands:
        print(f"[OPLOG] {len(failed_demands)} 个需求ML插入失败, 使用greedy修复")
        prev_obj = state.objective()
        new_state = greedy_repair(state, rng)
        new_obj = new_state.objective()
        improvement = prev_obj - new_obj
        
        # 处理边界情况, 防止改进值为inf或nan
        if math.isnan(improvement):
            improvement = 0.0
        elif math.isinf(improvement):
            improvement = 1e6 if improvement > 0 else -1e6
        
        # 使用实际特征数据而不是手动构建, 注意这里需要使用new_state
        feature = _compute_feature(new_state)
        feat_length, label_length = state.tracker.update_ml_data(feature, improvement)
        print(f"[OPLOG] 已保存greedy修复的实际训练数据: 特征值长度={feat_length}, 标签长度={label_length}, 改进={improvement:.4f}")
        state = new_state

    # 保持在函数末尾做一次完整重算以保证一致性 (如果需要)
    state.compute_inventory()
    
    print(f"[OPLOG] ML修复完成: {total_predictions}次预测, {successful_insertions}次成功插入, {len(failed_demands)}个需求使用greedy修复")
    
    elapsed = time.time() - t0
    print(f"[OPLOG] learning_based_repair: {elapsed:.4f}s")
    return state


class LearningBasedRepairOperator:
    # Learning-based repair operator wrapper class
    def __init__(self, model_type: str='adaptive', 
                 min_score: float=0.4, 
                 initial_sample_size: int=20,
                 adaptive_sample_size: int=100,
                 retrain_interval: int=80):
        
        self.model_type = model_type
        self.min_score = min_score
        self.initial_sample_size = initial_sample_size
        self.adaptive_sample_size = adaptive_sample_size
        self.retrain_interval = retrain_interval
        self.__name__ = "learning_based_repair"
        
        # 验证参数有效性
        valid_models = ['linear', 'random_forest', 'adaptive']
        if model_type not in valid_models:
            print(f"[WARNING] 无效的model_type '{model_type}', 使用默认值 'adaptive'")
            self.model_type = 'adaptive'
        
        if retrain_interval < 10:
            print(f"[WARNING] retrain_interval过小 ({retrain_interval}), 设置为最小值 10")
            self.retrain_interval = 10

    def __call__(self, current: SolutionState, rng: rnd.Generator):
        return learning_based_repair(current, rng, self.model_type, self.min_score, self.initial_sample_size, self.adaptive_sample_size, self.retrain_interval)

def get_removal_list(state: SolutionState) -> Dict[Tuple[str, str], int]:
    # Returns a dict {(dealer, sku_id): unmet_quantity}
    shipped = state.compute_shipped()
    removal_list = {}
    
    for (dealer, sku_id), total_demand in state.data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        unmet_qty = total_demand - shipped_qty
        
        if unmet_qty > 0:
            removal_list[(dealer, sku_id)] = unmet_qty
    
    return removal_list


def smart_batch_repair(partial: SolutionState, rng: rnd.Generator):
    # Smart Batch Repair Algorithm (SBRA)
    # Key improvements:
    # 1. Resource pool pre-computation: avoid repeated inventory/availability calculation
    # 2. Simple batch clustering
    # 3. Multi-objective: balance demand satisfaction, vehicle count, resource utilization
    # 4. Smart vehicle type selection
    # 5. Adaptive timeout
    t0 = time.time()
    print(f"[OPLOG] 开始执行 smart_batch_repair 算子")
    
    state = partial
    data = state.data
    
    # 自适应参数设置
    num_demands = len(data.demands)
    timeout = min(60.0, max(10.0, num_demands * 0.1))  # 根据问题规模调整超时
    batch_size = min(20, max(5, num_demands // 10))   # 动态批次大小
    
    print(f"[OPLOG] 问题规模: {num_demands} 需求, 超时限制: {timeout}s, 批次大小: {batch_size}")
    
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
    # Precompute all available resources to avoid repeated calculation
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
    # Get current list of unsatisfied demands
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
    # Smart batch: simplified version, split demands into batches by batch_size. Can be extended for clustering by region/SKU features if needed.
    if not unsatisfied_demands:
        return []
    # Simplified: sort by priority and split into batches
    sorted_demands = sorted(unsatisfied_demands, key=lambda x: x['priority'], reverse=True)
    batches = []
    for i in range(0, len(sorted_demands), batch_size):
        batch = sorted_demands[i:i + batch_size]
        batches.append(batch)
    return batches


def _process_demand_batch(state: SolutionState, batch, resource_pool, data: DataALNS, rng: rnd.Generator):
    # Process a single demand batch
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

            # 查询真实可用库存
            prev_inv = state.s_ikt.get((plant, sku_id, day-1), 0)
            production = data.sku_prod_each_day.get((plant, sku_id, day), 0)
            available = max(0, prev_inv + production)

            # 选择最合适的车型
            optimal_veh_type = _select_optimal_vehicle_type(
                remain_demand, sku_id, data
            )

            # 计算实际装载量，不能超过真实库存
            load_qty = min(remain_demand, available_qty, available)
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
    # Find the best resource allocation scheme
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
    # Select the optimal vehicle type based on demand quantity
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
    # Final forced loading to ensure all demands are satisfied
    print(f"[OPLOG] 开始最终强制装载")
    
    shipped = state.compute_shipped()
    force_loaded = 0
    
    for (dealer, sku_id), demand in data.demands.items():
        remain_demand = demand - shipped.get((dealer, sku_id), 0)
        if remain_demand <= 0:
            continue

        supply_chain = data.construct_supply_chain()
        available_plants = [plant for (plant, dealer_key), skus in supply_chain.items()
                            if dealer_key == dealer and sku_id in skus]
        if not available_plants:
            print(f"[OPLOG] 警告: 无法为 {dealer}-{sku_id} 找到供应工厂")
            continue

        # 遍历所有工厂和天，优先用有库存的组合
        for plant in available_plants:
            for day in range(1, data.horizons+1):
                if remain_demand <= 0:
                    break
                prev_inv = state.s_ikt.get((plant, sku_id, day-1), 0)
                production = data.sku_prod_each_day.get((plant, sku_id, day), 0)
                available = max(0, prev_inv + production)
                # 统计当天已发货
                used = sum(veh.cargo.get((sku_id, day), 0) for veh in state.vehicles if veh.fact_id == plant and veh.day == day)
                real_avail = max(0, available - used)
                load_qty = min(remain_demand, real_avail)
                if load_qty <= 0:
                    continue
                veh_type = min(data.veh_type_cap.keys(), key=lambda x: data.veh_type_cap[x])
                vehicle = Vehicle(plant, dealer, veh_type, day, data)
                orders = {sku_id: load_qty}
                try:
                    veh_loading(state, vehicle, orders)
                    force_loaded += 1
                    remain_demand -= load_qty
                except Exception as e:
                    print(f"[OPLOG] 强制装载失败 {dealer}-{sku_id}: {e}")
            if remain_demand <= 0:
                break
        if remain_demand > 0:
            print(f"[OPLOG] 强制装载后仍有未满足: {dealer}-{sku_id} 剩余 {remain_demand}")
    print(f"[OPLOG] 强制装载完成, 处理 {force_loaded} 个需求")
