# 列生成算法的子模型
# 适用于模型2——分解方法
# 模型 2 的假设包括：多周期、SKU供给量 = 期初库存 + 生产数量 ≥ 需求量、生产基地的存储能力有上限
# 多生产基地、多经销商、多车型、多SKU
# 每家生产基地可以提供相同的SKU，不同生产基地提供的SKU之间存在交集
import time
import math
from gurobipy import GRB
import gurobipy as gp
from collections import defaultdict
from typing import Dict, Tuple, Set, List, Hashable
from dataclasses import dataclass, field
from optutility import LogPrinter, BigNum
from InputDataCG import *

# 创建LogPrinter的实例
log_printer = LogPrinter(time.time())
# # 通过设置force_plain=True，强制化使用纯文本输出，例如在重定向到文件时
# log_printer.set_output_mode(force_plain=True)

@dataclass
class SubModel:
    fact_id: str
    dealer_id: str
    vehicle_type: str
    cur_day: int
    data: InputDataCG

    # 设定环境参数
    # MIP模型的求解gap，当达到设定的gap后，模型停止，默认值为0
    param_gap_limit: float = field(init=False)
    # MIP模型求解的时间限制，默认值为60
    param_time_limit: float = field(init=False)
    # 是否输出gurobi的求解日志
    param_output_flag: int = field(init=False)

    # 所有SKU的集合
    all_skus: Set[str] = field(init=False)
    # 当前经销商需要的且当前生产基地能够提供的SKU集合
    require_skus: Set[str] = field(init=False)
    # 每种SKU的体积量化数值 {sku: size}
    sku_sizes: Dict[str, int] = field(init=False)
    # 经销商对SKU的需求 {(dealer, sku, day): qty}
    demands: Dict[Tuple[str, str, int], int] = field(init=False)
    # 经销商需要的SKU集合
    skus_dealer: Dict[str, Set[str]] = field(init=False)
    # 在每个周期内，生产基地计划生产的SKU数量 {(plant, sku, day): qty}
    sku_prod_each_day: Dict[Tuple[str, str, int], int] = field(init=False)

    # 每一种车辆类型的使用成本 {v_type: cost}
    veh_type_cost: Dict[str, int] = field(init=False)
    # 每一种车辆类型的容量上限 {v_type: capacity}
    veh_type_cap: Dict[str, int] = field(init=False)
    # 每一种车辆类型的最小起运量 {v_type: min_load}
    veh_type_min_load: Dict[str, int] = field(init=False)

    # 接下来，需要定义数学模型、目标函数和决策变量
    # 最后再定义约束条件
    # 定义子模型
    model: gp.Model = field(init=False)
    # 定义目标函数
    obj: gp.LinExpr = field(init=False)
    
    # 生产基地 i 中SKU k 的库存转移到周期 t 的数量，其中 t = 0, 1, ..., T。
    # 特别地，s_ik0表示生产基地 i 中SKU k 的期初库存。 {(plant, sku_id, day): float}
    s_ikt: Dict[Tuple[str, str, int], float] = field(init=False)
    s_indices: Set[Tuple[str, str, int]] = field(init=False)
    keys: Set[Tuple[str, str, int]] = field(init=False)
    
    
    # 定义决策变量
    # 子模型的决策变量，表示在周期t内，从生产基地 i 到经销商 j 的车辆类型 f 对应的一辆车辆中装载SKU k 的商品数量
    # 其中，生产基地 i 、经销商 j 、车辆类型 f、周期 t 都是已知的
    d_xs: gp.tupledict[str, gp.Var] = field(init=False)
    # 辅助变量：车辆类型 f 中装载的SKU体积是否低于最小起运量，如果是取1，否则取0
    p_f: gp.Var = field(init=False)

    # 定义需要向模型中添加的约束条件
    # 车辆类型 f 的容量约束
    cons_veh_cap: gp.Constr = field(init=False)
    # 车辆类型 f 的最小起运量约束
    cons_veh_min_load: gp.Constr = field(init=False)
    # 决策变量的取值范围约束
    cons_xs_bounds1: gp.tupledict[Hashable, gp.Constr] = field(init=False)
    cons_xs_bounds2: gp.tupledict[Hashable, gp.Constr] = field(init=False)

    def __post_init__(self):
        self.param_gap_limit = self.data.param_gap_limit
        self.param_time_limit = self.data.param_time_limit
        self.param_output_flag = self.data.param_output_flag

        self.all_skus = self.data.all_skus
        self.sku_sizes = self.data.sku_sizes
        self.demands = self.data.demands
        self.skus_dealer = self.data.skus_dealer
        self.sku_prod_each_day = self.data.sku_prod_each_day
        self.veh_type_cost = self.data.veh_type_cost
        self.veh_type_cap = self.data.veh_type_cap
        self.veh_type_min_load = self.data.veh_type_min_load
        

        # 子模型的目标函数，只有在获取到主模型约束条件的对偶价格之后，才会构建
        self.model = gp.Model()
        self.obj = gp.LinExpr()

        self.require_skus = self.data.available_skus_to_dealer(self.fact_id, self.dealer_id)
        self.d_xs = self.model.addVars(self.require_skus, lb=0, vtype=GRB.INTEGER, name="d_xs")
        self.p_f = self.model.addVar(lb=0, vtype=GRB.BINARY, name="p_f")
        
        self.s_indices = {(self.fact_id, sku_id, t) for sku_id in self.require_skus for t in range(self.cur_day + 1)}
        
        # 对s_ikt的取值进行初始化
        self.s_ikt = defaultdict(float)
        for (plant, sku_id, t) in self.s_indices:
            if t == 0 and (plant, sku_id) in self.data.sku_initial_inv:
                self.s_ikt[plant, sku_id, t] = self.data.sku_initial_inv[plant, sku_id]

        # 添加约束条件
        self.cons_veh_cap = self.model.addConstr((
            gp.quicksum(
                self.d_xs[sku_id] * self.sku_sizes[sku_id] for sku_id in self.require_skus
            ) <= self.veh_type_cap[self.vehicle_type]
        ), name="cons_veh_cap")

        self.cons_veh_min_load = self.model.addConstr((
            gp.quicksum(
                self.d_xs[sku_id] * self.sku_sizes[sku_id] for sku_id in self.require_skus
            ) + BigNum * self.p_f >= self.veh_type_min_load[self.vehicle_type]
        ), name="cons_veh_min_load")
        
        
        self.cons_xs_bounds1 = self.model.addConstrs((
            self.d_xs[sku_id] <= self.demands[self.dealer_id, sku_id, self.cur_day]
            for sku_id in self.require_skus
        ), name="cons_xs_bounds1")
        
        self.cons_xs_bounds2 = self.model.addConstrs((
            self.d_xs[sku_id] <= self.data.sku_prod_each_day.get((self.fact_id, sku_id, self.cur_day), 0) + \
                self.s_ikt[self.fact_id, sku_id, self.cur_day-1]
            for sku_id in self.require_skus
        ), name="cons_xs_bounds2")
        

        self.model.setParam(GRB.Param.OutputFlag, 0)

    def run(self):
        # log_printer.print("Optimizing sub model...")
        self.model.setObjective(self.obj, sense=GRB.MAXIMIZE)
        # self.model.Params.FeasibilityTol = 1e-9
        self.model.optimize()
        self.model.write("SubModel.lp")

        if self.model.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            return False
        return True

    def get_load_pattern(self):
        """
        retrieve load patterns and prepare to submit them to master model.
        """
        return {(sku_id, self.cur_day): var.Xn for sku_id, var in self.d_xs.items() if var.Xn > 0}
