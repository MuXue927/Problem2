# 列生成算法的主模型
# 适用于模型2——分解方法
# 模型 2 的假设包括：多周期、SKU供给量 = 期初库存 + 生产数量 ≥ 需求量、生产基地的存储能力有上限
# 多生产基地、多经销商、多车型、多SKU
# 每家生产基地可以提供相同的SKU，不同生产基地提供的SKU之间存在交集
import time
import random
from copy import deepcopy
from gurobipy import GRB
import gurobipy as gp
from vehicle import Vehicle
from typing import Dict, Tuple, Set, List
from dataclasses import dataclass, field
from optutility import LogPrinter, POSITIVEEPS
from InputDataCG import InputDataCG

# 创建LogPrinter的实例
log_printer = LogPrinter(time.time())
# # 通过设置force_plain=True，强制化使用纯文本输出，例如在重定向到文件时
# log_printer.set_output_mode(force_plain=True)

@dataclass
class MasterModel:
    cur_day: int     # 当前的规划周期
    data: InputDataCG
    # 对于多周期问题，默认规划周期长度为 1
    # 具体数值是warehouse_production.csv文件中produce_date这一列的最大值
    horizons = 1
    
    # 设定环境参数
    # MIP模型的求解gap，当达到设定的gap后，模型停止，默认值为0
    param_gap_limit: float = field(init=False)
    # MIP模型求解的时间限制，默认值为60
    param_time_limit: float = field(init=False)
    # 是否输出gurobi的求解日志
    param_output_flag: int = field(init=False)

    # 装载模式的编号，只要不重复就好，用作key的一部分
    pattern_id = 0
    # 记录进入主模型中的每一种装载模式，key是装载模式的名称，用d_xs的变量名表示
    # value是一种具体的装载模式，包括装载的SKU及其数量
    pattern_content = {}
    # 当前模型中存在多少个装载模式
    num_patterns = 0
    # 裁剪模型的时候，每次减少多少个装载模式，当前没有启动
    remove_batch = 100

    # 定义需要的数据接口，看数学模型中的符号定义
    # 生产基地的集合
    plants: Set[str] = field(init=False)
    # 经销商的集合
    dealers: Set[str] = field(init=False)
    # 所有SKU的集合
    all_skus: Set[str] = field(init=False)
    # 所有车辆类型集合
    all_veh_types: Set[str] = field(init=False)
    # 规划周期 t 的集合
    periods: Set[int] = field(init=False)

    # 每种SKU的体积量化数值 {sku: size}
    sku_sizes: Dict[str, int] = field(init=False)

    # 在每个周期内，生产基地计划生产的SKU数量 {(plant, sku, day): qty}
    sku_prod_each_day: Dict[Tuple[str, str, int], int] = field(init=False)
    
    # 在每个周期内，所有生产基地计划生产的SKU数量 {(sku_id, day): qty}
    sku_prod_total: Dict[Tuple[str, int], int] = field(init=False)
    
    # 生产基地中SKU的期初库存数量 {(plant, sku): qty}
    sku_initial_inv: Dict[Tuple[str, str], int] = field(init=False)
    
    # 生产基地 i 中SKU k 转移到周期 t 的数量 {(plant, sku, day): qty}
    historical_s_ikt: Dict[Tuple[str, str, int], int] = field(init=False)
    
    # 生产基地期初拥有的SKU集合
    skus_initial: Dict[str, Set[str]] = field(init=False)
    # 生产基地需要生产的SKU集合
    skus_prod: Dict[str, Set[str]] = field(init=False)
    # 生产基地可以提供的SKU集合
    skus_plant: Dict[str, Set[str]] = field(init=False)

    # 经销商对SKU的需求 {(dealer, sku, day): qty}
    demands: Dict[Tuple[str, str, int], int] = field(init=False)
    # 经销商需要的SKU集合
    skus_dealer: Dict[str, Set[str]] = field(init=False)

    # 每一种车辆类型的使用成本 {v_type: cost}
    veh_type_cost: Dict[str, int] = field(init=False)
    # 每一种车辆类型的容量上限 {v_type: capacity}
    veh_type_cap: Dict[str, int] = field(init=False)
    # 每一种车辆类型的最小起运量 {v_type: min_load}
    veh_type_min_load: Dict[str, int] = field(init=False)
    # 每个生产基地的库存上限 {plant: max_cap}
    plant_inv_limit: Dict[str, int] = field(init=False)

    # 接下来，需要定义数学模型、目标函数、人工变量和决策变量
    # 最后再定义约束条件
    # 定义主模型
    model: gp.Model = field(init=False)
    # 定义目标函数
    obj: gp.LinExpr = field(init=False)
    # 人工变量 alpha
    d_alpha: gp.tupledict[Tuple[str, str, int], gp.Var] = field(init=False)  # {(dealer, sku, day): gp.Var}
    # 人工变量 beta
    d_beta: gp.tupledict[Tuple[str, str, int], gp.Var] = field(init=False)  # {(plant, sku, day): gp.Var}
    # 人工变量 gamma
    d_gamma: gp.tupledict[Tuple[str, str, int], gp.Var] = field(init=False)  # {(plant, sku, day): gp.Var}
    # # 人工变量 delta
    # d_delta: gp.tupledict[Tuple[str, str, int], gp.Var] = field(init=False)  # {(dealer, sku, day): gp.Var}

    # 生产基地 i 中SKU k 的库存转移到周期 t 的数量，其中 t = 0, 1, ..., T。
    # 特别地，s_ik0表示生产基地 i 中SKU k 的期初库存。 {(plant, sku_id, day): gp.Var}
    s_ikt: gp.tupledict[Tuple[str, str, int], gp.Var] = field(init=False)

    # 定义决策变量 x_ijft
    # 主模型的决策变量，表示在周期 t 内，从生产基地 i 到经销商 j 的车辆类型 f 对应的一辆车辆的装载方案的使用数量
    # {(plant, dealer, v_type, day, load_pattern_id): gp.Var}
    d_xs: gp.tupledict[Tuple[str, str, str, int, int], gp.Var] = field(init=False)

    s_indices: Set[Tuple[str, str, int]] = field(init=False)

    # 定义需要向模型中添加的约束条件
    # 在周期 t 内，所有生产基地运往某个经销商 j 的SKU k 的总量，大于等于经销商 j 对SKU k 的需求
    cons_dealer_demands: gp.tupledict[Tuple[str, str, int], gp.Constr] = field(init=False)  # {(dealer, sku_id, day): gp.Constr}
    # # 在周期 t 内，生产基地 i 运往经销商 j 的SKU k 的数量 <= 在周期 t 内，该生产基地可以提供的SKU k 的数量
    # cons_sku_supply: gp.tupledict[Tuple[str, str, str, int], gp.Constr] = field(init=False)  # {(plant, dealer, sku_id, day): gp.Constr}
    
    # 在周期 t 内，从生产基地 i 运出去的SKU k 的数量，等于
    # 该周期内生产基地 i 生产这种SKU的数量，加上生产基地 i 中SKU k 的库存转移到周期 t-1 的数量，
    # 再减去生产基地 i 中SKU k 的库存转移到周期 t 的数量。
    cons_sku_transfer: gp.tupledict[Tuple[str, str, int], gp.Constr] = field(init=False)  # {(plant, sku_id, day): gp.Constr}
    # 在任意周期 t 内，生产基地 i 中剩余的所有SKU数量不得超过该基地的库存容量上限。
    cons_fact_inv_limit: gp.tupledict[Tuple[str, int], gp.Constr] = field(init=False)  # {(plant, day): gp.Constr}

    def __post_init__(self):
        self.horizons = self.data.horizons

        self.param_gap_limit = self.data.param_gap_limit
        self.param_time_limit = self.data.param_time_limit
        self.param_output_flag = self.data.param_output_flag

        self.plants = self.data.plants
        self.dealers = self.data.dealers
        self.all_skus = self.data.all_skus
        self.all_veh_types = self.data.all_veh_types
        self.sku_sizes = self.data.sku_sizes
        self.sku_prod_each_day = self.data.sku_prod_each_day
        
        self.sku_prod_total = self.data.sku_prod_total
        
        self.sku_initial_inv = self.data.sku_initial_inv
        self.historical_s_ikt = self.data.historical_s_ikt
        
        self.skus_initial = self.data.skus_initial
        self.skus_prod = self.data.skus_prod
        self.skus_plant = self.data.skus_plant
        self.demands = self.data.demands
        self.skus_dealer = self.data.skus_dealer
        self.veh_type_cost = self.data.veh_type_cost
        self.veh_type_cap = self.data.veh_type_cap
        self.veh_type_min_load = self.data.veh_type_min_load
        self.plant_inv_limit = self.data.plant_inv_limit

        self.model = gp.Model()
        self.obj = gp.LinExpr()

        self.periods = {t for t in range(1, self.horizons + 1)}

        triples = {(dealer, sku_id, self.cur_day) for (dealer, sku_id, day) in self.demands}
        self.d_alpha = self.model.addVars(triples, lb=0, vtype=GRB.CONTINUOUS, name="d_alpha")
        
        triple_plant_dealer_sku = {
            (plant, dealer, sku_id) for (plant, dealer), skus in self.data.construct_supply_chain().items()
            for sku_id in skus
        }
        self.triple_plant_sku_day = {(plant, sku_id, self.cur_day) for (plant, dealer, sku_id) in triple_plant_dealer_sku}
        self.d_beta = self.model.addVars(self.triple_plant_sku_day, lb=0, vtype=GRB.CONTINUOUS, name="d_beta")
        self.d_gamma = self.model.addVars(self.triple_plant_sku_day, lb=0, vtype=GRB.CONTINUOUS, name="d_gamma")
        
        
        self.s_indices = {(plant, sku_id, t) for (plant, dealer, sku_id) in triple_plant_dealer_sku 
                          for t in range(self.cur_day + 1)}
        self.s_ikt = self.model.addVars(self.s_indices, lb=0, vtype=GRB.CONTINUOUS, name="s")
        # 由于s_ik0代表的是生产基地 i 中拥有的SKU k 的期初库存，所以需要 **固定** 这部分变量的取值
        # 如何实现？同时设定这部分变量的上界和下界为某个常数
        for (plant, sku_id, t) in self.s_indices:
            if t == 0 and (plant, sku_id) in self.sku_initial_inv:
                self.s_ikt[plant, sku_id, t].setAttr('LB', self.sku_initial_inv[plant, sku_id])
                self.s_ikt[plant, sku_id, t].setAttr('UB', self.sku_initial_inv[plant, sku_id])

        self.d_xs = gp.tupledict()

        self.obj = (self.data.param_pun_factor1 * self.d_alpha.sum('*', '*', '*') +
                    self.data.param_pun_factor2 * self.d_beta.sum('*', '*', '*') +
                    self.data.param_pun_factor3 * self.d_gamma.sum('*', '*', '*')
                    )
        
        self.model.setObjective(self.obj, sense=GRB.MINIMIZE)

        self.cons_dealer_demands = self.model.addConstrs((
            self.d_alpha[dealer, sku_id, t] >= self.demands[dealer, sku_id, t]
            for (dealer, sku_id, t) in triples
        ), name="cons_dealer_demands")
        
        
        for (plant, sku_id, t) in self.historical_s_ikt:
            if t == self.cur_day:
                self.s_ikt[plant, sku_id, t].setAttr('LB', self.historical_s_ikt[plant, sku_id, t])
                self.s_ikt[plant, sku_id, t].setAttr('UB', self.historical_s_ikt[plant, sku_id, t])

        self.cons_sku_transfer = self.model.addConstrs((
            self.d_beta[plant, sku_id, t] - self.d_gamma[plant, sku_id, t] ==
            self.sku_prod_each_day.get((plant, sku_id, t), 0) + self.s_ikt[plant, sku_id, t-1] - 
            self.s_ikt[plant, sku_id, t]
            for (plant, sku_id, t) in self.triple_plant_sku_day
        ), name="cons_sku_transfer")
        
        self.cons_fact_inv_limit = self.model.addConstrs((
            self.s_ikt.sum(plant, '*', t) >= -1 * self.plant_inv_limit[plant]
            for plant in self.plants for t in self.periods
        ), name="cons_fact_inv_limit")

        # 启用求解日志
        self.model.setParam('OutputFlag', self.param_output_flag)
        # 设置模型求解gap
        self.model.Params.MIPGap = self.param_gap_limit
        # 设置求解时间
        self.model.Params.TimeLimit = self.param_time_limit

    def get_alpha_size(self):
        non_zero_alpha = {(dealer, sku_id, t) for (dealer, sku_id, t), var in self.d_alpha.items() if var.X > POSITIVEEPS}
        return len(non_zero_alpha)

    def get_beta_size(self):
        non_zero_beta = {(plant, sku_id, t) for (plant, sku_id, t), var in self.d_beta.items() if var.X > POSITIVEEPS}
        return len(non_zero_beta)

    def get_gamma_size(self):
        non_zero_gamma = {(plant, sku_id, t) for (plant, sku_id, t), var in self.d_gamma.items() if var.X > POSITIVEEPS}
        return len(non_zero_gamma)

    
    def deactivate_alpha(self):
        for (dealer, sku_id, t) in self.d_alpha:
            self.d_alpha[dealer, sku_id, t].setAttr('UB', 0)

    def deactivate_beta(self):
        for (plant, sku_id, t) in self.d_beta:
            self.d_beta[plant, sku_id, t].setAttr('UB', 0)

    def deactivate_gamma(self):
        for (plant, sku_id, t) in self.d_gamma:
            self.d_gamma[plant, sku_id, t].setAttr('UB', 0)

    def deactivate_artificial_vars(self):
        self.deactivate_alpha()
        self.deactivate_beta()
        self.deactivate_gamma()
        # self.deactivate_delta()

    def activate_alpha(self):
        for (dealer, sku_id, t) in self.d_alpha:
            self.d_alpha[dealer, sku_id, t].setAttr('UB', GRB.INFINITY)

    def activate_beta(self):
        for (plant, sku_id, t) in self.d_beta:
            self.d_beta[plant, sku_id, t].setAttr('UB', GRB.INFINITY)

    def activate_gamma(self):
        for (plant, sku_id, t) in self.d_gamma:
            self.d_gamma[plant, sku_id, t].setAttr('UB', GRB.INFINITY)

    def activate_artificial_vars(self):
        self.activate_alpha()
        self.activate_beta()
        self.activate_gamma()
        # self.activate_delta()

    def change_var_type(self, var_type=GRB.INTEGER):
        log_printer.print("Set variables in the master model to integer type")
        for var in self.model.getVars():
            var.setAttr('VType', var_type)
        self.model.Params.MIPGap = self.param_gap_limit
        self.model.Params.TimeLimit = self.param_time_limit
        self.model.setParam('OutputFlag', self.param_output_flag)

    def remove_partial_artificial_vars(self):
        """
        remove artificial variables whose value equals zero and update model.
        """
        remove_alpha = {(dealer, sku_id, t): var for (dealer, sku_id, t), var in self.d_alpha.items() if var.X <= POSITIVEEPS}
        reserve_alpha = {(dealer, sku_id, t): var for (dealer, sku_id, t), var in self.d_alpha.items() if var.X > POSITIVEEPS}
        self.d_alpha = gp.tupledict(reserve_alpha)

        remove_beta = {(plant, sku_id, t): var for (plant, sku_id, t), var in self.d_beta.items() if var.X <= POSITIVEEPS}
        reserve_beta = {(plant, sku_id, t): var for (plant, sku_id, t), var in self.d_beta.items() if var.X > POSITIVEEPS}
        self.d_beta = gp.tupledict(reserve_beta)

        remove_gamma = {(plant, sku_id, t): var for (plant, sku_id, t), var in self.d_gamma.items() if var.X <= POSITIVEEPS}
        reserve_gamma = {(plant, sku_id, t): var for (plant, sku_id, t), var in self.d_gamma.items() if var.X > POSITIVEEPS}
        self.d_gamma = gp.tupledict(reserve_gamma)

        self.model.remove(remove_alpha)
        self.model.remove(remove_beta)
        self.model.remove(remove_gamma)
        # self.model.remove(remove_delta)

        self.model.update()
        self.obj = self.model.getObjective()
        return self.obj.size()

    def insert_pattern(self, fact_id: str, dealer_id: str, vehicle_type: str,
                        obj_contribution: int, load_pattern: Dict[Tuple[str, int], int]):
        """
        insert load patterns submitted by sub models into master model

        :param fact_id: a specific plant
        :param dealer_id: a specific dealer
        :param vehicle_type: a specific vehicle type
        :param obj_contribution: the profit contribution of a specific load pattern
        :param load_pattern: a specific load pattern
        :return: None
        """
        pattern_column = gp.Column()
        
        for duet_sku_day, num in load_pattern.items():
            sku_id, day = duet_sku_day
            load_num = [num]
            cons_dealer_demands_impacted = [self.cons_dealer_demands[dealer_id, sku_id, day]]
            pattern_column.addTerms(load_num, cons_dealer_demands_impacted)
            

            cons_sku_transfer_impacted = [self.cons_sku_transfer[fact_id, sku_id, day]]
            pattern_column.addTerms(load_num, cons_sku_transfer_impacted)

            d_xs_name = f"d_xs_{fact_id}_{dealer_id}_{vehicle_type}_{day}_{self.pattern_id}"
            self.d_xs[fact_id, dealer_id, vehicle_type, day, self.pattern_id] = (
                self.model.addVar(lb=0, obj=obj_contribution, vtype=GRB.CONTINUOUS, 
                                  column=pattern_column, name=d_xs_name)
            )

            self.pattern_content[d_xs_name] = load_pattern

            self.pattern_id += 1
            self.num_patterns += 1
            self.model.update()

    def shrink(self):
        """
        remove load patterns when the number of load patterns is greater than predefined value.
        """
        log_printer.print("Start to shrink master model...")
        remove_pattern_keys = [key for key in self.d_xs if self.d_xs[key].X == 0]
        remove_pattern_nums = len(remove_pattern_keys)
        random.shuffle(remove_pattern_keys)
        while remove_pattern_nums > 0 and self.num_patterns > self.data.param_pattern_num_shrink_to:
            remove_key = remove_pattern_keys.pop()
            self.model.remove(self.d_xs[remove_key])
            self.d_xs.pop(remove_key)
            remove_pattern_nums -= 1
            self.num_patterns -= 1
        self.model.update()
        log_printer.print("Shrink Done!")

    def initialize_solution(self):
        triples = {(plant, dealer, day) for (plant, dealer) in self.data.construct_supply_chain()
                   for day in range(1, self.data.horizons + 1)}
        for (plant, dealer, day) in triples:
            # 取得最大容量的车型
            max_veh_type = max(self.all_veh_types, key=float)
            selected_vehicle = Vehicle(plant, dealer, max_veh_type, day, self.data)
            # 当前经销商需要的SKU种类和数量
            orders = {sku_id: qty for (dealer_id, sku_id, t), qty in self.demands.items() 
                      if dealer_id == dealer and t == day}
            for sku_id, qty in orders.items():
                # 生产基地可以提供经销商需要的SKU
                if sku_id in self.skus_plant[plant]:
                    while qty > 0:
                        qty = selected_vehicle.load(sku_id, qty)
                        if qty > 0:
                            self.insert_pattern(plant, dealer, max_veh_type,
                                                self.data.veh_type_cost[max_veh_type], selected_vehicle.cargo)
                            selected_vehicle.clear()
            if not selected_vehicle.is_empty():
                self.insert_pattern(plant, dealer, max_veh_type,
                                    self.data.veh_type_cost[max_veh_type], selected_vehicle.cargo)
                selected_vehicle.clear()
        pass
    
    def run(self):
        # log_printer.print("Optimizing master model...")
        self.model.optimize()
        self.model.write("MasterModel.lp")

        if self.model.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            log_printer.print("Exception: Master Model Is Infeasible or Unbounded!", color='bold red')
            self.model.computeIIS()
            self.model.write("Infeasible_MasterModel.ilp")

            if self.model.IISMinimal:  # 判断 model.computeIIS() 返回的是否是包含不可行约束条件数目最少的IIS
                log_printer.print('IIS is minimal \n')
            else:
                log_printer.print('IIS is not minimal \n')
            log_printer.print('\n The following constraint (s) cannot be satisfied: ')
            for c in self.model.getConstrs():
                if c.IISConstr:
                    log_printer.print('%s' % c.ConstrName)
            return False
        else:
            self.print_sol_summary()
            return True

    def print_sol_summary(self):
        real_obj = sum(var.X for var in self.d_xs.values())
        log_printer.print(
            f"\tRealObj={real_obj:.1f}"
            f"\tNo. Alpha={self.get_alpha_size()}"
            f"\tNo. Beta={self.get_beta_size()}"
            f"\tNo. Gamma={self.get_gamma_size()}"
            f"\tNo. Pattern={self.num_patterns}"
            f"\tTotalObj={self.model.ObjVal:.2f}"
        )
