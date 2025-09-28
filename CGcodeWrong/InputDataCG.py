# 定义多周期优化问题的数据接口
# 整理保存在datasets//multiple-periods文件夹中的数据
# 将每个数据集中提供的数据，按照你需要的方式呈现
import os
import json
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set

@dataclass
class InputDataCG:
    input_file_loc: str
    output_file_loc: str
    # specify which dataset to load
    dataset_name: str

    # 对于多周期问题，默认规划周期长度为 1
    horizons = 1
    # 默认从第一天开始规划
    day = 1

    # 设定环境参数
    # 主模型中对第一类人工变量的惩罚系数
    param_pun_factor1: int = field(init=False)
    # 主模型中对第二类人工变量的惩罚系数
    param_pun_factor2: int = field(init=False)
    # 主模型中对第三类人工变量的惩罚系数
    param_pun_factor3: int = field(init=False)
    # 主模型中对第四类人工变量的惩罚系数
    # param_pun_factor4: int = field(init=False)

    # 列生成算法中，每次减少多少个装载模式
    param_pattern_batch: int = field(init=False)
    # 列生成算法中，要保留的装载模式数量
    param_pattern_num_shrink_to: int = field(init=False)
    # 列生成算法中，要保留的装载模式的最大数量上限，当超过这个上限时，开始删除多余的列
    param_trigger_shrink_level: int = field(init=False)

    # MIP模型的求解gap，当达到设定的gap后，模型停止，默认值为0
    param_gap_limit: float = field(init=False)
    # MIP模型求解的时间限制，默认值为60
    param_time_limit: float = field(init=False)
    # 是否输出gurobi的求解日志
    param_output_flag: int = field(init=False)

    # 定义需要的数据接口，看数学模型中的符号定义
    # 生产基地的集合
    plants: Set[str] = field(init=False)
    # 经销商的集合
    dealers: Set[str] = field(init=False)
    # 所有SKU的集合
    all_skus: Set[str] = field(init=False)
    # 所有车辆类型集合
    all_veh_types: Set[str] = field(init=False)
    # 每种SKU的体积量化数值 {sku: size}
    sku_sizes: Dict[str, int] = field(init=False)

    # 在每个周期内，生产基地计划生产的SKU数量 {(plant, sku, day): qty}
    sku_prod_each_day: Dict[Tuple[str, str, int], int] = field(init=False)
    
    # 在每个周期内，所有生产基地计划生产的SKU数量 {(sku_id, day): qty}
    sku_prod_total: Dict[Tuple[str, int], int] = field(init=False)
    
    # 生产基地中SKU的期初库存数量 {(plant, sku): qty}
    sku_initial_inv: Dict[Tuple[str, str], int] = field(init=False)
    # 生产基地期初拥有的SKU集合
    skus_initial: Dict[str, Set[str]] = field(init=False)
    # 生产基地需要生产的SKU集合
    skus_prod: Dict[str, Set[str]] = field(init=False)
    # 生产基地可以提供的SKU集合
    skus_plant: Dict[str, Set[str]] = field(init=False)

    # 经销商对SKU的需求 {(dealer, sku, day): qty}
    demands: Dict[Tuple[str, str], int] = field(init=False)
    # 经销商需要的SKU集合
    skus_dealer: Dict[str, Set[str]] = field(init=False)
    
    #　生产基地 i 中SKU k 转移到周期 t 的数量 {(plant, sku, day): qty}
    # 这个数据接口用于在主模型中，更新决策变量s_ikt的值
    historical_s_ikt: Dict[Tuple[str, str, int], int] = field(init=False)

    # 每一种车辆类型的使用成本 {v_type: cost}
    veh_type_cost: Dict[str, int] = field(init=False)
    # 每一种车辆类型的容量上限 {v_type: capacity}
    veh_type_cap: Dict[str, int] = field(init=False)
    # 每一种车辆类型的最小起运量 {v_type: min_load}
    veh_type_min_load: Dict[str, int] = field(init=False)
    # 每个生产基地的库存上限 {plant: max_cap}
    plant_inv_limit: Dict[str, int] = field(init=False)

    df_order: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 到这里，列生成模型需要的所有数据接口，已经定义完毕
    # 接下来，要做的事情是读取并整理数据

    # 开头带双下划线，用于对象的数据封装，以此命名的属性或者方法为类的私有属性或者私有方法，用于隐藏数据
    # 但这种机制并不是很严格，类中所有以双下划线开头的名称都会自动变为"_类名__name"的新名称
    # 这种机制可以阻止继承类重新定义或者更改方法的实现
    # 这个函数用于给dataframe中每一列重新命名
    def __reorganize_dataframe(self, df_input: pd.DataFrame, title_map):
        ordered_cols = list(title_map.values())
        df_input.rename(columns=title_map, inplace=True)
        df_input.reindex(columns=ordered_cols)

    def load(self):
        input_file_loc = self.input_file_loc
        # reading param setting from json file
        with open(os.path.join(input_file_loc, 'model_config_cg.json')) as fp:
            param = json.load(fp)

        self.param_pun_factor1 = param['pun_factor1']
        self.param_pun_factor2 = param['pun_factor2']
        self.param_pun_factor3 = param['pun_factor3']
        # self.param_pun_factor4 = param['pun_factor4']
        self.param_pattern_batch = param['pattern_batch']
        self.param_pattern_num_shrink_to = param['pattern_num_shrink_to']
        self.param_trigger_shrink_level = param['trigger_shrink_level']

        self.param_gap_limit = param['gap_limit']
        self.param_time_limit = param['time_limit']
        self.param_output_flag = param['output_flag']

        # multiple-factor for sku size and vehicle capacity
        multiple_factor = 100
        # reading data from each dataset
        input_file_loc = os.path.join(input_file_loc, self.dataset_name)

        input_file1 = os.path.join(input_file_loc, "order.csv")
        input_file2 = os.path.join(input_file_loc, "plant_size_upper_limit.csv")
        input_file3 = os.path.join(input_file_loc, "product_size.csv")
        input_file4 = os.path.join(input_file_loc, "vehicle.csv")
        input_file5 = os.path.join(input_file_loc, "warehouse_production.csv")
        input_file6 = os.path.join(input_file_loc, "warehouse_storage.csv")

        # 经销商对SKU的需求信息
        df_order = pd.read_csv(input_file1, header=0)
        # 生产基地的库存容量上限
        df_plant_cap = pd.read_csv(input_file2, header=0)
        # SKU的体积信息
        df_sku_size = pd.read_csv(input_file3, header=0)
        # 所有车辆类型信息，包括车辆容量、最小起运量、使用成本
        df_all_veh = pd.read_csv(input_file4, header=0)
        # 生产基地在每个周期内生产的SKU数量
        df_sku_prod = pd.read_csv(input_file5, header=0)
        # 生产基地中SKU的期初库存
        df_sku_inv = pd.read_csv(input_file6, header=0)

        # 整理order.csv里面的数据
        title_map = {'client_code': 'dealer_id', 'product_code': 'sku_id', 'required_date': 'day', 'volume': 'order_qty'}
        self.__reorganize_dataframe(df_order, title_map)
        df_order['dealer_id'] = df_order['dealer_id'].astype(str)
        df_order['sku_id'] = df_order['sku_id'].astype(str)
        df_order['day'] = df_order['day'].astype(int)
        df_order['order_qty'] = df_order['order_qty'].astype(int)

        # 整理plant_size_upper_limit.csv里面的数据
        title_map = {'plant_code': 'fact_id', 'max_volume': 'max_cap'}
        self.__reorganize_dataframe(df_plant_cap, title_map)
        df_plant_cap['fact_id'] = df_plant_cap['fact_id'].astype(str)
        df_plant_cap['max_cap'] = df_plant_cap['max_cap'].astype(int)

        # 整理product_size.csv里面的数据
        title_map = {'product_code': 'sku_id', 'standard_size': 'size'}
        self.__reorganize_dataframe(df_sku_size, title_map)
        df_sku_size['sku_id'] = df_sku_size['sku_id'].astype(str)
        df_sku_size['size'] = df_sku_size['size'].astype(float)

        # 整理vehicle.csv里面的数据
        title_map = {'vehicle_type': 'vehicle_type', 'carry_standard_size': 'capacity',
                     'min_standard_size': 'min_load', 'cost_to_use': 'cost'}
        self.__reorganize_dataframe(df_all_veh, title_map)
        df_all_veh['vehicle_type'] = df_all_veh['vehicle_type'].astype(str)
        df_all_veh['capacity'] = df_all_veh['capacity'].astype(int)
        df_all_veh['min_load'] = df_all_veh['min_load'].astype(int)

        # 整理warehouse_production.csv里面的数据
        title_map = {'plant_code': 'fact_id', 'product_code': 'sku_id', 'produce_date': 'day', 'volume': 'prod_qty'}
        self.__reorganize_dataframe(df_sku_prod, title_map)
        df_sku_prod['fact_id'] = df_sku_prod['fact_id'].astype(str)
        df_sku_prod['sku_id'] = df_sku_prod['sku_id'].astype(str)
        df_sku_prod['day'] = df_sku_prod['day'].astype(int)
        df_sku_prod['prod_qty'] = df_sku_prod['prod_qty'].astype(int)

        # 整理warehouse_storage.csv里面的数据
        title_map = {'plant_code': 'fact_id', 'product_code': 'sku_id', 'volume': 'inv'}
        self.__reorganize_dataframe(df_sku_inv, title_map)
        df_sku_inv['fact_id'] = df_sku_inv['fact_id'].astype(str)
        df_sku_inv['sku_id'] = df_sku_inv['sku_id'].astype(str)
        df_sku_inv['inv'] = df_sku_inv['inv'].astype(int)

        # 在整理好数据之后，接下来需要计算出之前定义好的数据接口
        # 计算数据集中的最大规划周期
        if len(df_sku_prod.index) > 0:
            self.horizons = df_sku_prod['day'].max()
        plants = list(df_plant_cap['fact_id'].unique())
        self.plants = {plant for plant in plants}

        dealers = list(df_order['dealer_id'].unique())
        self.dealers = {dealer for dealer in dealers}

        all_skus = list(df_sku_size['sku_id'].unique())
        self.all_skus = {sku for sku in all_skus}

        all_veh_types = list(df_all_veh['vehicle_type'].unique())
        self.all_veh_types = {v_type for v_type in all_veh_types}

        # {sku_id: size}
        self.sku_sizes = dict(zip(df_sku_size['sku_id'], df_sku_size['size']))
        # {(plant, sku, day): prod_qty}
        self.sku_prod_each_day = df_sku_prod.set_index(['fact_id', 'sku_id', 'day'])['prod_qty'].to_dict()
        
        # {(sku_id, day): prod_qty}
        self.sku_prod_total = df_sku_prod.groupby(['sku_id', 'day'])['prod_qty'].sum().to_dict()
        
        # {(plant, sku): inv}
        self.sku_initial_inv = df_sku_inv.set_index(['fact_id', 'sku_id'])['inv'].to_dict()
        
        self.historical_s_ikt = {}
        for (plant, sku), inv in self.sku_initial_inv.items():
            self.historical_s_ikt[plant, sku, 0] = inv
            
        # {plant: {sku}}
        self.skus_initial = {}
        for (plant, sku) in self.sku_initial_inv:
            if plant not in self.skus_initial:
                self.skus_initial[plant] = set()
            self.skus_initial[plant].add(sku)
        
        self.skus_prod = {}
        for (plant, sku, day) in self.sku_prod_each_day:
            if plant not in self.skus_prod:
                self.skus_prod[plant] = set()
            self.skus_prod[plant].add(sku)
                
        
        # 这里需要注意的是，在计算生产基地可以提供的SKU集合时，需要同时检查期初库存和计划生产库存
        self.skus_plant = self.skus_initial | self.skus_prod

        # {(dealer, sku, day): demand}
        self.demands = df_order.set_index(['dealer_id', 'sku_id', 'day'])['order_qty'].to_dict()
        # {dealer: {sku}}
        self.skus_dealer = {}
        for (dealer, sku, day) in self.demands:
            if dealer not in self.skus_dealer:
                self.skus_dealer[dealer] = set()
            self.skus_dealer[dealer].add(sku)

        # {v_type: cost}
        self.veh_type_cost = dict(zip(df_all_veh['vehicle_type'], df_all_veh['cost']))
        # {v_type: cap}
        self.veh_type_cap = dict(zip(df_all_veh['vehicle_type'], df_all_veh['capacity']))
        # {v_type: min_load}
        self.veh_type_min_load = dict(zip(df_all_veh['vehicle_type'], df_all_veh['min_load']))

        # {plant: max_cap}
        self.plant_inv_limit = dict(zip(df_plant_cap['fact_id'], df_plant_cap['max_cap']))


    def available_skus_in_plant(self, plant_id: str):
        """
        find skus available in a specific plant
        """
        skus = {sku_id for fact_id, skus in self.skus_plant.items() for sku_id in skus if fact_id == plant_id}
        return skus

    def plan_dealer_skus(self, plant_id: str):
        """
        find dealer and sku tuples needed to be satisfied by a specific plant
        """
        # duet_dealer_sku = {
        #     (dealer, sku) for (dealer, sku) in self.demands for (plant, sku_id) in self.sku_inv
        #     if plant == plant_id and sku == sku_id
        # }
        duet_dealer_sku = {
            (dealer, sku) for (plant, dealer), skus_available in self.construct_supply_chain().items()
            for sku in skus_available if plant == plant_id
        }
        return duet_dealer_sku

    def available_skus_to_dealer(self, plant_id: str, dealer_id: str):
        """
        find skus available to a specific dealer in a specific plant
        """
        require_skus = {sku_id for (dealer, sku_id) in self.plan_dealer_skus(plant_id) if dealer == dealer_id}
        return require_skus

    def construct_supply_chain(self):
        """
        construct supply chain between plants and dealers
        """
        supply_chain = defaultdict(set)
        for plant, skus_supply in self.skus_plant.items():
            for dealer, skus_needed in self.skus_dealer.items():
                for sku in skus_needed:
                    if sku in skus_supply:
                        supply_chain[plant, dealer].add(sku)
        
        supply_chain = dict(supply_chain)
        return supply_chain
