# 定义多周期优化问题的数据接口
# 整理保存在datasets//multiple-periods文件夹中的数据
# 将每个数据集中提供的数据，按照你需要的方式呈现
import os
import json
import logging
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Union, Optional

@dataclass
class DataALNS:
    input_file_loc: str
    output_file_loc: str
    # specify which dataset to load
    dataset_name: str

    # 对于多周期问题，默认规划周期长度为 1
    horizons: int = 1
    # 默认从第一天开始规划
    day: int = 1

    # 惩罚因子
    param_pun_factor1: float = field(init=False)
    param_pun_factor2: float = field(init=False)
    param_pun_factor3: float = field(init=False)
    param_pun_factor4: float = field(init=False)
    param_pun_objective: float = field(init=False)

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
    sku_sizes: Dict[str, float] = field(init=False)

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

    # 经销商对SKU的需求 {(dealer, sku): qty}
    # 这里指的是当前经销商在所有周期内对SKU k 的总需求
    demands: Dict[Tuple[str, str], int] = field(init=False)
    # 经销商需要的SKU集合
    skus_dealer: Dict[str, Set[str]] = field(init=False)

    #　生产基地 i 中SKU k 转移到周期 t 的数量 {(plant, sku, day): qty}
    # 这个数据接口用于 更新决策变量s_ikt的值
    historical_s_ikt: Dict[Tuple[str, str, int], int] = field(init=False)

    # 每一种车辆类型的使用成本 {v_type: cost}
    veh_type_cost: Dict[str, float] = field(init=False)
    # 每一种车辆类型的容量上限 {v_type: capacity}
    veh_type_cap: Dict[str, float] = field(init=False)
    # 每一种车辆类型的最小起运量 {v_type: min_load}
    veh_type_min_load: Dict[str, float] = field(init=False)
    # 每个生产基地的库存上限 {plant: max_cap}
    plant_inv_limit: Dict[str, float] = field(init=False)

    df_order: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 到这里，列生成模型需要的所有数据接口，已经定义完毕
    # 接下来，要做的事情是读取并整理数据

    def __post_init__(self):
        """初始化后设置日志记录器（仅获取，不添加handler，避免重复日志）"""
        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_file_exists(self, file_path: str, file_description: str) -> bool:
        """验证文件是否存在"""
        if not os.path.exists(file_path):
            self.logger.error(f"{file_description} 文件不存在: {file_path}")
            return False
        return True

    def _load_csv_safely(self, file_path: str, file_description: str) -> Optional[pd.DataFrame]:
        """安全地加载CSV文件，并将所有主键列统一为str类型，防止merge类型不一致"""
        if not self._validate_file_exists(file_path, file_description):
            return None
        try:
            df = pd.read_csv(file_path, header=0)
            # 强制将所有可能作为key的列转为str，防止merge类型不一致
            key_cols = [col for col in ['product_code', 'plant_code', 'client_code', 'vehicle_type'] if col in df.columns]
            for col in key_cols:
                df[col] = df[col].astype(str)
            self.logger.info(f"成功加载 {file_description}: {len(df)} 行数据")
            return df
        except Exception as e:
            self.logger.error(f"加载 {file_description} 失败: {e}")
            return None

    def _reorganize_dataframe(self, df_input: pd.DataFrame, title_map: Dict[str, str]) -> pd.DataFrame:
        """重新组织DataFrame的列名和顺序"""
        if df_input is None or df_input.empty:
            self.logger.warning("输入DataFrame为空，跳过重新组织")
            return df_input
        
        # 检查所有必需的列是否存在
        missing_cols = set(title_map.keys()) - set(df_input.columns)
        if missing_cols:
            self.logger.error(f"DataFrame缺少必需的列: {missing_cols}")
            raise ValueError(f"DataFrame缺少必需的列: {missing_cols}")
        
        # 重命名列
        df_result = df_input.rename(columns=title_map)
        
        # 重新排序列（仅包含映射后的列）
        ordered_cols = list(title_map.values())
        df_result = df_result.reindex(columns=ordered_cols)
        
        return df_result

    def _convert_dataframe_types(self, df: pd.DataFrame, type_mapping: Dict[str, type]) -> pd.DataFrame:
        """安全地转换DataFrame的数据类型，并确保所有key列为str类型"""
        if df is None or df.empty:
            return df
        df_result = df.copy()
        # 先强制所有key列为str
        key_cols = [col for col in ['product_code', 'plant_code', 'client_code', 'vehicle_type'] if col in df_result.columns]
        for col in key_cols:
            df_result[col] = df_result[col].astype(str)
        for col, dtype in type_mapping.items():
            if col in df_result.columns:
                try:
                    if dtype == str:
                        df_result[col] = df_result[col].astype(str)
                    elif dtype == int:
                        df_result[col] = pd.to_numeric(df_result[col], errors='coerce').fillna(0).astype(int)
                    elif dtype == float:
                        df_result[col] = pd.to_numeric(df_result[col], errors='coerce').fillna(0.0)
                except Exception as e:
                    self.logger.warning(f"转换列 {col} 到类型 {dtype} 失败: {e}")
        return df_result

    def load(self):
        """加载和处理所有数据文件"""
        try:
            # 1. 加载配置参数
            self._load_parameters()
            
            # 2. 构建文件路径
            file_paths = self._construct_file_paths()
            
            # 3. 加载所有CSV文件
            dataframes = self._load_all_dataframes(file_paths)
            
            # 4. 处理和验证数据
            self._process_dataframes(dataframes)
            
            # 5. 计算派生的数据接口
            self._compute_derived_data()
            
            # 6. 验证数据完整性
            self._validate_data_integrity()
            
            self.logger.info("数据加载和处理完成")
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise

    def _load_parameters(self):
        """加载配置参数"""
        config_file = os.path.join(self.input_file_loc, 'model_config_alns.json')
        if not self._validate_file_exists(config_file, "配置文件"):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as fp:
                param = json.load(fp)
            
            self.param_pun_factor1 = float(param.get('pun_factor1', 0))
            self.param_pun_factor2 = float(param.get('pun_factor2', 0))
            self.param_pun_factor3 = float(param.get('pun_factor3', 0))
            self.param_pun_factor4 = float(param.get('pun_factor4', 0))
            self.param_pun_objective = float(param.get('pun_objective', 0))
            
            self.logger.info("配置参数加载成功")
            
        except Exception as e:
            self.logger.error(f"加载配置参数失败: {e}")
            raise

    def _construct_file_paths(self) -> Dict[str, str]:
        """构建所有数据文件的路径"""
        dataset_path = os.path.join(self.input_file_loc, self.dataset_name)
        
        file_paths = {
            'order': os.path.join(dataset_path, "order.csv"),
            'plant_cap': os.path.join(dataset_path, "plant_size_upper_limit.csv"),
            'sku_size': os.path.join(dataset_path, "product_size.csv"),
            'vehicle': os.path.join(dataset_path, "vehicle.csv"),
            'production': os.path.join(dataset_path, "warehouse_production.csv"),
            'inventory': os.path.join(dataset_path, "warehouse_storage.csv")
        }
        
        return file_paths

    def _load_all_dataframes(self, file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """加载所有CSV文件到DataFrame"""
        dataframes = {}
        file_descriptions = {
            'order': '订单需求信息',
            'plant_cap': '生产基地库存容量上限',
            'sku_size': 'SKU体积信息',
            'vehicle': '车辆类型信息',
            'production': '生产基地生产计划',
            'inventory': '生产基地期初库存'
        }
        
        for key, file_path in file_paths.items():
            df = self._load_csv_safely(file_path, file_descriptions[key])
            if df is None:
                raise FileNotFoundError(f"必需文件 {file_descriptions[key]} 加载失败")
            dataframes[key] = df
        
        return dataframes

    def _process_dataframes(self, dataframes: Dict[str, pd.DataFrame]):
        """处理所有DataFrame的格式化和类型转换"""
        
        # 处理订单数据
        title_map = {'client_code': 'dealer_id', 'product_code': 'sku_id', 'volume': 'order_qty'}
        type_mapping = {'dealer_id': str, 'sku_id': str, 'order_qty': int}
        df_order = self._reorganize_dataframe(dataframes['order'], title_map)
        self.df_order = self._convert_dataframe_types(df_order, type_mapping)

        # 处理生产基地容量数据
        title_map = {'plant_code': 'fact_id', 'max_volume': 'max_cap'}
        type_mapping = {'fact_id': str, 'max_cap': float}
        df_plant_cap = self._reorganize_dataframe(dataframes['plant_cap'], title_map)
        self._df_plant_cap = self._convert_dataframe_types(df_plant_cap, type_mapping)

        # 处理SKU尺寸数据
        title_map = {'product_code': 'sku_id', 'standard_size': 'size'}
        type_mapping = {'sku_id': str, 'size': float}
        df_sku_size = self._reorganize_dataframe(dataframes['sku_size'], title_map)
        self._df_sku_size = self._convert_dataframe_types(df_sku_size, type_mapping)

        # 处理车辆类型数据
        title_map = {'vehicle_type': 'vehicle_type', 'carry_standard_size': 'capacity',
                     'min_standard_size': 'min_load', 'cost_to_use': 'cost'}
        type_mapping = {'vehicle_type': str, 'capacity': float, 'min_load': float, 'cost': float}
        df_all_veh = self._reorganize_dataframe(dataframes['vehicle'], title_map)
        self._df_all_veh = self._convert_dataframe_types(df_all_veh, type_mapping)

        # 处理生产计划数据
        title_map = {'plant_code': 'fact_id', 'product_code': 'sku_id', 'produce_date': 'day', 'volume': 'prod_qty'}
        type_mapping = {'fact_id': str, 'sku_id': str, 'day': int, 'prod_qty': int}
        df_sku_prod = self._reorganize_dataframe(dataframes['production'], title_map)
        self._df_sku_prod = self._convert_dataframe_types(df_sku_prod, type_mapping)

        # 处理库存数据
        title_map = {'plant_code': 'fact_id', 'product_code': 'sku_id', 'volume': 'inv'}
        type_mapping = {'fact_id': str, 'sku_id': str, 'inv': int}
        df_sku_inv = self._reorganize_dataframe(dataframes['inventory'], title_map)
        self._df_sku_inv = self._convert_dataframe_types(df_sku_inv, type_mapping)

    def _compute_derived_data(self):
        """计算派生的数据接口"""
        
        # 计算规划周期
        if len(self._df_sku_prod) > 0:
            self.horizons = int(self._df_sku_prod['day'].max())
        
        # 构建基础集合
        self.plants = set(self._df_plant_cap['fact_id'].unique())
        self.dealers = set(self.df_order['dealer_id'].unique())
        self.all_skus = set(self._df_sku_size['sku_id'].unique())
        self.all_veh_types = set(self._df_all_veh['vehicle_type'].unique())

        # 构建字典映射
        self.sku_sizes = dict(zip(self._df_sku_size['sku_id'], self._df_sku_size['size']))
        
        # 生产数据处理
        self.sku_prod_each_day = self._df_sku_prod.set_index(['fact_id', 'sku_id', 'day'])['prod_qty'].to_dict()
        self.sku_prod_total = self._df_sku_prod.groupby(['sku_id', 'day'])['prod_qty'].sum().to_dict()
        
        # 库存数据处理
        self.sku_initial_inv = self._df_sku_inv.set_index(['fact_id', 'sku_id'])['inv'].to_dict()
        
        # 历史库存数据
        self.historical_s_ikt = {}
        for (plant, sku), inv in self.sku_initial_inv.items():
            self.historical_s_ikt[plant, sku, 0] = inv
        
        # 构建SKU集合映射
        self._compute_sku_mappings()
        
        # 需求数据处理
        self.demands = self.df_order.set_index(['dealer_id', 'sku_id'])['order_qty'].to_dict()
        self.skus_dealer = {}
        for (dealer, sku) in self.demands:
            if dealer not in self.skus_dealer:
                self.skus_dealer[dealer] = set()
            self.skus_dealer[dealer].add(sku)

        # 车辆数据处理
        self.veh_type_cost = dict(zip(self._df_all_veh['vehicle_type'], self._df_all_veh['cost']))
        self.veh_type_cap = dict(zip(self._df_all_veh['vehicle_type'], self._df_all_veh['capacity']))
        self.veh_type_min_load = dict(zip(self._df_all_veh['vehicle_type'], self._df_all_veh['min_load']))

        # 生产基地容量限制
        self.plant_inv_limit = dict(zip(self._df_plant_cap['fact_id'], self._df_plant_cap['max_cap']))

    def _compute_sku_mappings(self):
        """计算SKU映射关系"""
        
        # 计算期初库存SKU集合
        self.skus_initial = defaultdict(set)
        for (plant, sku) in self.sku_initial_inv:
            self.skus_initial[plant].add(sku)
        self.skus_initial = dict(self.skus_initial)
        
        # 计算生产SKU集合
        self.skus_prod = defaultdict(set)
        for (plant, sku, day) in self.sku_prod_each_day:
            self.skus_prod[plant].add(sku)
        self.skus_prod = dict(self.skus_prod)
        
        # 计算可提供SKU集合（修复原有逻辑错误）
        self.skus_plant = {}
        for plant in self.plants:
            initial_skus = self.skus_initial.get(plant, set())
            prod_skus = self.skus_prod.get(plant, set())
            self.skus_plant[plant] = initial_skus.union(prod_skus)

    def _validate_data_integrity(self):
        """验证数据完整性"""
        errors = []
        
        # 检查基础数据是否为空
        if not self.plants:
            errors.append("没有找到生产基地数据")
        if not self.dealers:
            errors.append("没有找到经销商数据")
        if not self.all_skus:
            errors.append("没有找到SKU数据")
        if not self.all_veh_types:
            errors.append("没有找到车辆类型数据")
        
        # 检查数据一致性
        order_skus = set(sku for dealer, sku in self.demands.keys())
        missing_sku_sizes = order_skus - set(self.sku_sizes.keys())
        if missing_sku_sizes:
            errors.append(f"订单中的SKU缺少尺寸信息: {missing_sku_sizes}")
        
        # 检查生产基地是否有可提供的SKU
        plants_with_skus = set(self.skus_plant.keys())
        plants_without_skus = self.plants - plants_with_skus
        if plants_without_skus:
            self.logger.warning(f"以下生产基地没有可提供的SKU: {plants_without_skus}")
        
        if errors:
            error_msg = "数据完整性验证失败:\n" + "\n".join(f"- {error}" for error in errors)
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info("数据完整性验证通过")


    def available_skus_in_plant(self, plant_id: str) -> Set[str]:
        """
        获取指定生产基地可提供的所有SKU
        
        Args:
            plant_id: 生产基地ID
            
        Returns:
            该生产基地可提供的SKU集合
        """
        if plant_id not in self.plants:
            self.logger.warning(f"生产基地 {plant_id} 不存在于数据中")
            return set()
        
        return self.skus_plant.get(plant_id, set())

    def plan_dealer_skus(self, plant_id: str) -> Set[Tuple[str, str]]:
        """
        获取指定生产基地需要满足的经销商-SKU组合
        
        Args:
            plant_id: 生产基地ID
            
        Returns:
            (经销商ID, SKU ID) 元组的集合
        """
        if plant_id not in self.plants:
            self.logger.warning(f"生产基地 {plant_id} 不存在于数据中")
            return set()
        
        supply_chain = self.construct_supply_chain()
        dealer_sku_pairs = set()
        
        for (plant, dealer), skus_available in supply_chain.items():
            if plant == plant_id:
                for sku in skus_available:
                    dealer_sku_pairs.add((dealer, sku))
        
        return dealer_sku_pairs

    def available_skus_to_dealer(self, plant_id: str, dealer_id: str) -> Set[str]:
        """
        获取指定生产基地可向指定经销商提供的SKU
        
        Args:
            plant_id: 生产基地ID
            dealer_id: 经销商ID
            
        Returns:
            可提供的SKU集合
        """
        if plant_id not in self.plants:
            self.logger.warning(f"生产基地 {plant_id} 不存在于数据中")
            return set()
        
        if dealer_id not in self.dealers:
            self.logger.warning(f"经销商 {dealer_id} 不存在于数据中")
            return set()
        
        supply_chain = self.construct_supply_chain()
        return supply_chain.get((plant_id, dealer_id), set())

    def construct_supply_chain(self) -> Dict[Tuple[str, str], Set[str]]:
        """
        构建生产基地和经销商之间的供应链关系
        
        Returns:
            {(生产基地ID, 经销商ID): 可供应的SKU集合} 的字典
        """
        if not hasattr(self, '_supply_chain_cache'):
            supply_chain = defaultdict(set)
            
            for plant, skus_supply in self.skus_plant.items():
                for dealer, skus_needed in self.skus_dealer.items():
                    # 找到交集：该生产基地可提供且该经销商需要的SKU
                    common_skus = skus_supply.intersection(skus_needed)
                    if common_skus:
                        supply_chain[plant, dealer] = common_skus
            
            self._supply_chain_cache = dict(supply_chain)
        
        return self._supply_chain_cache

    # 新增的实用方法
    def get_total_demand_for_sku(self, sku_id: str) -> int:
        """
        获取指定SKU的总需求量
        
        Args:
            sku_id: SKU ID
            
        Returns:
            总需求量
        """
        total_demand = 0
        for (dealer, sku), demand in self.demands.items():
            if sku == sku_id:
                total_demand += demand
        
        return total_demand

    def get_total_production_for_sku(self, sku_id: str, day: Optional[int] = None) -> int:
        """
        获取指定SKU的总生产量
        
        Args:
            sku_id: SKU ID
            day: 指定天数，如果为None则返回所有天数的总和
            
        Returns:
            总生产量
        """
        total_production = 0
        
        if day is not None:
            # 获取指定天数的生产量
            total_production = self.sku_prod_total.get((sku_id, day), 0)
        else:
            # 获取所有天数的生产量
            for (sku, d), prod_qty in self.sku_prod_total.items():
                if sku == sku_id:
                    total_production += prod_qty
        
        return total_production

    def get_total_initial_inventory_for_sku(self, sku_id: str) -> int:
        """
        获取指定SKU的总期初库存量
        
        Args:
            sku_id: SKU ID
            
        Returns:
            总期初库存量
        """
        total_inventory = 0
        for (plant, sku), inv in self.sku_initial_inv.items():
            if sku == sku_id:
                total_inventory += inv
        
        return total_inventory

    def get_sku_supply_demand_balance(self, sku_id: str) -> Dict[str, Union[int, float]]:
        """
        获取指定SKU的供需平衡信息
        
        Args:
            sku_id: SKU ID
            
        Returns:
            包含总需求、总供应、平衡状态的字典
        """
        total_demand = self.get_total_demand_for_sku(sku_id)
        total_initial_inv = self.get_total_initial_inventory_for_sku(sku_id)
        total_production = self.get_total_production_for_sku(sku_id)
        total_supply = total_initial_inv + total_production
        
        balance = total_supply - total_demand
        balance_ratio = total_supply / total_demand if total_demand > 0 else float('inf')
        
        return {
            'sku_id': sku_id,
            'total_demand': total_demand,
            'total_initial_inventory': total_initial_inv,
            'total_production': total_production,
            'total_supply': total_supply,
            'balance': balance,
            'balance_ratio': balance_ratio,
            'status': 'surplus' if balance > 0 else 'deficit' if balance < 0 else 'balanced'
        }

    def get_summary_statistics(self) -> Dict[str, Union[int, float]]:
        """
        获取数据的汇总统计信息
        
        Returns:
            包含各种统计信息的字典
        """
        total_demand = sum(self.demands.values())
        total_initial_inv = sum(self.sku_initial_inv.values())
        total_production = sum(self.sku_prod_total.values())
        
        return {
            'num_plants': len(self.plants),
            'num_dealers': len(self.dealers),
            'num_skus': len(self.all_skus),
            'num_vehicle_types': len(self.all_veh_types),
            'planning_horizon': self.horizons,
            'total_demand': total_demand,
            'total_initial_inventory': total_initial_inv,
            'total_production': total_production,
            'total_supply': total_initial_inv + total_production,
            'supply_demand_ratio': (total_initial_inv + total_production) / total_demand if total_demand > 0 else float('inf')
        }
