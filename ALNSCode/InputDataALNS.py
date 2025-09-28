"""
InputDataALNS
=============

模块职责
    提供 ALNS 求解所需的全部静态输入数据读取 / 结构化 / 派生指标生成 / 一致性校验。
    该模块将多个原始 CSV / JSON (配置) 文件整合为 DataALNS 数据容器对象, 为后续
    初始解构建、算子 (destroy/repair)、库存演化、代价评估、ML 选择器提供统一的数据接口。

数据来源 (dataset 目录下 expected files)
    - model_config_alns.json                惩罚/目标参数 (惩罚系数、罚值等)
    - order.csv                             需求：经销商对 SKU 的订单量
    - plant_size_upper_limit.csv            生产基地库存容量上限
    - product_size.csv                      SKU 标准体积
    - vehicle.csv                           车辆类型 (容量 / 最小起运量 / 使用成本)
    - warehouse_production.csv              每日生产计划 (plant, sku, day, volume)
    - warehouse_storage.csv                 期初库存 (plant, sku, volume)

核心字段 (加载完成后保证存在)
    horizons                规划总周期 (由生产计划最大 day 推导；若无生产则保持默认 1)
    plants / dealers        生产基地集合, 经销商集合
    all_skus                SKU 集合
    all_veh_types           车辆类型集合
    sku_sizes               SKU → 体积
    sku_prod_each_day       (plant, sku, day) → 当日生产量
    sku_prod_total          (sku, day) → 所有基地该日生产量
    sku_initial_inv         (plant, sku) → 期初库存
    historical_s_ikt        (plant, sku, 0) → 期初库存 (为解初始化递推起点)
    skus_initial            plant → 期初库存中出现的 SKU 集
    skus_prod               plant → 生产计划中出现的 SKU 集
    skus_plant              plant → 可供 SKU 集 (期初 ∪ 生产)
    demands                 (dealer, sku) → 总需求量
    skus_dealer             dealer → 需求的 SKU 集
    veh_type_cost/cap/min_load
    plant_inv_limit         plant → 库存上限
    df_order                规范化后的订单 DataFrame (保留以便统计 / 调试)
    _supply_chain_cache     惰性缓存 {(plant, dealer): intersect_skus}

加载流程 (load()):
    1. _load_parameters               读取 JSON 配置 (惩罚参数)
    2. _construct_file_paths          构建该 dataset 的文件路径
    3. _load_all_dataframes           逐表安全读取 (类型统一, 关键列转 str)
    4. _process_dataframes            列重命名/重排/类型转换
    5. _compute_derived_data          推导集合/映射/派生指标
    6. _validate_data_integrity       校验缺失与一致性 (必要/警告)
    7. 成功则 DataALNS 即为可用状态

设计要点
    - 严格区分“原始 DataFrame” (_df_xxx) 与 “最终结构化字典”
    - 统一 key 列类型为 str, 避免 merge / dict 使用冲突
    - 增量构建 supply_chain (惰性缓存) 避免重复交集计算
    - 将需求/生产/库存/车辆数据清晰拆分，便于后续扩展 (如添加价格/区域等属性)
    - 不在本模块引入业务逻辑 (如装载/库存递推)，保持单一职责

异常与日志
    - 缺失必需文件 → raise
    - DataFrame 缺失必要列 → raise
    - 数据一致性警告 (如某 plant 无可供 SKU) → warning
    - 所有 IO/转换失败 → error + raise / 跳过 (视严重性)

可拓展建议 (未实现)
    - 加入 schema 校验 (pydantic / jsonschema)
    - 加入懒加载模式 (按需加载大型表)
    - 增加字段级缓存摘要 (hash) 便于调试版本差异
    - 支持多 dataset 合并 (跨区域/多业务线统一)

使用示例
    data = DataALNS(input_root, output_root, 'datasetA')
    data.load()
    print(data.horizons, len(data.demands), data.plant_inv_limit)

注意
    本次整理仅重构注释/文档/导入分组；逻辑实现保持不变。
"""

# =========================
# 标准库
# =========================
import os
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Union, Optional

# =========================
# 第三方库
# =========================
import pandas as pd


@dataclass
class DataALNS:
    """
    DataALNS
    --------
    ALNS 求解主程序统一依赖的数据容器。封装原始数据读取、转换、派生及校验过程。
    """
    input_file_loc: str
    output_file_loc: str
    dataset_name: str  # 指定数据集目录名 (子目录)

    # 规划周期: 默认 1 (加载后若生产计划包含更大 day 将被更新)
    horizons: int = 1
    # 当前起始日 (保留字段，若需滚动规划可扩展)
    day: int = 1

    # ==== 惩罚参数 (从配置文件加载) ====
    param_pun_factor1: float = field(init=False)
    param_pun_factor2: float = field(init=False)
    param_pun_factor3: float = field(init=False)
    param_pun_factor4: float = field(init=False)
    param_pun_objective: float = field(init=False)

    # ==== 基础集合 / 属性 ====
    plants: Set[str] = field(init=False)
    dealers: Set[str] = field(init=False)
    all_skus: Set[str] = field(init=False)
    all_veh_types: Set[str] = field(init=False)
    sku_sizes: Dict[str, float] = field(init=False)

    # 生产：逐日 & 汇总
    sku_prod_each_day: Dict[Tuple[str, str, int], int] = field(init=False)   # (plant, sku, day) → qty
    sku_prod_total: Dict[Tuple[str, int], int] = field(init=False)           # (sku, day) → aggregated qty

    # 库存
    sku_initial_inv: Dict[Tuple[str, str], int] = field(init=False)          # (plant, sku) → init qty
    skus_initial: Dict[str, Set[str]] = field(init=False)                    # plant → init SKUs
    skus_prod: Dict[str, Set[str]] = field(init=False)                       # plant → produced SKUs
    skus_plant: Dict[str, Set[str]] = field(init=False)                      # plant → available SKUs (init ∪ produced)

    # 需求
    demands: Dict[Tuple[str, str], int] = field(init=False)                  # (dealer, sku) → demand
    skus_dealer: Dict[str, Set[str]] = field(init=False)                     # dealer → demanded SKUs

    # 历史库存 (递推初值), 仅 day=0
    historical_s_ikt: Dict[Tuple[str, str, int], int] = field(init=False)    # (plant, sku, 0) → init inv

    # 车辆参数
    veh_type_cost: Dict[str, float] = field(init=False)
    veh_type_cap: Dict[str, float] = field(init=False)
    veh_type_min_load: Dict[str, float] = field(init=False)

    # 库存容量上限
    plant_inv_limit: Dict[str, float] = field(init=False)

    # 原始订单 DataFrame (规范化后存档，统计或调试使用)
    df_order: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ------------------------------------------------------------------
    # 初始化: 仅设置日志器 (避免重复添加 handler)
    # ------------------------------------------------------------------
    def __post_init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # 基础文件 / DataFrame 读取工具
    # ------------------------------------------------------------------
    def _validate_file_exists(self, file_path: str, file_description: str) -> bool:
        """验证文件是否存在 (不存在则记录错误并返回 False)"""
        if not os.path.exists(file_path):
            self.logger.error(f"{file_description} 文件不存在: {file_path}")
            return False
        return True

    def _load_csv_safely(
        self,
        file_path: str,
        file_description: str,
        required_cols: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        安全读取 CSV (增强版):
            - 存在性检查
            - 读取为 DataFrame
            - (新增) 必需列提前校验: required_cols 若有缺失立即报错 (早于后续重命名阶段)
            - 关键 key 列统一转 str
            - 异常捕获记录日志
        参数:
            file_path        文件路径
            file_description 日志描述
            required_cols    该原始文件应至少包含的原始列集合 (未重命名之前)
        返回:
            DataFrame 或 None (若文件不存在 / 读取失败)
        """
        if not self._validate_file_exists(file_path, file_description):
            return None
        try:
            df = pd.read_csv(file_path, header=0)
            # 早期必需列检查
            if required_cols:
                missing = set(required_cols) - set(df.columns)
                if missing:
                    self.logger.error(
                        f"{file_description} 缺少必需列: {missing} (文件: {file_path})"
                    )
                    raise ValueError(f"{file_description} 缺少必需列: {missing}")
            key_cols = [c for c in ['product_code', 'plant_code', 'client_code', 'vehicle_type'] if c in df.columns]
            for col in key_cols:
                df[col] = df[col].astype(str)
            self.logger.info(f"成功加载 {file_description}: {len(df)} 行数据")
            return df
        except Exception as e:
            self.logger.error(f"加载 {file_description} 失败: {e}")
            return None

    def _reorganize_dataframe(self, df_input: pd.DataFrame, title_map: Dict[str, str]) -> pd.DataFrame:
        """
        重新组织 DataFrame:
            - 列重命名 (title_map: 原名→新名)
            - 列顺序标准化 (仅保留映射后的列)
            - 缺失列抛异常
        """
        if df_input is None or df_input.empty:
            self.logger.warning("输入DataFrame为空，跳过重新组织")
            return df_input

        missing_cols = set(title_map.keys()) - set(df_input.columns)
        if missing_cols:
            self.logger.error(f"DataFrame缺少必需的列: {missing_cols}")
            raise ValueError(f"DataFrame缺少必需的列: {missing_cols}")

        df_result = df_input.rename(columns=title_map)
        ordered_cols = list(title_map.values())
        return df_result.reindex(columns=ordered_cols)

    def _convert_dataframe_types(self, df: pd.DataFrame, type_mapping: Dict[str, type]) -> pd.DataFrame:
        """
        将指定列转换为给定类型:
            - 不存在列安全跳过
            - 数值列: 使用 to_numeric(errors='coerce') + fillna 防止异常
            - key 列统一 str
        """
        if df is None or df.empty:
            return df
        df_result = df.copy()
        key_cols = [c for c in ['product_code', 'plant_code', 'client_code', 'vehicle_type'] if c in df_result.columns]
        for col in key_cols:
            df_result[col] = df_result[col].astype(str)
        for col, dtype in type_mapping.items():
            if col not in df_result.columns:
                continue
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

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def load(self):
        """
        加载并构建所有数据接口 (幂等):
            若任一步骤失败 → 抛出异常 (保证外部使用时状态一致)
        """
        try:
            self._load_parameters()
            file_paths = self._construct_file_paths()
            dataframes = self._load_all_dataframes(file_paths)
            self._process_dataframes(dataframes)
            self._compute_derived_data()
            self._validate_data_integrity()
            self.logger.info("数据加载和处理完成")
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise

    # ------------------------------------------------------------------
    # 步骤 1: 配置参数
    # ------------------------------------------------------------------
    def _load_parameters(self):
        """读取 JSON 配置中的惩罚参数 (缺失则报错)"""
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

    # ------------------------------------------------------------------
    # 步骤 2: 构建文件路径
    # ------------------------------------------------------------------
    def _construct_file_paths(self) -> Dict[str, str]:
        """构建数据集目录内各数据文件路径映射"""
        dataset_path = os.path.join(self.input_file_loc, self.dataset_name)
        return {
            'order': os.path.join(dataset_path, "order.csv"),
            'plant_cap': os.path.join(dataset_path, "plant_size_upper_limit.csv"),
            'sku_size': os.path.join(dataset_path, "product_size.csv"),
            'vehicle': os.path.join(dataset_path, "vehicle.csv"),
            'production': os.path.join(dataset_path, "warehouse_production.csv"),
            'inventory': os.path.join(dataset_path, "warehouse_storage.csv"),
        }

    # ------------------------------------------------------------------
    # 步骤 3: 读取所有 DataFrame
    # ------------------------------------------------------------------
    def _load_all_dataframes(self, file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """批量加载 CSV → DataFrame 映射 (任意失败即终止)"""
        dataframes = {}
        file_desc = {
            'order': '订单需求信息',
            'plant_cap': '生产基地库存容量上限',
            'sku_size': 'SKU体积信息',
            'vehicle': '车辆类型信息',
            'production': '生产基地生产计划',
            'inventory': '生产基地期初库存',
        }
        # 为每个文件定义最小必需列 (原始列名)
        required_columns = {
            'order': ['client_code', 'product_code', 'volume'],
            'plant_cap': ['plant_code', 'max_volume'],
            'sku_size': ['product_code', 'standard_size'],
            'vehicle': ['vehicle_type', 'carry_standard_size', 'min_standard_size', 'cost_to_use'],
            'production': ['plant_code', 'product_code', 'produce_date', 'volume'],
            'inventory': ['plant_code', 'product_code', 'volume'],
        }
        for key, path in file_paths.items():
            df = self._load_csv_safely(
                path,
                file_desc[key],
                required_cols=required_columns.get(key)
            )
            if df is None:
                raise FileNotFoundError(f"必需文件 {file_desc[key]} 加载失败")
            dataframes[key] = df
        return dataframes

    # ------------------------------------------------------------------
    # 步骤 4: 规范化各 DataFrame
    # ------------------------------------------------------------------
    def _process_dataframes(self, dataframes: Dict[str, pd.DataFrame]):
        """列映射 + 类型转换 (生成 _df_* 中间结果)"""

        # 订单
        title_map = {'client_code': 'dealer_id', 'product_code': 'sku_id', 'volume': 'order_qty'}
        type_mapping = {'dealer_id': str, 'sku_id': str, 'order_qty': int}
        df_order = self._reorganize_dataframe(dataframes['order'], title_map)
        self.df_order = self._convert_dataframe_types(df_order, type_mapping)

        # 基地容量
        title_map = {'plant_code': 'fact_id', 'max_volume': 'max_cap'}
        type_mapping = {'fact_id': str, 'max_cap': float}
        df_plant_cap = self._reorganize_dataframe(dataframes['plant_cap'], title_map)
        self._df_plant_cap = self._convert_dataframe_types(df_plant_cap, type_mapping)

        # SKU 尺寸
        title_map = {'product_code': 'sku_id', 'standard_size': 'size'}
        type_mapping = {'sku_id': str, 'size': float}
        df_sku_size = self._reorganize_dataframe(dataframes['sku_size'], title_map)
        self._df_sku_size = self._convert_dataframe_types(df_sku_size, type_mapping)

        # 车辆
        title_map = {
            'vehicle_type': 'vehicle_type',
            'carry_standard_size': 'capacity',
            'min_standard_size': 'min_load',
            'cost_to_use': 'cost',
        }
        type_mapping = {'vehicle_type': str, 'capacity': float, 'min_load': float, 'cost': float}
        df_all_veh = self._reorganize_dataframe(dataframes['vehicle'], title_map)
        self._df_all_veh = self._convert_dataframe_types(df_all_veh, type_mapping)

        # 生产计划
        title_map = {'plant_code': 'fact_id', 'product_code': 'sku_id', 'produce_date': 'day', 'volume': 'prod_qty'}
        type_mapping = {'fact_id': str, 'sku_id': str, 'day': int, 'prod_qty': int}
        df_sku_prod = self._reorganize_dataframe(dataframes['production'], title_map)
        self._df_sku_prod = self._convert_dataframe_types(df_sku_prod, type_mapping)

        # 期初库存
        title_map = {'plant_code': 'fact_id', 'product_code': 'sku_id', 'volume': 'inv'}
        type_mapping = {'fact_id': str, 'sku_id': str, 'inv': int}
        df_sku_inv = self._reorganize_dataframe(dataframes['inventory'], title_map)
        self._df_sku_inv = self._convert_dataframe_types(df_sku_inv, type_mapping)

    # ------------------------------------------------------------------
    # 步骤 5: 派生结构计算
    # ------------------------------------------------------------------
    def _compute_derived_data(self):
        """聚合集合 / 映射 / 期初递推结构"""
        # 规划周期
        if len(self._df_sku_prod) > 0:
            self.horizons = int(self._df_sku_prod['day'].max())

        # 基础集合
        self.plants = set(self._df_plant_cap['fact_id'].unique())
        self.dealers = set(self.df_order['dealer_id'].unique())
        self.all_skus = set(self._df_sku_size['sku_id'].unique())
        self.all_veh_types = set(self._df_all_veh['vehicle_type'].unique())

        # SKU 尺寸
        self.sku_sizes = dict(zip(self._df_sku_size['sku_id'], self._df_sku_size['size']))

        # 生产
        self.sku_prod_each_day = self._df_sku_prod.set_index(['fact_id', 'sku_id', 'day'])['prod_qty'].to_dict()
        self.sku_prod_total = self._df_sku_prod.groupby(['sku_id', 'day'])['prod_qty'].sum().to_dict()

        # 期初库存
        self.sku_initial_inv = self._df_sku_inv.set_index(['fact_id', 'sku_id'])['inv'].to_dict()

        # 历史库存 (仅 day=0)
        self.historical_s_ikt = {}
        for (plant, sku), inv in self.sku_initial_inv.items():
            self.historical_s_ikt[plant, sku, 0] = inv

        # SKU 可用集合映射
        self._compute_sku_mappings()

        # 需求
        self.demands = self.df_order.set_index(['dealer_id', 'sku_id'])['order_qty'].to_dict()
        self.skus_dealer = {}
        for (dealer, sku) in self.demands:
            self.skus_dealer.setdefault(dealer, set()).add(sku)

        # 车辆
        self.veh_type_cost = dict(zip(self._df_all_veh['vehicle_type'], self._df_all_veh['cost']))
        self.veh_type_cap = dict(zip(self._df_all_veh['vehicle_type'], self._df_all_veh['capacity']))
        self.veh_type_min_load = dict(zip(self._df_all_veh['vehicle_type'], self._df_all_veh['min_load']))

        # 基地容量上限
        self.plant_inv_limit = dict(zip(self._df_plant_cap['fact_id'], self._df_plant_cap['max_cap']))

    def _compute_sku_mappings(self):
        """构建期初/生产/综合 SKU 集合映射 (保持原逻辑不变)"""
        # 期初库存 SKU
        self.skus_initial = defaultdict(set)
        for (plant, sku) in self.sku_initial_inv:
            self.skus_initial[plant].add(sku)
        self.skus_initial = dict(self.skus_initial)

        # 生产 SKU
        self.skus_prod = defaultdict(set)
        for (plant, sku, day) in self.sku_prod_each_day:
            self.skus_prod[plant].add(sku)
        self.skus_prod = dict(self.skus_prod)

        # 可供 (期初 ∪ 生产)
        self.skus_plant = {}
        for plant in self.plants:
            initial = self.skus_initial.get(plant, set())
            prod = self.skus_prod.get(plant, set())
            self.skus_plant[plant] = initial.union(prod)

    # ------------------------------------------------------------------
    # 步骤 6: 完整性校验
    # ------------------------------------------------------------------
    def _validate_data_integrity(self):
        """对关键集合/映射进行存在性与交叉一致性检查"""
        errors = []

        # 必要集合
        if not self.plants:
            errors.append("没有找到生产基地数据")
        if not self.dealers:
            errors.append("没有找到经销商数据")
        if not self.all_skus:
            errors.append("没有找到SKU数据")
        if not self.all_veh_types:
            errors.append("没有找到车辆类型数据")

        # SKU 尺寸缺失
        order_skus = {sku for dealer, sku in self.demands.keys()}
        missing_sku_sizes = order_skus - set(self.sku_sizes.keys())
        if missing_sku_sizes:
            errors.append(f"订单中的SKU缺少尺寸信息: {missing_sku_sizes}")

        # 基地无可供 SKU (仅警告)
        plants_with_skus = set(self.skus_plant.keys())
        plants_without = self.plants - plants_with_skus
        if plants_without:
            self.logger.warning(f"以下生产基地没有可提供的SKU: {plants_without}")

        if errors:
            msg = "数据完整性验证失败:\n" + "\n".join(f"- {e}" for e in errors)
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("数据完整性验证通过")

    # ------------------------------------------------------------------
    # 供应链与访问帮助函数
    # ------------------------------------------------------------------
    def available_skus_in_plant(self, plant_id: str) -> Set[str]:
        """返回某生产基地可提供的全部 SKU 集 (不存在则空集, 并记录警告)"""
        if plant_id not in self.plants:
            self.logger.warning(f"生产基地 {plant_id} 不存在于数据中")
            return set()
        return self.skus_plant.get(plant_id, set())

    def plan_dealer_skus(self, plant_id: str) -> Set[Tuple[str, str]]:
        """返回该生产基地潜在服务的 (经销商, SKU) 组合 (依据 supply_chain 交集)"""
        if plant_id not in self.plants:
            self.logger.warning(f"生产基地 {plant_id} 不存在于数据中")
            return set()
        supply_chain = self.construct_supply_chain()
        result = set()
        for (plant, dealer), skus in supply_chain.items():
            if plant == plant_id:
                for sku in skus:
                    result.add((dealer, sku))
        return result

    def available_skus_to_dealer(self, plant_id: str, dealer_id: str) -> Set[str]:
        """返回生产基地对指定经销商可供 SKU 集合 (不存在返回空集并警告)"""
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
        构建供应链关系:
            {(plant, dealer): intersect_skus} 其中 intersect_skus 为 plant 可供且 dealer 需求
        结果缓存以避免重复计算。
        """
        if not hasattr(self, '_supply_chain_cache'):
            supply_chain = defaultdict(set)
            for plant, skus_supply in self.skus_plant.items():
                for dealer, skus_need in self.skus_dealer.items():
                    inter = skus_supply.intersection(skus_need)
                    if inter:
                        supply_chain[plant, dealer] = inter
            self._supply_chain_cache = dict(supply_chain)
        return self._supply_chain_cache

    # ------------------------------------------------------------------
    # 指标汇总 / 供需平衡分析
    # ------------------------------------------------------------------
    def get_total_demand_for_sku(self, sku_id: str) -> int:
        """聚合指定 SKU 的总需求量"""
        total = 0
        for (dealer, sku), demand in self.demands.items():
            if sku == sku_id:
                total += demand
        return total

    def get_total_production_for_sku(self, sku_id: str, day: Optional[int] = None) -> int:
        """返回指定 SKU 在某一日或所有日的总生产量"""
        if day is not None:
            return self.sku_prod_total.get((sku_id, day), 0)
        total = 0
        for (sku, d), qty in self.sku_prod_total.items():
            if sku == sku_id:
                total += qty
        return total

    def get_total_initial_inventory_for_sku(self, sku_id: str) -> int:
        """返回指定 SKU 的期初总库存"""
        total = 0
        for (plant, sku), inv in self.sku_initial_inv.items():
            if sku == sku_id:
                total += inv
        return total

    def get_sku_supply_demand_balance(self, sku_id: str) -> Dict[str, Union[int, float]]:
        """
        供需平衡:
            total_supply = initial + production_all
            balance = supply - demand
            balance_ratio = supply / demand (无需求 → inf)
        """
        total_demand = self.get_total_demand_for_sku(sku_id)
        total_init = self.get_total_initial_inventory_for_sku(sku_id)
        total_prod = self.get_total_production_for_sku(sku_id)
        total_supply = total_init + total_prod
        balance = total_supply - total_demand
        ratio = total_supply / total_demand if total_demand > 0 else float('inf')
        return {
            'sku_id': sku_id,
            'total_demand': total_demand,
            'total_initial_inventory': total_init,
            'total_production': total_prod,
            'total_supply': total_supply,
            'balance': balance,
            'balance_ratio': ratio,
            'status': 'surplus' if balance > 0 else 'deficit' if balance < 0 else 'balanced',
        }

    def get_summary_statistics(self) -> Dict[str, Union[int, float]]:
        """整体汇总指标 (用于日志或快速 sanity check)"""
        total_demand = sum(self.demands.values())
        total_initial = sum(self.sku_initial_inv.values())
        total_prod = sum(self.sku_prod_total.values())
        total_supply = total_initial + total_prod
        return {
            'num_plants': len(self.plants),
            'num_dealers': len(self.dealers),
            'num_skus': len(self.all_skus),
            'num_vehicle_types': len(self.all_veh_types),
            'planning_horizon': self.horizons,
            'total_demand': total_demand,
            'total_initial_inventory': total_initial,
            'total_production': total_prod,
            'total_supply': total_supply,
            'supply_demand_ratio': total_supply / total_demand if total_demand > 0 else float('inf'),
        }
