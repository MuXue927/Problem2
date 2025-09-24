import os
import csv
import sys
import time
import datetime
import logging
import pandas as pd
import numpy as np
import numpy.random as rnd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from alnsopt import *
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import SegmentedRouletteWheel
from alns.Result import Result
from optutility import LogPrinter
from alnstrack import ALNSTracker, calculate_gap
from combined_stopping import create_standard_combined_criterion
from functools import partial
from alns_config import ALNSConfig
from visualization import *

SEED = ALNSConfig.SEED
DATASET_TYPE = ALNSConfig.DATASET_TYPE   
DATASET_IDX = ALNSConfig.DATASET_IDX      
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alns_optimization.log', mode='w')  # 只写入日志文件, 不输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 检查文件是否被占用
def is_file_in_use(file_path):
    """
    检查文件是否被占用（被其他程序打开）
    """
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, 'a'):
            pass
        return False
    except PermissionError:
        return True

class ALNSOptimizer:
    """ALNS优化器主类, 封装了整个优化流程"""
    
    def __init__(self, input_file_loc: str = None, output_file_loc: str = None):
        """
        初始化ALNS优化器
        
        Parameters:
        -----------
        input_file_loc : str, optional
            输入文件位置, 如果为None则使用默认路径
        output_file_loc : str, optional  
            输出文件位置, 如果为None则使用默认路径
        """
        # 设置默认路径
        if input_file_loc is None or output_file_loc is None:
            current_dir = Path(__file__).parent
            par_path = current_dir.parent
            
            if input_file_loc is None:
                input_file_loc = par_path / 'datasets' / 'multiple-periods' / DATASET_TYPE
            if output_file_loc is None:
                output_file_loc = par_path / 'OutPut-ALNS' / 'multiple-periods' / DATASET_TYPE
        
        self.input_file_loc = Path(input_file_loc)
        self.output_file_loc = Path(output_file_loc)
        
        # 初始化日志打印器
        self.log_printer = LogPrinter(time.time())
        
        # 初始化数据和状态
        self.data: Optional[DataALNS] = None
        self.best_solution: Optional[SolutionState] = None
        self.result: Optional[Result] = None
        self.tracker: Optional[ALNSTracker] = None

    def load_data(self, dataset_name: str) -> bool:
        """
        加载数据
        
        Parameters:
        -----------
        dataset_name : str
            数据集名称
            
        Returns:
        --------
        bool : 是否加载成功
        """
        try:
            self.log_printer.print("Loading Data...")
            start_time = time.time()
            
            # 构建完整的输出路径
            full_output_path = self.output_file_loc / dataset_name
            
            self.data = DataALNS(
                str(self.input_file_loc), 
                str(full_output_path), 
                dataset_name
            )
            self.data.load()
            
            end_time = time.time()
            self.log_printer.print(f"Data loading elapsed time: {end_time - start_time:.2f}s")
            
            # 清理输出文件
            self._clear_output_files(full_output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            self.log_printer.print(f"Error loading data: {e}", color='bold red')
            return False

    def _clear_output_files(self, output_path: Path):
        """清理输出文件"""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 定义需要清理的文件列表
            files_to_clear = {
                'opt_result.csv': ['day', 'plant_code', 'client_code', 
                                 'product_code', 'vehicle_id', 'vehicle_type', 'qty'],
                'non_fulfill.csv': ['client_code', 'product_code', 'volume']
            }
            
            # 清理并初始化每个文件
            for file_name, header in files_to_clear.items():
                file_path = output_path / file_name
                with open(file_path, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(header)
                    
        except Exception as e:
            logger.error(f"清理输出文件失败: {e}")

    def create_initial_solution(self) -> Optional[SolutionState]:
        """
        创建初始解
        
        Returns:
        --------
        SolutionState or None : 初始解, 如果创建失败则返回None
        """
        if self.data is None:
            self.log_printer.print("Error: Data not loaded", color='bold red')
            return None
            
        try:
            self.log_printer.print("Initializing the solution...")
            
            # 创建SolutionState的实例
            state = SolutionState(self.data)
            
            # 获取初始解
            init_sol = initial_solution(state, rng=rnd.default_rng(seed=SEED))
            
            # 对初始解进行评估
            initial_feasibility, violations = init_sol.validate()
            
            self.log_printer.print(f"Initial Solution Objective: {init_sol.objective():.2f}")
            self.log_printer.print(f"Vehicles Used in Initial Solution: {len(init_sol.vehicles)}")
            self.log_printer.print(f"Initial Solution Cost: {init_sol.calculate_cost():.2f}")
            
            if not initial_feasibility:
                self.log_printer.print("Warning: The initial solution is infeasible!", color='bold red')
                self._print_violation_details(violations)
                
                # 根据配置决定是否终止程序
                if getattr(ALNSConfig, 'TERMINATE_ON_INFEASIBLE_INITIAL', True):
                    self.log_printer.print("Terminating due to infeasible initial solution.", color='bold red')
                    return None
                else:
                    self.log_printer.print("Continuing with infeasible initial solution.", color='yellow')
            
            return init_sol
            
        except Exception as e:
            logger.error(f"创建初始解失败: {e}")
            self.log_printer.print(f"Error creating initial solution: {e}", color='bold red')
            return None

    def _print_violation_details(self, violations: Dict):
        """打印违约详情"""
        if violations.get('negative_inventory'):
            self.log_printer.print("Negative inventory violations:")
            for key, inv in violations['negative_inventory']:
                self.log_printer.print(f"  Plant: {key[0]}, SKU: {key[1]}, Day: {key[2]}, Inventory: {inv}")
        
        if violations.get('veh_overload'):
            self.log_printer.print("Vehicle overload violations:")
            for v in violations['veh_overload']:
                veh = v['veh']
                self.log_printer.print(f"  Vehicle: Plant {veh.fact_id}, Dealer {veh.dealer_id}, Type {veh.type}, Day {veh.day}, Loaded: {v['loaded']}, Capacity: {v['cap']}")
        
        if violations.get('unmet_demand'):
            self.log_printer.print("Unmet demand violations:")
            for d in violations['unmet_demand']:
                self.log_printer.print(f"  Dealer: {d['dealer']}, SKU: {d['sku_id']}, Demand: {d['demand']}, Shipped: {d['shipped']}")

    def setup_alns(self) -> ALNS:
        """
        设置ALNS算法实例
        
        Returns:
        --------
        ALNS : 配置好的ALNS实例
        """
        try:
            # 创建ALNS的实例
            alns = ALNS(rng=rnd.default_rng(seed=SEED))
            
            # 注册destroy算子
            self._register_destroy_operators(alns)
            
            # 注册repair算子
            self._register_repair_operators(alns)
            
            return alns
            
        except Exception as e:
            logger.error(f"ALNS设置失败: {e}")
            raise

    def _register_destroy_operators(self, alns: ALNS):
        """注册destroy算子"""
        self.log_printer.print("注册destroy算子...")
        
        try:
            # 获取基础参数
            destroy_params = ALNSConfig.get_destroy_params()
            
            # 注册random removal算子
            random_removal_op = self._create_named_partial(
                random_removal, 
                name="random_removal",
                degree=destroy_params['random_removal_degree']
            )
            alns.add_destroy_operator(random_removal_op)
            
            # 注册 shaw removal算子
            shaw_removal_op = self._create_named_partial(
                shaw_removal,
                name="shaw_removal",
                degree=destroy_params['shaw_removal_degree']
            )
            alns.add_destroy_operator(shaw_removal_op)
            
            # 注册 periodic shaw removal算子
            
            # periodic_shaw_removal_op = self._create_named_partial(
            #     periodic_shaw_removal,
            #     name="periodic_shaw_removal",
            #     **ALNSConfig.PERIODIC_SHAW_PARAMS
            # )

            # another way to register the operator with parameters
            periodic_shaw_removal_op = PeriodicShawRemovalOperator(**ALNSConfig.PERIODIC_SHAW_PARAMS)
            alns.add_destroy_operator(periodic_shaw_removal_op)
            
            # 注册其他不需要传递参数的destroy算子
            destroy_operators = [
                worst_removal,
                infeasible_removal,
                surplus_inventory_removal,
                path_removal
            ]
            for operator in destroy_operators:
                alns.add_destroy_operator(operator)
            
            self.log_printer.print("所有destroy算子注册完成")
            
        except Exception as e:
            logger.error(f"注册destroy算子失败: {e}")
            raise

    def _register_repair_operators(self, alns: ALNS):
        """注册repair算子"""
        self.log_printer.print("注册repair算子...")
        
        try:
            # 注册 learning_based_repair 算子
            learning_based_repair_op = LearningBasedRepairOperator(**ALNSConfig.LEARNING_BASED_REPAIR_PARAMS)
            alns.add_repair_operator(learning_based_repair_op)
            
            # 注册其他不需要传递参数的repair算子
            repair_operators = [
                smart_batch_repair,
                greedy_repair,
                inventory_balance_repair,
                urgency_repair,
                infeasible_repair,
                local_search_repair
            ]
            
            for operator in repair_operators:
                alns.add_repair_operator(operator)
            
            self.log_printer.print("所有repair算子注册完成")
            
        except Exception as e:
            logger.error(f"注册repair算子失败: {e}")
            raise

    def _create_named_partial(self, func, name=None, **kwargs):
        """
        创建带有__name__属性的partial对象, 以兼容ALNS库
        
        Parameters:
        -----------
        func : callable
            要创建partial的函数
        name : str, optional
            partial对象的名称, 如果不提供则自动生成
        **kwargs : dict
            要固定的关键字参数
        
        Returns:
        --------
        partial object with __name__ attribute
        """
        partial_func = partial(func, **kwargs)
        
        if name is None:
            # 自动生成名称
            param_str = "_".join([f"{k}_{v}" for k, v in kwargs.items()])
            name = f"{func.__name__}_{param_str}"
        
        partial_func.__name__ = name
        return partial_func

    def run_optimization(self, dataset_name: str) -> bool:
        """
        运行完整的优化流程
        
        Parameters:
        -----------
        dataset_name : str
            数据集名称
            
        Returns:
        --------
        bool : 是否优化成功
        """
        try:
            self.log_printer.print(f"开始优化流程! / {datetime.datetime.now()}")
            optimization_start_time = time.time()

            # 1. 加载数据
            t0 = time.time()
            self.log_printer.print("Step 1: 加载数据...")
            if not self.load_data(dataset_name):
                return False
            
            # 新增: 绘制供应链网络图
            plot_supply_chain_network_3d_enhanced(self.data, str(self.output_file_loc / dataset_name))
            
            self.log_printer.print(f"Step 1耗时: {time.time() - t0:.2f}s")

            # 2. 创建初始解
            t0 = time.time()
            self.log_printer.print("Step 2: 创建初始解...")
            init_sol = self.create_initial_solution()
            if init_sol is None:
                return False
            self.log_printer.print(f"Step 2耗时: {time.time() - t0:.2f}s")

            # 3. 设置ALNS算法
            t0 = time.time()
            self.log_printer.print("Step 3: 设置ALNS算法...")
            alns = self.setup_alns()
            self.log_printer.print(f"Step 3耗时: {time.time() - t0:.2f}s")

            # 4. 配置算子选择机制
            t0 = time.time()
            self.log_printer.print("Step 4: 配置算子选择机制...")
            select = SegmentedRouletteWheel(
                scores=ALNSConfig.ROULETTE_SCORES,
                decay=ALNSConfig.ROULETTE_DECAY,
                seg_length=ALNSConfig.ROULETTE_SEG_LENGTH,
                num_destroy=7,
                num_repair=7
            )
            self.log_printer.print(f"Step 4耗时: {time.time() - t0:.2f}s")

            # 5. 配置接受准则
            t0 = time.time()
            self.log_printer.print("Step 5: 配置接受准则...")
            sa_accept = SimulatedAnnealing(
                start_temperature=ALNSConfig.SA_START_TEMP, 
                end_temperature=ALNSConfig.SA_END_TEMP, 
                step=ALNSConfig.SA_STEP, 
                method="exponential"
            )
            self.log_printer.print(f"Step 5耗时: {time.time() - t0:.2f}s")

            # 6. 配置停止准则
            t0 = time.time()
            self.log_printer.print("Step 6: 配置停止准则...")
            
            # 使用组合停止准则：
            # 小规模数据集: 300次迭代 OR 900秒运行时间
            # 中规模数据集: 600次迭代 OR 1800秒运行时间
            # 大规模数据集: 1000次迭代 OR 3600秒运行时间
            stop = create_standard_combined_criterion(
                max_iterations=ALNSConfig.MAX_ITERATIONS,  # 300次迭代
                max_runtime=ALNSConfig.MAX_RUNTIME,        # 900秒
                max_no_improvement=None  # 暂不使用无改进停止条件
            )
            
            self.log_printer.print(f"组合停止准则配置: 最大{ALNSConfig.MAX_ITERATIONS}次迭代 或 最大{ALNSConfig.MAX_RUNTIME}秒运行时间")
            self.log_printer.print(f"Step 6耗时: {time.time() - t0:.2f}s")

            # 7. 设置追踪器和回调函数
            t0 = time.time()
            self.log_printer.print("Step 7: 设置追踪器和回调函数...")
            self.tracker = ALNSTracker()
            # 新增: 将tracker注入到初始解中, 以便在repair算子中使用
            init_sol.set_tracker(self.tracker)
            

            def callback_on_best(state: SolutionState, rng, **kwargs) -> bool:
                is_feasible, violations = state.validate()
                if is_feasible:
                    self.log_printer.print(f"Found feasible solution with cost: {state.calculate_cost():.2f}")
                else:
                    unmet_count = len(violations.get('unmet_demand', []))
                    if unmet_count > 0:
                        self.log_printer.print(f"Best solution has {unmet_count} unmet demands")
                return True

            def callback_on_iteration(state: SolutionState, rng, **kwargs) -> bool:
                """在每次迭代时监控停止准则状态"""
                # 每50次迭代报告一次状态
                if hasattr(self.tracker, '_iteration_count'):
                    iteration = self.tracker._iteration_count
                    if iteration % 50 == 0:
                        status = stop.get_status()
                        elapsed = status['elapsed_time']
                        self.log_printer.print(f"迭代 {iteration}: 运行时间 {elapsed:.1f}s")
                
                # 调用原有的追踪器逻辑
                return self.tracker.on_iteration(state, rng, **kwargs)

            alns.on_best(callback_on_best)
            alns.on_accept(callback_on_iteration)
            self.log_printer.print(f"Step 7耗时: {time.time() - t0:.2f}s")

            # 8. 运行ALNS算法
            t0 = time.time()
            self.log_printer.print("Step 8: 运行ALNS算法...")
            self.result = alns.iterate(init_sol, select, sa_accept, stop)
            self.log_printer.print(f"Step 8耗时: {time.time() - t0:.2f}s")

            optimization_end_time = time.time()
            self.log_printer.print(f"优化总耗时: {optimization_end_time - optimization_start_time:.2f}s")

            # 9. 处理结果
            t0 = time.time()
            self.log_printer.print("Step 9: 处理结果...")
            self._process_results()
            self.log_printer.print(f"Step 9耗时: {time.time() - t0:.2f}s")

            # 10. 生成报告和图表
            t0 = time.time()
            self.log_printer.print("Step 10: 生成报告和图表...")
            self._generate_reports()
            self.log_printer.print(f"Step 10耗时: {time.time() - t0:.2f}s")

            self.log_printer.print_title(f"优化完成! {datetime.datetime.now()}")

            return True

        except Exception as e:
            logger.error(f"优化过程失败: {e}")
            self.log_printer.print(f"Optimization failed: {e}", color='bold red')
            return False

    def _process_results(self):
        """处理优化结果"""
        if self.result is None:
            return
            
        try:
            # 获取最终解
            self.best_solution = self.result.best_state
            
            # 输出统计信息
            tracker_stats = self.tracker.get_statistics()
            self.log_printer.print(f"Total iterations tracked: {tracker_stats['total_iterations']}")
            self.log_printer.print(f"Best objective found: {tracker_stats['best_objective']:.4f}")
            self.log_printer.print(f"Final Gap: {tracker_stats['final_gap']:.2f}%")
            
            # 计算最终Gap
            final_current_obj = self.result.statistics.objectives[-1]
            final_best_obj = self.best_solution.objective()
            final_gap = calculate_gap(final_current_obj, final_best_obj)
            self.log_printer.print(f"ALNS Final Gap: {final_gap:.2f}%")
            
            # 验证最终解
            final_feasibility, violations = self.best_solution.validate()
            if not final_feasibility:
                self.log_printer.print("Warning: The final best solution is infeasible!", color='bold red')
            
            self.log_printer.print(f"Best Heuristic Solution Objective: {self.best_solution.objective():.2f}")
            self.log_printer.print(f"Vehicles Used in Best Solution: {len(self.best_solution.vehicles)}")
            self.log_printer.print(f"Best Solution Cost: {self.best_solution.calculate_cost():.2f}")
            
            # 更新数据和输出结果
            self._update_demands()
            self._update_inventory()
            self._output_results()
            self._generate_statistics()
            
        except Exception as e:
            logger.error(f"处理结果失败: {e}")
            raise

    def _update_demands(self):
        """更新需求数据"""
        try:
            # 更新需求
            for veh in self.best_solution.vehicles:
                for (sku_id, day), qty in veh.cargo.items():
                    key = (veh.dealer_id, sku_id)
                    if key in self.data.demands:
                        self.data.demands[key] -= qty
            
            # 获取未满足的需求数据
            non_fulfill_data = []
            for (dealer, sku_id), qty in self.data.demands.items():
                if qty > 0:
                    non_fulfill_data.append({
                        'client_code': dealer,
                        'product_code': sku_id,
                        'volume': qty
                    })
            
            # 保存未满足需求
            if non_fulfill_data:
                df_non_fulfill = pd.DataFrame(non_fulfill_data)
                output_file = Path(self.data.output_file_loc) / 'non_fulfill.csv'
                # 检查文件是否被占用
                if is_file_in_use(output_file):
                    self.log_printer.print(f"文件 {output_file} 正在被占用, 请关闭后重试。", color='bold red')
                    return
                df_non_fulfill.to_csv(output_file, mode='a', header=False, index=False)
            
            # 更新数据对象中的需求
            self.data.demands = {(dealer, sku_id): qty for (dealer, sku_id), qty in self.data.demands.items() if qty > 0}
            
        except Exception as e:
            logger.error(f"更新需求失败: {e}")
            raise

    def _update_inventory(self):
        """更新库存数据"""
        try:
            sku_inv_left = {}
            
            # 遍历每个工厂的每种SKU
            for plant_id in self.data.plants:
                for sku_id in self.data.all_skus:
                    # 获取期初库存
                    current_inv = self.data.sku_initial_inv.get((plant_id, sku_id), 0)
                    
                    # 按时间顺序计算每天结束时的库存
                    for day in range(1, self.data.horizons + 1):
                        # 加上当天生产量
                        current_inv += self.data.sku_prod_each_day.get((plant_id, sku_id, day), 0)
                        
                        # 减去当天配送量
                        shipped_qty = 0
                        for veh in self.best_solution.vehicles:
                            if veh.fact_id == plant_id and (sku_id, day) in veh.cargo:
                                shipped_qty += veh.cargo[(sku_id, day)]
                        current_inv -= shipped_qty
                        
                        # 保存当天结束时的库存
                        sku_inv_left[(plant_id, sku_id, day)] = current_inv
            
            # 创建并保存库存DataFrame
            df_sku_inv_left = pd.DataFrame([
                {'plant_code': plant, 'product_code': sku_id, 'day': day, 'volume': qty}
                for (plant, sku_id, day), qty in sku_inv_left.items()
            ])
            output_file = Path(self.data.output_file_loc) / 'sku_inv_left.csv'
            if is_file_in_use(output_file):
                self.log_printer.print(f"文件 {output_file} 正在被占用, 请关闭后重试。", color='bold red')
                return
            df_sku_inv_left.to_csv(output_file, index=False)
            
        except Exception as e:
            logger.error(f"更新库存失败: {e}")
            raise

    def _output_results(self):
        """输出结果到文件"""
        try:
            output_file = Path(self.data.output_file_loc) / 'opt_result.csv'
            df_result = self._create_result_dataframe()
            if is_file_in_use(output_file):
                self.log_printer.print(f"文件 {output_file} 正在被占用, 请关闭后重试。", color='bold red')
                return
            df_result.to_csv(output_file, mode='a', header=False, index=False)
            
        except Exception as e:
            logger.error(f"输出结果失败: {e}")
            raise

    def _create_result_dataframe(self) -> pd.DataFrame:
        """创建结果DataFrame"""
        rows = []
        for idx, veh in enumerate(self.best_solution.vehicles):
            for (sku_id, day), qty in veh.cargo.items():
                rows.append({
                    'day': day,
                    'plant_code': veh.fact_id,
                    'client_code': veh.dealer_id,
                    'product_code': sku_id,
                    'vehicle_id': idx + 1,
                    'vehicle_type': veh.type,
                    'qty': qty
                })
        return pd.DataFrame(rows)

    def _generate_statistics(self):
        """生成统计信息"""
        try:
            dataset = self.data.dataset_name
            output_location = Path(self.data.output_file_loc)
            input_location = self.input_file_loc / dataset
            
            # 读取结果文件
            df_result = pd.read_csv(output_location / 'opt_result.csv')
            df_sku_size = pd.read_csv(input_location / 'product_size.csv')
            df_all_veh = pd.read_csv(input_location / 'vehicle.csv')
            
            # 合并数据前, 确保主键列类型一致（全部转为str）
            if 'product_code' in df_result.columns:
                df_result['product_code'] = df_result['product_code'].astype(str)
            if 'product_code' in df_sku_size.columns:
                df_sku_size['product_code'] = df_sku_size['product_code'].astype(str)
            if 'vehicle_type' in df_result.columns:
                df_result['vehicle_type'] = df_result['vehicle_type'].astype(str)
            if 'vehicle_type' in df_all_veh.columns:
                df_all_veh['vehicle_type'] = df_all_veh['vehicle_type'].astype(str)
            df_result = pd.merge(df_result, df_sku_size, on=['product_code'])
            df_result = pd.merge(df_result, df_all_veh, on=['vehicle_type'])
            
            # 删除不需要的列
            if 'cost_to_use' in df_result.columns:
                del df_result['cost_to_use']
            
            # 计算占用体积
            df_result['occupied_size'] = df_result['qty'] * df_result['standard_size']
            
            # 保存详细结果
            details_file = output_location / 'opt_details.csv'
            if is_file_in_use(details_file):
                self.log_printer.print(f"文件 {details_file} 正在被占用, 请关闭后重试。", color='bold red')
                return
            df_result.to_csv(details_file, index=False)
            
            # 生成汇总统计
            df_stat1 = df_result.copy()
            if 'product_code' in df_stat1.columns:
                df_stat1 = df_stat1.drop(['product_code', 'standard_size'], axis=1)
            
            df_summary = df_stat1.groupby(['day', 'plant_code', 'client_code', 'vehicle_id']).agg({
                'vehicle_type': 'first', 
                'qty': 'sum', 
                'carry_standard_size': 'first',
                'min_standard_size': 'first', 
                'occupied_size': 'sum'
            })
            
            # 重置索引
            df_summary = df_summary.reset_index()
            
            summary_file = output_location / 'opt_summary.csv'
            if is_file_in_use(summary_file):
                self.log_printer.print(f"文件 {summary_file} 正在被占用, 请关闭后重试。", color='bold red')
                return
            df_summary.to_csv(summary_file, index=False)
            
            # 验证容量约束
            self._validate_capacity_constraints(df_summary)
            
        except Exception as e:
            logger.error(f"生成统计信息失败: {e}")
            raise

    def _validate_capacity_constraints(self, df_summary: pd.DataFrame):
        """验证容量约束"""
        try:
            # 检查是否所有车辆装载的SKU体积满足车辆容量约束
            is_cap_exceed = pd.Series(df_summary['occupied_size'] <= df_summary['carry_standard_size'])
            
            if is_cap_exceed.all():
                self.log_printer.print("✓ 所有车辆的装载体积都满足容量约束")
            else:
                self.log_printer.print("⚠️ 部分车辆的装载体积超出容量限制", color='yellow')
                df_exceed = df_summary[df_summary["occupied_size"] > df_summary["carry_standard_size"]]
                
                df_exceed_info = df_exceed.loc[:, ['vehicle_id', 'vehicle_type', 'carry_standard_size', 'occupied_size']]
                df_exceed_info['exceeded_volume'] = df_exceed_info['occupied_size'] - df_exceed_info['carry_standard_size']
                
                output_path = Path(self.data.output_file_loc) / "extra_volume.csv"
                if is_file_in_use(output_path):
                    self.log_printer.print(f"文件 {output_path} 正在被占用, 请关闭后重试。", color='bold red')
                    return
                df_exceed_info.to_csv(output_path, index=False)
                self.log_printer.print(f"超容量信息已保存到: {output_path}")
                
        except Exception as e:
            logger.error(f"验证容量约束失败: {e}")

    def _generate_reports(self):
        """生成报告和图表"""
        try:
            if self.result is None or self.best_solution is None:
                return
            # 创建图像目录
            images_dir = Path(self.data.output_file_loc) / 'images'
            images_dir.mkdir(exist_ok=True)
            # 绘制目标函数变化
            plot_objective_changes(self.result, str(self.data.output_file_loc))
            # 绘制算子性能（调用可视化模块）
            plot_operator_performance(self.best_solution, self.result)
            # 绘制Gap变化
            plot_gap_changes(self.tracker, str(self.data.output_file_loc))
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            self.log_printer.print(f"Warning: Failed to generate reports: {e}", color='yellow')


    def _plot_operator_performance(self):
        """绘制算子性能（简化版本）"""
        try:
            # 这里实现一个简化的算子性能图表
            # 原始版本过于复杂, 简化为基本的柱状图
            
            destroy_counts = self.result.statistics.destroy_operator_counts
            repair_counts = self.result.statistics.repair_operator_counts
            
            # 创建简化的性能图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Destroy operators
            if destroy_counts:
                operators = list(destroy_counts.keys())
                totals = [sum(counts[:4]) for counts in destroy_counts.values()]  # 前4个结果的总和
                ax1.bar(operators, totals, color='skyblue')
                ax1.set_title('Destroy Operators Performance')
                ax1.set_ylabel('Total Usage')
                ax1.tick_params(axis='x', rotation=45)
            
            # Repair operators  
            if repair_counts:
                operators = list(repair_counts.keys())
                totals = [sum(counts[:4]) for counts in repair_counts.values()]
                ax2.bar(operators, totals, color='lightcoral')
                ax2.set_title('Repair Operators Performance')
                ax2.set_ylabel('Total Usage')
                ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            file_path = Path(self.data.output_file_loc) / 'images' / 'Performance_of_Operators.svg'
            plt.savefig(file_path, dpi=600, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制算子性能失败: {e}")


def run_model(input_file_loc=None, output_file_loc=None):
    """
    运行模型的主函数
    
    Parameters:
    -----------
    input_file_loc : str, optional
        输入文件位置
    output_file_loc : str, optional
        输出文件位置
    """
    try:
        # 使用默认路径如果未提供
        if input_file_loc is None:
            current_dir = Path(__file__).parent
            par_path = current_dir.parent
            input_file_loc = par_path / 'datasets' / 'multiple-periods' / DATASET_TYPE
        if output_file_loc is None:
            current_dir = Path(__file__).parent
            par_path = current_dir.parent
            output_file_loc = par_path / 'OutPut-ALNS' / 'multiple-periods' / DATASET_TYPE

        dataset_name = f'dataset_{DATASET_IDX}'

        # 创建优化器实例
        optimizer = ALNSOptimizer(input_file_loc, output_file_loc)
        # 运行优化
        success = optimizer.run_optimization(dataset_name)
        if success:
            print("=" * 100)
            optimizer.log_printer.print_title("The Vehicle Loading Plan is Done!")
        else:
            print("优化过程失败")
            sys.exit(1)
    except Exception as e:
        logger.error(f"运行模型失败: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 执行整个优化过程
    run_model()
    