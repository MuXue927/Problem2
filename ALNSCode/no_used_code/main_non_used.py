import os
import csv
import sys
import time
import datetime
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from alnsopt import *
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import SegmentedRouletteWheel
from alns.stop import MaxRuntime, MaxIterations, NoImprovement
from alns.Result import Result
from optutility import LogPrinter
from alnstrack import ALNSTracker, calculate_gap
from accept import DemandConstraintAccept
from functools import partial
from alns_config import ALNSConfig

SEED = ALNSConfig.SEED

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('alns_optimization.log')
    ]
)
logger = logging.getLogger(__name__)

class ALNSOptimizer:
    """ALNS优化器主类，封装了整个优化流程"""
    
    def __init__(self, input_file_loc: str = None, output_file_loc: str = None):
        """
        初始化ALNS优化器
        
        Parameters:
        -----------
        input_file_loc : str, optional
            输入文件位置，如果为None则使用默认路径
        output_file_loc : str, optional  
            输出文件位置，如果为None则使用默认路径
        """
        # 设置默认路径
        if input_file_loc is None or output_file_loc is None:
            current_dir = Path(__file__).parent
            par_path = current_dir.parent
            
            if input_file_loc is None:
                input_file_loc = par_path / 'datasets' / 'multiple-periods' / 'small'
            if output_file_loc is None:
                output_file_loc = par_path / 'OutPut-ALNS' / 'multiple-periods' / 'small'
        
        self.input_file_loc = Path(input_file_loc)
        self.output_file_loc = Path(output_file_loc)
        
        # 初始化日志打印器
        self.log_printer = LogPrinter(time.time())
        
        # 初始化数据和状态
        self.data: Optional[DataALNS] = None
        self.best_solution: Optional[SolutionState] = None
        self.result: Optional[Result] = None
        self.tracker: Optional[ALNSTracker] = None


    def load_data(self, dataset_name: str = 'dataset_1') -> bool:
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
                'opt_result.csv': ['Index', 'day', 'plant_code', 'client_code', 
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
        SolutionState or None : 初始解，如果创建失败则返回None
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
                if ALNSConfig.TERMINATE_ON_INFEASIBLE_INITIAL:
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
                name="random_removal_default",
                degree=destroy_params['random_removal_degree']
            )
            alns.add_destroy_operator(random_removal_op)
            
            # 注册其他destroy算子
            destroy_operators = [
                worst_removal,
                infeasible_removal,
                surplus_inventory_removal,
                path_removal,
                shaw_removal
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
            repair_operators = [
                demand_first_repair,
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
    """
    创建带有__name__属性的partial对象，以兼容ALNS库
    
    Parameters:
    -----------
    func : callable
        要创建partial的函数
    name : str, optional
        partial对象的名称，如果不提供则自动生成
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


def register_destroy_operators(alns, log_printer=None):
    """
    注册所有destroy算子到ALNS实例
    
    Parameters:
    -----------
    alns : ALNS
        ALNS算法实例
    log_printer : LogPrinter, optional
        日志打印器
    """
    if log_printer:
        log_printer.print("注册destroy算子...")
    
    # 获取基础参数
    destroy_params = ALNSConfig.get_destroy_params()
    
    # 注册单个random removal算子
    random_removal_op = create_named_partial(
        random_removal, 
        name="random_removal_default",
        degree=destroy_params['random_removal_degree']
    )
    alns.add_destroy_operator(random_removal_op)
    
    # 注册多个random removal变体（可选）
    # for variant in ALNSConfig.get_random_removal_variants():
    #     variant_op = create_named_partial(
    #         random_removal,
    #         name=variant['name'],
    #         degree=variant['degree']
    #     )
    #     alns.add_destroy_operator(variant_op)
    
    # 注册其他destroy算子
    alns.add_destroy_operator(worst_removal)
    alns.add_destroy_operator(infeasible_removal)
    alns.add_destroy_operator(surplus_inventory_removal)
    alns.add_destroy_operator(path_removal)
    alns.add_destroy_operator(shaw_removal)
    
    if log_printer:
        log_printer.print("所有destroy算子注册完成")


def run_model(input_file_loc=input_loc, output_file_loc=output_loc):
    medium_path = 'dataset_1'
    output_file_loc = os.path.join(output_file_loc, medium_path)
    global data
    
    log_printer.print("Loading Data...")
    start_time = time.time()
    data = DataALNS(input_file_loc, output_file_loc, medium_path)
    data.load()
    end_time = time.time()
    log_printer.print(f"Elapsed time: {end_time - start_time: .2f}s")
    
    clear_output_file(output_file_loc)
    
    log_printer.print(f"Start optimizing! / Now {datetime.datetime.now()}")
    # 创建SolutionState的实例
    state = SolutionState(data)
    # 获取初始解
    log_printer.print("Initializing the solution...")
    
    init_sol = initial_solution(state, rng=rnd.default_rng(seed=SEED))
    # 对初始解进行评估,如果初始解不可行，则输出关于不可行解的信息并终止程序
    initial_feasibility, violations = init_sol.validate()
    
    log_printer.print(f"Initial Solution Objective: {init_sol.objective(): .2f}")
    log_printer.print(f"Vehicles Used in Initial Solution: {len(init_sol.vehicles)}")
    log_printer.print(f"Initial Solution Cost: {init_sol.calculate_cost(): .2f}")
    
    if not initial_feasibility:
        log_printer.print("Warning: The initial solution is infeasible!", color='bold red')
        # 输出详细 violations 信息
        if violations['negative_inventory']:
            log_printer.print(f"Negative inventory violations:")
            for key, inv in violations['negative_inventory']:
                log_printer.print(f"  Plant: {key[0]}, SKU: {key[1]}, Day: {key[2]}, Inventory: {inv}")
        if violations['veh_overload']:
            log_printer.print(f"Vehicle overload violations:")
            for v in violations['veh_overload']:
                veh = v['veh']
                log_printer.print(f"  Vehicle: Plant {veh.fact_id}, Dealer {veh.dealer_id}, Type {veh.type}, Day {veh.day}, Loaded: {v['loaded']}, Capacity: {v['cap']}")
        if violations['unmet_demand']:
            log_printer.print(f"Unmet demand violations:")
            for d in violations['unmet_demand']:
                log_printer.print(f"  Dealer: {d['dealer']}, SKU: {d['sku_id']}, Demand: {d['demand']}, Shipped: {d['shipped']}")
        # 终止程序
        import sys
        sys.exit(1)
    
    
    # 创建ALNS的实例
    alns = ALNS(rng=rnd.default_rng(seed=SEED))
    
    # 注册destroy算子
    register_destroy_operators(alns, log_printer)

    # 添加repair算子
    alns.add_repair_operator(demand_first_repair)  
    alns.add_repair_operator(greedy_repair)
    alns.add_repair_operator(inventory_balance_repair)
    alns.add_repair_operator(urgency_repair)
    alns.add_repair_operator(infeasible_repair)
    alns.add_repair_operator(local_search_repair)
    
    # alns的算子选择机制
    select = SegmentedRouletteWheel(
        scores=ALNSConfig.ROULETTE_SCORES,
        decay=ALNSConfig.ROULETTE_DECAY,
        seg_length=ALNSConfig.ROULETTE_SEG_LENGTH,
        num_destroy=6,
        num_repair=6
    )
    
    
    sa_accept = SimulatedAnnealing(
        start_temperature=ALNSConfig.SA_START_TEMP, 
        end_temperature=ALNSConfig.SA_END_TEMP, 
        step=ALNSConfig.SA_STEP, 
        method="exponential"
    )
    
    # # 使用自定义接受准则，确保只接受满足所有需求的解
    # accept = DemandConstraintAccept(sa_accept, data)
    
    # alns的停止准则
    stop = MaxRuntime(max_runtime=ALNSConfig.MAX_RUNTIME)
    
    # 添加验证最优解的回调函数
    def callback_on_best(state: SolutionState, rng, **kwargs) -> bool:
        """
        回调函数，在找到新的最优解时被调用
        主要用于记录和验证解的质量，不应中断搜索过程
        
        Parameters:
        -----------
        state : SolutionState
            当前的解状态
        rng : numpy.random.Generator
            随机数生成器
        **kwargs : dict
            其他关键字参数
        
        Returns:
        --------
        bool : 始终返回True以继续搜索过程
        """
        # 验证解的可行性（无副作用）
        is_feasible, violations = state.validate()
        
        # 仅在找到完全可行解时记录成功信息
        if is_feasible:
            log_printer.print(f"Found feasible solution with cost: {state.calculate_cost():.2f}")
        else:
            # 简化日志，避免过多输出干扰搜索过程
            unmet_count = len(violations.get('unmet_demand', []))
            if unmet_count > 0:
                log_printer.print(f"Best solution has {unmet_count} unmet demands")
        
        # 始终返回True以继续搜索，让ALNS自然终止
        return True

    # 创建追踪器
    tracker = ALNSTracker()
    
    # 注册回调函数
    alns.on_best(callback_on_best)  # 仅在找到最优解时调用
    # 只注册on_accept回调以跟踪迭代过程，避免重复调用
    alns.on_accept(tracker.on_iteration)
    
    # 运行ALNS算法
    result = alns.iterate(
        init_sol,
        select,
        sa_accept,
        stop
    )
    
    end_time = time.time()
    log_printer.print(f"Elapsed time: {end_time - start_time: .2f}s")
    
    # 获取追踪器的统计信息
    tracker_stats = tracker.get_statistics()
    
    # 输出最终统计信息
    log_printer.print(f"Total iterations tracked: {tracker_stats['total_iterations']}")
    log_printer.print(f"Best objective found: {tracker_stats['best_objective']:.4f}")
    log_printer.print(f"Final Gap: {tracker_stats['final_gap']:.2f}%")
    
    # 计算最终的Gap（使用ALNS结果）
    final_current_obj = result.statistics.objectives[-1]
    final_best_obj = result.best_state.objective()
    
    final_gap = calculate_gap(final_current_obj, final_best_obj)
    log_printer.print(f"ALNS Final Gap: {final_gap:.2f}%")
    
    # 绘制求解过程中目标函数值的变化
    plot_obj_changes(state, result)
    
    # 绘制算子性能
    plot_operator_performance(state, result)
    
    # 绘制Gap的变化
    plot_gap_changes(tracker)
    
    # 获取最终的解
    best = result.best_state
    
    # 对最终解进行评估
    final_feasibility, violations = best.validate()
    if not final_feasibility:
        log_printer.print("Warning: The final best solution is infeasible!", color='bold red')
    
    log_printer.print(f"Best Heuristic Solution Objective is {best.objective(): .2f}")
    log_printer.print(f"Vehicles Used in Best Solution: {len(best.vehicles)}")
    log_printer.print(f"Best Solution Cost: {best.calculate_cost(): .2f}")
    
    # 更新需求
    update_demands(best)
    # 更新库存
    update_inventory(best)
    # 输出结果
    output_result(best)
    # 统计结果
    result_state(data)
    print("=" * 100 + "\n")
    
    delete_index()
    aggregate_result(data)
    pass
    log_printer.print_title(f"The Vehicle Loading Plan is Done! {datetime.datetime.now()}")


def plot_obj_changes(state: SolutionState, result: Result):
    plt.figure(figsize=(10, 6))
    result.plot_objectives(title=f'Changes of Objective')
    file_loc = os.path.join(data.output_file_loc, 'images')
    if not os.path.exists(file_loc):
        os.makedirs(file_loc)
    file_name = f'Objective.svg'
    file = os.path.join(file_loc, file_name)
    plt.savefig(file, dpi=600)
    plt.close()


def plot_operator_performance(state: SolutionState, result: Result):
    # 创建图形
    plt.figure(figsize=(15, 12))
    
    # 创建子图网格，2行2列，左侧放图，右侧放表
    gs = plt.GridSpec(2, 2, width_ratios=[1.8, 1.2], height_ratios=[1, 1])  # 调整右侧比例，给表格更多空间
    
    # 创建左侧的柱状图子图
    ax1 = plt.subplot(gs[0, 0])  # 上半部分
    ax2 = plt.subplot(gs[1, 0])  # 下半部分
    
    # 创建右侧的表格子图
    ax_table1 = plt.subplot(gs[0, 1])  # 上半部分的表格
    ax_table2 = plt.subplot(gs[1, 1])  # 下半部分的表格
    
    # 获取destroy和repair算子的统计数据
    destroy_counts = result.statistics.destroy_operator_counts
    repair_counts = result.statistics.repair_operator_counts
    
    # 使用科研论文风格的配色方案
    colors = ['#3C9BC9', '#76CBB4', '#FC757B', '#FFE59B']  # 淡蓝、淡绿、淡红、淡黄
    
    # 绘制destroy operators柱状图
    plot_operator_group(ax1, destroy_counts, "Destroy operators", colors)
    
    # 绘制repair operators柱状图
    plot_operator_group(ax2, repair_counts, "Repair operators", colors)
    
    # 创建表格数据
    destroy_data = []
    repair_data = []
    
    # 准备表格数据
    for op_name, counts in destroy_counts.items():
        destroy_data.append([op_name.replace('_', ' ').title()] + list(counts[:4]))  # 格式化算子名称
    
    for op_name, counts in repair_counts.items():
        repair_data.append([op_name.replace('_', ' ').title()] + list(counts[:4]))  # 格式化算子名称
    
    # 表格列标签
    columns = ['Operator', 'Best', 'Better', 'Accepted', 'Rejected']
    
    # 在右侧创建表格
    ax_table1.axis('off')
    ax_table2.axis('off')
    
    # 计算第一列的最大宽度（考虑所有行，包括表头）
    max_width_first_col = max(
        max(len(str(row[0])) for row in destroy_data),
        max(len(str(row[0])) for row in repair_data),
        len(columns[0])  # 考虑表头的长度
    )
    
    # 计算其他列的宽度（考虑数据和表头）
    other_col_widths = []
    for i in range(1, 5):  # 对于每一列（除第一列外）
        # 获取该列所有数据的最大长度
        max_data_width = max(
            max(len(str(row[i])) for row in destroy_data),
            max(len(str(row[i])) for row in repair_data),
            len(columns[i])  # 考虑表头的长度
        )
        other_col_widths.append(max_data_width)
    
    # 设置列宽度（调整每列的相对宽度，并添加一些额外空间）
    col_widths = [max_width_first_col/8]  # 第一列宽度，减小除数以增加宽度
    # 为其他列添加宽度，根据内容长度动态调整
    col_widths.extend([width/6 + 0.15 for width in other_col_widths])  # 增加其他列的宽度和间隙
    
    # 创建destroy operators表格
    table1 = ax_table1.table(
        cellText=destroy_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        bbox=[0.1, 0.15, 1.1, 0.6],  # 调整表格位置和整体宽度
        colWidths=col_widths
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1.2, 1.5)
    
    # 设置表格样式
    for (i, j), cell in table1._cells.items():
        if i == 0:  # 表头行
            cell.set_facecolor('#E6E6E6')
            cell.set_text_props(weight='bold')
        if j == 0:  # 第一列
            cell.set_text_props(wrap=True)
    
    # 创建repair operators表格
    table2 = ax_table2.table(
        cellText=repair_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        bbox=[0.1, 0.15, 1.1, 0.6],  # 调整表格位置和整体宽度
        colWidths=col_widths
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.2, 1.5)
    
    # 设置表格样式
    for (i, j), cell in table2._cells.items():
        if i == 0:  # 表头行
            cell.set_facecolor('#E6E6E6')
            cell.set_text_props(weight='bold')
        if j == 0:  # 第一列
            cell.set_text_props(wrap=True)
    
    # 添加表格标题
    ax_table1.text(0.5, 0.85, 'Destroy Operators Performance', 
                   ha='center', va='center', fontsize=10, weight='bold')
    ax_table2.text(0.5, 0.85, 'Repair Operators Performance',
                   ha='center', va='center', fontsize=10, weight='bold')
    
    # 调整布局
    plt.subplots_adjust(
        left=0.1,
        right=0.98,
        bottom=0.2,  # 增加底部空间以容纳图例
        top=0.92,    # 减小top值，为总标题留出更多空间
        wspace=0.3,
        hspace=0.4
    )
    
    # 添加图例
    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(
        handles,
        ['Best', 'Better', 'Accepted', 'Rejected'],
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.5, 0.05),  # 调整图例位置
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10
    )
    
    # 添加总标题
    plt.suptitle(f'Performance of Operators', y=0.98, weight='bold')
    
    # 保存图形
    file_loc = os.path.join(data.output_file_loc, 'images')
    if not os.path.exists(file_loc):
        os.makedirs(file_loc)
    file_name = f'Performance of Operators.svg'
    file = os.path.join(file_loc, file_name)
    
    # 保存时确保所有内容都在可见区域内
    plt.savefig(file, dpi=600, bbox_inches='tight', pad_inches=0.2)
    
    # 关闭图形，释放内存
    plt.close()


def plot_operator_group(ax, operator_counts, title, colors):
    """辅助函数：绘制单个算子组的柱状图"""
    operator_names = [name.replace('_', ' ').title() for name in operator_counts.keys()]
    operator_counts = np.array(list(operator_counts.values()))
    cumulative_counts = operator_counts[:, :4].cumsum(axis=1)
    
    # 绘制堆叠条形图
    for idx in range(4):
        widths = operator_counts[:, idx]
        starts = cumulative_counts[:, idx] - widths
        bars = ax.barh(operator_names, widths, left=starts, height=0.5, 
                      color=colors[idx], label=['Best', 'Better', 'Accepted', 'Rejected'][idx])
    
    ax.set_title(title, pad=20, weight='bold')
    ax.set_xlabel("Iterations where operator resulted in this outcome (#)", labelpad=10)
    ax.set_ylabel("Operator", labelpad=10)
    
    # 添加网格线
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def clear_output_file(output_file_loc=output_loc):
    """清理输出文件，在每个数据集的文件夹中创建新的结果文件"""
    if not os.path.exists(output_file_loc):
        os.makedirs(output_file_loc)
    
    # 定义需要清理的文件列表
    files_to_clear = {
        'opt_result.csv': ['Index', 'day', 'plant_code', 'client_code', 'product_code', 'vehicle_id', 'vehicle_type', 'qty'],
        'non_fulfill.csv': ['client_code', 'product_code', 'volume']
    }
    
    # 清理并初始化每个文件
    for file_name, header in files_to_clear.items():
        with open(os.path.join(output_file_loc, file_name), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)


def output_result(state: SolutionState):
    """在对应数据集的文件夹中保存结果"""
    output_file = os.path.join(data.output_file_loc, 'opt_result.csv')
    df_result = create_dataframe(state)
    # header=False表示不将dataframe的列名写入文件
    df_result.to_csv(output_file, mode='a', header=False)
    

def update_demands(state: SolutionState):
    global data
    # 更新需求
    for veh in state.vehicles:
        for (sku_id, day), qty in veh.cargo.items():
            data.demands[veh.dealer_id, sku_id] -= qty
    
    # 获取未满足的需求数据
    non_fulfill_data = []
    for (dealer, sku_id), qty in data.demands.items():
        if qty > 0:
            non_fulfill_data.append({
                'client_code': dealer,
                'product_code': sku_id,
                'volume': qty
            })
    
    
    # 创建DataFrame并保存
    if non_fulfill_data:
        df_non_fulfill = pd.DataFrame(non_fulfill_data)
        output_file = os.path.join(data.output_file_loc, 'non_fulfill.csv')
        
        # 追加数据（不包含表头）
        df_non_fulfill.to_csv(output_file, mode='a', header=False, index=False)
    
    # 更新demands，只保留未满足的需求
    data.demands = {(dealer, sku_id): qty for (dealer, sku_id), qty in data.demands.items() if qty > 0}
    data.df_order = pd.DataFrame([
        {'client_code': dealer, 'product_code': sku_id, 'volume': qty}
        for (dealer, sku_id), qty in data.demands.items()
    ])


def update_inventory(state: SolutionState):
    global data
    # 保存所有库存信息，包括期初库存(day=0)
    sku_inv_left = {}
    
    # 遍历每个工厂的每种SKU
    for plant_id in data.plants:
        for sku_id in data.all_skus:
            # 获取期初库存
            current_inv = data.sku_initial_inv.get((plant_id, sku_id), 0)
            
            # 按时间顺序计算每天结束时的库存
            for day in range(1, data.horizons + 1):
                # 加上当天生产量
                current_inv += data.sku_prod_each_day.get((plant_id, sku_id, day), 0)
                
                # 减去当天配送量
                shipped_qty = 0
                for veh in state.vehicles:
                    if veh.fact_id == plant_id and (sku_id, day) in veh.cargo:
                        shipped_qty += veh.cargo[(sku_id, day)]
                current_inv -= shipped_qty
                
                # 保存当天结束时的库存
                sku_inv_left[(plant_id, sku_id, day)] = current_inv
    
    # 创建DataFrame，只包含day>=1的记录
    df_sku_inv_left = pd.DataFrame([
        {'plant_code': plant, 'product_code': sku_id, 'day': day, 'volume': qty}
        for (plant, sku_id, day), qty in sku_inv_left.items()
    ])
    
    output_file = os.path.join(data.output_file_loc, 'sku_inv_left.csv')
    df_sku_inv_left.to_csv(output_file, index=False)


def create_dataframe(state: SolutionState):
    rows = []
    for idx, veh in enumerate(state.vehicles):
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
    

def delete_index():
    file_loc = data.output_file_loc
    file_names = ['opt_result.csv', 'opt_details.csv', 'opt_summary.csv']
    for file_name in file_names:
        file_path = os.path.join(file_loc, file_name)
        df = pd.read_csv(file_path)
        df = df.drop(columns=['Index'])
        df.to_csv(file_path, index=False)


def aggregate_result(data: DataALNS):
    dataset = data.dataset_name
    output_location = data.output_file_loc
    
    df_summary = pd.read_csv(os.path.join(output_location, 'opt_summary.csv'))
    df_summary = df_summary.groupby(['day', 'plant_code', 'client_code', 'vehicle_id']).agg(
        {'vehicle_type': 'first', 'qty': 'sum', 'carry_standard_size': 'first',
        'min_standard_size': 'first', 'occupied_size': 'sum'})
    
    df_summary.to_csv(os.path.join(output_location, 'opt_summary.csv'))


def result_state(data: DataALNS):
    """处理并保存统计结果到对应的数据集文件夹"""
    dataset = data.dataset_name
    output_location = data.output_file_loc
    input_location = os.path.join(data.input_file_loc, dataset)
    
    df_result = pd.read_csv(os.path.join(output_location, 'opt_result.csv'))

    df_sku_size = pd.read_csv(os.path.join(input_location, 'product_size.csv'))

    # after merge, df_result has the 'standard_size' column, that means sku_size
    df_result = pd.merge(df_result, df_sku_size, on=['product_code'])

    df_all_veh = pd.read_csv(os.path.join(input_location, 'vehicle.csv'))
    # after merge, df_result has the 'carry_standard_size' column, 'min_standard_size' column
    # and 'cost_to_use' column
    df_result = pd.merge(df_result, df_all_veh, on=['vehicle_type'])
    del df_result['cost_to_use']

    df_result['occupied_size'] = df_result['qty'] * df_result['standard_size']
    df_result.to_csv(os.path.join(output_location, 'opt_details.csv'), index=False)

    df_stat1 = df_result.copy()
    df_stat1 = df_stat1.drop(['product_code', 'standard_size'], axis=1)

    df_tmp = df_stat1.groupby(['Index', 'day', 'plant_code', 'client_code', 'vehicle_id']).agg(
        {'vehicle_type': 'first', 'qty': 'sum', 'carry_standard_size': 'first',
        'min_standard_size': 'first', 'occupied_size': 'sum'})

    df_tmp.to_csv(os.path.join(output_location, 'opt_summary.csv'))

    df_tmp = df_tmp.reset_index()
    # 判断是否所有车辆中装载的SKU体积满足车辆容量约束
    is_cap_exceed = pd.Series(df_tmp['occupied_size'] <= df_tmp['carry_standard_size'])
    if is_cap_exceed.all():  # 所有值都为True，则返回True
        log_printer.print("Congratulations! "
                        "The volume of SKUs loaded in each vehicle is equal or smaller than vehicle's capacity.")
    else:
        log_printer.print("Warning! The volume of SKUs loaded in some vehicles has exceeded their capacities.")
        df_exceed = df_tmp[df_tmp["occupied_size"] > df_tmp["carry_standard_size"]]

        df_exceed_info = df_exceed.loc[:, ['vehicle_id', 'vehicle_type', 'carry_standard_size', 'occupied_size']]
        df_exceed_info['exceeded_volume'] = df_exceed_info['occupied_size'] - df_exceed_info['carry_standard_size']

        df_exceed_info.to_csv(os.path.join(output_location, "extra_volume.csv"), index=False)
        log_printer.print("The relevant information of this warning has been recorded.")
    
            
def plot_gap_changes(tracker: ALNSTracker):
    """
    绘制Gap的变化曲线
    
    Parameters:
    -----------
    tracker : ALNSTracker
        ALNS跟踪器
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(tracker.gaps) + 1), tracker.gaps)
    plt.title(f'Changes of Gap')
    plt.ylabel('Gap (%)')
    plt.xlabel('Iteration (#)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    file_loc = os.path.join(data.output_file_loc, 'images')
    if not os.path.exists(file_loc):
        os.makedirs(file_loc)
    file_name = f'Gaps.svg'
    file = os.path.join(file_loc, file_name)
    plt.savefig(file, dpi=600)
    plt.close()


if __name__ == "__main__":
    run_model()
