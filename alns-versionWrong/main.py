import os
import csv
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from alnsopt import *
from alns import ALNS
from alns.accept import SimulatedAnnealing, RecordToRecordTravel
from alns.select import RouletteWheel, SegmentedRouletteWheel
from alns.stop import MaxRuntime, MaxIterations, NoImprovement
from alns.Result import Result
from alns.Outcome import Outcome
from optutility import LogPrinter
from alnstrack import ALNSTracker, calculate_gap
SEED = 15926535

data: DataALNS
# 获取当前.py文件所在的目录，这样可以确保路径正确，即使文件被移动到其他位置也不会影响
current_dir = os.path.dirname(__file__)
# 当前.py文件所在目录的上一级目录
par_path = os.path.dirname(current_dir)
input_loc = os.path.join(par_path, 'datasets', 'multiple-periods', 'small')
output_loc = os.path.join(par_path, 'OutPut-ALNS', 'multiple-periods', 'small')

log_printer = LogPrinter(time.time())


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
    
    for day in range(1, data.horizons + 1):
        log_printer.print(f"Start optimizing! Day {day} / Now {datetime.datetime.now()}")
        # 创建SolutionState的实例
        state = SolutionState(day, data)
        # 获取初始解
        log_printer.print("Initializing the solution...")
        init_sol = initial_solution(state, rng=rnd.default_rng(seed=SEED))
        log_printer.print(f"Initial Solution Objective: {init_sol.objective(): .2f}")
        
        # 创建ALNS的实例
        alns = ALNS(rng=rnd.default_rng(seed=SEED))
        # 添加destroy算子
        alns.add_destroy_operator(random_removal)
        alns.add_destroy_operator(worst_removal)
        alns.add_destroy_operator(infeasible_removal)
        alns.add_destroy_operator(demand_removal)
        # 添加repair算子
        alns.add_repair_operator(greedy_repair)
        alns.add_repair_operator(inventory_balance_repair)
        alns.add_repair_operator(urgency_repair)
        
        # alns的算子选择机制
        select = SegmentedRouletteWheel(
            scores=[5, 2, 1, 0.5],
            decay=0.8,
            seg_length=500,
            num_destroy=4,
            num_repair=3
        )
        
        # alns接受新解的机制
        accept = SimulatedAnnealing(start_temperature=1_000, end_temperature=1, 
                            step=1 - 1e-3, method="exponential")
        
        # alns的停止准则
        stop = MaxRuntime(max_runtime=120)
        # stop = MaxIterations(max_iterations=1_000)
        # stop = NoImprovement(max_iterations=1_000)
        
        # 运行alns算法
        log_printer.print("Running ALNS...")
        start_time = time.time()
        
        # 创建Gap结果输出文件路径
        file_loc = os.path.join(output_file_loc, 'gaps')
        if not os.path.exists(file_loc):
            os.makedirs(file_loc)
        gap_output_file = os.path.join(file_loc, f'gaps_day_{day}.csv')
        
        # 创建一个跟踪器来记录ALNS迭代过程中的信息
        tracker = ALNSTracker(output_file=gap_output_file)
        
        # 初始化跟踪器的初始解
        tracker.update(init_sol)
        
        # 定义回调函数，用于在每次迭代后更新跟踪器
        # 使用python中闭包的特性，回调函数可以访问外部作用域中的tracker变量
        def callback_on_iteration(state, rng):
            tracker.update(state)
        
        # 注册回调函数，在每种结果类型后都调用
        alns.on_best(callback_on_iteration)
        alns.on_better(callback_on_iteration)
        alns.on_accept(callback_on_iteration)
        alns.on_reject(callback_on_iteration)
        
        result = alns.iterate(init_sol, select, accept, stop)
        
        end_time = time.time()
        log_printer.print(f"Elapsed time: {end_time - start_time: .2f}s")
        
        # 计算最终的Gap
        final_current_obj = result.statistics.objectives[-1]
        final_best_obj = result.best_state.objective()
        final_gap = calculate_gap(final_current_obj, final_best_obj)
        log_printer.print(f"Final Gap: {final_gap:.2f}%")
        
        # 绘制求解过程中目标函数值的变化
        plot_obj_changes(state, result)
        
        # 绘制算子性能
        plot_operator_performance(state, result)
        
        # 绘制Gap的变化
        plot_gap_changes(tracker, day)
        
        # 获取最终的解
        best = result.best_state
        log_printer.print(f"Beset Heuristic Solution Objective is {best.objective(): .2f}")
        
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
    result.plot_objectives(title=f'Changes of Objective in Day {state.day}')
    file_loc = os.path.join(data.output_file_loc, 'images')
    if not os.path.exists(file_loc):
        os.makedirs(file_loc)
    file_name = f'Objective in Day {state.day}.svg'
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
    plt.suptitle(f'Performance of Operators in Day {state.day}', y=0.98, weight='bold')
    
    # 保存图形
    file_loc = os.path.join(data.output_file_loc, 'images')
    if not os.path.exists(file_loc):
        os.makedirs(file_loc)
    file_name = f'Performance of Operators in Day {state.day}.svg'
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
        'extral_fulfill.csv': ['client_code', 'product_code', 'day', 'volume']
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
            data.demands[veh.dealer_id, sku_id, day] -= qty
    
    # 获取超额满足的需求数据
    extral_fulfill_data = []
    for (dealer, sku_id, day), qty in data.demands.items():
        if qty <= 0:
            extral_fulfill_data.append({
                'client_code': dealer,
                'product_code': sku_id,
                'day': day,
                'volume': -qty  # 转换为正值表示超额满足量
            })
    
    # 创建DataFrame并保存
    if extral_fulfill_data:
        df_extra_fulfill = pd.DataFrame(extral_fulfill_data)
        output_file = os.path.join(data.output_file_loc, 'extral_fulfill.csv')
        
        # 追加数据（不包含表头）
        df_extra_fulfill.to_csv(output_file, mode='a', header=False, index=False)
    
    # 更新demands，只保留未满足的需求
    data.demands = {(dealer, sku_id, day): qty for (dealer, sku_id, day), qty in data.demands.items() if qty > 0}
    data.df_order = pd.DataFrame([
        {'client_code': dealer, 'product_code': sku_id, 'day': day, 'volume': qty}
        for (dealer, sku_id, day), qty in data.demands.items()
    ])


def update_inventory(state: SolutionState):
    global data
    # sku_inv_left是生产基地中剩余的库存，qty=0表示没有剩余库存
    # 对于多周期问题，只统计 t=T 时的剩余库存
    sku_inv_left = {(plant, sku_id, day): qty for (plant, sku_id, day), qty in state.s_ikt.items() 
                    if day == data.horizons}
    
    df_sku_inv_left = pd.DataFrame({
        (plant, sku_id, day, qty) for (plant, sku_id, day), qty in sku_inv_left.items()
    }, columns=['plant_code', 'product_code', 'day', 'volume'])
    
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
    
            
def plot_gap_changes(tracker: ALNSTracker, day: int):
    """
    绘制Gap的变化曲线
    
    Parameters:
    -----------
    tracker : ALNSTracker
        ALNS跟踪器
    day : int
        当前优化的天数
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(tracker.gaps) + 1), tracker.gaps)
    plt.title(f'Changes of Gap in Day {day}')
    plt.ylabel('Gap (%)')
    plt.xlabel('Iteration (#)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    file_loc = os.path.join(data.output_file_loc, 'images')
    if not os.path.exists(file_loc):
        os.makedirs(file_loc)
    file_name = f'Gap in Day {day}.svg'
    file = os.path.join(file_loc, file_name)
    plt.savefig(file, dpi=600)
    plt.close()


if __name__ == "__main__":
    run_model()
