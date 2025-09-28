import sys
import os
from pathlib import Path
import pytest

from ALNSCode.InputDataALNS import DataALNS
from ALNSCode.visualization import plot_supply_chain_network_3d_enhanced
from ALNSCode.alns_config import default_config as ALNSConfig

# --- 配置 ---
# 从配置文件中读取数据集类型，并指定一个用于测试的数据集索引
DATASET_TYPE = ALNSConfig.DATASET_TYPE
DATASET_IDX = 1  # 使用第一个数据集作为测试样本
dataset_name = f'dataset_{DATASET_IDX}'

# --- 路径设置 ---
# 动态计算输入和输出目录的绝对路径
# 这样做可以确保脚本在不同环境下都能正确找到文件
try:
    # 假设脚本位于 ALNSCode/test/ 目录下
    par_path = Path(__file__).resolve().parent.parent.parent
except NameError:
    # 如果在交互式环境（如IPython）中运行，__file__可能未定义
    # 这种情况，我们假定当前工作目录是项目根目录
    par_path = Path.cwd()
    
input_file_loc = par_path / 'datasets' / 'multiple-periods' / DATASET_TYPE
output_file_loc = par_path / 'OutPut-ALNS' / 'multiple-periods' / DATASET_TYPE

def test_plot_3d_network():
    """
    测试 `plot_supply_chain_network_3d_enhanced` 函数的核心功能。
    
    该测试执行以下步骤：
    1. 初始化 DataALNS 对象并加载指定的数据集。
    2. 调用绘图函数生成3D供应链网络图。
    3. 检查预期的HTML输出文件是否已成功创建。
    """
    print("--- 开始测试3D供应链网络可视化功能 ---")
    
    # 1. 初始化并加载数据
    print(f"正在加载数据集: {DATASET_TYPE}/{dataset_name}...")
    try:
        data = DataALNS(input_file_loc, output_file_loc, dataset_name)
        data.load()
        print("数据加载成功。")
    except Exception as e:
        raise Exception(f"数据加载失败，错误: {e}")

    # 2. 调用绘图函数
    print("正在生成3D网络图...")
    try:
        # 构建完整的文件输出路径
        full_output_loc = os.path.join(output_file_loc, dataset_name)
        plot_supply_chain_network_3d_enhanced(data, full_output_loc)
        print("绘图函数执行完毕。")
    except Exception as e:
        raise Exception(f"调用 `plot_supply_chain_network_3d_enhanced` 函数时出错: {e}")

    # 3. 验证输出文件是否已创建
    expected_output_path = os.path.join(full_output_loc, 'images', 'supply_chain_network_3d_enhanced.html')
    print(f"检查输出文件是否存在于: {expected_output_path}")
    
    assert os.path.exists(expected_output_path), f"测试失败：未在指定位置找到输出文件 {expected_output_path}"
    
    print(f"--- 测试成功！增强版3D网络图已保存至: {expected_output_path} ---")

if __name__ == '__main__':
    # 当直接运行此脚本时，执行测试函数
    test_plot_3d_network()
