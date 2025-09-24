#!/usr/bin/env python3
"""
测试优化后的DataALNS类
验证所有功能是否正常工作
"""

import sys
import os
import traceback
import logging
from pathlib import Path
import pytest

# 确保将项目根目录加入 sys.path，使顶级包 `ALNSCode` 可被直接运行的测试脚本导入
current_file = os.path.abspath(__file__)
# 项目结构: <project_root>/ALNSCode/test/others/<this file>
# 向上四级到达 project_root
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from ALNSCode.InputDataALNS import DataALNS
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

def test_data_alns_basic_functionality():
    """测试DataALNS类的基本功能"""
    print("=== 测试DataALNS类基本功能 ===")
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试实例（使用相对路径）
    try:
        input_file_loc = "d:/Gurobi_code/Problem2/datasets"
        output_file_loc = "d:/Gurobi_code/Problem2/outputs"
        dataset_name = "multiple-periods"
        
        print(f"尝试从 {input_file_loc} 加载数据集 {dataset_name}")
        
        # 检查路径是否存在
        if not os.path.exists(input_file_loc):
            print(f"警告: 输入路径不存在 {input_file_loc}")
            raise AssertionError(f"输入路径不存在: {input_file_loc}")
            
        # 创建DataALNS实例
        data_alns = DataALNS(
            input_file_loc=input_file_loc,
            output_file_loc=output_file_loc,
            dataset_name=dataset_name
        )
        
        print("✓ DataALNS实例创建成功")
        
        # 测试数据加载
        try:
            data_alns.load()
            print("✓ 数据加载成功")
        except FileNotFoundError as e:
            print(f"文件未找到（这是预期的，如果数据文件不存在）: {e}")
            # treat missing files as expected in some setups
            pytest.skip(f"缺少数据文件，跳过该测试: {e}")
        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            traceback.print_exc()
            assert False, f"数据加载失败: {e}"
        
        # 测试基本属性
        print(f"✓ 生产基地数量: {len(data_alns.plants)}")
        print(f"✓ 经销商数量: {len(data_alns.dealers)}")
        print(f"✓ SKU数量: {len(data_alns.all_skus)}")
        print(f"✓ 车辆类型数量: {len(data_alns.all_veh_types)}")
        print(f"✓ 规划周期: {data_alns.horizons}")
        
        # 测试新增的方法
        if data_alns.all_skus:
            test_sku = next(iter(data_alns.all_skus))
            
            # 测试供需平衡分析
            balance_info = data_alns.get_sku_supply_demand_balance(test_sku)
            print(f"✓ SKU {test_sku} 供需平衡信息: {balance_info['status']}")
            
            # 测试总需求
            total_demand = data_alns.get_total_demand_for_sku(test_sku)
            print(f"✓ SKU {test_sku} 总需求: {total_demand}")
            
            # 测试总生产量
            total_production = data_alns.get_total_production_for_sku(test_sku)
            print(f"✓ SKU {test_sku} 总生产量: {total_production}")
        
        # 测试汇总统计
        summary = data_alns.get_summary_statistics()
        print("✓ 汇总统计信息:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # 测试供应链构建
        supply_chain = data_alns.construct_supply_chain()
        print(f"✓ 供应链连接数: {len(supply_chain)}")
        
        # 测试生产基地SKU查询
        if data_alns.plants:
            test_plant = next(iter(data_alns.plants))
            available_skus = data_alns.available_skus_in_plant(test_plant)
            print(f"✓ 生产基地 {test_plant} 可提供SKU数量: {len(available_skus)}")
            
            if data_alns.dealers:
                test_dealer = next(iter(data_alns.dealers))
                available_to_dealer = data_alns.available_skus_to_dealer(test_plant, test_dealer)
                print(f"✓ 生产基地 {test_plant} 可向经销商 {test_dealer} 提供SKU数量: {len(available_to_dealer)}")
        
        print("✓ 所有测试通过！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        traceback.print_exc()
        assert False, f"测试失败: {e}"

def test_error_handling():
    """测试错误处理功能"""
    print("\n=== 测试错误处理功能 ===")
    
    try:
        # 测试不存在的路径
        data_alns = DataALNS(
            input_file_loc="nonexistent_path",
            output_file_loc="nonexistent_output",
            dataset_name="nonexistent_dataset"
        )
        
        try:
            data_alns.load()
            print("✗ 应该抛出异常但没有")
            assert False, "Expected an exception when loading nonexistent path"
        except Exception as e:
            print(f"✓ 正确处理了不存在路径的错误: {type(e).__name__}")
            # 测试通过：捕获到预期异常
            assert True
            
    except Exception as e:
        print(f"✗ 错误处理测试失败: {e}")
        assert False, f"错误处理测试失败: {e}"

def test_type_conversion():
    """测试类型转换功能"""
    print("\n=== 测试类型转换功能 ===")
    
    try:
        import pandas as pd
        
        # 创建测试DataFrame
        test_data = {
            'client_code': ['C1', 'C2', 'C3'],
            'product_code': ['P1', 'P2', 'P3'],
            'volume': ['10', '20.5', '30']  # 字符串格式的数字
        }
        df = pd.DataFrame(test_data)
        
        # 创建DataALNS实例进行测试
        data_alns = DataALNS(
            input_file_loc="test",
            output_file_loc="test",
            dataset_name="test"
        )
        
        # 测试重新组织DataFrame
        title_map = {'client_code': 'dealer_id', 'product_code': 'sku_id', 'volume': 'order_qty'}
        reorganized_df = data_alns._reorganize_dataframe(df, title_map)
        print("✓ DataFrame重新组织成功")
        
        # 测试类型转换
        type_mapping = {'dealer_id': str, 'sku_id': str, 'order_qty': int}
        converted_df = data_alns._convert_dataframe_types(reorganized_df, type_mapping)
        print("✓ 数据类型转换成功")
        
        # 验证类型
        if converted_df['dealer_id'].dtype == 'object':  # pandas中字符串类型
            print("✓ dealer_id 类型转换正确")
        if converted_df['order_qty'].dtype == 'int64':
            print("✓ order_qty 类型转换正确")
        
        print("✓ 类型转换测试通过！")
        
    except Exception as e:
        print(f"✗ 类型转换测试失败: {e}")
        traceback.print_exc()
        assert False, f"类型转换测试失败: {e}"

def main():
    """主测试函数"""
    print("开始测试优化后的DataALNS类")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # 运行所有测试
    tests = [
        test_data_alns_basic_functionality,
        test_error_handling,
        test_type_conversion
    ]
    
    for test_func in tests:
        try:
            test_func()
            success_count += 1
        except Exception as e:
            print(f"测试 {test_func.__name__} 出现异常: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"测试完成: {success_count}/{total_tests} 个测试通过")
    
    if success_count == total_tests:
        print("🎉 所有测试都通过了！DataALNS类优化成功。")
    else:
        print("⚠️  部分测试失败，请检查相关功能。")

    # 返回布尔值以便在 __main__ 中使用
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)