#!/usr/bin/env python3
"""
测试优化后的OutPutData和check_solution逻辑
"""

import sys
import os

from ALNSCode.OutputDataALNS import OutPutData
from ALNSCode.check_solution import check_solution
import pandas as pd


def test_output_data():
    """测试OutPutData类的优化逻辑"""
    print("=== 测试OutPutData类 ===\n")
    
    # 获取路径
    current_dir = os.path.dirname(__file__)
    par_path = os.path.dirname(current_dir)
    output_loc = os.path.join(par_path, 'OutPut-ALNS', 'multiple-periods', 'small')
    
    try:
        # 创建OutPutData实例
        output_data = OutPutData(output_loc, 'dataset_1')
        
        print("1. 测试数据加载:")
        output_data.load()
        
        print("\n2. 测试统计信息:")
        stats = output_data.get_summary_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n3. 测试数据访问:")
        print(f"   订单履行记录数: {len(output_data.order_fulfill)}")
        print(f"   车辆装载记录数: {len(output_data.vehicle_load)}")
        print(f"   未满足需求记录数: {len(output_data.non_fulfill)}")
        print(f"   库存记录数: {len(output_data.sku_inv_left)}")
        
        # 显示一些示例数据
        if output_data.order_fulfill:
            sample_key = list(output_data.order_fulfill.keys())[0]
            print(f"\n   订单履行示例: {sample_key} -> {output_data.order_fulfill[sample_key]}")
        
        print("\n✅ OutPutData测试通过")
        
    except Exception as e:
        print(f"❌ OutPutData测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_check_solution():
    """测试check_solution的优化逻辑"""
    print("\n=== 测试check_solution函数 ===\n")
    
    # 获取路径
    current_dir = os.path.dirname(__file__)
    par_path = os.path.dirname(current_dir)
    input_loc = os.path.join(par_path, 'datasets', 'multiple-periods', 'small')
    output_loc = os.path.join(par_path, 'OutPut-ALNS', 'multiple-periods', 'small')
    
    try:
        print("开始解决方案验证...")
        result = check_solution(input_loc, output_loc)
        
        status = "可行" if result else "不可行"
        print(f"\n验证结果: 解决方案是 {status} 的")
        print("\n✅ check_solution测试完成")
        
    except Exception as e:
        print(f"❌ check_solution测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_missing_files():
    """测试缺失文件的处理"""
    print("\n=== 测试缺失文件处理 ===\n")
    
    try:
        # 使用不存在的路径
        output_data = OutPutData("/nonexistent/path", "nonexistent_dataset")
        output_data.load()
        
        print("处理缺失文件的结果:")
        stats = output_data.get_summary_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n✅ 缺失文件处理测试通过")
        
    except Exception as e:
        print(f"❌ 缺失文件处理测试失败: {str(e)}")


if __name__ == "__main__":
    print("开始测试优化后的逻辑...\n")
    
    test_output_data()
    test_check_solution()
    test_missing_files()
    
    print("\n所有测试完成！")