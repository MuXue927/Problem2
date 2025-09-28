#!/usr/bin/env python3
"""
调试版本的ALNS主程序，添加库存一致性验证
"""
import os
import sys
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent

from ALNSCode.main import ALNSOptimizer
from ALNSCode.alns_config import default_config as ALNSConfig

def debug_main():
    """调试版本的主函数，增加库存验证"""
    
    print("🚀 开始调试运行ALNS算法...")
    
    # 创建优化器
    optimizer = ALNSOptimizer()
    
    # 加载数据
    dataset_name = f"dataset_{ALNSConfig.DATASET_IDX}"
    if not optimizer.load_data(dataset_name):
        print("❌ 数据加载失败")
        raise AssertionError("数据加载失败")
    
    # 创建初始解
    initial_solution = optimizer.create_initial_solution()
    if initial_solution is None:
        print("❌ 初始解创建失败")
        raise AssertionError("初始解创建失败")
    
    print("✅ 初始解创建成功")
    
    # 验证初始解
    initial_feasible, initial_violations = initial_solution.validate()
    print(f"📊 初始解可行性: {initial_feasible}")
    if initial_violations['negative_inventory']:
        print(f"⚠️  初始解负库存问题: {len(initial_violations['negative_inventory'])} 个")
    
    # 运行ALNS优化
    print("🔄 开始ALNS优化...")
    success = optimizer.run_optimization(dataset_name)
    
    if success:
        print("✅ ALNS优化完成")
        
        # 验证最终解
        final_feasible, final_violations = optimizer.best_solution.validate()
        print(f"📊 最终解可行性: {final_feasible}")
        if final_violations['negative_inventory']:
            print(f"⚠️  最终解负库存问题: {len(final_violations['negative_inventory'])} 个")
        
        # 处理结果
        print("💾 处理和保存结果...")
        optimizer._process_results()
        
        print("🎉 调试运行完成!")
        return
    else:
        print("❌ ALNS优化失败")
        raise AssertionError("ALNS 优化失败")

if __name__ == "__main__":
    success = debug_main()
    if success:
        print("✨ 调试运行成功完成")
        
        # 运行快速验证
        print("\n🔍 运行快速验证...")
        os.system("python quick_test.py")
    else:
        print("💥 调试运行失败")
