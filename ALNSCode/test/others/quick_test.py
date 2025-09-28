#!/usr/bin/env python3
"""
快速验证库存一致性的脚本
"""
import os
import sys
from pathlib import Path

current_dir = Path(__file__).parent

from ALNSCode.InputDataALNS import DataALNS
from ALNSCode.OutputDataALNS import OutPutData
from ALNSCode.alns_config import default_config as ALNSConfig

def quick_validation():
    """快速验证库存一致性"""
    input_loc = current_dir.parent / 'datasets' / 'multiple-periods' / 'small'
    output_loc = current_dir.parent / 'OutPut-ALNS' / 'multiple-periods' / 'small'
    dataset_name = f'dataset_{ALNSConfig.DATASET_IDX}'
    try:
        print("🔍 开始快速验证...")
        dataset_output_loc = output_loc / dataset_name
        input_data = DataALNS(str(input_loc), str(dataset_output_loc), dataset_name)
        input_data.load()
        output_data = OutPutData(str(output_loc), dataset_name)
        output_data.load()
        print(f"📈 数据概览: 工厂: {len(input_data.plants)}, 经销商: {len(input_data.dealers)}, SKU: {len(input_data.all_skus)}")
        output_stats = output_data.get_summary_stats()
        print(f"   输出统计: {output_stats}")
        print("🔬 检查库存流转...")
        
        violations = []
        for plant_id in input_data.plants:
            for sku_id in input_data.all_skus:
                current_inv = input_data.sku_initial_inv.get((plant_id, sku_id), 0)
                for day in range(1, input_data.horizons + 1):
                    current_inv += input_data.sku_prod_each_day.get((plant_id, sku_id, day), 0)
                    for dealer_id in input_data.dealers:
                        current_inv -= output_data.order_fulfill.get((day, plant_id, dealer_id, sku_id), 0)
                    if current_inv < 0:
                        violations.append(f"工厂 {plant_id} 的 SKU {sku_id} 在第 {day} 天出现负库存: {current_inv}")
        
        print("="*60)
        print("🎯 验证结果:")
        
        if len(violations) == 0:
            print("✅ 库存验证通过！没有发现负库存问题。\n🎉 修复成功！")
        else:
            print(f"❌ 发现 {len(violations)} 个库存违反:")
            for i, violation in enumerate(violations[:5]):
                print(f"   {i+1}. {violation}")
            if len(violations) > 5:
                print(f"   ... 还有 {len(violations) - 5} 个违反")
            print("🔧 需要进一步修复。")
        print("="*60)
        
        assert len(violations) == 0, f"Found {len(violations)} inventory violations"
    
    except AssertionError:
        raise
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        raise

if __name__ == "__main__":
    try:
        quick_validation()
        print("✨ 验证完成：修复生效！")
    except AssertionError as ae:
        print("⚠️  验证完成：仍需修复。", ae)
    except Exception as e:
        print("⚠️  验证过程出错：", e)
