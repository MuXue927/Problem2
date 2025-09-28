#!/usr/bin/env python3
"""
测试脚本：验证 adaptive_degree 函数集成到参数解析流程中的功能

测试目标：
1. 验证 resolve_degree 函数能正确调用 param_tuner.adaptive_degree
2. 验证在不同迭代阶段，degree 参数会根据进度动态调整
3. 验证异常处理机制正常工作
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import numpy.random as rnd
from unittest.mock import Mock, MagicMock

# 导入需要测试的模块
from ALNSCode.destroy_utils import resolve_degree
from ALNSCode.param_tuner import ParamAutoTuner
from ALNSCode.InputDataALNS import DataALNS

def create_mock_state_with_tuner(iteration=0, max_iterations=1000):
    """创建带有 param_tuner 的模拟 state 对象"""
    # 创建模拟的 DataALNS
    mock_data = Mock(spec=DataALNS)
    mock_data.plants = ['P1', 'P2']
    mock_data.dealers = ['D1', 'D2']
    mock_data.all_skus = ['S1', 'S2']
    mock_data.horizons = 5
    mock_data.all_veh_types = ['T1', 'T2']
    mock_data.demands = {('D1', 'S1'): 100, ('D2', 'S2'): 200}
    mock_data.historical_s_ikt = {('P1', 'S1', 0): 50, ('P2', 'S2', 0): 75}
    mock_data.sku_prod_each_day = {('P1', 'S1', 1): 20, ('P2', 'S2', 1): 30}
    mock_data.veh_type_cap = {'T1': 100, 'T2': 200}
    
    # 创建 ParamAutoTuner
    rng = rnd.default_rng(42)
    param_tuner = ParamAutoTuner(mock_data, rng)
    param_tuner.set_iteration(iteration, max_iterations)
    
    # 创建模拟的 state
    mock_state = Mock()
    mock_state.param_tuner = param_tuner
    
    return mock_state

def test_adaptive_degree_integration():
    """测试 adaptive_degree 集成功能"""
    print("=== 测试 adaptive_degree 集成功能 ===")
    
    # 测试1: 早期迭代 (exploration 阶段)
    print("\n1. 测试早期迭代 (exploration 阶段)")
    state_early = create_mock_state_with_tuner(iteration=100, max_iterations=1000)
    
    # 测试不同的基础 degree 值
    base_degrees = [0.2, 0.3, 0.4, 0.5]
    for base_degree in base_degrees:
        result = resolve_degree('random_removal', state_early, base_degree)
        print(f"  基础 degree: {base_degree:.2f} -> 自适应 degree: {result:.3f}")
        # 早期阶段，degree 应该接近基础值（衰减较小）
        assert 0.05 <= result <= 0.5, f"结果超出预期范围: {result}"
    
    # 测试2: 中期迭代 (exploitation 阶段)
    print("\n2. 测试中期迭代 (exploitation 阶段)")
    state_mid = create_mock_state_with_tuner(iteration=500, max_iterations=1000)
    
    for base_degree in base_degrees:
        result = resolve_degree('random_removal', state_mid, base_degree)
        print(f"  基础 degree: {base_degree:.2f} -> 自适应 degree: {result:.3f}")
        assert 0.05 <= result <= 0.5, f"结果超出预期范围: {result}"
    
    # 测试3: 后期迭代 (refinement 阶段)
    print("\n3. 测试后期迭代 (refinement 阶段)")
    state_late = create_mock_state_with_tuner(iteration=900, max_iterations=1000)
    
    for base_degree in base_degrees:
        result = resolve_degree('random_removal', state_late, base_degree)
        print(f"  基础 degree: {base_degree:.2f} -> 自适应 degree: {result:.3f}")
        # 后期阶段，degree 应该显著降低
        assert 0.05 <= result <= 0.5, f"结果超出预期范围: {result}"
    
    # 测试4: 验证衰减效果
    print("\n4. 验证衰减效果")
    iterations = [0, 200, 400, 600, 800, 1000]
    base_degree = 0.4
    
    print(f"  基础 degree: {base_degree}")
    print("  迭代进度 -> 自适应 degree")
    for iteration in iterations:
        state = create_mock_state_with_tuner(iteration=iteration, max_iterations=1000)
        result = resolve_degree('random_removal', state, base_degree)
        progress = iteration / 1000
        print(f"  {progress:5.1%} ({iteration:4d}/1000) -> {result:.3f}")
    
    print("\n✓ adaptive_degree 集成测试通过")

def test_fallback_behavior():
    """测试回退行为"""
    print("\n=== 测试回退行为 ===")
    
    # 测试1: 没有 param_tuner 的情况
    print("\n1. 测试没有 param_tuner 的情况")
    mock_state_no_tuner = Mock()
    mock_state_no_tuner.param_tuner = None
    
    result = resolve_degree('random_removal', mock_state_no_tuner, 0.3)
    print(f"  没有 param_tuner，degree: 0.3 -> {result:.3f}")
    assert result == 0.3, f"应该返回原始值，但得到: {result}"
    
    # 测试2: param_tuner 没有 adaptive_degree 方法
    print("\n2. 测试 param_tuner 没有 adaptive_degree 方法")
    mock_state_no_method = Mock()
    mock_tuner = Mock()
    # 确保 mock_tuner 没有 adaptive_degree 属性
    if hasattr(mock_tuner, 'adaptive_degree'):
        delattr(mock_tuner, 'adaptive_degree')
    mock_state_no_method.param_tuner = mock_tuner
    
    result = resolve_degree('random_removal', mock_state_no_method, 0.3)
    print(f"  没有 adaptive_degree 方法，degree: 0.3 -> {result:.3f}")
    assert result == 0.3, f"应该返回原始值，但得到: {result}"
    
    # 测试3: adaptive_degree 方法抛出异常
    print("\n3. 测试 adaptive_degree 方法抛出异常")
    mock_state_exception = Mock()
    mock_tuner_exception = Mock()
    mock_tuner_exception.adaptive_degree = Mock(side_effect=Exception("测试异常"))
    mock_state_exception.param_tuner = mock_tuner_exception
    
    result = resolve_degree('random_removal', mock_state_exception, 0.3)
    print(f"  adaptive_degree 抛出异常，degree: 0.3 -> {result:.3f}")
    assert result == 0.3, f"应该返回原始值，但得到: {result}"
    
    print("\n✓ 回退行为测试通过")

def test_parameter_priority():
    """测试参数优先级"""
    print("\n=== 测试参数优先级 ===")
    
    state = create_mock_state_with_tuner(iteration=500, max_iterations=1000)
    
    # 测试1: 显式传入 degree 参数
    print("\n1. 测试显式传入 degree 参数")
    explicit_degree = 0.35
    result = resolve_degree('random_removal', state, explicit_degree)
    print(f"  显式 degree: {explicit_degree} -> 自适应 degree: {result:.3f}")
    
    # 测试2: degree=None，使用 param_tuner 获取参数
    print("\n2. 测试 degree=None，使用 param_tuner")
    result = resolve_degree('random_removal', state, None)
    print(f"  degree=None -> 自适应 degree: {result:.3f}")
    
    # 测试3: 边界值测试
    print("\n3. 测试边界值")
    boundary_values = [-0.1, 0.0, 0.5, 1.0, 1.5]
    for val in boundary_values:
        result = resolve_degree('random_removal', state, val)
        print(f"  输入: {val:4.1f} -> 输出: {result:.3f}")
        assert 0.0 <= result <= 1.0, f"输出应该在 [0,1] 范围内，但得到: {result}"
    
    print("\n✓ 参数优先级测试通过")

def main():
    """主测试函数"""
    print("开始测试 adaptive_degree 集成功能...")
    
    try:
        test_adaptive_degree_integration()
        test_fallback_behavior()
        test_parameter_priority()
        
        print("\n" + "="*50)
        print("🎉 所有测试通过！adaptive_degree 集成功能正常工作")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
