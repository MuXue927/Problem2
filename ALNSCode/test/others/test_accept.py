#!/usr/bin/env python3
"""
测试DemandConstraintAccept类优化后的逻辑
"""

from ALNSCode.accept import DemandConstraintAccept
from alns.accept import SimulatedAnnealing
import random
import numpy as np


class MockDataALNS:
    """模拟DataALNS类用于测试"""
    def __init__(self):
        self.demands = {
            ('dealer1', 'sku1'): 100,
            ('dealer1', 'sku2'): 50,
            ('dealer2', 'sku1'): 80,
        }


class MockSolutionState:
    """模拟SolutionState类用于测试"""
    def __init__(self, objective_value=1000.0, shipped_quantities=None, is_feasible=True):
        self._objective_value = objective_value
        self._shipped_quantities = shipped_quantities or {}
        self._is_feasible = is_feasible
    
    def objective(self):
        return self._objective_value
    
    def validate(self):
        """模拟验证方法"""
        violations = {'unmet_demand': []}
        
        if not self._is_feasible:
            # 模拟一些违反需求的情况
            violations['unmet_demand'] = [
                {'dealer': 'dealer1', 'sku_id': 'sku1', 'demand': 100, 'shipped': 80},
                {'dealer': 'dealer2', 'sku_id': 'sku1', 'demand': 80, 'shipped': 60}
            ]
        
        return self._is_feasible, violations
    
    def compute_shipped(self):
        return self._shipped_quantities


def test_demand_constraint_accept():
    """测试DemandConstraintAccept类的逻辑"""
    print("=== 测试DemandConstraintAccept类的逻辑 ===\n")
    
    # 创建测试数据
    data = MockDataALNS()
    
    # 创建基础接受准则（模拟退火）
    sa_accept = SimulatedAnnealing(
        start_temperature=1000.0,
        end_temperature=1.0,
        step=0.01,
        method="linear"
    )
    
    # 创建自定义接受准则
    demand_accept = DemandConstraintAccept(sa_accept, data, penalty_factor=500.0)
    
    print("1. 测试初始状态:")
    stats = demand_accept.get_statistics()
    print(f"   初始统计信息: {stats}")
    
    # 创建随机数生成器
    rng = np.random.default_rng(42)
    
    # 测试场景
    test_scenarios = [
        # (old_obj, new_obj, new_feasible, description)
        (1000.0, 900.0, True, "可行解且更优"),
        (900.0, 950.0, True, "可行解但较差"),
        (950.0, 800.0, False, "不可行解但目标值更优"),
        (800.0, 850.0, True, "可行解且适中"),
        (850.0, 1200.0, False, "不可行解且较差"),
    ]
    
    print("\n2. 测试不同场景:")
    
    best_solution = MockSolutionState(800.0, is_feasible=True)  # 最优解
    current_solution = MockSolutionState(1000.0, is_feasible=True)  # 当前解
    
    for i, (old_obj, new_obj, new_feasible, description) in enumerate(test_scenarios, 1):
        current_solution._objective_value = old_obj
        candidate_solution = MockSolutionState(new_obj, is_feasible=new_feasible)
        
        # 调用接受准则 - 使用正确的参数顺序
        accept = demand_accept(rng, best_solution, current_solution, candidate_solution)
        
        print(f"   场景 {i} - {description}:")
        print(f"     当前目标值: {old_obj:.1f}")
        print(f"     候选目标值: {new_obj:.1f}")
        print(f"     候选可行性: {new_feasible}")
        print(f"     接受结果: {'接受' if accept else '拒绝'}")
        print()
    
    print("3. 测试大量迭代（观察统计信息）:")
    
    # 模拟100次迭代
    for i in range(95):  # 加上前面5次，总共100次
        # 随机生成解
        current_obj = random.uniform(500, 1500)
        candidate_obj = random.uniform(500, 1500)
        candidate_feasible = random.random() > 0.3  # 70%概率可行
        
        current_solution._objective_value = current_obj
        candidate_solution = MockSolutionState(candidate_obj, is_feasible=candidate_feasible)
        demand_accept(rng, best_solution, current_solution, candidate_solution)
    
    print("\n4. 最终统计信息:")
    final_stats = demand_accept.get_statistics()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_demand_constraint_accept()