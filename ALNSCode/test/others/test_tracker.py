#!/usr/bin/env python3
"""
测试ALNSTracker类优化后的逻辑
"""

from ALNSCode.alnstrack import ALNSTracker, calculate_gap
import random


class MockSolutionState:
    """模拟SolutionState类用于测试"""
    def __init__(self, objective_value=1000.0, is_feasible=True):
        self._objective_value = objective_value
        self._is_feasible = is_feasible
    
    def objective(self):
        return self._objective_value
    
    def validate(self):
        return self._is_feasible, {}
    
    def set_objective(self, value):
        self._objective_value = value
    
    def set_feasible(self, feasible):
        self._is_feasible = feasible


def test_tracker_logic():
    """测试追踪器的逻辑"""
    print("=== 测试ALNSTracker类的逻辑 ===\n")
    
    # 创建追踪器
    tracker = ALNSTracker()
    
    print("1. 测试初始状态:")
    print(f"   初始迭代数: {tracker.iteration}")
    print(f"   初始最优目标值: {tracker.best_obj}")
    print(f"   初始当前目标值: {tracker.current_obj}")
    
    # 模拟几次迭代
    rng = random.Random(42)
    test_scenarios = [
        (1000.0, True),   # 第一个可行解
        (950.0, True),    # 更好的可行解
        (980.0, False),   # 更差的不可行解  
        (920.0, True),    # 最优的可行解
        (940.0, True),    # 较好的可行解
    ]
    
    print("\n2. 模拟迭代过程:")
    for i, (obj, feasible) in enumerate(test_scenarios, 1):
        # 创建模拟状态
        state = MockSolutionState(obj, feasible)
        
        # 调用回调函数
        tracker.on_iteration(state, rng)
        
        feasible_str = "可行" if feasible else "不可行"
        print(f"   迭代 {i}: 目标值={obj:.1f} ({feasible_str}), 当前最优={tracker.best_obj:.1f}, Gap={calculate_gap(tracker.current_obj, tracker.best_obj):.2f}%")
    
    print("\n3. 测试统计信息:")
    stats = tracker.get_statistics()
    for key, value in stats.items():
        if key in ['objectives_history', 'gaps_history', 'best_solution']:
            if isinstance(value, list):
                print(f"   {key}: [长度={len(value)}]")
            else:
                print(f"   {key}: 存在={value is not None}")
        elif key == 'elapsed_time':
            print(f"   {key}: {value:.4f}s")
        else:
            print(f"   {key}: {value}")
    
    print("\n4. 测试Gap计算函数:")
    test_cases = [
        (1000, 800),  # 正常情况
        (800, 800),   # 相等情况
        (0, 0),       # 都为0
        (100, 0),     # 最优为0
    ]
    
    for current, best in test_cases:
        gap = calculate_gap(current, best)
        print(f"   Current: {current}, Best: {best} => Gap: {gap:.2f}%")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_tracker_logic()