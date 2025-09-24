"""
针对ALNS场景的functools.partial实用示例
展示如何在您的项目中使用partial
"""
from functools import partial
import random
from typing import Dict, List, Any

# ========================================
# 模拟您的ALNS场景
# ========================================

class MockState:
    """模拟SolutionState"""
    def __init__(self, vehicles_count=10):
        self.vehicles_count = vehicles_count
    
    def copy(self):
        return MockState(self.vehicles_count)

class MockRNG:
    """模拟随机数生成器"""
    def choice(self, seq, size=None, replace=True):
        if size is None:
            return random.choice(seq)
        return random.choices(seq, k=size) if replace else random.sample(seq, size)

# ========================================
# 1. 基础destroy算子（带参数）
# ========================================

def random_removal(state: MockState, rng: MockRNG, degree: float = 0.25) -> MockState:
    """随机移除算子"""
    new_state = state.copy()
    remove_count = int(new_state.vehicles_count * degree)
    new_state.vehicles_count -= remove_count
    print(f"Random removal: 移除比例={degree:.2%}, 移除了{remove_count}辆车, 剩余{new_state.vehicles_count}辆")
    return new_state

def worst_removal(state: MockState, rng: MockRNG, max_remove_ratio: float = 0.5) -> MockState:
    """最差移除算子"""
    new_state = state.copy()
    max_remove = int(new_state.vehicles_count * max_remove_ratio)
    actual_remove = min(max_remove, random.randint(1, max_remove))
    new_state.vehicles_count -= actual_remove
    print(f"Worst removal: 最大移除比例={max_remove_ratio:.2%}, 实际移除了{actual_remove}辆车, 剩余{new_state.vehicles_count}辆")
    return new_state

def shaw_removal(state: MockState, rng: MockRNG, relatedness_threshold: float = 0.7) -> MockState:
    """Shaw移除算子"""
    new_state = state.copy()
    remove_count = random.randint(1, int(new_state.vehicles_count * 0.4))
    new_state.vehicles_count -= remove_count
    print(f"Shaw removal: 相关性阈值={relatedness_threshold:.2f}, 移除了{remove_count}辆车, 剩余{new_state.vehicles_count}辆")
    return new_state

# ========================================
# 2. 使用partial创建不同配置的算子
# ========================================

def create_alns_operators():
    """创建ALNS算子的工厂函数"""
    
    # 不同强度的random removal
    operators = {
        'random_gentle': partial(random_removal, degree=0.15),      # 温和移除15%
        'random_normal': partial(random_removal, degree=0.25),      # 正常移除25%
        'random_aggressive': partial(random_removal, degree=0.35),  # 激进移除35%
        
        # 不同配置的worst removal
        'worst_conservative': partial(worst_removal, max_remove_ratio=0.3),
        'worst_moderate': partial(worst_removal, max_remove_ratio=0.5),
        'worst_aggressive': partial(worst_removal, max_remove_ratio=0.7),
        
        # 不同阈值的shaw removal
        'shaw_strict': partial(shaw_removal, relatedness_threshold=0.9),
        'shaw_normal': partial(shaw_removal, relatedness_threshold=0.7),
        'shaw_loose': partial(shaw_removal, relatedness_threshold=0.5),
    }
    
    return operators

# ========================================
# 3. 模拟ALNS框架
# ========================================

class SimpleALNS:
    """简化的ALNS框架"""
    
    def __init__(self):
        self.destroy_operators = []
        self.operator_names = []
    
    def add_destroy_operator(self, operator_func, name: str = None):
        """添加destroy算子"""
        self.destroy_operators.append(operator_func)
        if name:
            self.operator_names.append(name)
        else:
            self.operator_names.append(f"operator_{len(self.destroy_operators)}")
        print(f"✓ 已注册算子: {self.operator_names[-1]}")
    
    def run_iteration(self, state: MockState, rng: MockRNG):
        """运行一次迭代"""
        # 随机选择一个算子
        idx = random.randint(0, len(self.destroy_operators) - 1)
        operator = self.destroy_operators[idx]
        operator_name = self.operator_names[idx]
        
        print(f"\n--- 第{getattr(self, '_iteration', 0) + 1}次迭代 ---")
        print(f"选择算子: {operator_name}")
        print(f"初始状态: {state.vehicles_count}辆车")
        
        # 执行算子
        new_state = operator(state, rng)
        
        setattr(self, '_iteration', getattr(self, '_iteration', 0) + 1)
        return new_state

# ========================================
# 4. 配置驱动的算子管理
# ========================================

class OperatorConfig:
    """算子配置类"""
    
    # Random removal配置
    RANDOM_REMOVAL_CONFIGS = [
        {'degree': 0.15, 'name': 'random_gentle'},
        {'degree': 0.25, 'name': 'random_normal'},
        {'degree': 0.35, 'name': 'random_aggressive'},
    ]
    
    # Worst removal配置
    WORST_REMOVAL_CONFIGS = [
        {'max_remove_ratio': 0.3, 'name': 'worst_conservative'},
        {'max_remove_ratio': 0.5, 'name': 'worst_moderate'},
        {'max_remove_ratio': 0.7, 'name': 'worst_aggressive'},
    ]
    
    # Shaw removal配置
    SHAW_REMOVAL_CONFIGS = [
        {'relatedness_threshold': 0.9, 'name': 'shaw_strict'},
        {'relatedness_threshold': 0.7, 'name': 'shaw_normal'},
        {'relatedness_threshold': 0.5, 'name': 'shaw_loose'},
    ]

def register_operators_from_config(alns: SimpleALNS):
    """从配置注册算子"""
    
    # 注册random removal算子
    for config in OperatorConfig.RANDOM_REMOVAL_CONFIGS:
        name = config.pop('name')
        operator = partial(random_removal, **config)
        alns.add_destroy_operator(operator, name)
    
    # 注册worst removal算子
    for config in OperatorConfig.WORST_REMOVAL_CONFIGS:
        name = config.pop('name')
        operator = partial(worst_removal, **config)
        alns.add_destroy_operator(operator, name)
    
    # 注册shaw removal算子
    for config in OperatorConfig.SHAW_REMOVAL_CONFIGS:
        name = config.pop('name')
        operator = partial(shaw_removal, **config)
        alns.add_destroy_operator(operator, name)

# ========================================
# 5. 运行示例
# ========================================

def main():
    """主函数演示"""
    print("=== functools.partial 在 ALNS 中的应用 ===\n")
    
    # 创建ALNS实例
    alns = SimpleALNS()
    
    # 方法1: 手动创建算子
    print("方法1: 手动使用partial创建算子")
    alns.add_destroy_operator(partial(random_removal, degree=0.2), "manual_random")
    alns.add_destroy_operator(partial(worst_removal, max_remove_ratio=0.4), "manual_worst")
    
    print("\n" + "="*50)
    
    # 方法2: 使用工厂函数
    print("\n方法2: 使用工厂函数创建算子")
    operators = create_alns_operators()
    for name, operator in list(operators.items())[:3]:  # 只取前3个作演示
        alns.add_destroy_operator(operator, name)
    
    print("\n" + "="*50)
    
    # 方法3: 从配置注册
    print("\n方法3: 从配置文件注册算子")
    config_alns = SimpleALNS()
    register_operators_from_config(config_alns)
    
    print("\n" + "="*50)
    
    # 运行几次迭代
    print("\n运行示例:")
    state = MockState(vehicles_count=20)
    rng = MockRNG()
    
    for i in range(3):
        state = config_alns.run_iteration(state, rng)
    
    print(f"\n最终状态: {state.vehicles_count}辆车")

# ========================================
# 6. 实际项目中的最佳实践
# ========================================

def best_practices_demo():
    """最佳实践演示"""
    print("\n\n=== 最佳实践 ===")
    
    # 1. 使用描述性的partial名称
    random_light = partial(random_removal, degree=0.1)
    random_medium = partial(random_removal, degree=0.25)
    random_heavy = partial(random_removal, degree=0.4)
    
    # 2. 创建partial工厂
    def create_random_removal(degree: float, name: str = None):
        """创建random removal算子的工厂函数"""
        operator = partial(random_removal, degree=degree)
        operator.name = name or f"random_{degree:.0%}"
        operator.degree = degree  # 保存参数以便调试
        return operator
    
    # 3. 参数验证
    def validated_random_removal(degree: float):
        """带参数验证的partial创建"""
        if not 0 < degree < 1:
            raise ValueError(f"degree必须在(0,1)范围内，得到: {degree}")
        return partial(random_removal, degree=degree)
    
    # 演示
    light_removal = create_random_removal(0.1, "轻度移除")
    medium_removal = create_random_removal(0.25, "中度移除")
    
    print(f"创建的算子: {light_removal.name}, 参数: {light_removal.degree}")
    print(f"创建的算子: {medium_removal.name}, 参数: {medium_removal.degree}")
    
    # 4. partial对象的自省
    print(f"\npartial对象信息:")
    print(f"函数: {light_removal.func.__name__}")
    print(f"固定参数: {light_removal.keywords}")

if __name__ == "__main__":
    main()
    best_practices_demo()
    
    print("\n=== 总结 ===")
    print("在ALNS中使用partial的优势:")
    print("✓ 参数化算子配置")
    print("✓ 代码复用和模块化")
    print("✓ 配置驱动的算子管理")
    print("✓ 清晰的算子命名和组织")
    print("✓ 便于参数调优和实验")