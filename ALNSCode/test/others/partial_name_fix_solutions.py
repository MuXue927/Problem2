"""
解决 functools.partial 在 ALNS 中的 __name__ 属性错误

问题描述:
ALNS库在添加算子时会尝试访问算子的__name__属性来记录日志，
但functools.partial对象没有__name__属性，导致AttributeError。

错误信息:
AttributeError: 'functools.partial' object has no attribute '__name__'
"""

from functools import partial

# ========================================
# 解决方案1: 手动添加__name__属性 (简单直接)
# ========================================

def solution_1_manual():
    """手动为partial对象添加__name__属性"""
    
    def sample_function(x, y, param=1):
        return x + y + param
    
    # 创建partial对象
    partial_func = partial(sample_function, param=10)
    
    # 手动添加__name__属性
    partial_func.__name__ = "sample_function_param_10"
    
    print(f"解决方案1 - 手动设置名称: {partial_func.__name__}")
    return partial_func

# ========================================
# 解决方案2: 创建helper函数 (推荐)
# ========================================

def create_named_partial(func, name=None, **kwargs):
    """
    创建带有__name__属性的partial对象
    
    Parameters:
    -----------
    func : callable
        要创建partial的函数
    name : str, optional
        partial对象的名称，如果不提供则自动生成
    **kwargs : dict
        要固定的关键字参数
    
    Returns:
    --------
    partial object with __name__ attribute
    """
    partial_func = partial(func, **kwargs)
    
    if name is None:
        # 自动生成描述性名称
        param_str = "_".join([f"{k}_{v}" for k, v in kwargs.items()])
        name = f"{func.__name__}_{param_str}"
    
    partial_func.__name__ = name
    return partial_func

def solution_2_helper():
    """使用helper函数创建named partial"""
    
    def sample_function(x, y, param=1):
        return x + y + param
    
    # 使用helper函数
    partial_func = create_named_partial(
        sample_function, 
        name="custom_sample_func",
        param=20
    )
    
    print(f"解决方案2 - Helper函数: {partial_func.__name__}")
    return partial_func

# ========================================
# 解决方案3: 包装类方法
# ========================================

class NamedPartialWrapper:
    """
    partial对象的包装类，提供__name__属性
    """
    def __init__(self, func, name=None, **kwargs):
        self.partial_func = partial(func, **kwargs)
        self.__name__ = name or f"{func.__name__}_wrapper"
        
        # 保留原函数的一些重要属性
        self.func = func
        self.keywords = kwargs
    
    def __call__(self, *args, **kwargs):
        """使对象可调用"""
        return self.partial_func(*args, **kwargs)

def solution_3_wrapper():
    """使用包装类创建named partial"""
    
    def sample_function(x, y, param=1):
        return x + y + param
    
    # 使用包装类
    wrapped_func = NamedPartialWrapper(
        sample_function,
        name="wrapped_sample_func",
        param=30
    )
    
    print(f"解决方案3 - 包装类: {wrapped_func.__name__}")
    return wrapped_func

# ========================================
# 针对ALNS的实际应用
# ========================================

def mock_random_removal(state, rng, degree=0.25):
    """模拟ALNS中的random_removal函数"""
    return f"removed {int(degree*100)}% vehicles"

class MockALNS:
    """模拟ALNS类"""
    def __init__(self):
        self.operators = []
    
    def add_destroy_operator(self, operator):
        """模拟添加destroy算子"""
        # 这里会尝试访问operator.__name__
        print(f"Adding operator: {operator.__name__}")
        self.operators.append(operator)

def alns_application_demo():
    """演示在ALNS中的应用"""
    print("\n=== ALNS实际应用演示 ===")
    
    alns = MockALNS()
    
    # 方法1: 手动设置
    print("\n方法1: 手动设置__name__")
    op1 = partial(mock_random_removal, degree=0.2)
    op1.__name__ = "random_removal_light"
    alns.add_destroy_operator(op1)
    
    # 方法2: 使用helper函数
    print("\n方法2: 使用helper函数")
    op2 = create_named_partial(
        mock_random_removal,
        name="random_removal_moderate",
        degree=0.3
    )
    alns.add_destroy_operator(op2)
    
    # 方法3: 使用包装类
    print("\n方法3: 使用包装类")
    op3 = NamedPartialWrapper(
        mock_random_removal,
        name="random_removal_aggressive",
        degree=0.4
    )
    alns.add_destroy_operator(op3)
    
    # 测试调用
    print(f"\n测试调用:")
    print(f"op1 结果: {op1('state', 'rng')}")
    print(f"op2 结果: {op2('state', 'rng')}")
    print(f"op3 结果: {op3('state', 'rng')}")

# ========================================
# 最佳实践建议
# ========================================

def best_practices():
    """最佳实践建议"""
    print("\n=== 最佳实践建议 ===")
    print("1. 推荐使用解决方案2 (helper函数)")
    print("   - 代码清晰，易于维护")
    print("   - 自动生成描述性名称")
    print("   - 支持自定义名称")
    
    print("\n2. 命名建议:")
    print("   - 包含函数名和关键参数")
    print("   - 使用描述性词汇 (gentle, moderate, aggressive)")
    print("   - 保持一致的命名风格")
    
    print("\n3. 组织建议:")
    print("   - 将算子创建逻辑封装到单独函数中")
    print("   - 使用配置文件管理参数")
    print("   - 为不同强度的算子提供预设配置")

if __name__ == "__main__":
    print("=== functools.partial __name__ 属性问题解决方案 ===")
    
    # 运行所有解决方案
    solution_1_manual()
    solution_2_helper()
    solution_3_wrapper()
    
    # ALNS应用演示
    alns_application_demo()
    
    # 最佳实践
    best_practices()
    
    print("\n=== 总结 ===")
    print("✓ 问题已解决：为partial对象添加__name__属性")
    print("✓ 推荐方案：使用create_named_partial helper函数")
    print("✓ 适用场景：所有需要__name__属性的函数对象")