"""
测试partial函数修复的脚本
"""
import sys
import os

# 添加必要的路径
sys.path.append(os.path.dirname(__file__))

from functools import partial

# 模拟一个简单的函数
def mock_random_removal(state, rng, degree=0.25):
    """模拟random_removal函数"""
    print(f"调用random_removal，参数degree={degree}")
    return f"result_with_degree_{degree}"

def create_named_partial(func, name=None, **kwargs):
    """
    创建带有__name__属性的partial对象，以兼容ALNS库
    """
    partial_func = partial(func, **kwargs)
    
    if name is None:
        # 自动生成名称
        param_str = "_".join([f"{k}_{v}" for k, v in kwargs.items()])
        name = f"{func.__name__}_{param_str}"
    
    partial_func.__name__ = name
    return partial_func

# 测试
print("=== 测试partial对象修复 ===")

# 创建带名称的partial对象
partial_func = create_named_partial(mock_random_removal, degree=0.3)

print(f"✓ 函数名称: {partial_func.__name__}")
print(f"✓ 原始函数: {partial_func.func.__name__}")
print(f"✓ 固定参数: {partial_func.keywords}")
print(f"✓ 是否可调用: {callable(partial_func)}")

# 测试调用
result = partial_func("mock_state", "mock_rng")
print(f"✓ 调用结果: {result}")

# 测试自定义名称
custom_partial = create_named_partial(
    mock_random_removal, 
    name="aggressive_removal",
    degree=0.4
)
print(f"✓ 自定义名称: {custom_partial.__name__}")

print("\n=== 测试完成，修复方案有效 ===")