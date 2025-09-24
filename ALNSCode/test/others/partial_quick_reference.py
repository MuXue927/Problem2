"""
functools.partial 快速参考指南
"""

# ========================================
# 核心概念
# ========================================

"""
partial(func, *args, **keywords) 
创建一个新的可调用对象，其行为类似于使用预设参数调用func

主要用途：
1. 固定函数的某些参数
2. 创建专门化的函数版本
3. 回调函数和事件处理
4. 函数式编程
"""

from functools import partial

# ========================================
# 基础语法示例
# ========================================

def multiply(x, y, z=1):
    return x * y * z

# 固定位置参数
double = partial(multiply, 2)           # 固定x=2
print(double(3))        # 输出: 6 (相当于multiply(2, 3))

# 固定关键字参数  
multiply_z10 = partial(multiply, z=10)  # 固定z=10
print(multiply_z10(2, 3))  # 输出: 60 (相当于multiply(2, 3, z=10))

# 固定多个参数
times_six = partial(multiply, 2, 3)     # 固定x=2, y=3
print(times_six())      # 输出: 6 (相当于multiply(2, 3))

# ========================================
# 在您的ALNS项目中的应用
# ========================================

# 原始函数
def random_removal(state, rng, degree=0.25):
    # 移除逻辑
    pass

# 使用partial创建不同配置的算子
gentle_removal = partial(random_removal, degree=0.15)  # 移除15%
normal_removal = partial(random_removal, degree=0.25)  # 移除25%
aggressive_removal = partial(random_removal, degree=0.35)  # 移除35%

# 在ALNS中注册
# alns.add_destroy_operator(gentle_removal)
# alns.add_destroy_operator(normal_removal)  
# alns.add_destroy_operator(aggressive_removal)

# ========================================
# 常用模式
# ========================================

# 1. 配置驱动
REMOVAL_CONFIGS = [
    {'degree': 0.1, 'name': 'light'},
    {'degree': 0.25, 'name': 'medium'},
    {'degree': 0.4, 'name': 'heavy'}
]

operators = {}
for config in REMOVAL_CONFIGS:
    name = config.pop('name')
    operators[name] = partial(random_removal, **config)

# 2. 工厂函数
def create_removal_operator(degree, name=None):
    op = partial(random_removal, degree=degree)
    op.name = name or f"removal_{degree:.0%}"
    return op

# 3. 链式partial
base_func = partial(complex_function, param1=value1)
specialized_func = partial(base_func, param2=value2)

# ========================================
# partial对象的属性
# ========================================

p = partial(multiply, 2, z=10)
print(p.func)       # 原始函数
print(p.args)       # 固定的位置参数: (2,)
print(p.keywords)   # 固定的关键字参数: {'z': 10}

# ========================================
# 注意事项
# ========================================

"""
1. 位置参数按顺序固定，不能跳过
2. 关键字参数可以被后续调用覆盖
3. partial对象是可调用的
4. 比lambda更适合简单参数固定场景
5. 对于复杂逻辑，考虑使用类包装器

正确: partial(func, a, b, c=1)
错误: partial(func, a, c=1, b)  # 不能在关键字参数后使用位置参数
"""

# ========================================
# partial vs lambda
# ========================================

# partial: 简洁，专门用于参数固定
add_10 = partial(lambda x, y: x + y, 10)

# lambda: 更灵活，可以有复杂逻辑
add_10_lambda = lambda y: 10 + y

# 选择原则：
# - 简单参数固定 → 使用partial
# - 需要复杂逻辑 → 使用lambda或函数