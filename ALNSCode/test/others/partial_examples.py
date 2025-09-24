"""
functools.partial 详细用法示例
"""
from functools import partial

# ========================================
# 1. 基本用法示例
# ========================================

def multiply(x, y, z=1):
    """简单的乘法函数"""
    return x * y * z

# 原始函数调用
print("=== 基本用法 ===")
print(f"multiply(2, 3, 4) = {multiply(2, 3, 4)}")

# 使用partial固定第一个参数
double = partial(multiply, 2)  # 固定 x=2
print(f"double(3, 4) = {double(3, 4)}")  # 相当于 multiply(2, 3, 4)
print(f"double(5) = {double(5)}")        # 相当于 multiply(2, 5, 1)

# 使用partial固定多个参数
multiply_by_6 = partial(multiply, 2, 3)  # 固定 x=2, y=3
print(f"multiply_by_6(4) = {multiply_by_6(4)}")  # 相当于 multiply(2, 3, 4)

# 使用partial固定关键字参数
multiply_with_z = partial(multiply, z=10)  # 固定 z=10
print(f"multiply_with_z(2, 3) = {multiply_with_z(2, 3)}")  # 相当于 multiply(2, 3, 10)

# ========================================
# 2. 在ALNS中的实际应用
# ========================================

print("\n=== ALNS中的应用 ===")

def remove_vehicles(state, rng, degree=0.25, method="random"):
    """模拟车辆移除函数"""
    print(f"移除 {degree*100:.1f}% 的车辆，使用 {method} 方法")
    return f"移除了 {int(10 * degree)} 辆车"

# 创建不同配置的移除算子
random_25 = partial(remove_vehicles, degree=0.25, method="random")
random_30 = partial(remove_vehicles, degree=0.30, method="random")
worst_20 = partial(remove_vehicles, degree=0.20, method="worst")

# 模拟ALNS调用
class MockState:
    pass

class MockRNG:
    pass

state = MockState()
rng = MockRNG()

print("调用不同的移除算子:")
print(f"random_25: {random_25(state, rng)}")
print(f"random_30: {random_30(state, rng)}")
print(f"worst_20: {worst_20(state, rng)}")

# ========================================
# 3. 高级用法
# ========================================

print("\n=== 高级用法 ===")

# 3.1 链式使用partial
def power_function(base, exponent, multiplier=1):
    return (base ** exponent) * multiplier

# 先固定指数为2（平方函数）
square_func = partial(power_function, exponent=2)
# 再固定乘数为3
triple_square = partial(square_func, multiplier=3)

print(f"triple_square(4) = {triple_square(4)}")  # (4^2) * 3 = 48

# 3.2 partial对象的属性
print(f"\npartial对象的属性:")
print(f"triple_square.func = {triple_square.func}")
print(f"triple_square.args = {triple_square.args}")
print(f"triple_square.keywords = {triple_square.keywords}")

# ========================================
# 4. 实际场景：事件处理
# ========================================

print("\n=== 实际场景：事件处理 ===")

def log_event(message, level="INFO", timestamp=True):
    """日志记录函数"""
    prefix = "[TIMESTAMP] " if timestamp else ""
    return f"{prefix}[{level}] {message}"

# 创建不同级别的日志函数
info_log = partial(log_event, level="INFO")
error_log = partial(log_event, level="ERROR", timestamp=True)
debug_log = partial(log_event, level="DEBUG", timestamp=False)

print(info_log("程序启动"))
print(error_log("发生错误"))
print(debug_log("调试信息"))

# ========================================
# 5. 与lambda的比较
# ========================================

print("\n=== partial vs lambda ===")

def calculate(a, b, operation="add"):
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b

# 使用partial
add_func = partial(calculate, operation="add")
multiply_func = partial(calculate, operation="multiply")

# 使用lambda（等价但更啰嗦）
add_lambda = lambda a, b: calculate(a, b, operation="add")
multiply_lambda = lambda a, b: calculate(a, b, operation="multiply")

print(f"partial版本: add_func(3, 4) = {add_func(3, 4)}")
print(f"lambda版本: add_lambda(3, 4) = {add_lambda(3, 4)}")

# ========================================
# 6. 在类方法中使用partial
# ========================================

print("\n=== 在类中使用partial ===")

class DataProcessor:
    def __init__(self, default_format="json"):
        self.default_format = default_format
    
    def process_data(self, data, format_type=None, validate=True):
        format_type = format_type or self.default_format
        validation = "已验证" if validate else "未验证"
        return f"处理数据: {data}, 格式: {format_type}, {validation}"

processor = DataProcessor()

# 创建预配置的处理函数
json_processor = partial(processor.process_data, format_type="json", validate=True)
xml_processor = partial(processor.process_data, format_type="xml", validate=False)

print(json_processor("用户数据"))
print(xml_processor("配置数据"))

# ========================================
# 7. 在ALNS算子注册中的应用模式
# ========================================

print("\n=== ALNS算子注册模式 ===")

class ALNSSimulator:
    def __init__(self):
        self.operators = []
    
    def add_destroy_operator(self, operator_func):
        """添加破坏算子"""
        self.operators.append(operator_func)
        print(f"已注册算子: {operator_func}")
    
    def run_operators(self, state, rng):
        """运行所有算子"""
        for op in self.operators:
            result = op(state, rng)
            print(f"算子执行结果: {result}")

def destroy_operator(state, rng, removal_rate=0.25, strategy="random"):
    """通用破坏算子"""
    return f"使用{strategy}策略移除{removal_rate*100:.0f}%的元素"

# 模拟ALNS使用方式
alns = ALNSSimulator()

# 注册不同配置的算子
alns.add_destroy_operator(partial(destroy_operator, removal_rate=0.2, strategy="random"))
alns.add_destroy_operator(partial(destroy_operator, removal_rate=0.3, strategy="worst"))
alns.add_destroy_operator(partial(destroy_operator, removal_rate=0.25, strategy="shaw"))

# 运行算子
print("\n运行算子:")
alns.run_operators("mock_state", "mock_rng")

# ========================================
# 8. partial的注意事项
# ========================================

print("\n=== 注意事项 ===")

def demo_function(a, b, c=3, d=4):
    return f"a={a}, b={b}, c={c}, d={d}"

# 注意1: 位置参数的顺序很重要
partial_func1 = partial(demo_function, 1)  # 固定a=1
print(f"partial_func1(2, c=30) = {partial_func1(2, c=30)}")

# 注意2: 关键字参数可以被覆盖
partial_func2 = partial(demo_function, c=100)  # 预设c=100
print(f"partial_func2(1, 2) = {partial_func2(1, 2)}")
print(f"partial_func2(1, 2, c=200) = {partial_func2(1, 2, c=200)}")  # c被覆盖

# 注意3: partial对象是可调用的
print(f"partial对象是否可调用: {callable(partial_func1)}")

print("\n=== 总结 ===")
print("partial的主要优势:")
print("1. 代码复用性强")
print("2. 参数预设灵活")
print("3. 比lambda表达式更清晰")
print("4. 适合函数式编程风格")
print("5. 在回调函数和事件处理中很有用")