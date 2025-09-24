# 读取gurobi的求解日志文件
# 使用标准库re从中提取Incumbent、BestBd、Time这三列的信息
# 将这些做可视化呈现，观察模型求解过程中Gap的变化
import os
import re
import matplotlib.pyplot as plt


def read_log_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def parse_gurobi_log(content):
    # * 匹配0次、1次或者多次
    # + 匹配1次或者多次
    # . 通配符，除与换行符之外的其他字符都匹配，只能匹配一个字符，不能匹配0个或者2个
    # 字符集 -- 将子串放在一对方括号内，字符集只能匹配一个字符
    # 管道字符 | 表示2选1
    # 子模式 -- 将子串放在一对圆括号内，可以在子模式后面加上问号，表示可选可不选
    # 脱字符^用来表示字符串的开头，美元符号$表示字符串的结尾
    # \s 匹配Unicode空白字符，包括[\t\n\r\f\v]等
    # \d 匹配任意十进制数码，比如[0-9]，还包括许多其他的数码类字符
    # \w 匹配Unicode单词类字符，包括所有Unicode字母数字类字符，以及下划线
    # \s*([H*]?) -- 匹配可能存在的前导空白字符，然后可选地匹配 'H' 或 '*'。
    # \s*(\d+)\s+(\d+) -- 匹配已经探索过的节点数量和未探索过的节点数量
    # \s+([-\d.]+|infeasible|cutoff|\s*) -- 匹配 Obj 列，可以是数字、'infeasible'、'cutoff' 或空白。
    # \s+\d*\s+\d* -- 匹配 Depth 和 IntInf 列（我们不捕获这些值）。
    # \s+([-\d.]+|\s*) -- 匹配 Incumbent 列，可以是数字或 '-'（表示无穷大）。
    # \s+([\d.]+) --  匹配 BestBd 列。
    # (?:\s+(?:[\d.]+%|-))?\s+ -- 匹配可选的 Gap 列，可以是百分比或 '-'
    # .*?(\d+)s -- 匹配Time列
    pattern = r'^\s*([H*]?)\s*(\d+)\s+(\d+)\s+([-\d.]+|infeasible|cutoff|\s*)\s+\d*\s+\d*\s+([-\d.]+|\s*)\s+([\d.]+)(?:\s+(?:[\d.]+%|-))?\s+.*?(\d+)s$'
    # re.finditer(pattern, string, flags=0)
    # 针对正则表达式 pattern 在 string 里的所有非重叠匹配返回一个产生 Match 对象的 iterator。
    # string 将被从左至右地扫描，并且匹配也将按被找到的顺序返回。 空匹配也会被包括在结果中。
    # 表达式的行为可通过指定 flags 值来修改。 值可以是任意 flags 变量，可使用按位 OR (| 运算符) 进行组合。
    # re.MULTILINE是re标准库中的一个flag，使用模式字符'^'匹配每一行的开头，使用'$'匹配每一行的末尾。
    matches = re.finditer(pattern, content, re.MULTILINE)

    data = []
    for match in matches:
        # Match.groups(default=None)  返回一个元组，包含所有匹配的子组
        h_start, expl, un_expl, obj, incumbent, best_bd, time = match.groups()
        data.append({
            'feasible_tag': h_start,
            'expl_node': int(expl),
            'un_expl_node': int(un_expl),
            'obj': obj if obj and obj != 'infeasible' and obj != 'cutoff' else None,
            'incumbent': float(incumbent) if incumbent and incumbent != '-' else float('inf'),
            'best_bound': float(best_bd),
            'time': int(time)
        })
    return data

def plot_gurobi_progress(data):
    # 创建矢量图文件的保存路径
    # fig_file_path1 = os.path.join('opt_progress', 'single-period', 'PDF')
    # fig_file_path2 = os.path.join('opt_progress', 'single-period', 'SVG')
    fig_file_path1 = os.path.join('opt_progress', 'multiple-periods', 'PDF')
    if not os.path.exists(fig_file_path1):
        os.makedirs(fig_file_path1)

    fig_file_path2 = os.path.join('opt_progress', 'multiple-periods', 'SVG')
    if not os.path.exists(fig_file_path2):
        os.makedirs(fig_file_path2)

    times = [info['time'] for info in data]
    incumbents = [info['incumbent'] for info in data]
    best_bds = [info['best_bound'] for info in data]

    plt.figure(figsize=(6, 4))
    plt.plot(times, incumbents, '-', label='Incumbent')
    plt.plot(times, best_bds, '--', label='Best Bound')
    plt.xlabel('Time (s)')
    plt.ylabel('Primal (Dual) Objective Value')
    plt.title('Gurobi Optimization Progress')
    plt.legend()
    plt.yscale('log')  # use log scale for y-axis
    plt.tight_layout()


    fig_file = os.path.join(fig_file_path1, 'gurobi_opt_progress_1-3.pdf')
    plt.savefig(fig_file, dpi=1000)
    fig_file1 = os.path.join(fig_file_path2, 'gurobi_opt_progress_1-3.svg')
    plt.savefig(fig_file1, dpi=1000)
    plt.show()

# # 单周期问题的文件读取路径
# input_file_loc = os.path.join('solve-logs', 'single-period', 'small', 'A01')
# log_file = os.path.join(input_file_loc, 'Windows PowerShell.txt')

# 多周期问题的文件读取路径
input_file_loc = os.path.join('logs-cg', 'multiple-periods', 'small', 'dataset_1')
log_file = os.path.join(input_file_loc, 'Windows PowerShell.txt')

log_content = read_log_file(log_file)
data = parse_gurobi_log(log_content)
plot_gurobi_progress(data)
