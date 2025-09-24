import os
import csv
import time
from optutility import LogPrinter

log_printer = LogPrinter(time.time())

# 定义一个类来跟踪ALNS迭代过程中的信息
class ALNSTracker:
    def __init__(self, output_file=None):
        self.iteration = 0
        self.current_obj = float('inf')
        self.best_obj = float('inf')
        self.objectives = []
        self.gaps = []
        self.output_file = output_file
        
        # 如果指定了输出文件，则创建文件并写入表头
        if self.output_file:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Iteration', 'Current_Obj', 'Best_Obj', 'Gap'])
        
    def update(self, state):
        """更新当前解的目标函数值"""
        self.iteration += 1
        self.current_obj = state.objective()
        self.objectives.append(self.current_obj)
        self.best_obj = min(self.best_obj, self.current_obj)
        
        # 计算Gap
        gap = calculate_gap(self.current_obj, self.best_obj)
        self.gaps.append(gap)
        
        # 输出迭代信息
        log_printer.print(f"Iteration: {self.iteration}\t Current Obj: {self.current_obj:.2f}\t Best Obj: {self.best_obj:.2f}\t Gap: {gap:.2f}%")
        
        # 如果指定了输出文件，则将结果写入文件
        if self.output_file:
            with open(self.output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.iteration, self.current_obj, self.best_obj, gap])


# 定义一个函数用于计算Gap
def calculate_gap(current_obj, best_obj):
    """
    计算Gap值, 根据公式: Gap = (z_c - z_b) / z_c * 100%
    其中z_c是当前解的目标函数值, z_b是最好解的目标函数值
    
    Parameters:
    -----------
    current_obj : float
        当前解的目标函数值
    best_obj : float
        最好解的目标函数值
        
    Returns:
    --------
    float
        Gap值, 以百分比表示
    """
    if current_obj == 0:  # 避免除以零
        return 0
    return (current_obj - best_obj) / current_obj * 100