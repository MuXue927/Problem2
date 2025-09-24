import os
import csv
import time
import copy
from .optutility import LogPrinter
import numpy.random as rnd
from typing import Optional, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .alnsopt import SolutionState

log_printer = LogPrinter(time.time())

# 定义一个类来跟踪ALNS迭代过程中的信息
class ALNSTracker:
    def __init__(self, output_file=None):
        self.iteration = 0
        self.current_obj = float('inf')  # 当前解的目标函数值
        self.best_obj = float('inf')     # 最优解的目标函数值
        self.best_solution = None
        self.start_time = time.time()
        self.objectives = []  # 存储每次迭代的目标函数值
        self.gaps = []
        self.output_file = output_file
        
        # 新增: 用于 learning_based_repair 的数据存储
        self.features = []  # list of list, [[demand, size, day, inv, util], ...]
        self.labels = []    # list of float, [improvement1, improvement2, ...]
        
        # 新增: ML模型缓存机制
        self._cached_model: Optional[Any] = None   # 缓存的机器学习模型 (Ridge或RandomForest)
        self._cached_scaler: Optional[Any] = None  # 缓存的特征标准化器 (StandardScaler)
        self._last_train_iteration: Optional[int] = None  # 上次训练模型的迭代次数
        
        # 如果指定了输出文件，则创建文件并写入表头
        if self.output_file:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Iteration', 'Current_Obj', 'Best_Obj', 'Gap'])
        
    def on_iteration(self, state: 'SolutionState', rng: rnd.Generator, **kwargs):
        """
        每次迭代后的回调函数
        简化逻辑，直接追踪迭代进度
        """
        # 增加迭代计数器
        self.iteration += 1
        
        # 获取当前解的目标函数值
        current_obj = state.objective()
        self.current_obj = current_obj
        self.objectives.append(current_obj)
        
        # 检查是否找到更好的可行解
        is_feasible, violations = state.validate()
        if current_obj < self.best_obj and is_feasible:
            self.best_obj = current_obj
            self.best_solution = copy.deepcopy(state)
            
            # 计算并打印gap（当前解等于最优解时gap为0）
            gap = calculate_gap(current_obj, self.best_obj)
            elapsed_time = time.time() - self.start_time
            log_printer.print(f"Iteration {self.iteration}: New best feasible solution found!", color='bold green')
            log_printer.print(f"Objective: {self.best_obj:.2f}, Gap: {gap:.2%}, Time: {elapsed_time:.2f}s", color='bold green')
        
        # 计算当前Gap
        gap = calculate_gap(self.current_obj, self.best_obj)
        self.gaps.append(gap)
        
        # 每隔100次迭代输出一次信息
        if self.iteration % 100 == 0:
            log_printer.print(f"Iteration: {self.iteration}\t Current Obj: {self.current_obj:.4f}\t Best Obj: {self.best_obj:.4f}\t Gap: {gap:.2f}%")
        
        # 如果指定了输出文件，则将结果写入文件
        if self.output_file:
            with open(self.output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.iteration, self.current_obj, self.best_obj, gap])
    
    # 新增: 用于 learning_based_repair 的数据更新方法
    def update_ml_data(self, feature: list, label: float) -> Tuple[int, int]:
        """
        添加一个新的feature和label到追踪器中,
        feature: list, 特征向量 [demand, size, day, inv, util],
        label: float, 该特征对应的目标函数改进值
        """
        self.features.append(feature)
        self.labels.append(label)
        # 可选优化：限存储大小，避免内存溢出
        if len(self.features) > 1000:  # 例如，保留最近1000个样本
            self.features = self.features[-1000:]
            self.labels = self.labels[-1000:]
        return len(self.features), len(self.labels)
    
    # 新增: ML模型缓存管理方法
    def cache_ml_model(self, model: Any, scaler: Any, iteration: int):
        """
        缓存训练好的ML模型和相关信息
        
        Parameters:
        -----------
        model : Any
            训练好的ML模型对象
        scaler : Any
            特征标准化器对象
        iteration : int
            当前训练的迭代数
        """
        self._cached_model = model
        self._cached_scaler = scaler
        self._last_train_iteration = iteration
        print(f"[Tracker] Cached ML model at iteration {iteration}")
        
    def get_cached_model(self) -> Tuple[Optional[Any], Optional[Any], Optional[int]]:
        """
        获取缓存的ML模型和相关信息
        
        Returns:
        --------
        Tuple[Optional[Any], Optional[Any], Optional[int]]
            (model, scaler, last_train_iteration)
        """
        return self._cached_model, self._cached_scaler, self._last_train_iteration
    
    def has_cached_model(self) -> bool:
        """
        检查是否有缓存的ML模型
        
        Returns:
        --------
        bool
            如果有缓存的模型则返回True, 否则返回False
        """
        return (
            self._cached_model is not None and 
            self._cached_scaler is not None and 
            self._last_train_iteration is not None
        )
        
    def clear_ml_cache(self):
        """
        清除缓存的ML模型和相关信息, 用于重置或者内存管理
        """
        self._cached_model = None
        self._cached_scaler = None
        self._last_train_iteration = None
        print("[Tracker] Cleared ML model cache")
        
    def get_ml_cache_info(self) -> dict:
        """
        获取当前ML模型缓存的信息
        
        Returns:
        --------
        dict
            包含缓存状态信息的字典
        """
        return {
            'has_model': self.has_cached_model(),
            'last_train_iteration': self._last_train_iteration,
            'model_type': type(self._cached_model).__name__ if self._cached_model else None,
            'scaler_type': type(self._cached_scaler).__name__ if self._cached_scaler else None,
            'data_samples': len(self.features)  # 当前存储的训练样本数量
        }
    
    
    def get_statistics(self):
        """
        获取追踪器的统计信息
        
        Returns:
        --------
        dict
            包含统计信息的字典
        """
        return {
            'total_iterations': self.iteration,
            'best_objective': self.best_obj,
            'current_objective': self.current_obj,
            'final_gap': calculate_gap(self.current_obj, self.best_obj) if self.objectives else 0.0,
            'objectives_history': self.objectives.copy(),
            'gaps_history': self.gaps.copy(),
            'elapsed_time': time.time() - self.start_time,
            'best_solution': self.best_solution,
            'features': self.features.copy(),  # ML特征数据
            'labels': self.labels.copy(),       # ML标签数据
            'ml_cache_info': self.get_ml_cache_info()  # ML缓存信息 
        }


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
    if current_obj == best_obj == 0:
        return 0.0
    if current_obj == 0 and best_obj > 0:
        return float('inf')
    return abs(current_obj - best_obj) / abs(current_obj) * 100