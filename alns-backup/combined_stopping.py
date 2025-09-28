"""
组合停止准则：支持多个停止条件的逻辑组合
"""

import time
from typing import List, Optional, Union
from numpy.random import Generator
from alns.State import State
from alns.stop import MaxRuntime, MaxIterations, NoImprovement


class CombinedStoppingCriterion:
    """
    组合停止准则: 支持多个停止条件, 满足任一条件即停止 (OR逻辑)
    
    这个类允许组合多个停止准则, 如最大迭代次数、最大运行时间、无改进迭代等。
    当任意一个条件满足时, 算法就会停止。
    """
    
    def __init__(self, *criteria):
        """
        初始化组合停止准则
        
        Parameters
        ----------
        *criteria
            可变数量的停止准则对象, 支持：
            - MaxRuntime(max_runtime=900)
            - MaxIterations(max_iterations=300) 
            - NoImprovement(max_iterations=100)
            - 或任何实现了 StoppingCriterion 协议的对象
        """
        if not criteria:
            raise ValueError("至少需要提供一个停止准则")
        
        self.criteria = list(criteria)
        self._start_time = None
        
    def add_criterion(self, criterion):
        """添加额外的停止准则"""
        self.criteria.append(criterion)
        
    def __call__(self, rng: Generator, best: State, current: State) -> bool:
        """
        检查是否应该停止优化
        
        Returns
        -------
        bool
            如果任意一个停止准则满足, 返回 True; 否则返回 False
        """
        # 记录开始时间（用于运行时统计）
        if self._start_time is None:
            self._start_time = time.perf_counter()
        
        # 检查每个停止准则
        for i, criterion in enumerate(self.criteria):
            try:
                if criterion(rng, best, current):
                    # 记录是哪个准则触发了停止
                    criterion_name = type(criterion).__name__
                    elapsed_time = time.perf_counter() - self._start_time
                    print(f"[STOP] {criterion_name} 触发停止条件 (运行时间: {elapsed_time:.2f}s)")
                    return True
            except Exception as e:
                print(f"[WARNING] 停止准则 {type(criterion).__name__} 评估失败: {e}")
                continue
                
        return False
    
    def get_status(self) -> dict:
        """
        获取当前各停止准则的状态信息
        
        Returns
        -------
        dict
            包含各停止准则当前状态的字典
        """
        status = {
            'elapsed_time': time.perf_counter() - self._start_time if self._start_time else 0,
            'criteria_status': []
        }
        
        for criterion in self.criteria:
            criterion_info = {
                'type': type(criterion).__name__,
                'current_value': None,
                'max_value': None
            }
            
            # 获取具体准则的状态信息
            if hasattr(criterion, '_current_iteration'):
                criterion_info['current_value'] = criterion._current_iteration
                criterion_info['max_value'] = criterion.max_iterations
            elif hasattr(criterion, '_start_runtime') and criterion._start_runtime is not None:
                current_runtime = time.perf_counter() - criterion._start_runtime
                criterion_info['current_value'] = current_runtime
                criterion_info['max_value'] = criterion.max_runtime
                
            status['criteria_status'].append(criterion_info)
            
        return status


def create_standard_combined_criterion(max_iterations: int = 300, 
                                     max_runtime: float = 900,
                                     max_no_improvement: Optional[int] = None) -> CombinedStoppingCriterion:
    """
    创建标准的组合停止准则
    
    Parameters
    ----------
    max_iterations : int
        最大迭代次数, 默认300
    max_runtime : float
        最大运行时间（秒）, 默认900
    max_no_improvement : int, optional
        最大无改进迭代次数, 如果提供则添加此准则
        
    Returns
    -------
    CombinedStoppingCriterion
        配置好的组合停止准则
    """
    criteria = [
        MaxIterations(max_iterations=max_iterations),
        MaxRuntime(max_runtime=max_runtime)
    ]
    
    if max_no_improvement is not None:
        criteria.append(NoImprovement(max_iterations=max_no_improvement))
    
    return CombinedStoppingCriterion(*criteria)


if __name__ == "__main__":
    # 测试示例
    print("=== 组合停止准则测试 ===")
    
    # 创建组合停止准则
    combined_stop = create_standard_combined_criterion(
        max_iterations=300,
        max_runtime=900,
        max_no_improvement=100
    )
    
    print(f"创建了包含 {len(combined_stop.criteria)} 个停止准则的组合条件:")
    for criterion in combined_stop.criteria:
        print(f"  - {type(criterion).__name__}")
    
    print("测试完成")