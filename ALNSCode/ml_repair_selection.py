"""
利用基于机器学习的算子选择机制, 自动选择最佳修复算子
"""
import time
import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Callable, Optional
import numpy.random as rnd

from .alnsopt import SolutionState
from .ml_operator_selector import MLOperatorSelector
from .alns_config import default_config as ALNSConfig

# 导入所需的修复算子
from .repair_operators import (
    greedy_repair,
    local_search_repair,
    inventory_balance_repair,
    infeasible_repair,
    smart_batch_repair,
    regret_based_repair
)

# Helper: sanitize and align ParamTuner kwargs to operator signatures
def _sanitize_repair_params(op_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(params, dict) or not params:
        return {}
    allowed = {
        "greedy_repair": {"demand_weight", "stock_weight"},
        "local_search_repair": {"max_iter"},
        "inventory_balance_repair": set(),
        "infeasible_repair": set(),
        "smart_batch_repair": {"max_iter", "batch_size", "timeout"},
        "regret_based_repair": {"k", "topN", "time_limit"},
    }
    allow = allowed.get(op_name, set())
    return {k: v for k, v in params.items() if k in allow}


def ml_based_repair(current: SolutionState, rng: rnd.Generator, 
                   min_sample_size: int = None) -> SolutionState:
    """
    基于机器学习的修复算子
    
    使用ML选择器自动选择最适合当前解状态的修复算子和参数。
    在数据不足时，随机选择一个修复算子。
    
    Args:
        current: 当前解状态
        rng: 随机数生成器
        min_sample_size: 最小样本数量，低于此数量时随机选择算子
    
    Returns:
        修改后的解状态
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行 ml_based_repair 算子")
    state = current.copy()

    # 若未显式传入 min_sample_size，则从集中配置中读取默认值（向后兼容）
    if min_sample_size is None:
        min_sample_size = getattr(ALNSConfig, "ML_INITIAL_SAMPLE_SIZE", 20)
    
    # 检查ML选择器是否可用
    if not state.ml_selector:
        print(f"[OPLOG] ml_based_repair: ML选择器不可用, 回退到贪心修复")
        from .repair_operators import greedy_repair
        greedy_params = ALNSConfig.get_repair_params().get("greedy_repair", {})
        return greedy_repair(state, rng, **(greedy_params or {}))
    
    # 获取可用的修复算子列表
    repair_operators = {
        "greedy_repair": lambda s, r, **kwargs: greedy_repair(s, r, **kwargs),
        "local_search_repair": lambda s, r, **kwargs: local_search_repair(s, r, **kwargs),
        "inventory_balance_repair": lambda s, r, **kwargs: inventory_balance_repair(s, r, **kwargs),
        "infeasible_repair": lambda s, r, **kwargs: infeasible_repair(s, r, **kwargs),
        "smart_batch_repair": lambda s, r, **kwargs: smart_batch_repair(s, r, **kwargs),
        "regret_based_repair": lambda s, r, **kwargs: regret_based_repair(s, r, **kwargs)
    }
    
    # 检查训练数据是否足够
    if len(state.ml_selector.repair_features) < min_sample_size:
        # 数据不足，随机选择一个修复算子
        print(f"[OPLOG] ml_based_repair: 训练数据不足 ({len(state.ml_selector.repair_features)} < {min_sample_size})，随机选择")
        op_name = rng.choice(list(repair_operators.keys()))
        operator = repair_operators[op_name]
        
        # 获取参数
        params = {}
        if state.param_tuner:
            params = state.param_tuner.get_operator_params(op_name)
        params = _sanitize_repair_params(op_name, params)
        
        # 执行算子
        prev_obj = state.objective()
        new_state = operator(state, rng, **params)
        new_obj = new_state.objective()
        
        # 计算改进值
        improvement = prev_obj - new_obj
        if math.isnan(improvement):
            improvement = 0.0
        elif math.isinf(improvement):
            improvement = 1e6 if improvement > 0 else -1e6
        
        # 记录性能（使用操作前状态特征，避免泄漏后验信息）
        state.ml_selector.record_operator_performance(op_name, 'repair', state, improvement)
        
        print(f"[OPLOG] ml_based_repair: 随机选择 {op_name}，改进值 {improvement:.4f}")
        return new_state
    
    # 训练数据足够，使用ML选择器选择最佳算子
    state.ml_selector.train_models()
    
    # 选择最佳算子
    op_name, params = state.ml_selector.select_best_operator(
        state, 'repair', list(repair_operators.keys())
    )
    params = _sanitize_repair_params(op_name, params)
    
    operator = repair_operators[op_name]
    
    # 执行算子
    prev_obj = state.objective()
    new_state = operator(state, rng, **params)
    new_obj = new_state.objective()
    
    # 计算改进值
    improvement = prev_obj - new_obj
    if math.isnan(improvement):
        improvement = 0.0
    elif math.isinf(improvement):
        improvement = 1e6 if improvement > 0 else -1e6
    
    # 记录性能（使用操作前状态特征，避免泄漏后验信息）
    state.ml_selector.record_operator_performance(op_name, 'repair', state, improvement)
    
    print(f"[OPLOG] ml_based_repair: 选择 {op_name}，改进值 {improvement:.4f} ({time.time() - t0:.4f}s)")
    return new_state
