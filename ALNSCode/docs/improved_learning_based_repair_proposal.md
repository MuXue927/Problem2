# Improved Learning-Based Repair Operator with Diverse Training Strategy

## Overview

This document presents an improved version of the `learning_based_repair` operator that addresses the bootstrapping problem by using random selection from all available repair operators during the initial training phase, instead of relying solely on the greedy repair operator.

## Problem Analysis

The original implementation suffered from a chicken-and-egg problem:
- The operator required sufficient training data to function
- But it only collected data when successfully performing ML-based insertions
- During bootstrapping, it fell back to greedy repair but didn't save training data
- This created an unbreakable cycle where the operator never accumulated enough data

## Proposed Solution

### Key Improvements

1. **Diverse Operator Selection**: When the tracker is unavailable, contains no data, or lacks sufficient data, randomly select from all six repair operators instead of defaulting to greedy repair.

2. **Enhanced Training Data Collection**: During the training phase, replace all instances of greedy repair with random operator selection to generate diverse training examples.

3. **Objective Improvement Evaluation**: Evaluate the improvement in objective function after applying any repair operator and use it as the label for training data.

### Available Repair Operators

The system has 6 repair operators available:
- `smart_batch_repair`: Intelligent batch processing with resource optimization
- `greedy_repair`: Priority-based greedy assignment
- `inventory_balance_repair`: Focuses on balancing inventory levels
- `urgency_repair`: Prioritizes based on demand urgency
- `infeasible_repair`: Fixes infeasible solutions (negative inventory)
- `local_search_repair`: Local search optimization

## Pseudocode Implementation

```python
def improved_learning_based_repair(partial: SolutionState, rng: rnd.Generator,
                                  model_type: str='adaptive', min_score: float=0.4,
                                  initial_sample_size: int=20, adaptive_sample_size: int=100,
                                  retrain_interval: int=80) -> SolutionState:
    """
    Improved learning-based repair operator with diverse training strategy.
    
    When insufficient data is available, randomly selects from all repair operators
    to generate diverse training examples instead of defaulting to greedy repair.
    """
    t0 = time.time()
    print(f"[OPLOG] 开始执行改进版 learning_based_repair 算子")
    state = partial
    data = state.data
    
    # Define all available repair operators for random selection
    REPAIR_OPERATORS = [
        smart_batch_repair,
        greedy_repair,
        inventory_balance_repair,
        urgency_repair,
        infeasible_repair,
        local_search_repair
    ]
    
    # step 1: 检查tracker可用性
    if state.tracker is None:
        print(f"[OPLOG] 无tracker引用, 随机选择修复算子")
        selected_operator = rng.choice(REPAIR_OPERATORS)
        print(f"[OPLOG] 选择算子: {selected_operator.__name__}")
        return selected_operator(state, rng)
    
    tracker_stats = state.tracker.get_statistics()
    features_data = state.tracker.features
    labels_data = state.tracker.labels
    current_iteration = tracker_stats['total_iterations']
    
    # Modified: When no data exists, randomly select operator and collect training data
    if not features_data or not labels_data:
        print(f"[OPLOG] 正在构建多样化训练数据...")
        print(f"[OPLOG] tracker中无数据, 随机选择修复算子并记录数据")
        
        # Record objective before repair
        prev_obj = state.objective()
        
        # Randomly select a repair operator
        selected_operator = rng.choice(REPAIR_OPERATORS)
        print(f"[OPLOG] 选择算子: {selected_operator.__name__}")
        
        # Apply the selected operator
        new_state = selected_operator(state, rng)
        
        # Calculate improvement
        new_obj = new_state.objective()
        actual_improvement = prev_obj - new_obj
        
        # Construct representative features (same as original implementation)
        _construct_training_data(new_state, actual_improvement)
        return new_state
        
    
    # Modified: During initial training phase, use random operator selection
    if len(labels_data) < initial_sample_size:
        print(f"[OPLOG] 训练数据不足 {len(labels_data)} < {initial_sample_size}, 随机选择算子并记录数据")
        
        # Record objective before repair
        prev_obj = state.objective()
        
        # Randomly select a repair operator (could be any, including greedy)
        selected_operator = rng.choice(REPAIR_OPERATORS)
        print(f"[OPLOG] 选择算子: {selected_operator.__name__}")
        
        # Apply the selected operator
        new_state = selected_operator(state, rng)
        
        # Calculate improvement and save training data
        new_obj = new_state.objective()
        actual_improvement = prev_obj - new_obj
        _construct_training_data(new_state, actual_improvement)
        return new_state
    
    # step 2-7: ML-based repair logic (unchanged from original implementation)
    # ... existing ML training and prediction code ...
    
    # When ML fails and falls back to repair, also use random selection
    if failed_demands:
        print(f"[OPLOG] {len(failed_demands)} 个需求ML插入失败, 随机选择算子修复")
        
        # Record objective before repair
        prev_obj = state.objective()
        
        # Randomly select a repair operator
        selected_operator = rng.choice(REPAIR_OPERATORS)
        print(f"[OPLOG] 选择算子: {selected_operator.__name__}")
        
        # Apply the selected operator
        new_state = selected_operator(state, rng)
        
        # Calculate improvement and save training data
        new_obj = new_state.objective()
        actual_improvement = prev_obj - new_obj
        _construct_training_data(new_state, actual_improvement)
        state = new_state

    state.compute_inventory()
    
    print(f"[OPLOG] 改进版ML修复完成: {total_predictions}次预测, {successful_insertions}次成功插入")
    
    elapsed = time.time() - t0
    print(f"[OPLOG] improved_learning_based_repair: {elapsed:.4f}s")
    return state


def _construct_training_data(partial: SolutionState, improvement: float):
    """
    构造训练数据 (保持原有逻辑不变)
    """
    state = partial
    data = state.data
    
    avg_demand = np.mean(list(data.demands.values())) if data.demands else 1.0
    avg_sku_size = np.mean([data.sku_sizes[sku] for sku in data.all_skus]) if data.all_skus else 1.0
    
    periods = list(range(1, data.horizons + 1))
    avg_day = float(periods[len(periods) // 2]) if periods else 1.0
    
    avg_inventory = np.mean([inv for (plant, sku_id, day), inv in state.s_ikt.items() if day == avg_day-1]) if state.s_ikt else 1.0
    avg_capacity_util = float(rnd.uniform(0, 1))
    
    feature = [
        avg_demand,
        avg_sku_size,
        avg_day,
        avg_inventory,
        avg_capacity_util
    ]
    
    feat_length, label_length = state.tracker.update_ml_data(feature, improvement)
    print(f"构建训练数据: 特征长度 {feat_length}, 标签长度 {label_length}, 目标函数改进 {improvement:.4f}")


class ImprovedLearningBasedRepairOperator:
    """
    改进版带参数的学习-based修复算子包装类
    """
    def __init__(self, model_type: str='adaptive', 
                 min_score: float=0.4, 
                 initial_sample_size: int=20,
                 adaptive_sample_size: int=100,
                 retrain_interval: int=80):
        
        self.model_type = model_type
        self.min_score = min_score
        self.initial_sample_size = initial_sample_size
        self.adaptive_sample_size = adaptive_sample_size
        self.retrain_interval = retrain_interval
        self.__name__ = "improved_learning_based_repair"
        
        # 验证参数有效性
        valid_models = ['linear', 'random_forest', 'adaptive']
        if model_type not in valid_models:
            print(f"[WARNING] 无效的model_type '{model_type}', 使用默认值 'adaptive'")
            self.model_type = 'adaptive'
        
        if retrain_interval < 10:
            print(f"[WARNING] retrain_interval过小 ({retrain_interval}), 设置为最小值 10")
            self.retrain_interval = 10

    def __call__(self, current: SolutionState, rng: rnd.Generator):
        return improved_learning_based_repair(current, rng, self.model_type, self.min_score, 
                                            self.initial_sample_size, self.adaptive_sample_size, 
                                            self.retrain_interval)
```

## Key Changes from Original Implementation

### 1. Random Operator Selection During Bootstrapping
```python
# Original: Always use greedy
return greedy_repair(state, rng)

# Improved: Random selection from all operators
selected_operator = rng.choice(REPAIR_OPERATORS)
return selected_operator(state, rng)
```

### 2. Diverse Training Data Collection
```python
# Original: Only greedy during training phase
if len(labels_data) < initial_sample_size:
    new_state, improvement = _call_greedy_repair(state, rng)
    _construct_training_data(new_state, improvement)

# Improved: Random operator selection
if len(labels_data) < initial_sample_size:
    prev_obj = state.objective()
    selected_operator = rng.choice(REPAIR_OPERATORS)
    new_state = selected_operator(state, rng)
    new_obj = new_state.objective()
    actual_improvement = prev_obj - new_obj
    _construct_training_data(new_state, actual_improvement)
```

### 3. Enhanced Fallback Strategy
```python
# Original: Fallback to greedy
if failed_demands:
    new_state, improvement = _call_greedy_repair(state, rng)
    _construct_training_data(new_state, improvement)

# Improved: Random selection for fallback
if failed_demands:
    prev_obj = state.objective()
    selected_operator = rng.choice(REPAIR_OPERATORS)
    new_state = selected_operator(state, rng)
    new_obj = new_state.objective()
    actual_improvement = prev_obj - new_obj
    _construct_training_data(new_state, actual_improvement)
```

## Expected Benefits

1. **More Diverse Training Data**: The ML model learns from various repair strategies rather than just greedy approaches.

2. **Better Generalization**: Exposure to different operators helps the model understand which strategies work better in different scenarios.

3. **Reduced Bias**: Avoids over-reliance on greedy repair patterns during training.

4. **Improved Performance**: The model can potentially discover better repair patterns by learning from multiple approaches.

5. **Robust Bootstrapping**: The operator can start learning immediately with diverse examples.

## Implementation Notes

- All repair operators share the same function signature, making them interchangeable.
- The random selection uses the provided `rng` parameter for reproducibility.
- Training data construction remains the same to maintain compatibility.
- The improvement maintains backward compatibility with existing ALNS configuration.

## Integration with Existing ALNS Framework

To use this improved operator, replace the registration in `main.py`:

```python
# Original
learning_based_repair_op = LearningBasedRepairOperator(**ALNSConfig.LEARNING_BASED_REPAIR_PARAMS)
alns.add_repair_operator(learning_based_repair_op)

# Improved
improved_learning_based_repair_op = ImprovedLearningBasedRepairOperator(**ALNSConfig.LEARNING_BASED_REPAIR_PARAMS)
alns.add_repair_operator(improved_learning_based_repair_op)
```

## Conclusion

This improvement addresses the fundamental bootstrapping limitation by providing diverse training data from multiple repair operators. The approach is feasible, maintains system compatibility, and should lead to better ML model performance through more comprehensive training examples.</content>
<parameter name="filePath">d:\Gurobi_code\Problem2\improved_learning_based_repair_proposal.md