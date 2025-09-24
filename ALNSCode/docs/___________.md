# 针对ALNS算法的改进Operators设计与伪代码

## 引言

本Markdown文件组织了两个改进operators的设计思路和对应的伪代码：周期相关 removal operator（作为destroy operator，用于破坏当前解）和学习-based repair operator（作为repair operator，用于修复破坏后的解）。这些改进基于现有ALNS实现代码（例如SolutionState、Vehicle类和alnsopt.py中的函数），并考虑了问题特征（如多周期库存转移s_ikt、车辆容量约束）。设计思路来源于ALNS综述文章（PDF）的相关变体（如Shaw removal的扩展和history/learning-based repair），并针对多周期动态和适应性进行了优化。

文件结构：

* **Destroy Operator** ：周期相关 removal的设计思路和伪代码。
* **Repair Operator** ：学习-based repair的设计思路和伪代码。
* **实施建议** ：如何集成到现有代码，以及潜在优势与风险。

最后更新日期：2025年9月20日。

## Destroy Operator：周期相关 Removal (Time-related Shaw Variant)

### 设计思路

这个operator扩展了Shaw removal（文章中常见destroy operator，用于基于相似性移除相关元素群）。核心是融入周期（time）维度，确保移除考虑时间依赖（如相邻天库存转移），避免破坏后导致跨周期负库存或供需失衡。这提升了算法对多周期问题的处理能力，比标准random removal更具针对性。

* **相似度函数** ：定义周期相关性 = α * 时间差（|day_i - day_j|） + β * 需求相似（经销商需求向量余弦相似度） + γ * 库存转移相似（s_ikt差异的欧氏距离）。权重α/β/γ可调（e.g., 0.4/0.3/0.3），强调时间依赖。
* **移除过程** ：从当前解（vehicles列表）中选种子元素（随机或高成本车辆），计算所有元素的相关性，移除一群高相关元素（degree比例，0.2-0.4）。使用K-means聚类（numpy）分组周期相关簇，增加结构性破坏。
* **问题适应** ：针对多周期（horizons），优先移除导致库存不稳的周期群（如生产计划密集天）。集成supply_chain，确保移除后不破坏供应链；添加噪声（uniform随机）避免重复（文章建议）。
* **参数** ：degree=0.3（从alns_config.py继承）；簇数k=3-5（基于horizons）；噪声幅度=0.1。
* **与现有代码整合** ：在alnsopt.py的destroy函数中调用compute_inventory()预计算s_ikt；使用DataALNS的sku_prod_each_day获取生产相似。
* **优势** ：提升多周期动态优化，减少punish_negative_inventory；创新性高（周期特定变体，可论文突出）。
* **风险** ：聚类计算开销中（O(n^2)相似度），可优化为采样子集；若horizons大，需限簇规模。

### 伪代码

```python
def time_related_shaw_removal(state: SolutionState, rng: rnd.Generator, degree: float = 0.3, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3, k_clusters: int = 3) -> SolutionState:
    import numpy as np
    from sklearn.cluster import KMeans

    # Step 1: 预计算所有分配元素的特征向量（周期、需求、库存）
    state.compute_inventory()  # 更新s_ikt
    allocations = []  # list of (veh, sku, day, feature)
    for veh in state.vehicles:
        plant = veh.fact_id
        dealer = veh.dealer_id
        day = veh.day
        demand_vec = np.array([state.data.demands.get((dealer, sku), 0) for sku in state.data.all_skus])  # 需求向量
        for (sku, _day), qty in list(veh.cargo.items()):  # 嵌套循环提取sku
            if _day != day: continue  # 只考虑当前周期
            inv_transfer = state.s_ikt.get((plant, sku, day), 0)  # sku-specific
            feature = np.array([day, demand_vec.mean(), inv_transfer])  # 简化特征
            allocations.append((veh, sku, day, feature))

    if not allocations:
        return state

    # Step 2: 使用K-means聚类分组周期相关簇
    features = np.array([a[3] for a in allocations])
    try:
        kmeans = KMeans(n_clusters=min(k_clusters, len(allocations)), random_state=rng.integers(0, 1000))
        labels = kmeans.fit_predict(features)
    except ValueError:  # 处理簇数无效
        return state

    # Step 3: 选种子簇（随机或高成本），计算相似度并移除
    unique_labels = np.unique(labels)
    if len(unique_labels) == 0: return state
    seed_cluster = rng.choice(unique_labels)
    cluster_indices = np.where(labels == seed_cluster)[0]
    cluster_allocs = [allocations[i] for i in cluster_indices]

    if len(cluster_allocs) <= 1: return state  # 不足以计算相似

    similarities = []
    seed_feature = cluster_allocs[0][3]
    for alloc in cluster_allocs[1:]:
        feat = alloc[3]
        time_diff = abs(feat[0] - seed_feature[0])
        demand_cos = np.dot(feat[1:], seed_feature[1:]) / (np.linalg.norm(feat[1:]) * np.linalg.norm(seed_feature[1:]) + 1e-8)  # 避免除零
        inv_dist = abs(feat[2] - seed_feature[2])
        sim = alpha * time_diff + beta * (1 - demand_cos) + gamma * inv_dist
        noise = rng.uniform(-0.1, 0.1) * sim
        sim += noise
        similarities.append((alloc, sim))

    # Step 4: 排序并移除高相关（低sim）元素
    similarities.sort(key=lambda x: x[1])  # 升序（更相关先移除）
    num_remove = int(len(cluster_allocs) * degree)
    for i in range(min(num_remove, len(similarities))):
        veh, sku, day, _ = similarities[i][0]
        qty = veh.cargo.pop((sku, day), 0)
        sku_size = state.data.sku_sizes[sku]
        veh.capacity += qty * sku_size
        if veh.is_empty():
            state.vehicles.remove(veh)

    return state
```

## Repair Operator：学习-based Repair (ML Variant) - 重新设计

### 架构分析与问题识别

经过对现有代码的深入分析，发现tracker并非SolutionState的属性，而是ALNSOptimizer的属性。这导致原设计中的`state.tracker`访问方式不可行。现有架构如下：

- **ALNSOptimizer类**：包含`self.tracker: ALNSTracker`实例，负责整个优化流程
- **SolutionState类**：独立的解状态表示，不包含tracker引用
- **ALNSTracker类**：已具备`features`和`labels`属性，通过`update_ml_data()`方法收集ML数据
- **ALNS框架**：通过回调函数机制运行，repair operators作为独立函数执行

### 重新设计的解决方案

**核心思路**：在SolutionState中添加tracker引用，使repair operator能够访问历史数据。这种设计保持了架构的清洁性，同时支持ML功能。

#### 方案要点

* **SolutionState扩展**：添加可选的`tracker`属性，通过`set_tracker()`方法在运行时注入
* **数据流重设计**：repair operator通过`state.tracker`访问历史数据，无需修改函数签名
* **向后兼容**：tracker属性为可选，不影响现有代码运行
* **集成方式**：在ALNSOptimizer.run_optimization()中，创建初始解后立即注入tracker引用

#### 技术细节

* **模型选择**：使用sklearn的RandomForestRegressor（轻量、稳定）。输入特征：[demand, sku_size, day, inventory, capacity_utilization]；输出：预测插入质量分数
* **训练策略**：动态训练（每50-100迭代重训），初期无数据时fallback到greedy_repair
* **特征工程**：标准化处理，避免量纲差异影响；添加交互特征（如demand*inventory）
* **数据管理**：限制历史数据大小（1000样本），使用滑动窗口避免内存溢出
* **性能优化**：训练开销控制在可接受范围内（<1秒/次），避免影响迭代效率

#### 风险控制

* **过拟合防护**：小样本时使用简单模型（线性回归），大样本时使用随机森林
* **鲁棒性保证**：预测失败时自动降级到传统repair方法
* **数值稳定**：特征归一化，标签平滑处理极值

### 设计思路优势

1. **架构优雅**：最小化对现有代码的侵入性修改
2. **扩展性强**：其他repair operators也可复用tracker机制
3. **创新性高**：ML与ALNS的深度融合，适合论文贡献
4. **实用性好**：动态学习提升算法效率，减少无效搜索

### 修改后的SolutionState类扩展

```python
@dataclass
class SolutionState:
    data: DataALNS
    vehicles: List[Vehicle] = field(default_factory=list)
    s_ikt: Dict[Tuple[str, str, int], int] = field(default_factory=dict)
    s_indices: Set[Tuple[str, str, int]] = field(default_factory=set)
    tracker: Optional['ALNSTracker'] = None  # 新增：tracker引用

    def __post_init__(self):
        self.vehicles = []
        self.s_ikt = {}
        self.s_indices = self.construct_indices()
        self._iteration_count = 0
        
        # 对s_ikt进行初始化, s_ik0表示期初库存
        for (plant, sku_id, day), inv in self.data.historical_s_ikt.items():
            if day == 0:
                self.s_ikt[plant, sku_id, day] = inv
    
    def set_tracker(self, tracker: 'ALNSTracker'):
        """设置tracker引用，用于ML-based operators"""
        self.tracker = tracker
    
    def copy(self):
        """复制当前解时保持tracker引用"""
        new_state = SolutionState(self.data)
        new_state.vehicles = copy.deepcopy(self.vehicles)
        new_state.s_ikt = copy.deepcopy(self.s_ikt)
        new_state.s_indices = copy.deepcopy(self.s_indices)
        new_state.tracker = self.tracker  # 保持tracker引用
        return new_state
```

### ALNSOptimizer集成修改

```python
class ALNSOptimizer:
    # ... 现有代码 ...
    
    def run_optimization(self, dataset_name: str) -> bool:
        """运行完整的优化流程 - 修改版"""
        try:
            # ... 前面的步骤保持不变 ...
            
            # 2. 创建初始解
            init_sol = self.create_initial_solution()
            if init_sol is None:
                return False
            
            # 7. 设置追踪器和回调函数
            self.tracker = ALNSTracker()
            
            # 新增：将tracker注入到解状态中
            init_sol.set_tracker(self.tracker)
            
            # ... 其余代码保持不变 ...
```

### 重新设计的Learning-based Repair伪代码（修正版）

```python
def learning_based_repair(state: SolutionState, rng: rnd.Generator, 
                         model_type: str = 'adaptive', min_score: float = 0.5, 
                         retrain_interval: int = 100) -> SolutionState:
    """
    基于机器学习的修复算子 - 完整实现版
    
    Parameters:
    -----------
    state : SolutionState
        当前解状态（现在包含tracker引用）
    rng : rnd.Generator  
        随机数生成器
    model_type : str
        模型类型：'simple'(Ridge), 'sklearn'(RandomForest), 'adaptive'(自动选择)
    min_score : float
        最小可接受预测分数阈值
    retrain_interval : int
        模型重训练间隔（迭代次数）
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Step 1: 检查tracker可用性，无tracker时fallback到greedy
    if state.tracker is None:
        return greedy_repair(state, rng)
    
    tracker_stats = state.tracker.get_statistics()
    features_data = state.tracker.features
    labels_data = state.tracker.labels
    current_iteration = tracker_stats['total_iterations']
    
    # 初始阶段数据不足时使用greedy
    if len(features_data) < 50:
        return greedy_repair(state, rng)

    # Step 2: 检查是否需要重训练模型
    # 使用tracker的缓存机制，避免频繁重训练
    need_retrain = True
    model = None
    scaler = None
    
    # 检查tracker中是否有缓存的模型和上次训练迭代数
    if state.tracker.has_cached_model():
        cached_model, cached_scaler, last_train_iter = state.tracker.get_cached_model()
        if current_iteration - last_train_iter < retrain_interval:
            # 距离上次训练未达到间隔，使用缓存模型
            need_retrain = False
            model = cached_model
            scaler = cached_scaler
            print(f"[OPLOG] 使用缓存模型，距上次训练 {current_iteration - last_train_iter} 迭代")
        else:
            print(f"[OPLOG] 缓存模型过期 ({current_iteration - last_train_iter} > {retrain_interval})，需要重新训练")
    else:
        print("[OPLOG] 无缓存模型，开始首次训练")
    if need_retrain:
        try:
            X = np.array(features_data)
            y = np.array(labels_data)
            
            # 特征标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 根据model_type参数选择模型
            if model_type == 'simple':
                # 强制使用简单模型
                model = Ridge(alpha=1.0, random_state=rng.integers(0, 1000))
                print(f"[OPLOG] 使用指定的简单模型 (Ridge)")
                
            elif model_type == 'sklearn':
                # 强制使用复杂模型
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    random_state=rng.integers(0, 1000),
                    n_jobs=1
                )
                print(f"[OPLOG] 使用指定的随机森林模型")
                
            elif model_type == 'adaptive':
                # 自动根据数据量选择模型
                if len(X) < 200:
                    model = Ridge(alpha=1.0, random_state=rng.integers(0, 1000))
                    print(f"[OPLOG] 自适应选择简单模型 (数据量: {len(X)})")
                else:
                    model = RandomForestRegressor(
                        n_estimators=50,
                        max_depth=10,
                        random_state=rng.integers(0, 1000),
                        n_jobs=1
                    )
                    print(f"[OPLOG] 自适应选择随机森林模型 (数据量: {len(X)})")
            else:
                # 未知模型类型，fallback到自适应
                print(f"[OPLOG] 未知模型类型 '{model_type}'，使用自适应模式")
                model = Ridge(alpha=1.0, random_state=rng.integers(0, 1000)) if len(X) < 200 else RandomForestRegressor(n_estimators=50, max_depth=10, random_state=rng.integers(0, 1000), n_jobs=1)
            
            # 训练模型
            model.fit(X_scaled, y)
            
            # 缓存模型和相关信息
            state.tracker.cache_ml_model(model, scaler, current_iteration)
            print(f"[OPLOG] 模型训练完成，已缓存到tracker")
            
        except Exception as e:
            # 模型训练失败时降级到greedy
            print(f"[OPLOG] 模型训练失败 ({type(e).__name__}), 使用greedy修复")
            return greedy_repair(state, rng)

    # Step 4: 获取待插入的未满足需求
    removal_list = get_removal_list(state)
    if not removal_list:
        return state

    # Step 5: 为每个待插入需求生成候选并预测分数
    total_predictions = 0
    successful_insertions = 0
    
    for (dealer, sku_id), remain_qty in removal_list.items():
        candidates = []
        
        # 生成候选插入位置
        available_plants = state.data.get_available_plants_for_dealer_sku(dealer, sku_id)
        
        for plant in available_plants:
            for day in range(1, state.data.horizons + 1):
                for veh_type in state.data.all_veh_types:
                    try:
                        # 提取特征向量
                        demand = state.data.demands.get((dealer, sku_id), 0)
                        sku_size = state.data.sku_sizes[sku_id] * remain_qty
                        inventory = state.s_ikt.get((plant, sku_id, day), 0)
                        veh_capacity = state.data.veh_type_cap[veh_type]
                        capacity_util = sku_size / veh_capacity if veh_capacity > 0 else 0
                        
                        # 构建特征向量[demand, sku_size, day, inventory, capacity_util]
                        feature_vector = np.array([[demand, sku_size, day, inventory, capacity_util]])
                        feature_scaled = scaler.transform(feature_vector)
                        
                        # 预测插入质量分数
                        pred_score = model.predict(feature_scaled)[0]
                        total_predictions += 1
                        
                        # 只保留高质量候选
                        if pred_score >= min_score:
                            candidates.append((plant, veh_type, day, pred_score, feature_vector[0]))
                            
                    except Exception:
                        continue  # 跳过异常候选

        # Step 6: 选择最佳候选进行插入
        if candidates:
            # 按预测分数降序排序
            candidates.sort(key=lambda x: x[3], reverse=True)
            
            for plant, veh_type, day, pred_score, raw_feature in candidates:
                try:
                    # 尝试插入
                    veh = Vehicle(plant, dealer, veh_type, day, state.data)
                    orders = {sku_id: remain_qty}
                    
                    # 记录插入前的目标函数值
                    prev_obj = state.objective()
                    
                    # 执行插入
                    success = veh_loading(state, veh, orders)
                    if success and veh.cargo:
                        state.vehicles.append(veh)
                        state.compute_inventory()
                        
                        # 计算实际改进并更新tracker
                        new_obj = state.objective()
                        actual_improvement = prev_obj - new_obj
                        
                        # 更新ML训练数据
                        state.tracker.update_ml_data(raw_feature.tolist(), actual_improvement)
                        successful_insertions += 1
                        
                        break  # 成功插入，处理下一个需求
                        
                except Exception:
                    continue  # 插入失败，尝试下一个候选
        
        # 如果所有候选都失败，使用greedy作为fallback
        if not candidates:
            # 对剩余未处理的需求使用greedy方法
            greedy_state = greedy_repair(state, rng)
            state.vehicles = greedy_state.vehicles
            state.s_ikt = greedy_state.s_ikt

    # Step 7: 记录性能统计
    print(f"[OPLOG] ML修复完成: {total_predictions}次预测, {successful_insertions}次成功插入")
    
    return state


class LearningBasedRepairOperator:
    """
    带参数的基于学习的修复算子包装类 - 完整版
    """
    def __init__(self, model_type: str = 'adaptive', min_score: float = 0.5, retrain_interval: int = 100):
        self.model_type = model_type
        self.min_score = min_score
        self.retrain_interval = retrain_interval
        self.__name__ = "learning_based_repair"
        
        # 验证参数有效性
        valid_models = ['simple', 'sklearn', 'adaptive']
        if model_type not in valid_models:
            print(f"[WARNING] 无效的model_type '{model_type}', 使用默认值 'adaptive'")
            self.model_type = 'adaptive'
        
        if retrain_interval < 10:
            print(f"[WARNING] retrain_interval过小 ({retrain_interval}), 设置为最小值 10")
            self.retrain_interval = 10

    def __call__(self, current: SolutionState, rng: rnd.Generator):
        return learning_based_repair(current, rng, self.model_type, self.min_score, self.retrain_interval)
```

## 参数设计原理与使用建议

### model_type参数详解

**设计目的**：提供灵活的模型选择策略，适应不同问题规模和数据特征

**参数选项**：
- `'simple'`：强制使用Ridge回归，适用于数据量小、特征维度低的场景
- `'sklearn'`：强制使用RandomForest，适用于数据量大、特征复杂的场景  
- `'adaptive'`（推荐）：自动根据数据量选择最优模型

**使用建议**：
```python
# 小规模问题（<10基地，<7天期）
LEARNING_BASED_REPAIR_PARAMS = {
    'model_type': 'simple',
    'min_score': 0.3,
    'retrain_interval': 50
}

# 中大规模问题（>10基地，>7天期）
LEARNING_BASED_REPAIR_PARAMS = {
    'model_type': 'sklearn', 
    'min_score': 0.5,
    'retrain_interval': 100
}

# 通用配置（推荐）
LEARNING_BASED_REPAIR_PARAMS = {
    'model_type': 'adaptive',
    'min_score': 0.4,
    'retrain_interval': 80
}
```

### retrain_interval参数详解

**设计目的**：控制模型重训练频率，平衡学习效果与计算开销

**工作机制**：
- 在tracker中缓存训练好的模型和标准化器
- 记录上次训练的迭代数(`_last_train_iteration`)
- 只有当迭代间隔达到`retrain_interval`时才重新训练
- 否则复用缓存的模型，显著减少计算开销

**参数影响**：
- **过小**（<20）：频繁重训练，计算开销大，可能导致过拟合
- **过大**（>200）：模型更新不及时，无法适应新的搜索模式
- **适中**（50-150）：平衡学习效果与性能，推荐区间

**自适应建议**：
```python
# 根据问题规模动态调整
if horizons <= 5:
    retrain_interval = 50   # 小问题，快速学习
elif horizons <= 10: 
    retrain_interval = 100  # 中等问题，标准配置
else:
    retrain_interval = 150  # 大问题，稳定优先
```

### 参数协同优化

**协同关系**：
- `model_type='simple'` + `retrain_interval=50`：快速学习，适合探索阶段
- `model_type='sklearn'` + `retrain_interval=100`：稳定学习，适合收敛阶段
- `model_type='adaptive'` + `retrain_interval=80`：动态平衡，通用配置

**性能监控指标**：
- 训练时间占比：< 5%（避免影响整体效率）
- 预测准确性：相关系数 > 0.3（确保学习有效）
- 成功插入率：> 60%（验证模型实用性）

### 当前实现中的修正需要

基于代码审查，当前实现需要以下修正：

1. **model_type参数使用**：
   - 当前代码忽略了用户指定的model_type
   - 需要按照伪代码实现显式的模型选择逻辑

2. **retrain_interval参数使用**：
   - 当前每次调用都重新训练，未考虑训练间隔
   - 需要在tracker中添加模型缓存机制

3. **性能优化**：
   - 添加训练时间监控，避免ML开销过大
   - 增加预测统计，评估模型有效性

这些修正将显著提升算法的实用性和可配置性，使learning-based repair成为一个真正智能化的ALNS组件。

## 实施建议 - 更新版

### 代码审查总结

✅ **正确实现的部分**：
- SolutionState正确添加了tracker属性和相关方法
- ALNSOptimizer正确注入了tracker引用
- 算子注册和参数配置机制完整
- 基本的ML训练和预测逻辑正确
- ALNSTracker已具备ML数据存储功能

❌ **需要修正的问题**：
- `model_type`参数未被使用，代码硬编码了模型选择
- `retrain_interval`参数被忽略，每次都重新训练
- **新增**：ALNSTracker缺少模型缓存属性和方法
- 缺少模型缓存机制，造成不必要的计算开销
- 缺少性能监控和统计信息

1. **修改SolutionState类**：
   - 添加可选的`tracker`属性和`set_tracker()`方法
   - 在`copy()`方法中保持tracker引用
   - 确保向后兼容性（tracker为None时正常运行）

2. **修改ALNSOptimizer类**：
   - 在`run_optimization()`中创建初始解后立即调用`init_sol.set_tracker(self.tracker)`
   - 确保所有复制的解状态都保持tracker引用

3. **注册新的repair operator**：
   - 在`_register_repair_operators()`中添加`learning_based_repair`
   - 在alnsopt.py中实现完整的repair函数

4. **扩展ALNSTracker**（需要修改）：
   - 添加模型缓存属性：`_cached_model`、`_cached_scaler`、`_last_train_iteration`
   - 添加缓存管理方法：`cache_ml_model()`、`get_cached_model()`、`has_cached_model()`、`clear_ml_cache()`、`get_ml_cache_info()`
   - 在`get_statistics()`中包含ML缓存信息

### 测试与优化策略

1. **渐进式测试**：
   - 先在小规模数据集上验证基本功能
   - 逐步增加数据规模，监控性能指标
   - 对比ML-repair与传统repair的效果差异

2. **性能监控**：
   - 跟踪模型训练时间（目标<1秒/次）
   - 监控内存使用（features列表大小控制）
   - 记录预测准确性和实际改进相关性

3. **参数调优**：
   - `retrain_interval`：根据问题规模调整（50-200迭代）
   - `min_score`：根据历史数据分布调整阈值
   - 模型超参数：树的数量、深度等

### 架构优势与创新点

1. **架构设计优势**：
   - **最小侵入性**：只需在两个关键点修改（SolutionState和ALNSOptimizer）
   - **向后兼容**：tracker为可选，不影响现有算子运行
   - **扩展性强**：其他算子也可利用tracker机制进行增强
   - **解耦合好**：ML逻辑封装在repair operator内部

2. **论文贡献点**：
   - **方法创新**：首次将ML与ALNS的repair机制深度融合
   - **自适应性**：动态学习最优插入策略，优于静态启发式
   - **实用性强**：在多周期库存约束下显著提升求解效率
   - **架构可扩展**：为未来ML-ALNS混合算法提供范例

3. **与现有方法对比**：
   - vs **传统ALNS**：增加学习能力，减少盲目搜索
   - vs **静态ML**：动态适应问题特征，持续改进
   - vs **纯启发式**：基于历史数据决策，更加智能

### 潜在挑战与解决方案

1. **过拟合风险**：
   - **问题**：训练数据不足时模型可能过拟合
   - **解决**：使用模型复杂度随数据量调整的策略（Ridge→RandomForest）

2. **计算开销**：
   - **问题**：ML训练可能增加计算时间
   - **解决**：控制训练频率和模型复杂度，设置时间阈值

3. **数据质量**：
   - **问题**：early iteration的数据质量可能较低
   - **解决**：使用数据过滤和平滑策略，延迟ML启动时机

4. **维度诅咒**：
   - **问题**：高维特征空间可能导致稀疏性
   - **解决**：特征选择和降维，重点关注最相关特征

### 实验设计建议

1. **对照实验**：
   - 基线：原始ALNS（无ML）
   - 变体1：ML-repair（本设计）
   - 变体2：ML-destroy（如果实现）

2. **评估指标**：
   - **解质量**：最终目标函数值、可行性
   - **收敛速度**：到达最优解的迭代次数
   - **计算效率**：总运行时间、每迭代平均时间
   - **学习效果**：预测准确性随迭代的改进

3. **数据集设计**：
   - 小规模：验证正确性（3天期，5基地5经销商）
   - 中规模：性能测试（7天期，10基地15经销商）
   - 大规模：扩展性验证（14天期，20基地30经销商）

通过这种重新设计，learning-based repair operator能够在现有架构下正确运行，同时为算法引入显著的创新元素和实际性能提升。

# ALNSTracker扩展 - 完整版

## 现有实现分析

通过代码分析发现，ALNSTracker已经具备了支持learning-based repair所需的基本功能：

1. **已有属性**：`self.features = []` 和 `self.labels = []` 已在当前实现中存在
2. **数据更新方法**：`update_ml_data(feature, label)` 方法已实现
3. **数据访问**：`get_statistics()` 方法已返回features和labels
4. **内存管理**：已实现1000样本的滑动窗口机制

## 需要添加的模型缓存功能

为了支持learning-based repair operator中的模型缓存机制，需要为ALNSTracker添加以下属性和方法：

### 新增属性
- `_cached_model`: 缓存训练好的ML模型对象
- `_cached_scaler`: 缓存特征标准化器对象  
- `_last_train_iteration`: 记录上次模型训练的迭代数

### 新增方法
- `cache_ml_model(model, scaler, iteration)`: 缓存模型和相关信息
- `get_cached_model()`: 获取缓存的模型
- `clear_ml_cache()`: 清除ML缓存（用于重置或内存管理）

## 完整的ALNSTracker扩展实现

```python
import os
import csv
import time
import copy
from optutility import LogPrinter
import numpy.random as rnd
from alnsopt import SolutionState
from typing import Optional, Any, Tuple

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
        self._cached_model: Optional[Any] = None      # 缓存的ML模型
        self._cached_scaler: Optional[Any] = None     # 缓存的特征标准化器
        self._last_train_iteration: Optional[int] = None  # 上次训练的迭代数
        
        # 如果指定了输出文件，则创建文件并写入表头
        if self.output_file:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Iteration', 'Current_Obj', 'Best_Obj', 'Gap'])
        
    def on_iteration(self, state: SolutionState, rng: rnd.Generator, **kwargs):
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
    def update_ml_data(self, feature: list, label: float):
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
        print(f"[OPLOG] ML模型已缓存，训练迭代: {iteration}")
    
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
        检查是否有缓存的模型
        
        Returns:
        --------
        bool
            是否有缓存的模型
        """
        return (self._cached_model is not None and 
                self._cached_scaler is not None and 
                self._last_train_iteration is not None)
    
    def clear_ml_cache(self):
        """
        清除ML模型缓存，用于重置或内存管理
        """
        self._cached_model = None
        self._cached_scaler = None
        self._last_train_iteration = None
        print("[OPLOG] ML模型缓存已清除")
    
    def get_ml_cache_info(self) -> dict:
        """
        获取ML缓存的详细信息
        
        Returns:
        --------
        dict
            包含缓存状态信息的字典
        """
        return {
            'has_cache': self.has_cached_model(),
            'last_train_iteration': self._last_train_iteration,
            'model_type': type(self._cached_model).__name__ if self._cached_model else None,
            'scaler_type': type(self._cached_scaler).__name__ if self._cached_scaler else None,
            'data_samples': len(self.features)
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
```

## 扩展功能详解

### 1. 模型缓存机制

**核心属性**：
- `_cached_model`: 存储训练好的ML模型（Ridge或RandomForest）
- `_cached_scaler`: 存储特征标准化器（StandardScaler）
- `_last_train_iteration`: 记录模型训练时的迭代数

**缓存管理方法**：
- `cache_ml_model()`: 保存模型、标准化器和训练迭代数
- `get_cached_model()`: 返回缓存的三元组
- `has_cached_model()`: 检查缓存完整性
- `clear_ml_cache()`: 清除缓存（用于重置）

### 2. 缓存状态监控

**信息获取方法**：
- `get_ml_cache_info()`: 返回详细的缓存状态信息
- 在`get_statistics()`中包含ML缓存信息

**监控指标**：
- 缓存是否存在
- 上次训练迭代数
- 模型和标准化器类型
- 当前数据样本数量

### 3. 内存和性能优化

**自动内存管理**：
- 特征和标签数据限制在1000个样本
- 模型缓存按需更新，避免频繁训练

**性能监控**：
- 记录训练迭代间隔
- 提供缓存使用统计
- 支持缓存状态查询

## 与Learning-based Repair的集成

### 缓存检查逻辑

```python
# 在repair operator中的使用
if state.tracker.has_cached_model():
    model, scaler, last_iter = state.tracker.get_cached_model()
    if current_iteration - last_iter < retrain_interval:
        # 使用缓存模型
        print(f"使用缓存模型，距上次训练 {current_iteration - last_iter} 迭代")
    else:
        # 需要重新训练
        print("缓存模型过期，需要重新训练")
else:
    # 首次训练
    print("无缓存模型，开始训练")
```

### 缓存更新逻辑

```python
# 训练完成后更新缓存
state.tracker.cache_ml_model(model, scaler, current_iteration)
```

## 架构优势

1. **高效缓存**：避免重复训练，显著提升性能
2. **状态透明**：提供完整的缓存状态监控
3. **内存安全**：自动管理内存使用
4. **向后兼容**：不影响现有功能
5. **扩展性强**：支持未来更多ML组件

这种扩展使ALNSTracker成为一个完整的ML支持跟踪器，为learning-based repair operator提供了必要的缓存和管理功能。

## 完整的实施步骤

### 1. 修改ALNSTracker类
按照"ALNSTracker扩展 - 完整版"章节中的完整实现，添加模型缓存功能。

### 2. 修改Learning-based Repair Operator  
按照修正后的伪代码实现，使用新的缓存管理方法。

### 3. 更新参数配置
确保alns_config.py中的参数配置与新的实现兼容。

### 4. 测试验证
- 验证模型缓存功能正常工作
- 检查训练间隔控制正确
- 确认性能监控信息准确
- 测试内存使用在合理范围内

通过这些修改，learning-based repair operator将具备完整的ML缓存和性能优化功能。
