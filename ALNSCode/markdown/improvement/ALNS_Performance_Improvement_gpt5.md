# 多周期产品配送 ALNS 性能改进方案 (草稿)

> 状态: 草稿 v0.1  (仅包含：问题概述、瓶颈诊断假设、测量/剖析计划)  
> 后续待补充：优化策略矩阵、伪代码、路线图、风险 & 收益评估、参考文献。

## 1. 问题概述
中小规模数据集 (30 个小规模算例) 上当前 ALNS 实现能够在合理时间内得到可行解。然而在中等规模数据集上：
- 初始解生成耗时 ~300s；
- 第一次迭代完成时间约 1043.8s；
- 1800s 总时限内仅完成 12 次迭代；
- 大规模数据上 38 分钟仍未得到初始解。  
这导致：
1. 迭代次数过少 → 学习型修复算子无法积累足够样本；
2. 算子调用频率极低，ALNS 自适应机制难以发挥；
3. 初始阶段时间消耗过大，整体 wall-clock 性能严重不足。

## 2. 主要复杂度与潜在瓶颈假设
结合代码结构(`alnsopt.py`)与日志特征，总结如下候选瓶颈：

### 2.1 初始解 `initial_solution()`
- 重复调用 `state.compute_inventory()` (双重嵌套：经销商→工厂→天→车型)；
- 每次车辆装载 `veh_loading()` 内又调用 `state.compute_inventory()`；
- 多次计算 `state.compute_shipped()` 来判断剩余需求；
- 需求遍历顺序固定（大需求优先）缺乏剪枝或聚合，导致在大数据集上呈现 O(|D| * |Plants_j| * T * |VehTypes|) 次库存与需求重复统计。  

### 2.2 库存计算 `compute_inventory()`
- 每次全量重算所有 `(plant, sku, day)`，复杂度 O(|Plants| * |Skus| * T)。
- 许多场景下只是局部新增/删除少量车辆；缺乏增量更新与脏标记 (dirty flag) 机制。  

### 2.3 装载函数 `veh_loading()`
- 内部用 `used_inv` 统计同基地当日装载量，重复扫描 `state.vehicles`；
- 车辆容量与可用库存循环中反复整数除法与哈希查找；
- 可能频繁创建“空车再丢弃”，缺少对“剩余需求+剩余容量”提前终止判断；
- 结尾再次调用 `state.compute_inventory()` 触发全量刷新。  

### 2.4 破坏算子 (Removal Operators)
- 多数算子在操作后立即调用 `compute_inventory()` 全量更新；
- 某些算子（如 `periodic_shaw_removal`）做特征构造+KMeans，频繁在车辆很多时构造高维数组；
- `periodic_shaw_removal` 中 `demand_vector.mean()` 为每个车辆-货物对重复生成整向量。  

### 2.5 修复算子 (Repair Operators)
- `greedy_repair()` 与 `smart_batch_repair()` 内部频繁调用 `construct_supply_chain()`；
- 多层循环里调用 `compute_inventory()`；
- `smart_batch_repair()` 的资源池只构建一次，但后续未与车辆新增差量同步（可能引入不一致或需要额外开销重新修复）；
- `learning_based_repair()`：
  - 每个候选插入 (工厂 × 天 × 车型) 均做 `model.predict()` → O(|Plants_j| * T * |VehTypes| * |Unmet|)；
  - `state.objective()` 内部会调用 `validate()` → `compute_inventory()` → 再次全量计算。

### 2.6 目标函数与验证 `objective() / validate()`
- `objective()` 调用 `compute_inventory()`；
- `validate()` 又进行多次汇总 (`compute_shipped()`, 遍历库存、车辆)；
- 在迭代内部被高频调用（评估候选解、局部搜索、ML 插入等），成为热点函数。  

### 2.7 数据结构方面
- `vehicles` 为列表，移除操作 O(n)；按 key 检索需多次线性遍历；
- 缺乏针对 (plant, day) 或 (dealer, sku) 的索引缓存；
- 多次使用 `construct_supply_chain()` 临时构建映射（可能已在数据载入阶段可一次性持久化）。  

### 2.8 不必要的对象/内存开销
- KMeans 聚类对象频繁创建与销毁；
- 重复 `np.mean`、`np.std` 在小向量上开销与 Python 循环交织；
- 深拷贝 `state.copy()` 发生在所有算子入口，深度复制 vehicles 可能占主导。  

### 2.9 学习阶段延迟影响性能
- 初始解过慢 → 首轮迭代更慢 → 训练样本不足 → ML 算子退化为随机/贪心；
- 在少样本阶段仍然付出特征计算与候选生成成本。  

### 2.10 日志与打印 I/O
- 大量 `[OPLOG]` print 调用在 tight loop；
- I/O 可在 Windows 环境中显著拖慢（行缓冲/编码）。

## 3. 序列化的性能测量与剖析计划
目标：以“度量→定位→验证”循环驱动优化，避免盲目修改。

### 3.1 指标体系 (KPI)
| 类别 | 指标 | 说明 |
|------|------|------|
| Startup | T_init | 初始解生成耗时 (s) |
| Iteration | T_iter_mean / median | 单次完整迭代平均 & 中位耗时 |
| Iteration | I_per_600s | 600 秒内可完成迭代数 |
| Hotspot | #inventory_calls / s | 每秒库存重算次数 |
| Hotspot | %time_objective | `objective()+validate()` 占总 CPU 比 |
| Operator | T_destroy_avg / T_repair_avg | 各类算子平均耗时分布 |
| ML | sample_growth_rate | 每分钟新增训练样本数 |
| Memory | peak_RSS | 峰值常驻内存 |

### 3.2 度量工具组合
- 宏观时间线：自定义 `EventLogger`（写入 CSV：阶段名、t_start、t_end、meta）。
- 函数级 CPU：`cProfile` + `pstats` 按累计时间排序；
- 行级热点：对 `initial_solution`, `veh_loading`, `compute_inventory`, `objective` 使用 `line_profiler`；
- 内存：`tracemalloc`（采样快照前后比较 KMeans / 深拷贝）。
- 预测阶段：统计 `learning_based_repair` 中候选数量、过滤后数量、成功插入数。  

### 3.3 最小侵入式埋点建议
伪代码示例：
```python
class EventLogger:
    def __init__(self, path):
        self.f = open(path, 'w', newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow(['phase', 't_start', 't_end', 'elapsed', 'meta'])
    def log(self, phase, t0, meta=''):
        t1 = time.perf_counter()
        self.writer.writerow([phase, t0, t1, t1 - t0, meta])
    def close(self):
        self.f.close()
```
使用方式：
```python
logger.log('initial_solution', t0, meta=f'veh={len(state.vehicles)}')
```

### 3.4 典型剖析流程脚本化
1. 运行一次 baseline：记录 KPI；
2. 启用 `cProfile`：`python -m cProfile -o prof.out main.py`；
3. 分析：
```python
import pstats
p = pstats.Stats('prof.out').sort_stats('cumtime')
p.print_stats(30)
```
4. 对 hotspot 再加 `line_profiler`（仅包裹 4~6 个函数）；
5. 建立 `profiling/` 目录存放快照与报表；
6. 优化后重复步骤 1 对比比率 (speedup >= 2x 视为显著)。

### 3.5 验收门槛定义 (Performance Gates)
| Gate | 现状 (估) | 目标 Phase 1 | 目标 Phase 2 |
|------|-----------|--------------|--------------|
| T_init | ~300s | < 60s | < 20s |
| I_per_600s | 12/1800s ≈ 0.4/min | >= 5/min | >= 15/min |
| %time_objective | (待测) | < 25% | < 10% |
| #inventory_calls | (待测) | -50% | -80% |
| ML 样本 10 分钟 | 很少 | >= 200 | >= 600 |

---

(以下章节原为占位，已补充完整内容如下)

## 4. 性能优化策略总览
为避免“大杂烩式”修改，按影响域分层：

| 层级 | 目标 | 核心策略 | 预期收益 |
|------|------|----------|----------|
| A. 度量与可观测性 | 获取可信基线 | KPI / 采样 / 事件日志 | 决策有数据支撑 |
| B. 算法结构 | 降低初始解与单次迭代成本 | 分解、惰性计算、预算控制 | 初始解 & 迭代速度 2~5x |
| C. 数据结构 & 缓存 | 降低热点函数复杂度 | 增量库存、索引映射、结果缓存 | 减少重复 O(N) 扫描 |
| D. 算子级微优化 | 缩短算子调用耗时 | 减少深拷贝/打印/冗余聚类 | 释放 CPU 20~40% |
| E. ML 集成优化 | 加速进入有效学习阶段 | 分层候选过滤 / 热启动 / 异步训练 | 样本积累速率 3~5x |
| F. 并行与混合 | 利用多核 / 数学规划补强 | 轻量并行销毁、列池预解、局部 MILP | 质量提升 + 时间稳定 |
| G. 可靠性 & 降级 | 在超时/退化时保持进展 | 时间预算 / 看门狗 / 降级路径 | 避免卡死，保证最差性能 |

### 4.1 初始解生成优化
问题：当前全量贪心 + 多次全量库存/发货统计，耗时过长。

改进策略：
1. 分阶段构造 (Two-Stage Heuristic)：
   - Stage 1：仅按聚合需求 (dealer 总体量 或 dealer-sku 但不分日) 估算所需车辆数（近似容量向上取整），快速生成“骨架车辆”(无具体日)。
   - Stage 2：再按时间维度细分，将需求切成时间片，填充骨架。
2. 批量库存前推 (Forward Sweep)：一次性按 (plant, sku) 预计算每期“最大可用库存前缀”数组，用于 O(1) 估算可用量，避免每次局部构造时全量重算。
3. 按 SKU 分组 (SKU Clustering by Size Ratio)：先处理大体积 SKU（减少车辆碎片化）。
4. 引入时间预算 (Time Cap)：例如初始解最长期限 60s；超时则输出当前部分可行解 + 调用快速修复算子补足。
5. 可行性优先级 → “先可行后质量”：早期不检查所有罚项，只保证 (库存 >=0, 满足最小需求比例 θ)。
6. 减少 `compute_shipped()` 调用：维护增量字典 shipped_cache，车辆加入时更新。

### 4.2 库存计算增量化
现状：每次 `compute_inventory()` 全遍历所有 (plant, sku, day)。

策略：
1. 引入脏区 (Dirty Regions)：记录新增/删除车辆影响的 (plant, sku, day) 三元组最小 & 最大 day，局部重算区间。
2. 预先计算 prefix：`prefix_prod[plant, sku][day] = sum_{t<=day} production`，用于 O(1) 快速恢复区间值。
3. 车辆发货量映射：`ship_amount[(plant, sku)][day]` 数组（或 list），更新车辆时 O(1) 调整，再滚动计算库存。
4. 只在需要验证 / 目标函数时才保证库存一致；算子内部可“延迟校正” (lazy evaluation)。
5. 提供 `ensure_inventory(up_to_day=None)` 接口，允许只更新到指定天数。

### 4.3 车辆与需求索引结构
添加以下辅助索引（保持与原列表并存，不改变外部接口）：
```text
vehicles_by_fact_day[(plant, day)] -> list[veh_id]
vehicles_by_dealer_sku[(dealer, sku)] -> cumulative shipped qty (int)
```
目的：
* O(1)/O(log n) 查询某工厂在某天已有装载总量；
* 快速获取 dealer-sku 未满足需求无需重新聚合。

### 4.4 装载函数 `veh_loading` 优化
1. 预传入 (plant, day) 剩余容量指针结构（例如 `remaining_capacity[(plant, day, veh_type)]`）。
2. 使用 while 循环→单次计算最大可装 SKU 数量时避免整数除法多次重复。
3. 避免创建空车：先算最大可装 `max_qty = min(remain_demand, available, cap_limit)`，如果 `max_qty <=0` 直接 break/换车型。
4. 可选：批量装载多个 SKU（若后续扩展多 SKU 单车）。

### 4.5 破坏算子优化
1. 拆分“结构变更”与“库存更新”：先应用一批移除，批末统一调用一次增量库存刷新。
2. `periodic_shaw_removal`：
   - 预缓存 `dealer_avg_demand`；
   - 按需（车辆数超过阈值 M）才启用聚类；
   - 引入 `max_cluster_sample`，随机抽样减少 KMeans 输入规模。
3. 提供轻量替代算子 (例如 simple_day_removal) 在大规模阶段早期占较高权重。

### 4.6 修复算子优化
1. `greedy_repair`：合并同 dealer-sku 的工厂遍历，将 `construct_supply_chain()` 外提。
2. `smart_batch_repair`：资源池与库存保持同步（当车辆加载后仅更新受影响条目）。
3. 引入 “Repair Budget” 参数：每次 ALNS 迭代允许修复算子最多消耗 X ms，超时提前返回（保证迭代数量）。
4. 优化 ML repair：加入双层候选筛选：
   - Level 1（启发式粗筛）：过滤掉库存不足、车型容量明显不匹配的候选；
   - Level 2（ML 预测）：仅对通过粗筛的少量候选做 `model.predict()`。
5. 训练触发条件：除固定 `retrain_interval`，再加“新样本增长幅度阈值”触发（如新增样本 > 上次训练后 +30%）。

### 4.7 目标函数与验证分离
1. 提供 `objective_fast(assume_feasible=True)`：跳过 `validate()`；
2. 每 N 次（或当新增船量 > 阈值）才做一次完整 `validate()`；
3. 在 ALNS 接受准则（如 SA 温度计算）处仅使用 fast 版本加 penalty 缓存；
4. 引入 penalty 缓存表：若只改变少量车辆，只局部更新违规项贡献。

### 4.8 深拷贝与状态管理
1. `state.copy()` 仅对 vehicles 做结构化浅拷 + 共享不可变数据；
2. 提供“应用差分 (diff) 回滚”机制：破坏算子记录被删车辆、修复算子记录新增车辆，在拒绝解时快速撤销。
3. 采用对象池 (Object Pool) 重复利用 `Vehicle` 实例，减少 GC 频率。

### 4.9 ML 模型集成加速
1. Warm Start：启动时快速生成 5~10 个“不同随机种子”初始解，对其应用简单修复收集样本（可并行）。
2. 特征归一化：在 tracker 内集中做一次 batch 标准化（避免重复 scaler.transform 开销）。
3. 异步训练 (Optional)：主线程把新样本追加到共享队列，独立线程周期训练并替换模型（需线程安全读写锁）。
4. 模型类型自适应：数据量大后切换到 **极端随机树 ExtraTreesRegressor**（往往比 RF 更快）；
5. 失败回退策略：连续 K 次（如 3 次）预测插入全部无改进 → 调整 min_score 或降级到 greedy。

### 4.10 并行与混合策略
1. 多起点并行 (Parallel Multi-Start)：在时间预算内并行构造多个初始解，选最优进入主循环；
2. 并行算子候选：对同一解并行生成多个破坏+修复候选，快速择优（需注意线程间不可共享可变状态）；
3. 混合 MIP 局部强化：选择 (plant, dealer, subset sku, small time window) 构建小型 MIP 提升局部质量（限制求解 ≤ 3s）。
4. 列池 (Pattern Pool)：缓存历史高效车辆装载模式，在修复阶段快速重放（pattern reuse）。

### 4.11 降级与容错
1. Watchdog：若最近 5 分钟迭代数 < 阈值，自动降低破坏复杂度、关闭 KMeans、停用 ML。
2. 超时降级：算子内部利用 `if time.perf_counter()-t0 > budget: break`；
3. 训练失败/异常 try-except 必须返回原 state（已具备部分基础）。

## 5. 关键伪代码示例

### 5.1 增量库存更新结构
```python
class InventoryManager:
    def __init__(self, data):
        self.prefix_prod = precompute_prefix_production(data)
        self.ship = defaultdict(lambda: array('i', [0]*(data.horizons+1)))
        self.stock = {}  # (plant, sku, day) -> value
        self.dirty = set()  # {(plant, sku)} 需要刷新

    def add_vehicle(self, veh):
        for (sku, day), qty in veh.cargo.items():
            self.ship[(veh.fact_id, sku)][day] += qty
            self.dirty.add((veh.fact_id, sku))

    def remove_vehicle(self, veh):
        for (sku, day), qty in veh.cargo.items():
            self.ship[(veh.fact_id, sku)][day] -= qty
            self.dirty.add((veh.fact_id, sku))

    def ensure(self, up_to_day=None):
        for (plant, sku) in list(self.dirty):
            prev = initial_stock(plant, sku)
            for d in range(1, horizons+1):
                if up_to_day and d > up_to_day: break
                prod = self.prefix_prod[(plant, sku)][d] - self.prefix_prod[(plant, sku)][d-1]
                shipped = self.ship[(plant, sku)][d]
                curr = prev + prod - shipped
                self.stock[(plant, sku, d)] = curr
                prev = curr
            self.dirty.remove((plant, sku))
```

### 5.2 时间预算控制迭代
```python
def alns_loop(state, operators, time_limit, max_iters):
    start = time.perf_counter()
    for it in range(max_iters):
        if time.perf_counter() - start > time_limit: break
        iter_start = time.perf_counter()
        destroy = select_destroy(operators, budget_class='fast')
        repair  = select_repair(operators, remaining_time=time_limit-(time.perf_counter()-start))
        cand = repair(destroy(state))
        # fast objective
        obj = objective_fast(cand)
        accept = accept_rule(obj, state)
        if accept: state = cand
        record_iter_metrics(it, time.perf_counter()-iter_start)
```

### 5.3 ML 候选两阶段筛选
```python
def generate_candidates(unmet, data):
    # Level 1: 规则过滤
    raw = []
    for plant in feasible_plants(unmet):
        for day in feasible_days(unmet):
            if quick_inventory_check(plant, unmet.sku, day) <= 0: continue
            best_types = pick_vehicle_types(unmet.qty, top_k=2)
            for vt in best_types:
                raw.append((plant, day, vt))
    return raw

def score_with_model(cands, model, scaler):
    feats = build_feature_matrix(cands)
    scores = model.predict(scaler.transform(feats))
    return [c for c, s in zip(cands, scores) if s >= MIN_SCORE]
```

### 5.4 Pattern Pool 结构
```python
class PatternPool:
    def __init__(self, cap=500):
        self.pool = []
        self.cap = cap
    def add(self, veh):
        pattern = tuple(sorted(veh.cargo.items()))
        self.pool.append((pattern, veh.type))
        if len(self.pool) > self.cap: self.pool.pop(0)
    def sample(self, rng, k=5):
        return rng.sample(self.pool, min(k, len(self.pool)))
```

## 6. 实施路线图 (Roadmap)
### Phase 0 (Baseline Ready)
 - [ ] 引入 KPI 采集与 cProfile 流程
 - [ ] 记录基线：T_init, 迭代速率, %time_objective

### Phase 1 (高性价比 Quick Wins)
 - [ ] 关闭/可配置 verbose print → 日志级别分层
 - [ ] 缓存 supply_chain & dealer_avg_demand
 - [ ] 减少 `compute_inventory()` 调用：合并到算子批末
 - [ ] 初始解添加时间上限 + shipped_cache
 - [ ] objective_fast / 延迟 validate

### Phase 2 (结构性改造)
 - [ ] 引入 InventoryManager 增量库存
 - [ ] vehicles 索引 (plant, day) 与 (dealer, sku)
 - [ ] 修复算子合并 `construct_supply_chain()` 提前外提
 - [ ] ML repair 两阶段候选生成
 - [ ] Pattern Pool + Warm Start 多初始解

### Phase 3 (高级与混合)
 - [ ] 异步模型训练线程
 - [ ] 并行多算子候选 (线程或进程池 + 只读拷贝)
 - [ ] 局部 MIP 强化 (小窗口 MILP) 原型
 - [ ] Watchdog 自适应降级

### Phase 4 (精细化与稳定性)
 - [ ] 回滚 diff 机制替代深拷贝
 - [ ] 对象池化/内存剖析调优
 - [ ] 进一步 SIMD / NumPy 化关键循环

## 7. 风险 & 收益矩阵 (示例)
| 优化项 | 预期收益 | 实现复杂度 | 风险 | 优先级 |
|--------|----------|------------|------|--------|
| 减少 print / 合并库存重算 | 20~30% 启动加速 | 低 | 几乎无 | 高 |
| 初始解时间上限 + 分阶段 | 初始解 5~10x 加速 | 中 | 可行性初期略差 | 高 |
| 增量库存管理 | 迭代次数 2~3x | 中高 | 一致性 bug | 高 |
| 索引结构 (plant-day) | 减少 O(n) 扫描 | 中 | 维护成本 | 高 |
| ML 两阶段候选 | 预测耗时下降 50%+ | 中 | 过度过滤 | 中高 |
| 异步训练 | 主迭代更顺畅 | 高 | 线程安全 | 中 |
| 多起点 Warm Start | 更好初期质量 | 中 | 资源占用 | 中 |
| Pattern Pool | 修复多样性 & 速度 | 中 | 过时模式污染 | 中 |
| 局部 MIP 强化 | 质量提升 | 高 | 开发复杂 | 低中 |
| 回滚 diff 替深拷贝 | 迭代内存/时间优化 | 高 | 实现易出错 | 中 |

## 8. 评估框架与回归测试建议
1. 建立一组代表性数据集：small / medium / large（规模参数表）。
2. 对每次结构性修改：运行 3 次取平均（消除随机性）并记录 JSON 报告。
3. 建立 `perf_baseline.json`，使用脚本比较：若任一 KPI 劣化 > 10% 则标记回归。
4. 建立 “搜索质量” 指标：最终可行成本 / 已知下界 (来自 CG 或 Monolithic) 的 GAP。
5. 引入统计显著性：迭代成本分布用 Mann–Whitney U 检验差异是否显著 (可选)。

## 9. 参考文献 (精选)
1. Ropke, S., & Pisinger, D. (2006). An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows. Transportation Science.
2. Pisinger, D., & Ropke, S. (2010). Large Neighborhood Search. In Handbook of Metaheuristics.
3. Shaw, P. (1998). Using constraint programming and local search methods to solve vehicle routing problems. CP Workshop.
4. Aires, A. et al. (2020). A survey on machine learning in large-scale combinatorial optimization. (综述 ML 与组合优化融合路径)。
5. Bengio, Y., Lodi, A., & Prouvost, A. (2021). Machine Learning for Combinatorial Optimization: A Methodological Tour d'Horizon. Eur. J. OR.
6. Vidal, T. (2022). Hybrid genetic search for the CVRP: open-source implementation and new variants. Computers & OR. (混合启发式结构借鉴)。
7. Lu, H. et al. (2019). Learning heuristics for routing problems via deep reinforcement learning. (ML 加速启发式思想参考)。
8. Birattari, M. (2009). The automatic design of metaheuristics (参数自适应策略参考)。
9. Gurobi Optimization, LLC. (2024). Gurobi Documentation (局部 MIP 强化可参考官方性能调参章节)。
10. Johnson, S. (优化日志 I/O 性能常见实践博文/行业经验)。

## 10. 总结与下一步
该方案强调“先度量、后分层重构、再增量增强”，优先解决初始解与库存重算导致的结构性瓶颈，再通过 ML 策略精简与并行化增强扩大迭代数量。建议路线：
1. 立即落地 Phase 0/1 → 取得首轮 ≥3x 迭代速率提升；
2. 验证增量库存模块正确性（对比全量结果哈希校验）；
3. 引入 Warm Start + Pattern Pool 改善早期样本质量；
4. 再推进 ML 候选两阶段过滤与异步训练；
5. 大规模数据集回归对比 GAP 与时间曲线。  

如需，我可再进一步把其中某一模块（例如增量库存管理或初始解两阶段构造）细化为实现任务清单。

