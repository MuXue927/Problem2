# Improving Performance of ALNS Algorithm for Large-Scale Datasets

## 1. Introduction

The Adaptive Large Neighborhood Search (ALNS) algorithm is a powerful metaheuristic framework for solving combinatorial optimization problems, particularly vehicle routing and supply chain logistics, as evidenced by its widespread adoption in the literature (Mara et al., 2022). However, empirical testing on medium- and large-scale datasets reveals significant performance bottlenecks in the current implementation. Specifically:

- **Initial Solution Generation**: Takes ~300 seconds on medium datasets and >38 minutes on large ones, delaying the iterative phase.
- **Iteration Efficiency**: Only 12 iterations completed in ~2103 seconds under a 1800-second runtime limit for medium datasets.
- **Operator Overhead**: Frequent calls to destroy/repair operators, exacerbated by inefficient implementations, dominate runtime.
- **ML Integration Challenges**: The learning-based repair operator requires substantial samples for training but cannot accumulate them due to the overall slowdown.

These issues render the algorithm unsuitable for real-world production environments with large datasets. This document outlines a structured analysis of the bottlenecks and proposes targeted, non-code-modifying improvements. Suggestions focus on algorithmic redesign, parallelization, and hybridization, with pseudocode for key enhancements. Implementation of these ideas can be evaluated incrementally to restore efficiency.

## 2. Problem Analysis

To address the issues systematically, we break down the bottlenecks into sub-problems, aligning with the user's preference for manageable analysis:

### 2.1 Initial Solution Generation Bottleneck
- **Root Cause**: The `initial_solution` function in `alnsopt.py` uses greedy batch processing (`_process_demand_batch`) with exhaustive searches over supply chains and vehicle selections. For large datasets (e.g., many SKUs, plants, dealers), the O(n^2) or higher complexity in `_find_best_allocation` and `_select_optimal_vehicle_type` scales poorly.
- **Impact**: Delays entry into ALNS iterations, preventing sample accumulation for ML operators.
- **Metrics**: 300s (medium) to 38min+ (large), vs. <10s target.

### 2.2 Destroy and Repair Operator Inefficiencies
- **Root Cause**: Operators like `random_removal`, `shaw_removal`, and `learning_based_repair` involve repeated computations (e.g., inventory updates, clustering in periodic Shaw removal). ML-based repair adds overhead from feature extraction and model inference/training without caching or early stopping.
- **Impact**: Each iteration takes ~88s (1043s / 12 iterations), far exceeding typical ALNS benchmarks (Pisinger & Ropke, 2007).
- **Metrics**: Frequent low-impact operators (e.g., aggressive random removal) waste cycles.

### 2.3 Overall Framework Overhead
- **Root Cause**: Sequential execution in the ALNS loop; lack of adaptive operator selection tuning; no parallelization despite independent sub-routines.
- **Impact**: Exacerbates ML sample starvation, leading to suboptimal repairs.
- **Metrics**: Total iterations <20 in runtime limits, vs. 600+ target.

### 2.4 ML-Specific Challenges
- **Root Cause**: Training triggers (e.g., every 80 iterations) are unmet; features/labels accumulate slowly in `ALNSTracker`.
- **Impact**: Fallback to heuristic repairs reduces solution quality.

## 3. Improvement Suggestions

Suggestions are categorized by sub-problem, prioritizing high-impact, low-complexity changes. Each includes rationale, estimated benefit, and pseudocode where applicable. Focus on redesign principles from ALNS literature (e.g., adaptive weights, hybridization) to ensure scalability.

### 3.1 Optimize Initial Solution Generation
**Rationale**: Shift from exhaustive search to a multi-stage heuristic: (1) Pre-compute supply-demand matches; (2) Use priority queues for allocation; (3) Parallelize batch processing. This reduces complexity to O(n log n) and leverages domain knowledge (e.g., sort by demand urgency).

**Estimated Benefit**: Reduce time to <30s (medium) / <5min (large); enables faster ML sample bootstrap.

**Pseudocode: Multi-Stage Initial Solution Generator**
```
Algorithm MultiStageInitialSolution(state, data, rng):
    // Stage 1: Pre-compute priorities (O(n log n))
    demands_sorted = sort(demands by urgency: demand_qty / available_supply)
    supply_priorities = build_priority_queue(supply_chains by capacity / distance)
    
    // Stage 2: Greedy allocation with parallelism
    parallel_for each batch in demands_sorted:
        for each demand in batch:
            best_alloc = supply_priorities.extract_max(compatible with demand)
            if best_alloc:
                veh_type = select_vehicle_by_bin_packing(demand, data.veh_types)  // Use first-fit decreasing
                load_vehicle(state, veh_type, demand)
    
    // Stage 3: Force-fill residuals (sequential fallback)
    force_load_remaining_demands(state, data)
    return state
```

**Implementation Notes**: Integrate as a drop-in replacement for `initial_solution`. Reference: Bin-packing heuristics from Martello & Toth (1990) for vehicle selection.

### 3.2 Enhance Destroy and Repair Operator Efficiency
**Rationale**: (a) Cache intermediate states (e.g., inventory snapshots) to avoid recomputation; (b) Introduce operator-specific early termination (e.g., limit removals to 10% of solution size); (c) Adaptive degree tuning based on solution density. For ML-repair, use online learning with mini-batches.

**Estimated Benefit**: Per-iteration time <10s; 50x more iterations in same runtime.

**Suggestions**:
- **Destroy Operators**: Limit `degree` dynamically: `effective_degree = min(base_degree, 0.1 * solution_size)`. Use incremental updates instead of full recomputes.
- **Repair Operators**: For `learning_based_repair`, implement warm-start: Retrain only on recent samples (last 100) if iteration > threshold.
- **General**: Profile operators via `ALNSTracker` to disable low-performers (<1% success rate) after 50 iterations.

**Pseudocode: Cached Incremental Destroy (e.g., Random Removal)**
```
Algorithm IncrementalRandomDestroy(state, degree, cache):
    removals = sample(random_indices, size = min(degree * n, max_removals=0.1*n))
    delta_cost = 0
    for idx in removals:
        item = state.solution[idx]
        delta_cost += compute_marginal_cost(item)  // Cached lookup
        remove_item(state, idx)  // O(1) swap-and-pop
        cache.update(item, delta_cost)  // Store for rollback
    return state, delta_cost
```

**Implementation Notes**: Add a `SolutionCache` dataclass in `SolutionState` for deltas. Reference: Efficiency tweaks in Coelho et al. (2012) for ALNS operators.

### 3.3 Framework-Level Optimizations
**Rationale**: (a) Parallelize independent iterations or operator evaluations; (b) Tune roulette wheel more aggressively (e.g., steeper decay); (c) Hybridize with exact methods for sub-problems (e.g., MIP for small batches).

**Estimated Benefit**: 2-5x speedup via parallelism; better convergence.

**Suggestions**:
- **Parallelization**: Use multiprocessing for operator evaluations (e.g., evaluate 3 candidate repairs in parallel).
- **Adaptive Selection**: Increase `ROULETTE_DECAY` to 0.9; segment into more phases (e.g., exploration vs. exploitation).
- **Hybridization**: Embed a lightweight MIP solver (e.g., via PuLP) for final repairs on small neighborhoods.

**Pseudocode: Parallel Operator Evaluation**
```
Algorithm ParallelRepairEvaluation(state, repair_candidates, num_threads=4):
    futures = []
    for repair in repair_candidates:
        future = thread_pool.submit(repair.apply, copy(state))  // Shallow copy for speed
        futures.append(future)
    results = [f.result() for f in futures]
    best_repair = argmin(results by objective)
    return apply(best_repair, state)
```

**Implementation Notes**: Leverage Python's `concurrent.futures`. Reference: Parallel ALNS in Vidal et al. (2012) for VRP.

### 3.4 Address ML Integration Bottlenecks
**Rationale**: Bootstrap with synthetic samples; use transfer learning from small-dataset models; trigger training on wall-clock time (e.g., every 60s) instead of iterations.

**Estimated Benefit**: Viable ML repairs after 5-10 iterations; 10-20% better solutions.

**Suggestions**:
- **Bootstrap**: Generate 50 synthetic features/labels from initial solution perturbations.
- **Online Adaptation**: Use incremental learners (e.g., switch to SGDClassifier if samples >200).
- **Fallback**: If samples < threshold, degrade to ensemble of heuristics.

**Pseudocode: Time-Triggered ML Retrain**
```
Algorithm TimeTriggeredRetrain(tracker, threshold_samples=100, interval_sec=60):
    if len(tracker.features) >= threshold_samples and (time.now() - last_train_time > interval_sec):
        X = tracker.scaler.fit_transform(tracker.features[-recent_n:])
        model.fit(X, tracker.labels[-recent_n:])
        tracker.cache_ml_model(model, scaler, current_iteration)
        last_train_time = time.now()
```

**Implementation Notes**: Modify `update_ml_data` in `ALNSTracker`. Reference: ML-enhanced ALNS in Hemmelmayr et al. (2012).

### 3.5 Stopping Criteria and Monitoring
**Rationale**: Add runtime-aware criteria; log operator timings for profiling.

**Suggestions**: Extend `CombinedStoppingCriterion` with `PerIterationTimeout` (e.g., 30s/iteration). Use `ALNSTracker` to log timings: `tracker.log_operator_time(op_name, duration)`.

**Estimated Benefit**: Prevents hangs; identifies slow components.

## 4. Evaluation Roadmap
1. **Profile First**: Run with timing logs to confirm bottlenecks (e.g., via `timeit` on operators).
2. **Incremental Testing**: Implement initial solution fix first; benchmark on medium datasets.
3. **Full Validation**: Test on 30 small/medium/large instances; target <1800s for 200+ iterations.
4. **Metrics**: Track iterations/sec, solution gap, ML sample accrual rate.

## 5. References
- Mara, S. T. W., et al. (2022). A survey of adaptive large neighborhood search algorithms and applications. *Computers & Operations Research*, 146, 105903. [DOI: 10.1016/j.cor.2022.105903](https://doi.org/10.1016/j.cor.2022.105903)
- Pisinger, D., & Ropke, S. (2007). A general heuristic for vehicle routing problems. *Computers & Operations Research*, 34(8), 2403-2435.
- Coelho, L. C., et al. (2012). Thirty years of inventory routing. *Transportation Science*, 48(1), 1-19. (For operator efficiency)
- Vidal, T., et al. (2012). A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time-windows. *Computers & Operations Research*, 40(1), 475-489. (For parallel ALNS)
- Hemmelmayr, V. C., et al. (2012). Adaptive large neighborhood search for the pickup and delivery problem with transshipment. *European Journal of Operational Research*, 217(2), 373-382. (For ML enhancements)
- Martello, S., & Toth, P. (1990). *Knapsack Problems: Algorithms and Computer Implementations*. Wiley. (For bin-packing heuristics)

This document provides a blueprint for enhancements. Prioritize Section 3.1 for immediate gains.