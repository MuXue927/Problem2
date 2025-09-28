# systemPatterns.md

Purpose
Capture architectural structure, design patterns, control flows, extensibility points, and constraint handling strategies for Problem2.

High-Level Layering
1. Data Layer
   - DataALNS loads raw CSV inputs (production, inventory, vehicle, demand, size) → reorganizes columns, enforces dtypes, derives mappings (supply chain, SKU aggregates).
2. State / Domain Layer
   - SolutionState stores vehicles, inventory trajectories, unmet demand, penalties; provides objective(), validate(), calculate_cost().
3. Operator Layer
   - Destroy & repair operators (pure functions or lightweight classes) mutate/copy state segments to explore neighborhoods.
4. Adaptation Layer
   - ParamAutoTuner dynamically adjusts operator parameters based on performance signals.
   - (Optional) MLOperatorSelector / ML-based destroy & repair augment selection intelligence.
5. Search Orchestration
   - ALNS (external library) coordinates iteration loop: select operators → apply → accept (SimulatedAnnealing) → record stats.
6. Tracking & Reporting
   - ALNSTracker collects iteration metrics, operator usage, model retraining triggers.
   - Visualization & CSV exporters generate analytical artifacts post-run.
7. Presentation & IO
   - LogPrinter for structured console/file output; CSV outputs for downstream analysis.

Key Design Patterns
- Strategy
  - Operator registration (destroy/repair) as interchangeable strategies.
- Partial Application Wrapper
  - _create_named_partial() binds parameter sets while preserving __name__ for ALNS library compatibility.
- Observer / Callback
  - alns.on_best(), alns.on_accept() used to inject tracking, feasibility checks, ML retraining intervals.
- Adaptive Feedback Loop
  - ParamAutoTuner updates operator parameter ranges / degree modulation using performance deltas.
- Facade
  - ALNSOptimizer wraps end-to-end orchestration (data → search → reporting) behind run_optimization().
- Template Workflow
  - Fixed step sequencing (load → init → register → configure → iterate → process → report).
- Data Pipeline
  - DataALNS: structured validation + derived indices (SKU mappings) → consumed by operators and penalty logic.

Constraint Handling Strategy
- Soft constraints penalized in objective via SolutionState (e.g., unmet demand, negative inventory, min-load violations).
- Hard-ish constraints validated post-operation (validate()) with potential termination if initial solution infeasible (configurable).
- Capacity compliance re-verified in summary stage (_validate_capacity_constraints()) for diagnostics.

Objective Composition (Conceptual)
objective = base_cost + Σ(penalties)
Where penalties cover:
- Unmet demand
- Negative inventory occurrences
- Vehicle load ratio violations (below minimum / above capacity)
(Exact numeric weighting encapsulated in SolutionState methods.)

Operator Ecosystem (Current)
Destroy:
- random_removal, shaw_removal, periodic_shaw_removal (class-based), worst_removal, infeasible_removal, surplus_inventory_removal, path_removal, ml_based_destroy (conditional).
Repair:
- greedy_repair, inventory_balance_repair, urgency_repair, infeasible_repair, local_search_repair, LearningBasedRepairOperator, ml_based_repair (conditional).

Adaptation & ML Hooks
- ParamAutoTuner tracks per-operator performance: success counts, average improvement windows, dynamic degree scaling.
- MLOperatorSelector accumulates feature vectors (state descriptors + operator encodings) → trains periodic models (trigger every N iterations) → informs operator ranking / selection (when enabled).
- Minimum sample thresholds (e.g., min_sample_size) gate ML-driven decisions to avoid premature overfitting.

State Integrity & Copy Semantics
- Operators should treat SolutionState immutably where possible (copy() when exploring) to prevent side-effects contaminating incumbent/best references.

Stopping Criteria (Combined)
- Iteration cap (ALNSConfig.MAX_ITERATIONS)
- Runtime cap (ALNSConfig.MAX_RUNTIME seconds)
- (Placeholder for future no-improvement window)
create_standard_combined_criterion() aggregates these; tracker reads status for periodic logging.

Randomness & Reproducibility
- Central RNG seed (ALNSConfig.SEED) passed to ALNS + local generation (rnd.default_rng) ensuring repeatable experiment runs.

Reporting & Analytics
- Iteration metrics: objective trajectory, gap progression.
- Operator usage counts & performance distributions.
- Parameter evolution (param_tuning_report/*).
- Final feasibility & capacity diagnostics (extra_volume.csv, non_fulfill.csv).

Extensibility Guidelines
To add a new operator:
1. Implement function(state: SolutionState, rng, **params) → mutated/new state
2. (Optional) Introduce parameter ranges in ParamAutoTuner
3. Register via ALNSOptimizer._register_destroy_operators/_register_repair_operators
4. Add visualization support if specialized metrics needed

To add new penalty logic:
1. Extend SolutionState calculation methods (e.g., punish_* pattern)
2. Update validate() for explicit detection
3. Reflect new metrics in tracking/reports if required

Potential Future Patterns
- Plugin discovery (entry points) for operator auto-registration
- Feature store abstraction for ML model training
- Constraint relaxation scheduling (progressive hardening strategy)

Risks / Trade-offs
- Penalty-based feasibility may delay full constraint satisfaction early (exploration advantage vs temporary infeasibility noise).
- ML layer adds overhead; must balance retraining interval with marginal gains.
- Partial application naming relies on manual __name__ assignment—must maintain uniqueness.

Technical Debt / Watchlist
- CG model correctness unresolved (avoid premature integration in adaptation loop).
- Lack of unified configuration object snapshotting (future: structured run manifest).
- Sparse ML feature engineering (opportunity: richer temporal features, inventory slack metrics).

Cross-References
- projectbrief.md (scope & success metrics)
- productContext.md (stakeholder + narrative)
- techContext.md (dependencies + runtime)
- activeContext.md (current focus)
- progress.md (implementation status)

Status
Initial architectural capture; refine as new patterns or refactors emerge.
