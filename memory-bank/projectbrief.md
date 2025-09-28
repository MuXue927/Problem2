# projectbrief.md

Title: Multi-Period Product Distribution Optimization (Problem2)

Objective
Provide a unified experimental and research platform for optimizing multi-period distribution from plants to dealers under production, inventory, vehicle capacity, minimum load, and demand satisfaction constraints using three methodological tracks:
1. ALNS (primary metaheuristic) with adaptive & ML-enhanced operator control
2. Column Generation (CG) prototype (model correctness under review)
3. Monolithic mathematical baseline (reference / validation)

Core Problem
Given:
- Multiple days (planning horizon)
- Plants with production and initial inventory
- Dealers with time-phased demand per SKU
- Vehicle types with capacity and minimum load usage cost implications
Decide vehicle loading and shipment allocation per day to minimize total penalty-adjusted objective (unmet demand, capacity misuse, infeasibility penalties) while respecting flow and inventory logic.

Scope (Inclusions)
- Data ingestion & normalization (CSV-based datasets)
- Initial constructive solution generation
- Adaptive large neighborhood search (destroy/repair operator ecosystem)
- Parameter auto-tuning & optional ML-based operator selection
- Iterative improvement tracking & acceptance (Simulated Annealing)
- Multi-criterion stopping (iterations / runtime)
- Structured output artifacts (CSV summaries, reports, plots)
- Validation utilities for major feasibility dimensions

Out of Scope (Current)
- Exact CG final validated model (flagged “may be doubtful”)
- Full stochastic demand modeling
- Continuous-time or multi-objective Pareto exploration
- Deployment / API service mode

Success Criteria
- Feasible solutions (no negative inventory, capacity violations, or unmet demand if solvable)
- Competitive objective value vs baselines
- Controlled gap convergence profile
- Diversity and effective usage of registered operators
- Reproducibility via fixed SEED & deterministic dataset references

Primary Entities
- DataALNS: Structured dataset loader + derived metrics
- SolutionState: Encapsulates decision representation & penalties
- Operators: Destroy (random, shaw, worst, infeasible-focused, surplus, path, periodic variants, ML) and Repair (greedy, inventory_balance, urgency, infeasible, local search, learning-based, ML)
- ParamAutoTuner / MLOperatorSelector: Adaptive control layer
- ALNSTracker: Iteration metrics + ML feature capture
- Visualization & reporting modules

Key Outputs
- opt_result.csv / opt_details.csv / opt_summary.csv
- non_fulfill.csv / extra_volume.csv (exception diagnostics)
- sku_inv_left.csv (inventory evolution)
- images/* (objective / gap / operator performance)
- param_tuning_report/* (adaptive parameter evolution)
- Logs: alns_optimization.log + console-styled prints

Operational Flow (High Level)
1. Load dataset & normalize structures
2. Build initial solution (heuristic + feasibility assessment)
3. Register operators (parameterized + ML-augmented if available)
4. Configure selection (SegmentedRouletteWheel), acceptance (SimulatedAnnealing), stopping (combined)
5. Iterate ALNS: destroy → repair → evaluate → accept/reject → track
6. Periodic ML model fitting (if enabled)
7. Final best solution extraction, validation, and artifact generation

Risks / Open Items
- CG mathematical formulation correctness unresolved
- Potential infeasible initial solution (config-controlled termination)
- ML operator performance dependent on feature richness and sample thresholds
- Large instance scaling (runtime vs iteration-based stopping)

Non-Functional Considerations
- Determinism: RNG seeded (ALNSConfig.SEED)
- Extensibility: Operator registration pattern + partial wrapper
- Maintainability: Separation of data, state, operators, adaptation, reporting
- Portability: Python >= 3.10, optional gurobipy for advanced components

Next Evolution (Planned Later)
- Formal validation & refactor of CG pipeline
- Benchmark harness across instance families
- CI workflow (editable install + headless plotting)
- Enhanced operator performance attribution (per-feature SHAP-like metrics)
- Structured ML dataset export for offline experimentation

Cross-References
- productContext.md: User / problem narrative
- systemPatterns.md: Architectural & pattern decisions
- techContext.md: Tooling & dependency baseline
- activeContext.md: Current working focus
- progress.md: Status log & outstanding tasks

Status
Initialized as part of first memory-bank population.
