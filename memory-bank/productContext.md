# productContext.md

Purpose
Describe the real-world problem framing, stakeholders, value proposition, usage scenarios, and experiential goals for Problem2.

Problem Narrative
A multi-period (discrete days) distribution planning problem:
- Plants produce SKUs over time with per-day production and initial inventories.
- Dealers (clients) issue time-phased deterministic demands by SKU.
- Vehicles of various types (capacity + minimum effective loading thresholds) move product plant→dealer.
- Objective: Minimize composite cost/penalty while satisfying (where possible) demand, honoring capacity, and preventing negative inventory.

Why This Exists
- Enable rapid experimentation with adaptive metaheuristics (ALNS) on structured supply allocation + loading.
- Provide a comparative baseline (Monolithic model) and exploratory Column Generation prototype.
- Offer a research playground for operator adaptation (parameter auto-tuning + ML-guided selection).

Primary Users
- Researcher exploring adaptive LNS performance and operator design.
- Engineer integrating new destroy/repair heuristics or ML feature sets.
- Analyst validating feasibility / constraint behaviors on new datasets.

Key User Goals
- Load a dataset quickly and obtain a feasible solution with diagnostics.
- Plug in new operators with minimal registration friction.
- Observe improvement trajectory (objective, gap, operator usage).
- Investigate infeasibility causes (unmet demand, inventory deficits, capacity breaches).
- Export structured artifacts for post-hoc analysis.

Value Proposition
- Unified pipeline from raw CSV inputs → optimization → rich reporting.
- Extensible operator ecosystem with adaptive and ML layers.
- Deterministic reproducibility for benchmarking.
- Clear separation of concerns (data, state, operators, adaptation, reporting).

Workflows (Happy Paths)
1) Standard Solve
   - pip install -e .
   - python -m ALNSCode.main
   - Inspect outputs (csv + images + logs)
2) Operator Development
   - Implement new destroy/repair function
   - Register via partial or class-based operator
   - Rerun solve, compare operator statistics
3) ML/Adaptive Tuning
   - Ensure param_tuner & ML modules import
   - Validate parameter evolution + model performance
4) Feasibility Diagnosis
   - Check validation summaries & violation prints
   - Inspect non_fulfill.csv / extra_volume.csv / inventory traces

Pain Points Addressed
- Manual parameter tuning → replaced with ParamAutoTuner adaptive updates
- Opaque operator contribution → usage & performance plots / counts
- Difficult reproducibility → centralized config (ALNSConfig + seeded RNG)
- Fragmented reporting → consolidated CSV + visualization suite

Data Characteristics (Conceptual)
- SKU-level production, initial inventory maps, vehicle fleet definitions, demand matrices
- Derived metrics: aggregated demands, supply chain mappings, usage summaries

Experience & Quality Goals
- Fast initial feedback (< seconds for data load + initial solution for small instances)
- Transparent logging (file + styled console)
- Fails fast with clear messages on misconfigured runs
- Modular extensibility without editing core loops

Constraints & Assumptions
- Demands deterministic (currently no stochastic sampling)
- Single-echelon distribution (plant→dealer) per planning horizon (no intermediate hubs)
- Penalty-based handling for infeasible transitional states (adaptive penalties inside SolutionState)
- Dataset directory layout stable (expected CSV names)

Future Extensions (Not Yet Implemented)
- Stochastic or scenario-based demand
- Multi-objective trade-off exploration
- Multi-echelon routing / consolidation
- Advanced ML feature engineering or offline model training pipeline

User Risks / Caveats
- CG model flagged “may be doubtful” (do not rely for correctness proof)
- ML performance depends on sample accumulation (warm-up iterations)
- Heavy instances can extend runtime despite iteration caps (monitor gap + logs)

Cross-References
- projectbrief.md (scope & success criteria)
- systemPatterns.md (architecture & design patterns)
- techContext.md (runtime & dependencies)
- activeContext.md (current working focus)
- progress.md (status tracking)

Status
Initialized with first memory-bank population; evolve as user stories or datasets broaden.
