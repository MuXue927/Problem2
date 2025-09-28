# techContext.md

Purpose
Record technologies, dependencies, tooling conventions, configuration levers, runtime behavior, and environment constraints.

Runtime & Language
- Python >= 3.10 (pattern matching availability, typing improvements)
- Deterministic RNG: numpy.random.default_rng seeded via ALNSConfig.SEED
- Optional commercial dependency: gurobipy (needed for CG / monolithic variants or advanced validation)

Core Dependencies (pyproject.toml)
- alns: Metaheuristic framework providing iteration engine, acceptance, operator selection infrastructure
- numpy / pandas: Numerical & tabular core
- matplotlib / plotly: Static & interactive visualization
- scikit-learn / mabwiser: ML modeling and multi-armed bandit style experimentation (basis for operator selection enhancements)
- networkx: Supply chain / graph-based visualization or structural analysis
- openpyxl: Excel ingestion if needed for xlsx inputs
- vrplib: Potential benchmark dataset integration
- rich: Styled logging output (colorization)
- gurobipy: External optimization engine (Column Generation / baseline solve paths)

Dev / Tooling Dependencies
- pytest / pytest-cov: Testing + coverage
- black: Formatting
- mypy: Static typing (incremental adoption potential)
- pyinstrument: Profiling

Packaging & Installation
- setuptools build backend
- find packages include pattern: ALNSCode*
- Editable install: pip install -e .
- Secondary local package: ALNS/ (embedded copy for controlled experiments); may also pip install -e ALNS

Configuration Surface (ALNSConfig exemplar)
- Random seed, dataset selection (DATASET_TYPE, DATASET_IDX)
- Operator degree parameters (random_removal_degree, shaw_removal_degree, etc.)
- Periodic / learning-based operator parameter dicts
- Roulette wheel scores / decay / segment length
- Simulated annealing temperatures & step
- Stopping thresholds: MAX_ITERATIONS, MAX_RUNTIME
- Feasibility handling flag: TERMINATE_ON_INFEASIBLE_INITIAL

Execution Modes
- Standard heuristic solve: python -m ALNSCode.main
- Dataset path resolution relative to package root (datasets/multiple-periods/<DATASET_TYPE>)
- Output directory root: OutPut-ALNS/multiple-periods/<DATASET_TYPE>/<dataset_name>/

Key Generated Artifacts
- opt_result.csv / incremental shipments appended
- opt_details.csv (joined with sizes & vehicle info)
- opt_summary.csv (vehicle-day aggregated loads)
- non_fulfill.csv, extra_volume.csv (exception diagnostics)
- sku_inv_left.csv (inventory timeline)
- images/* (gap / objective / operator stats)
- param_tuning_report/* (parameter/ML performance)
- alns_optimization.log (file logging)
- Visualization outputs include SVG (high resolution for publication)

Logging & Monitoring
- logging.basicConfig with FileHandler only (no console duplication)
- Custom LogPrinter abstracts formatting, ANSI color (rich dependency)
- Periodic iteration progress printed (every N iterations) referencing stop status

Adaptive / ML Components
- ParamAutoTuner:
  - Tracks improvement deltas
  - Maintains success counts & rolling performance windows
  - Adjusts degrees / parameters inside defined ranges
- MLOperatorSelector (conditional import):
  - Feature extraction from SolutionState
  - Model retrain interval (e.g., every 100 iterations)
  - Minimum sample gating (avoid premature predictions)
  - Caches model + scaler; supports warm restarts

Data Expectations
- CSV naming consistency (product_size.csv, vehicle.csv, etc.)
- Columns cast to numeric with explicit dtype enforcement
- Derived structures: supply chain mappings, SKU index dictionaries, demand aggregates

Performance Considerations
- Iteration loop cost dominated by operator application + validation
- ML retraining frequency tuned to avoid overhead spikes
- Potential future optimization: batch feature extraction caching, profiling-guided hotspots (see pyinstrument usage)

Testing Strategy (Current Gaps)
- Tests focus on initial solution & core feasibility
- Missing: systematic operator regression benchmarks, ML selector correctness tests, stress tests for large horizons
- Headless plotting advisory: use Agg backend (future explicit config in conftest)

Extensibility Hooks
- Add operator: implement function or class, register in optimizer
- Add penalty dimension: extend SolutionState and validation
- Add adaptive signal: augment ParamAutoTuner metrics
- Replace acceptance: swap SimulatedAnnealing in optimizer setup
- Add stop criterion: extend combined_stopping (time / iteration / stagnation / target objective)

Environment Constraints
- Windows (documented scripts PowerShell oriented)
- ExecutionPolicy Bypass invoked in helper scripts
- Headless CI (future) must enforce non-interactive matplotlib backend

Risks / Constraints
- gurobipy licensing may limit reproducibility outside licensed environments
- Embedded ALNS local package divergence risk vs upstream PyPI version
- ML performance sensitive to feature drift; absence of feature versioning

Future Technical Enhancements
- Central consolidated run manifest (JSON) capturing config + hash of operator set
- Operator performance DB (parquet) for multi-run analytical dashboards
- ML feature store abstraction with schema versioning
- Unified CLI entrypoint (problem2-cli) for scripted batch experiments

Cross-References
- projectbrief.md (objective & scope)
- productContext.md (stakeholder goals)
- systemPatterns.md (architectural layering)
- activeContext.md (current focus & near-term adjustments)
- progress.md (status evolution)

Status
Initial capture—update alongside dependency changes, new tooling, or architectural refactors.
