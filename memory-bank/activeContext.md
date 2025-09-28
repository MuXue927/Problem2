# activeContext.md

Purpose
Live snapshot of current focus, decisions, near-term actions, and evolving considerations. Updated whenever significant work occurs.

Current Date (UTC+8)
2025-09-26

Current Focus
- Initialize structured Memory Bank (core six documents) to persist project knowledge between stateless sessions.
- Ensure architectural, product, technical, and status dimensions are captured for future incremental updates.

Recent Actions (This Session)
- Created projectbrief.md (problem framing, scope, success metrics).
- Created productContext.md (stakeholders, workflows, value).
- Created systemPatterns.md (layering, design patterns, extensibility).
- Created techContext.md (dependencies, tooling, configuration surface).
- Drafting activeContext.md (this file).
- Pending: progress.md initialization.

Key Architectural Anchors
- ALNS metaheuristic pipeline with adaptive + optional ML operator selection.
- Penalty-mediated feasibility shaping inside SolutionState (soft constraint relaxation early).
- Extensible operator registration (partial binding preserves naming).
- Tracking (ALNSTracker) + reporting (plots, CSV artifacts, parameter reports).
- Config centralization via ALNSConfig controlling reproducibility & search dynamics.

Notable Risks / Flags
- Column Generation model correctness unresolved (marked “may be doubtful”).
- Initial solution feasibility not guaranteed; termination governed by TERMINATE_ON_INFEASIBLE_INITIAL.
- ML operator effectiveness dependent on sufficient iteration samples (warm-up needed).
- Potential scaling pressure for large horizons / dataset families (runtime vs iteration cap).

Immediate Next Steps
1. Create progress.md with current implementation status & open items.
2. (Future) Add operator_catalog.md cataloging destroy/repair semantics & parameter meanings.
3. (Future) Capture ML feature schema (state feature vector definition) in a dedicated doc.
4. (Future) Add testingStrategy.md (expand planned test coverage: ML correctness, operator regression).

Planned Enhancements (Short Horizon)
- Introduce CI workflow (editable install, headless matplotlib backend Agg).
- Replace return-based tests with assert patterns (cleanup technical debt).
- Export run manifest (JSON) summarizing config, seed, operator set, dataset id.

Deferred / Backlog
- Formal CG validation & potential refactor.
- Performance profiling focusing on high-cost operators (pyinstrument-guided).
- Feature engineering expansion for ML selector (inventory slack, time-window saturation metrics).
- Structured persistence of ParamAutoTuner evolution for multi-run learning.

Open Questions
- Do we need a reproducible benchmark suite baseline (instance set + reference metrics)?
- Threshold tuning: Optimal frequency for ML retraining vs marginal improvement (currently periodic).
- Whether to introduce no-improvement stopping condition soon.

Conventions Established
- All memory-bank markdown files cross-reference each other.
- Active context updated after each substantial structural or capability change.
- Use deterministic seed path (ALNSConfig.SEED) for reproducible experiments.

Monitoring Signals (Future Tracking)
- Gap decay curve smoothness
- Operator usage entropy (avoid domination)
- Average improvement per operator family over sliding window
- ML model retrain latency vs search iteration time

Next Update Trigger
- After adding progress.md OR after first CI/test modernization task execution.

Status
Active; awaiting progress.md creation to complete initial memory bank population.
