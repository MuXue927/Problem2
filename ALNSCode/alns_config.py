"""
Centralized ALNS configuration module.

Purpose:
- Collect all top-level algorithm configuration, per-operator default parameters and ranges,
  tuner defaults, ML settings, and helper methods to read/update configs.
- Keep this module free of runtime imports (SolutionState, Vehicle, etc.) to avoid circular imports.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional


@dataclass
class ALNSConfig:
    # --------------------------
    # Meta / run control
    # --------------------------
    SEED: int = 15926535
    LOG_LEVEL: str = "INFO"           # DEBUG / INFO / WARN / ERROR
    VERBOSE: bool = True

    # Feature toggles
    ENABLE_PARAM_TUNER: bool = False
    ENABLE_ML_OPERATORS: bool = False
    TRACKER_ENABLED: bool = False
    ENABLE_REPORTS: bool = False

    # Dataset defaults (can be overridden by user config)
    DATASET_TYPE: str = "large"
    DATASET_IDX: int = 1

    # --------------------------
    # Stopping / scheduling
    # --------------------------
    MAX_RUNTIME: int = 600                   # seconds (10 minutes) — increased for long-run profiling
    MAX_ITERATIONS: int = 1000
    MAX_ITERATIONS_NO_IMPROVEMENT: int = 100  # currently unused
    TERMINATE_ON_INFEASIBLE_INITIAL: bool = True

    # --------------------------
    # Simulated annealing / acceptance
    # --------------------------
    SA_START_TEMP: float = 1000.0
    SA_END_TEMP: float = 1.0
    SA_STEP: float = 1.0 - 1e-3

    # --------------------------
    # Selection / roulette wheel
    # --------------------------
    ROULETTE_SCORES: List[float] = field(default_factory=lambda: [5.0, 2.0, 1.0, 0.5])
    ROULETTE_DECAY: float = 0.8
    ROULETTE_SEG_LENGTH: int = 500

    # --------------------------
    # Initial solution / parallelism
    # --------------------------
    PARALLEL_DEMAND_THRESHOLD: int = 1000
    MAX_INIT_THREADS: int = 8
    INITIAL_SOLUTION_STRATEGY: str = "improved_greedy"  # placeholder key

    # --------------------------
    # Random removal variants (convenience)
    # --------------------------
    RANDOM_REMOVAL_VARIANTS: List[Dict[str, Any]] = field(default_factory=lambda: [
        {'degree': 0.15, 'name': 'random_gentle'},
        {'degree': 0.25, 'name': 'random_normal'},
        {'degree': 0.35, 'name': 'random_aggressive'},
    ])

    # --------------------------
    # ML / learning-based repair defaults
    # --------------------------
    ML_LEARNING_ENABLED: bool = False
    ML_MODEL_TYPE: str = "adaptive"   # 'linear', 'random_forest', 'adaptive'
    ML_MIN_SCORE: float = 0.4
    ML_INITIAL_SAMPLE_SIZE: int = 25
    ML_ADAPTIVE_SAMPLE_SIZE: int = 200
    ML_RETRAIN_INTERVAL: int = 80

    # --------------------------
    # Param tuner defaults (sync with ParamAutoTuner)
    # --------------------------
    TUNER_ENABLED: bool = False
    TUNER_LEARNING_RATE: float = 0.1
    TUNER_EXPLORATION_INITIAL: float = 0.3
    TUNER_PHASE_THRESHOLDS: Tuple[float, float] = (0.3, 0.7)  # exploration/exploitation/refinement splits

    # --------------------------
    # Operator defaults (params + ranges) - used by ParamAutoTuner._init_default_operators
    # Keep keys in sync with ParamAutoTuner usage.
    # --------------------------
    OPERATOR_DEFAULTS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        # Destroy operators
        "random_removal": {
            "params": {"degree": 0.25},
            "ranges": {"degree": (0.1, 0.5)}
        },
        "shaw_removal": {
            "params": {"degree": 0.3},
            "ranges": {"degree": (0.1, 0.5)}
        },
        "periodic_shaw_removal": {
            "params": {"degree": 0.3, "alpha": 0.4, "beta": 0.3, "gamma": 0.3, "k_clusters": 3},
            "ranges": {"degree": (0.1, 0.5), "alpha": (0.2, 0.6), "beta": (0.1, 0.5), "gamma": (0.1, 0.5), "k_clusters": (2, 10)}
        },
        "path_removal": {
            "params": {"degree": 0.5},
            "ranges": {"degree": (0.2, 0.7)}
        },
        "worst_removal": {
            "params": {"degree": 0.25, "value_bias": 2.0},
            "ranges": {"degree": (0.1, 0.5), "value_bias": (1.0, 3.0)}
        },
        "infeasible_removal": {
            "params": {},
            "ranges": {}
        },
        "surplus_inventory_removal": {
            "params": {"degree": 0.25},
            "ranges": {"degree": (0.1, 0.5)}
        },

        # Repair operators
        "greedy_repair": {
            "params": {"demand_weight": 0.8, "stock_weight": 0.2},
            "ranges": {"demand_weight": (0.0, 1.0), "stock_weight": (0.0, 1.0)}
        },
        "local_search_repair": {
            "params": {"max_iter": 10},
            "ranges": {}
        },
        "inventory_balance_repair": {
            "params": {},
            "ranges": {}
        },
        "infeasible_repair": {
            "params": {},
            "ranges": {}
        },
        "regret_based_repair": {
            "params": {"k": 2, "topN": 6},
            "ranges": {"k": (1, 5), "topN": (1, 20)}
        },
        "smart_batch_repair": {
            "params": {"batch_size": 10, "max_iter": 10, "timeout": None},
            "ranges": {"batch_size": (5, 50)}
        },
    })

    # --------------------------
    # Convenience - high-level mapping for destroy/repair param quick access
    # --------------------------
    DESTROY_DEFAULTS: Dict[str, Any] = field(default_factory=lambda: {
        "random_removal_degree": 0.25,
        "shaw_removal_degree": 0.3,
        "periodic_shaw_params": {"degree": 0.3, "alpha": 0.4, "beta": 0.3, "gamma": 0.3, "k_clusters": 3}
    })

    REPAIR_DEFAULTS: Dict[str, Any] = field(default_factory=lambda: {
        "greedy_repair": {"demand_weight": 0.8, "stock_weight": 0.2},
        "local_search_repair": {"max_iter": 10},
        "smart_batch_repair": {"max_iter": 10, "batch_size": 10, "timeout": None},
        "regret_based_repair": {"k": 2, "topN": 6, "time_limit": 10.0}
    })

    # --------------------------
    # periodic_shaw KMeans tuning knobs (micro-tuning / perf)
    # Exposed here so users/CI can reduce KMeans cost without code edits.
    # --------------------------
    PERIODIC_SHAW_N_INIT: int = 3
    PERIODIC_SHAW_MAX_ITER: int = 50
    PERIODIC_SHAW_TOL: float = 1e-3
    PERIODIC_SHAW_USE_FLOAT32: bool = True

    # --------------------------
    # Regret operator practical caps (safeguards for production)
    # These limit candidate branching / caching granularity.
    # --------------------------
    REGRET_MAX_TOPN: int = 12
    REGRET_MAX_K: int = 5

    # --------------------------
    # Regret simulation tuning (controls how many candidates are fully simulated
    # and the fast-estimator threshold to gate full simulations). Adjust to
    # trade off CPU vs solution-quality in production.
    # --------------------------
    # Maximum number of heuristic-selected candidates to perform full exact
    # simulation on (overrides the per-call fallback of max(6, topN*2)).
    # Keep conservative default but allow tightening via config.
    REGRET_SIM_MAX_SIMULATE: int = 4

    # Require the fast estimator improvement to exceed this threshold before
    # performing an expensive full simulation. Value is in the "fast-impr"
    # estimator scale (typically small); increase to be more conservative.
    # Lowering this value makes the gate stricter (fewer full sims). Set to 1e-3.
    REGRET_FULL_SIM_THRESHOLD: float = 0.001

    # --------------------------
    # Logging / persistence
    # --------------------------
    LOG_DIR: Optional[str] = "logs-alns"
    SAVE_BEST_EVERY_N_ITERS: int = 50

    # --------------------------
    # Helper instance methods
    # --------------------------
    def get_destroy_params(self) -> Dict[str, Any]:
        """Return a compact dict of commonly used destroy parameters."""
        return {
            'random_removal_degree': self.DESTROY_DEFAULTS.get('random_removal_degree', 0.25),
            'shaw_removal_degree': self.DESTROY_DEFAULTS.get('shaw_removal_degree', 0.3),
            'periodic_shaw_params': self.DESTROY_DEFAULTS.get('periodic_shaw_params', {})
        }

    def get_repair_params(self) -> Dict[str, Any]:
        """Return a compact dict of commonly used repair defaults."""
        return self.REPAIR_DEFAULTS.copy()

    def get_operator_default(self, op_name: str) -> Dict[str, Any]:
        """Return the operator default structure {params, ranges} for a given operator."""
        return self.OPERATOR_DEFAULTS.get(op_name, {"params": {}, "ranges": {}})

    def get_tuner_operator_defaults(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping suitable for ParamAutoTuner._init_default_operators:
        {op_name: {"params":..., "ranges":...}, ...}
        """
        return self.OPERATOR_DEFAULTS.copy()

    def update_from_dict(self, cfg: Dict[str, Any]):
        """
        Update configuration fields from a dict (in-place).
        Accepts nested structures for OPERATOR_DEFAULTS / DESTROY_DEFAULTS etc.
        """
        for k, v in cfg.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                # Allow updating nested dict entries
                if k in ("OPERATOR_DEFAULTS", "DESTROY_DEFAULTS", "REPAIR_DEFAULTS"):
                    existing = getattr(self, k, None)
                    if isinstance(existing, dict) and isinstance(v, dict):
                        existing.update(v)
                        setattr(self, k, existing)
                    else:
                        setattr(self, k, v)
                else:
                    # Unknown keys stored in a generic attribute map
                    setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the main configuration to a plain dict (shallow for convenience)."""
        out = {}
        for name, val in self.__dict__.items():
            out[name] = val
        return out


# Module-level default config instance for easy import/use
default_config = ALNSConfig()
