from ALNSCode.alns_config import default_config as cfg
# Short-run overrides for profiling
cfg.MAX_RUNTIME = 30
cfg.MAX_ITERATIONS = 50
# Optionally use a smaller dataset if available (leave as default if not)
# cfg.DATASET_TYPE = "small"

from ALNSCode.main import run_model

if __name__ == "__main__":
    # Run a single short profiling-friendly execution
    run_model()
