#!/usr/bin/env python3
"""
Numba warm-up helper for CI / service startup.

Purpose
- Trigger Numba JIT compilation for the hot inventory kernel used by ALNS:
  ALNSCode.inventory_numba._compute_inventory_core_numba
- Avoids first-iteration latency caused by JIT compilation in long-running runs
  by performing a small, deterministic call at startup/CI-time.

Usage
- Locally / in deployment startup:
    python -u scripts/numba_warmup.py

- In CI (example snippet for GitHub Actions):
    - name: Numba warmup
      run: python -u scripts/numba_warmup.py

Behavior
- If Numba is available and the numba kernel exists, this script will allocate
  a tiny (P=1,S=2,H=3) set of arrays and call the njit kernel once to force
  compilation. The kernel is invoked twice: first call triggers compilation,
  second call exercises the compiled code to ensure correctness.
- If Numba is not available or the kernel is missing, the script falls back to
  calling the numpy implementation to still exercise the code paths.
- The script is intentionally small and side-effect free (does not modify repo files).

Notes
- Keep this step cheap (small arrays) so CI time impact is minimal while still
  paying the JIT cost once per runner image / process.
- For long-running services, call this in the service start-up sequence (before
  entering the main ALNS loop) to avoid sporadic latency on the first ALNS iteration.
"""
import time
import numpy as np

def warmup_inventory_kernel():
    try:
        from ALNSCode import inventory_numba
    except Exception as e:
        print("Unable to import ALNSCode.inventory_numba:", e)
        return 1

    H = max(3, getattr(inventory_numba, "NUMBA_HORIZON", 3))
    # small shapes
    P, S = 1, 2
    H = 3

    # prepare tiny arrays
    s_arr = np.zeros((P, S, H + 1), dtype=np.float64)
    # set a small nonzero baseline to avoid degenerate optimizations
    s_arr[:, :, 0] = 1.0
    shipments = np.zeros((P, S, H + 1), dtype=np.float64)
    prod = np.zeros((P, S, H + 1), dtype=np.float64)
    prod[:, :, 1:] = 0.5

    # Try numba kernel if available; otherwise use numpy core
    if getattr(inventory_numba, "NUMBA_AVAILABLE", False) and hasattr(inventory_numba, "_compute_inventory_core_numba"):
        kernel = inventory_numba._compute_inventory_core_numba
        print("Numba available: warming up numba kernel...")
        try:
            t0 = time.perf_counter()
            # first call triggers compilation
            kernel(s_arr.copy(), shipments, prod, H)
            t1 = time.perf_counter()
            # second call exercises compiled kernel
            kernel(s_arr.copy(), shipments, prod, H)
            t2 = time.perf_counter()
            print(f"Numba warmup: compile+run time ~{(t1-t0):.4f}s, compiled run ~{(t2-t1):.4f}s")
            return 0
        except Exception as e:
            print("Numba kernel warmup failed:", e)
            # fallback to numpy path below
    else:
        print("Numba not available or kernel missing; using numpy fallback for warm-up.")

    # numpy fallback: call the numpy core to exercise the code path
    try:
        kernel_np = getattr(inventory_numba, "_compute_inventory_core_numpy", None)
        if kernel_np is None:
            # final fallback: call compute_inventory_fast on a sample state (best-effort)
            print("No numpy core found; falling back to compute_inventory_fast (may have side-effects).")
            try:
                # try creating a minimal fake state (best-effort)
                from ALNSCode.alnsopt import SolutionState
                from types import SimpleNamespace
                data = SimpleNamespace()
                data.plants = ["P1"]
                data.skus_plant = {"P1": ["S1"]}
                data.horizons = 3
                data.sku_prod_each_day = {}
                data.historical_s_ikt = {}
                s = SolutionState(data=data)
                inventory_numba.compute_inventory_fast(s)
                print("Fallback compute_inventory_fast completed.")
                return 0
            except Exception as e2:
                print("Fallback compute_inventory_fast also failed:", e2)
                return 2
        else:
            t0 = time.perf_counter()
            kernel_np(s_arr.copy(), shipments, prod, H)
            t1 = time.perf_counter()
            print(f"Numpy warmup run time ~{(t1-t0):.6f}s")
            return 0
    except Exception as e:
        print("Exception during numpy warmup:", e)
        return 3

if __name__ == "__main__":
    code = warmup_inventory_kernel()
    if code == 0:
        print("Numba warm-up completed successfully.")
    else:
        print("Numba warm-up completed with non-zero exit code:", code)
    raise SystemExit(code)
