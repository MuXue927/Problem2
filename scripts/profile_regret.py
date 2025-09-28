# lang: python
"""
Lightweight profiler for regret_based_repair.
Runs multiple short regret_based_repair calls on a real state to collect cProfile data.
Outputs:
 - scripts/prof_regret.out (raw)
 - scripts/prof_regret.txt (human-readable top 60 cumulative)
"""
import sys
from pathlib import Path
import time
import cProfile
import pstats
import io

this_file = Path(__file__).resolve()
project_root = this_file.parent.parent
sys.path.append(str(project_root))

from ALNSCode.InputDataALNS import DataALNS
from ALNSCode.alnsopt import SolutionState
from ALNSCode.repair_operators import regret_based_repair
from ALNSCode.alns_config import ALNSConfig

import numpy.random as rnd

OUT_DIR = Path(__file__).parent
PROF_OUT = OUT_DIR / "prof_regret.out"
PROF_TXT = OUT_DIR / "prof_regret.txt"

def build_state():
    DATASET_TYPE = ALNSConfig.DATASET_TYPE
    DATASET_IDX = getattr(ALNSConfig, "DATASET_IDX", 0)
    dataset_name = f"dataset_{DATASET_IDX}"
    current_dir = Path(__file__).parent.parent
    input_file_loc = current_dir / "datasets" / "multiple-periods" / DATASET_TYPE
    output_file_loc = current_dir / "OutPut-ALNS" / "multiple-periods" / DATASET_TYPE

    data = DataALNS(input_file_loc, output_file_loc, dataset_name)
    data.load()
    state = SolutionState(data)
    # create a small seed solution using initial_solution entrypoint if available
    try:
        from ALNSCode.alnsopt import initial_solution
        rng = rnd.default_rng(42)
        initial = initial_solution(state, rng)
        if initial is not None:
            state = initial
    except Exception:
        pass
    return state

def main():
    print("Profiling regret_based_repair (short runs)...")
    state = build_state()
    rng = rnd.default_rng(1234)

    # we will run multiple short regret calls; cap total wall time ~18s
    TARGET_SECONDS = 18.0
    start = time.time()
    pr = cProfile.Profile()
    runs = 0
    prof_active = False
    try:
        try:
            # try the simple enable/disable approach first
            pr.enable()
            prof_active = True
            while time.time() - start < TARGET_SECONDS:
                # use small internal time_limit to keep each call short but realistic
                try:
                    _ = regret_based_repair(state, rng, k=2, topN=6, time_limit=1.0)
                except Exception:
                    # ignore errors during profiling; continue
                    pass
                runs += 1
        except ValueError:
            # Another profiler is active (e.g. pyinstrument) and prevents cProfile from enabling.
            # Fall back to running the workload without cProfile in this process and record timing.
            print("Warning: another profiler is already active; running without cProfile in this process.")
            rr = 0
            t0 = time.time()
            while time.time() - t0 < TARGET_SECONDS:
                try:
                    _ = regret_based_repair(state, rng, k=2, topN=6, time_limit=1.0)
                except Exception:
                    pass
                rr += 1
            runs = rr
            elapsed = time.time() - start
    finally:
        # disable may raise if enable never succeeded; guard it
        if prof_active:
            try:
                pr.disable()
            except Exception:
                pass

    elapsed = time.time() - start
    pr.dump_stats(str(PROF_OUT))

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(60)
    txt = s.getvalue()
    with open(PROF_TXT, 'w', encoding='utf-8') as fh:
        fh.write(f"Elapsed: {elapsed:.4f} seconds\nRuns: {runs}\n\n")
        fh.write(txt)
    print(f"Profile complete. Elapsed: {elapsed:.2f}s Runs: {runs}")
    print(f"Raw profile: {PROF_OUT}")
    print(f"Summary: {PROF_TXT}")
    # print top lines
    print("\nTop profile lines (first 40 lines):\n")
    print(txt.splitlines()[:40])

if __name__ == '__main__':
    main()
