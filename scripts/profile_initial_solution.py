"""Profile initial_solution pipeline (cProfile + optional pyinstrument)

Usage:
  .\.venv\Scripts\python.exe scripts\profile_initial_solution.py

Outputs:
  - scripts/prof_initial_solution.out : raw cProfile output (pstats readable)
  - scripts/prof_initial_solution.txt : human-readable top-50 functions by cumulative time
  - scripts/pyinstrument_profile.html : optional, if pyinstrument is installed

This script profiles the flow: load data -> build state -> call initial_solution(state, rng)
"""
import sys
from pathlib import Path
import time
import cProfile
import pstats
import io

# ensure project root on path
this_file = Path(__file__).resolve()
project_root = this_file.parent.parent
sys.path.append(str(project_root))

try:
    from ALNSCode.alnsopt import initial_solution, SolutionState
    from ALNSCode.InputDataALNS import DataALNS
    from ALNSCode.alns_config import ALNSConfig
except Exception as e:
    print('Failed to import ALNSCode modules:', e)
    raise

import numpy.random as rnd

OUT_DIR = Path(__file__).parent
PROF_OUT = OUT_DIR / 'prof_initial_solution.out'
PROF_TXT = OUT_DIR / 'prof_initial_solution.txt'
PYINSTR_HTML = OUT_DIR / 'pyinstrument_profile.html'


def do_run(dataset_idx=None):
    DATASET_TYPE = ALNSConfig.DATASET_TYPE
    if dataset_idx is None:
        dataset_idx = ALNSConfig.DATASET_IDX

    dataset_name = f'dataset_{dataset_idx}'
    current_dir = Path(__file__).parent.parent
    input_file_loc = current_dir / 'datasets' / 'multiple-periods' / DATASET_TYPE
    output_file_loc = current_dir / 'OutPut-ALNS' / 'multiple-periods' / DATASET_TYPE

    data = DataALNS(input_file_loc, output_file_loc, dataset_name)
    data.load()

    state = SolutionState(data)
    rng = rnd.default_rng(42)

    sol = initial_solution(state, rng)
    return sol


def main():
    print('Profiling initial_solution pipeline...')
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    sol = do_run()
    pr.disable()
    elapsed = time.time() - t0

    # dump raw profile
    pr.dump_stats(str(PROF_OUT))

    # create human readable summary
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(50)

    txt = s.getvalue()
    with open(PROF_TXT, 'w', encoding='utf-8') as f:
        f.write(f'Elapsed: {elapsed:.4f} seconds\n\n')
        f.write(txt)

    print(f'Profile complete. Elapsed: {elapsed:.2f}s')
    print(f'Raw profile: {PROF_OUT}')
    print(f'Summary: {PROF_TXT}')

    # attempt pyinstrument if available
    try:
        from pyinstrument import Profiler
        print('Running pyinstrument profiler (pyinstrument found)...')
        profiler = Profiler()
        profiler.start()
        _ = do_run()
        profiler.stop()
        with open(PYINSTR_HTML, 'w', encoding='utf-8') as fh:
            fh.write(profiler.output_html())
        print(f'pyinstrument HTML saved to: {PYINSTR_HTML}')
    except Exception as e:
        print('pyinstrument not available or failed, skipping pyinstrument:', e)

    # print top lines to stdout
    print('\nTop profile lines (first 40 lines):\n')
    print(txt.splitlines()[:40])


if __name__ == '__main__':
    main()
