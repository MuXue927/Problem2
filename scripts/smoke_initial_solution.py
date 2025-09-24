"""Smoke runner: run initial_solution once for a single dataset without needing pytest.

Usage (PowerShell):
  # from project root
  .\.venv\Scripts\python.exe scripts\smoke_initial_solution.py

This script doesn't import the test file (which requires pytest). It directly uses
ALNSCode modules to build data, generate an initial solution, validate it and print a short summary.
"""
import sys
import os
import time
from pathlib import Path

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


def run(dataset_idx=None):
    DATASET_TYPE = ALNSConfig.DATASET_TYPE
    if dataset_idx is None:
        dataset_idx = ALNSConfig.DATASET_IDX

    dataset_name = f'dataset_{dataset_idx}'

    current_dir = Path(__file__).parent.parent
    input_file_loc = current_dir / 'datasets' / 'multiple-periods' / DATASET_TYPE
    output_file_loc = current_dir / 'OutPut-ALNS' / 'multiple-periods' / DATASET_TYPE

    print(f'Running smoke initial solution for {DATASET_TYPE} {dataset_name}...')

    data = DataALNS(input_file_loc, output_file_loc, dataset_name)
    data.load()

    state = SolutionState(data)
    rng = rnd.default_rng(42)

    t0 = time.time()
    sol = initial_solution(state, rng)
    elapsed = time.time() - t0

    feasible, violations = sol.validate()

    print(f'Elapsed: {elapsed:.2f}s; Feasible: {feasible}')
    neg_inv = violations.get('negative_inventory', [])
    veh_over_load = violations.get('veh_over_load', [])
    plant_inv_exceed = violations.get('plant_inv_exceed', [])
    print(f'Negative inventory violations: {len(neg_inv)}')
    print(f'Vehicle overload violations: {len(veh_over_load)}')
    print(f'Plant inventory exceed violations: {len(plant_inv_exceed)}')

    return sol


if __name__ == '__main__':
    # allow optional dataset index as CLI arg
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--idx', type=int, default=None, help='dataset index to run (overrides config)')
    args = p.parse_args()
    run(args.idx)
