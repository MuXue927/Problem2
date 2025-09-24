import sys
import os
import time
import datetime
import numpy as np
import pytest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ALNSCode.alnsopt import initial_solution, SolutionState
from ALNSCode.InputDataALNS import DataALNS

from ALNSCode.alns_config import ALNSConfig
DATASET_TYPE = ALNSConfig.DATASET_TYPE
DATASET_IDX = ALNSConfig.DATASET_IDX

import numpy.random as rnd


current_dir = Path(__file__).parent.parent
par_path = current_dir.parent
input_file_loc = par_path / 'datasets' / 'multiple-periods' / DATASET_TYPE
output_file_loc = par_path / 'OutPut-ALNS' / 'multiple-periods' / DATASET_TYPE

def test_initial_solution(dataset_name: str):

    print(f"Begin testing the initial solution generation algorithm / {datetime.datetime.now()}...")

    data = DataALNS(input_file_loc, output_file_loc, dataset_name)
    data.load()
    
    # 构造初始解状态
    state = SolutionState(data)
    rng = rnd.default_rng(42)
    
    # 生成初始解
    t0 = time.time()
    print(f'Generating initial solution for {DATASET_TYPE} dataset_{DATASET_IDX}...')
    result = initial_solution(state, rng)
    print(f'Initial solution generated in {time.time() - t0:.2f} seconds.')
    
    # 校验解的可行性
    feasible, violations = result.validate()
    print('Feasible:', feasible)
    
    neg_inv = violations.get('negative_inventory', [])
    veh_over_load = violations.get('veh_over_load', [])
    plant_inv_exceed = violations.get('plant_inv_exceed', [])
    
    print(f'Negative inventory violations: {len(neg_inv)}')
    print(f'Vehicle overload violations: {len(veh_over_load)}')
    print(f'Plant inventory exceed violations: {len(plant_inv_exceed)}')

    # 断言无严重违规
    assert feasible, f"Initial solution infeasible! Negative inventory: {len(neg_inv)}, Vehicle overload: {len(veh_over_load)}, Plant inventory exceed: {len(plant_inv_exceed)}."
   

if __name__ == '__main__':
    # 一次性测试当前数据集中的所有instances
    
    print(f'Testing all instances in {DATASET_TYPE} dataset...\n')
    t_start = time.time()
    
    TEST_ALL = True  # 是否一次性测试所有实例
    
    if TEST_ALL:
    
        total_time = 0   # record total time for all tests
        
        for idx in range(1, 31):
            DATASET_IDX = idx
            dataset_name = f'dataset_{DATASET_IDX}'
            print(f'\nTesting {DATASET_TYPE} dataset_{DATASET_IDX}...')
            t0 = time.time()
            test_initial_solution(dataset_name)
            duration = time.time() - t0
            total_time += duration
            print(f'Test completed in {duration:.2f} seconds.')
        
        print(f'\nAll tests in {DATASET_TYPE} dataset completed in {time.time() - t_start:.2f} seconds.')
        print(f'Average time per instance: {total_time / 30:.2f} seconds.')
        
    else:
        dataset_name = f'dataset_{DATASET_IDX}'
        print(f'\nTesting {DATASET_TYPE} dataset_{DATASET_IDX}...')
        t0 = time.time()
        test_initial_solution(dataset_name)
        duration = time.time() - t0
        print(f'Test completed in {duration:.2f} seconds.')
