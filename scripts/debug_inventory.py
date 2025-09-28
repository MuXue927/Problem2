#!/usr/bin/env python3
import numpy as np
from types import SimpleNamespace
from ALNSCode import inventory_numba
from ALNSCode.alnsopt import SolutionState

# Recreate minimal DummyData and vehicle as in tests
class DummyData:
    def __init__(self):
        self.plants = ["P1"]
        self.skus_plant = {"P1": ["A", "B"]}
        self.horizons = 3
        self.sku_prod_each_day = {
            ("P1", "A", 1): 2,
            ("P1", "A", 2): 1,
            ("P1", "B", 1): 0,
            ("P1", "B", 2): 3,
        }
        self.historical_s_ikt = {
            ("P1", "A", 0): 10,
            ("P1", "B", 0): 5,
        }
        self.sku_sizes = {"A": 1, "B": 1}
        self.veh_type_cap = {}
        self.veh_type_cost = {}
        self.veh_type_min_load = {}
        self.param_pun_factor1 = 1.0
        self.param_pun_factor3 = 1.0
        self.demands = {}
        self.plant_inv_limit = {"P1": 999999}

    def construct_supply_chain(self):
        return {("P1", "D1"): list(self.skus_plant.get("P1", []))}

def make_vehicle(fact_id, day, cargo):
    v = SimpleNamespace()
    v.fact_id = fact_id
    v.day = day
    v.cargo = cargo
    v.id = id(v)
    v.dealer_id = "D1"
    v.type = "T1"
    v.capacity = 1000
    return v

def main():
    np.set_printoptions(threshold=1000, edgeitems=50, linewidth=200)
    data = DummyData()
    state = SolutionState(data=data)
    v1 = make_vehicle("P1", 1, {("A", 1): 4, ("B", 2): 1})
    state.vehicles.append(v1)

    maps = inventory_numba.build_index_maps(data)
    s_arr, shipments, prod = inventory_numba.state_to_inventory_arrays(state, maps)

    # print arrays and shapes
    out_lines = []
    out_lines.append(f"maps.horizons = {maps.horizons}")
    out_lines.append(f"idx_to_plant = {maps.idx_to_plant}")
    out_lines.append(f"idx_to_sku = {maps.idx_to_sku}")
    out_lines.append(f"s_arr.shape = {s_arr.shape}")
    out_lines.append(f"shipments.shape = {shipments.shape}")
    out_lines.append(f"prod.shape = {prod.shape}")
    out_lines.append('s_arr (initial baseline slice):')
    out_lines.append(np.array2string(s_arr, separator=', '))
    out_lines.append('shipments:')
    out_lines.append(np.array2string(shipments, separator=', '))
    out_lines.append('prod:')
    out_lines.append(np.array2string(prod, separator=', '))
    # Run numpy core on a copy
    s_copy = s_arr.copy()
    inventory_numba._compute_inventory_core_numpy(s_copy, shipments, prod, maps.horizons)
    out_lines.append('s_copy after numpy core:')
    out_lines.append(np.array2string(s_copy, separator=', '))

    # Also run baseline python compute to compare
    # Build a temp state and run state.compute_inventory (pure python fallback path)
    state2 = SolutionState(data=data)
    state2.vehicles = [v1]
    state2.compute_inventory()
    out_lines.append('state2.s_ikt (baseline compute_inventory):')
    # order the output for readability
    keys = sorted(list(state2.s_ikt.keys()))
    for k in keys:
        out_lines.append(f"{k}: {state2.s_ikt[k]}")

    # write to file
    with open('debug_inventory_output.txt', 'w', encoding='utf-8') as f:
        f.write('\\n'.join(out_lines))

    print('Wrote debug_inventory_output.txt')

if __name__ == '__main__':
    main()
