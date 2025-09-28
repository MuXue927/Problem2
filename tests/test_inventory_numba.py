import time
from types import SimpleNamespace

import numpy as np

from ALNSCode import inventory_numba
from ALNSCode.alnsopt import SolutionState


class DummyData:
    def __init__(self):
        # single plant, two skus, horizon 3
        self.plants = ["P1"]
        self.skus_plant = {"P1": ["A", "B"]}
        self.horizons = 3
        # production: (plant, sku, day) -> qty
        self.sku_prod_each_day = {
            ("P1", "A", 1): 2,
            ("P1", "A", 2): 1,
            ("P1", "B", 1): 0,
            ("P1", "B", 2): 3,
        }
        # initial inventory day 0
        self.historical_s_ikt = {
            ("P1", "A", 0): 10,
            ("P1", "B", 0): 5,
        }
        # minimal other attributes to be compatible if referenced
        self.sku_sizes = {"A": 1, "B": 1}
        self.veh_type_cap = {}
        self.veh_type_cost = {}
        self.veh_type_min_load = {}
        self.param_pun_factor1 = 1.0
        self.param_pun_factor3 = 1.0
        self.demands = {}
        self.plant_inv_limit = {"P1": 999999}

    def construct_supply_chain(self):
        """
        Minimal construct_supply_chain used by SolutionState.construct_indices in tests.
        Returns a mapping {(plant, dealer): [sku list]}.
        """
        # single dealer 'D1' for plant P1 in test fixture
        return {("P1", "D1"): list(self.skus_plant.get("P1", []))}


def make_vehicle(fact_id, day, cargo):
    # simple object with required attributes
    v = SimpleNamespace()
    v.fact_id = fact_id
    v.day = day
    v.cargo = cargo  # dict of ((sku, day), qty)
    v.id = id(v)
    v.dealer_id = "D1"
    v.type = "T1"
    v.capacity = 1000
    return v


def baseline_compute(state):
    """
    Reproduce the pure-Python baseline from alnsopt.compute_inventory:
    iterate plant-sku pairs and days, using state.s_ikt baseline day0,
    state.data.sku_prod_each_day and shipped_by_plant_sku_day aggregated from vehicles.
    Returns a dict mapping (plant, sku, day) -> inventory
    """
    from collections import defaultdict

    data = state.data
    shipped_by_plant_sku_day = defaultdict(int)
    for veh in state.vehicles:
        fact = veh.fact_id
        d = veh.day
        for (sku_id, day_k), q in veh.cargo.items():
            if day_k == d:
                shipped_by_plant_sku_day[(fact, sku_id, day_k)] += q

    # collect plant-sku pairs from data.skus_plant
    plant_sku_pairs = {(plant, sku) for plant, skus in data.skus_plant.items() for sku in skus}
    s_ikt = {}
    # initialize day0 from historical
    for (p, sku, day), inv in data.historical_s_ikt.items():
        if day == 0:
            s_ikt[(p, sku, 0)] = inv

    for (plant, sku_id) in plant_sku_pairs:
        for day in range(1, data.horizons + 1):
            shipped_from_plant = shipped_by_plant_sku_day.get((plant, sku_id, day), 0)
            prev_inventory = s_ikt.get((plant, sku_id, day - 1), 0)
            production = data.sku_prod_each_day.get((plant, sku_id, day), 0)
            current_inventory = prev_inventory + production - shipped_from_plant
            s_ikt[(plant, sku_id, day)] = current_inventory
    return s_ikt


def test_numba_fast_matches_baseline_and_cache_behavior():
    data = DummyData()
    state = SolutionState(data=data)

    # create a vehicle shipping 5 of A on day 1
    v1 = make_vehicle("P1", 1, {("A", 1): 5})
    state.vehicles.append(v1)

    # baseline expected
    expected = baseline_compute(state)

    # run compute_inventory_fast (uses inventory_numba wrapper)
    ok = inventory_numba.compute_inventory_fast(state)
    assert ok is True

    # verify state.s_ikt matches expected for keys present
    for k, v in expected.items():
        assert abs(state.s_ikt.get(k, 0) - v) < 1e-9

    # test cache behavior: modify vehicle cargo and ensure recompute notices change
    # modify v1 cargo to 3 instead of 5
    v1.cargo = {("A", 1): 3}
    # recompute expected
    expected2 = baseline_compute(state)
    # call fast path again (should detect signature change and rebuild)
    ok2 = inventory_numba.compute_inventory_fast(state)
    assert ok2 is True
    for k, v in expected2.items():
        assert abs(state.s_ikt.get(k, 0) - v) < 1e-9


def test_numpy_core_equivalence():
    """Run a small end-to-end using the numpy core explicitly (no caching)"""
    data = DummyData()
    state = SolutionState(data=data)
    v1 = make_vehicle("P1", 1, {("A", 1): 4, ("B", 2): 1})
    state.vehicles.append(v1)

    maps = inventory_numba.build_index_maps(data)
    s_arr, shipments, prod = inventory_numba.state_to_inventory_arrays(state, maps)

    # run numpy core on a copy
    s_copy = s_arr.copy()
    inventory_numba._compute_inventory_core_numpy(s_copy, shipments, prod, maps.horizons)
    # write back to temp dict
    temp_state = SimpleNamespace()
    inventory_numba.arrays_to_state_s_ikt(temp_state, maps, s_copy)
    # compute baseline
    state2 = SolutionState(data=data)
    state2.vehicles = [v1]
    expected = baseline_compute(state2)

    # compare entries
    for pi, plant in enumerate(maps.idx_to_plant):
        for si, sku in enumerate(maps.idx_to_sku):
            for d in range(0, maps.horizons + 1):
                key = (plant, sku, d)
                val_expected = expected.get(key, 0)
                val_actual = temp_state.s_ikt.get(key, 0)
                assert abs(val_expected - val_actual) < 1e-9


if __name__ == "__main__":
    # quick manual run
    t0 = time.perf_counter()
    test_numba_fast_matches_baseline_and_cache_behavior()
    t1 = time.perf_counter()
    print("tests passed, time:", t1 - t0)
