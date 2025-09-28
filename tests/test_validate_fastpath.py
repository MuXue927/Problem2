import copy
from types import SimpleNamespace
import random
import pytest
import numpy as np

from ALNSCode.alnsopt import SolutionState
from ALNSCode import inventory_numba

# Helper builders (small, deterministic)
def make_data(P, S, H, seed=0):
    rnd = random.Random(seed)
    plants = [f"P{i+1}" for i in range(P)]
    skus = [f"S{j+1}" for j in range(S)]
    skus_plant = {}
    idx = 0
    for p in plants:
        count = max(1, S // P)
        skus_plant[p] = skus[idx: idx + count]
        idx += count
    sku_prod_each_day = {}
    historical_s_ikt = {}
    for p in plants:
        for s in skus_plant[p]:
            historical_s_ikt[(p, s, 0)] = float(rnd.randint(0, 50))
            for d in range(1, H+1):
                sku_prod_each_day[(p, s, d)] = float(rnd.randint(0, 5))
    data = SimpleNamespace()
    data.plants = plants
    data.skus_plant = skus_plant
    data.horizons = H
    data.sku_prod_each_day = sku_prod_each_day
    data.historical_s_ikt = historical_s_ikt
    data.sku_sizes = {s: 1 for s in skus}
    data.veh_type_cap = {"T1": 1000}
    data.veh_type_cost = {}
    data.veh_type_min_load = {}
    data.param_pun_factor1 = 1.0
    data.param_pun_factor3 = 1.0
    data.demands = {}
    data.plant_inv_limit = {p: 10**9 for p in plants}
    def construct_supply_chain():
        return {(p, "D1"): list(skus_plant[p]) for p in plants}
    data.construct_supply_chain = construct_supply_chain
    return data

def make_vehicle(fact_id, day, cargo):
    v = SimpleNamespace()
    v.fact_id = fact_id
    v.day = day
    v.cargo = cargo
    v.id = id(v)
    v.dealer_id = "D1"
    v.type = "T1"
    v.capacity = 1000
    def is_empty():
        return not bool(v.cargo)
    def load(sku, qty):
        v.cargo[(sku, day)] = v.cargo.get((sku, day), 0) + qty
    v.is_empty = is_empty
    v.load = load
    return v

# Baseline validator implementation (dict-based) used for assertions
def baseline_validate_dict(state: SolutionState):
    if not getattr(state, "s_initialized", False):
        state.compute_inventory()
    violations = {'negative_inventory': [], 'veh_over_load': [], 'plant_inv_exceed': []}
    for key, inv in state.s_ikt.items():
        if inv < 0:
            violations['negative_inventory'].append((key, inv))
    for veh in state.vehicles:
        try:
            loaded = state.compute_veh_load(veh)
            cap = state.data.veh_type_cap.get(veh.type, float('inf'))
            if loaded - cap > 1e-9:
                violations['veh_over_load'].append({'veh': veh, 'loaded': loaded, 'cap': cap})
        except Exception:
            violations['veh_over_load'].append({'veh': veh, 'error': 'compute_veh_load error'})
    from collections import defaultdict
    plant_day_inventory = defaultdict(float)
    for (plant, sku, day), inv in state.s_ikt.items():
        plant_day_inventory[(plant, day)] += float(inv)
    for (plant, day), total in plant_day_inventory.items():
        max_cap = state.data.plant_inv_limit.get(plant, float('inf'))
        if total - max_cap > 1e-9:
            violations['plant_inv_exceed'].append({'plant': plant, 'day': day, 'total_inv': total, 'max_cap': max_cap})
    is_feasible = not (violations['negative_inventory'] or violations['veh_over_load'] or violations['plant_inv_exceed'])
    return is_feasible, violations

def build_state_with_vehicles(data, vehicles):
    state = SolutionState(data=data)
    state.vehicles = list(vehicles)
    return state

def compare_violations(v1, v2):
    # Compare by keys and lengths; for negative_inventory compare sets of tuples to avoid ordering issues
    assert set([ (k[0], k[1], int(k[2]), float(v)) for (k,v) in v1['negative_inventory'] ]) == set([ (k[0], k[1], int(k[2]), float(v)) for (k,v) in v2['negative_inventory'] ])
    assert len(v1['veh_over_load']) == len(v2['veh_over_load'])
    assert len(v1['plant_inv_exceed']) == len(v2['plant_inv_exceed'])

def test_validate_equivalence_small():
    data = make_data(2, 6, 5, seed=1)
    # populate a few vehicles
    v1 = make_vehicle(data.plants[0], 1, {(list(data.skus_plant[data.plants[0]])[0], 1): 2})
    v2 = make_vehicle(data.plants[1], 2, {(list(data.skus_plant[data.plants[1]])[0], 2): 3})
    st = build_state_with_vehicles(data, [v1, v2])
    # run both validators
    fast_ok, fast_viol = st.validate()
    baseline_ok, baseline_viol = baseline_validate_dict(st)
    assert fast_ok == baseline_ok
    compare_violations(fast_viol, baseline_viol)

def test_validate_fallback_on_state_to_arrays_exception(monkeypatch):
    data = make_data(2, 6, 5, seed=2)
    v = make_vehicle(data.plants[0], 1, {(list(data.skus_plant[data.plants[0]])[0], 1): 1000})
    st = build_state_with_vehicles(data, [v])
    # monkeypatch the function used by validate's fast-path to force an exception
    orig = inventory_numba.state_to_inventory_arrays
    def broken_state_to_arrays(state, maps):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(inventory_numba, "state_to_inventory_arrays", broken_state_to_arrays)
    try:
        fast_ok, fast_viol = st.validate()  # should fallback and not raise
        baseline_ok, baseline_viol = baseline_validate_dict(st)
        assert fast_ok == baseline_ok
        compare_violations(fast_viol, baseline_viol)
    finally:
        monkeypatch.setattr(inventory_numba, "state_to_inventory_arrays", orig)

def test_negative_inventory_detection():
    data = make_data(1, 2, 3, seed=3)
    # set very small initial stock so shipments cause negative
    # override historical to zeros
    for p in data.plants:
        for s in data.skus_plant[p]:
            data.historical_s_ikt[(p, s, 0)] = 0.0
    # create a vehicle that ships more than available at day 1
    sku0 = list(data.skus_plant[data.plants[0]])[0]
    v = make_vehicle(data.plants[0], 1, {(sku0, 1): 10})
    st = build_state_with_vehicles(data, [v])
    ok, viol = st.validate()
    # expect at least one negative inventory violation
    assert len(viol['negative_inventory']) >= 1
