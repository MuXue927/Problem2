#!/usr/bin/env python3
"""
Micro-benchmark for SolutionState.validate().

Compares a pure-Python baseline (dict-iteration) vs the implemented fast-path
(which uses ALNSCode.inventory_numba + ndarray-based checks inside validate()).

Writes results to benchmark_validate_results.txt
"""
import time
import statistics
import random
from types import SimpleNamespace
from copy import deepcopy

from ALNSCode import inventory_numba
from ALNSCode.alnsopt import SolutionState

# Helper data/state builders (kept small and self-contained for reproducibility)
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

def make_vehicle(fact_id, day, cargo, seed=0):
    v = SimpleNamespace()
    v.fact_id = fact_id
    v.day = day
    v.cargo = cargo
    v.id = id(v)
    v.dealer_id = "D1"
    v.type = "T1"
    v.capacity = 1000
    # provide small helpers used by alns code
    def is_empty():
        return not bool(v.cargo)
    def load(sku, qty):
        v.cargo[(sku, day)] = v.cargo.get((sku, day), 0) + qty
    v.is_empty = is_empty
    v.load = load
    return v

def populate_vehicles(data, n_vehicles, seed=0):
    rnd = random.Random(seed)
    vehicles = []
    plants = data.plants
    for i in range(n_vehicles):
        p = rnd.choice(plants)
        d = rnd.randint(1, data.horizons)
        skus = data.skus_plant[p]
        k = rnd.randint(1, min(3, len(skus)))
        cargo = {}
        for _ in range(k):
            sku = rnd.choice(skus)
            qty = rnd.randint(1, 10)
            cargo[(sku, d)] = cargo.get((sku, d), 0) + qty
        vehicles.append(make_vehicle(p, d, cargo))
    return vehicles

def build_state(data, vehicles):
    state = SolutionState(data=data)
    state.vehicles = list(vehicles)
    return state

# Baseline validator: implements validate() fallback logic (dict iteration)
def baseline_validate(state: SolutionState):
    # ensure inventory baseline computed
    if not getattr(state, "s_initialized", False):
        state.compute_inventory()
    violations = {'negative_inventory': [], 'veh_over_load': [], 'plant_inv_exceed': []}
    # negative inventory via s_ikt dict
    for key, inv in state.s_ikt.items():
        if inv < 0:
            violations['negative_inventory'].append((key, inv))
    # veh over load
    for veh in state.vehicles:
        try:
            loaded = state.compute_veh_load(veh)
            cap = state.data.veh_type_cap.get(veh.type, float('inf'))
            if loaded - cap > 1e-9:
                info = {
                    'veh_id': getattr(veh, 'id', None),
                    'fact_id': getattr(veh, 'fact_id', None),
                    'dealer_id': getattr(veh, 'dealer_id', None),
                    'type': getattr(veh, 'type', None),
                    'day': getattr(veh, 'day', None),
                    'loaded': loaded,
                    'cap': cap,
                    'cargo': dict(getattr(veh, 'cargo', {}))
                }
                violations['veh_over_load'].append(info)
        except Exception:
            violations['veh_over_load'].append({'veh': veh, 'error': 'compute_veh_load error'})
    # plant day aggregation via dict
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

# Fast validator: call state.validate() which will attempt ndarray fast-path
def fast_validate(state: SolutionState):
    return state.validate()

# Timing helpers
def time_fn(fn, repeats=10):
    timings=[]
    for _ in range(repeats):
        t0=time.perf_counter()
        fn()
        t1=time.perf_counter()
        timings.append(t1-t0)
    return timings

def bench_instance(P,S,H,n_vehicles, repeats=12, warmup=3, seed=0):
    data = make_data(P,S,H,seed=seed)
    vehicles = populate_vehicles(data, n_vehicles, seed=seed+1)
    # prepare states for each run to avoid cross-call caching effects
    def run_baseline():
        st = build_state(data, vehicles)
        baseline_validate(st)
    def run_fast():
        st = build_state(data, vehicles)
        # warm fastpath: call once to ensure numba jit/warmup if applicable
        fast_validate(st)
    # warmup
    for _ in range(warmup):
        run_baseline()
        run_fast()
    base = time_fn(run_baseline, repeats=repeats)
    fast = time_fn(run_fast, repeats=repeats)
    return {
        'P':P,'S':S,'H':H,'V':n_vehicles,
        'baseline': base, 'fast': fast
    }

def summarize_timings(t):
    import statistics
    return {
        'mean': statistics.mean(t),
        'stdev': statistics.stdev(t) if len(t)>1 else 0.0,
        'min': min(t),
        'max': max(t)
    }

def run_benchmarks():
    scenarios = [
        {'P':1,'S':5 ,'H':5,  'V':10},
        {'P':5,'S':50,'H':30, 'V':200},
        {'P':10,'S':200,'H':60,'V':2000},
    ]
    results = []
    for sc in scenarios:
        print(f"Running validate scenario P={sc['P']} S={sc['S']} H={sc['H']} V={sc['V']}")
        res = bench_instance(sc['P'], sc['S'], sc['H'], sc['V'], repeats=8, warmup=3, seed=123)
        results.append(res)
    lines = []
    lines.append("validate benchmark summary")
    for r in results:
        lines.append(f"scenario P={r['P']} S={r['S']} H={r['H']} V={r['V']}")
        for k in ('baseline','fast'):
            stats = summarize_timings(r[k])
            lines.append(f"  {k}: mean={stats['mean']:.6f}s stdev={stats['stdev']:.6f}s min={stats['min']:.6f}s max={stats['max']:.6f}s")
        mean_base = statistics.mean(r['baseline'])
        mean_fast = statistics.mean(r['fast'])
        lines.append(f"  speedup fast vs baseline: {mean_base/mean_fast:.2f}x")
        lines.append("")
    with open('benchmark_validate_results.txt','w',encoding='utf-8') as f:
        f.write("\\n".join(lines))
    print("Wrote benchmark_validate_results.txt")

if __name__ == '__main__':
    run_benchmarks()
