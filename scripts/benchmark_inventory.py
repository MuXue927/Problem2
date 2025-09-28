#!/usr/bin/env python3
"""
Benchmark script: compare baseline Python compute_inventory vs numpy core vs numba/core fast path.

Saves results to benchmark_results.txt
"""
import time
import statistics
import random
from types import SimpleNamespace
from copy import deepcopy

from ALNSCode import inventory_numba
from ALNSCode.alnsopt import SolutionState

def make_data(P, S, H, seed=0):
    rnd = random.Random(seed)
    plants = [f"P{i+1}" for i in range(P)]
    # distribute SKUs across plants evenly
    skus = [f"S{j+1}" for j in range(S)]
    skus_plant = {}
    idx = 0
    for p in plants:
        count = max(1, S // P)
        skus_plant[p] = skus[idx: idx + count]
        idx += count
    # production and historical
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
    data.veh_type_cap = {}
    data.veh_type_cost = {}
    data.veh_type_min_load = {}
    data.param_pun_factor1 = 1.0
    data.param_pun_factor3 = 1.0
    data.demands = {}
    data.plant_inv_limit = {p: 10**9 for p in plants}
    def construct_supply_chain():
        # map each (plant,dealer) to skus (single dealer D1 per plant to keep simple)
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
    return v

def populate_vehicles(data, n_vehicles, seed=0):
    rnd = random.Random(seed)
    vehicles = []
    plants = data.plants
    for i in range(n_vehicles):
        p = rnd.choice(plants)
        d = rnd.randint(1, data.horizons)
        # select 1-3 skus for this vehicle
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

def time_fn(fn, repeats=10):
    timings=[]
    for _ in range(repeats):
        t0=time.perf_counter()
        fn()
        t1=time.perf_counter()
        timings.append(t1-t0)
    return timings

def bench_instance(P,S,H,n_vehicles, repeats=8, warmup=3, seed=0):
    data = make_data(P,S,H,seed=seed)
    vehicles = populate_vehicles(data, n_vehicles, seed=seed+1)
    # baseline: state.compute_inventory (pure-Python)
    def run_baseline():
        st = build_state(data, vehicles)
        st.compute_inventory()
    # numpy core: operate on arrays directly (no state writeback)
    maps = inventory_numba.build_index_maps(data)
    # prepare arrays from a sample state (should be equivalent per-run)
    sample_state = build_state(data, vehicles)
    s_arr_base, shipments_base, prod_base = inventory_numba.state_to_inventory_arrays(sample_state, maps)
    def run_numpy():
        s_copy = s_arr_base.copy()
        inventory_numba._compute_inventory_core_numpy(s_copy, shipments_base, prod_base, maps.horizons)

    # numba/core or fast path: prefer calling core if available to measure kernel
    has_numba_kernel = hasattr(inventory_numba, "_compute_inventory_core_numba") and inventory_numba.NUMBA_AVAILABLE
    if has_numba_kernel:
        def run_numba_kernel():
            s_copy = s_arr_base.copy()
            inventory_numba._compute_inventory_core_numba(s_copy, shipments_base, prod_base, maps.horizons)
    else:
        # fallback to compute_inventory_fast which includes caching/overhead
        def run_numba_kernel():
            st = build_state(data, vehicles)
            inventory_numba.compute_inventory_fast(st)

    # warmup
    for _ in range(warmup):
        run_baseline()
        run_numpy()
        run_numba_kernel()

    base = time_fn(run_baseline, repeats=repeats)
    numpy_times = time_fn(run_numpy, repeats=repeats)
    numba_times = time_fn(run_numba_kernel, repeats=repeats)

    return {
        'P':P,'S':S,'H':H,'V':n_vehicles,
        'baseline': base, 'numpy': numpy_times, 'numba': numba_times,
        'numba_kernel_available': has_numba_kernel
    }

def summarize_timings(t):
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
        print(f"Running scenario P={sc['P']} S={sc['S']} H={sc['H']} V={sc['V']}")
        res = bench_instance(sc['P'], sc['S'], sc['H'], sc['V'], repeats=6, warmup=2, seed=42)
        results.append(res)
    # write out summary
    lines = []
    lines.append("benchmark summary")
    for r in results:
        lines.append(f"scenario P={r['P']} S={r['S']} H={r['H']} V={r['V']}")
        for k in ('baseline','numpy','numba'):
            stats = summarize_timings(r[k])
            lines.append(f"  {k}: mean={stats['mean']:.6f}s stdev={stats['stdev']:.6f}s min={stats['min']:.6f}s max={stats['max']:.6f}s")
        if r.get('numba_kernel_available'):
            # compute speedups
            mean_base = statistics.mean(r['baseline'])
            mean_num = statistics.mean(r['numba'])
            mean_np = statistics.mean(r['numpy'])
            lines.append(f"  speedup numpy vs baseline: {mean_base/mean_np:.2f}x")
            lines.append(f"  speedup numba vs baseline: {mean_base/mean_num:.2f}x")
        else:
            mean_base = statistics.mean(r['baseline'])
            mean_np = statistics.mean(r['numpy'])
            mean_num = statistics.mean(r['numba'])
            lines.append(f"  speedup numpy vs baseline: {mean_base/mean_np:.2f}x")
            lines.append(f"  speedup fastpath vs baseline: {mean_base/mean_num:.2f}x (fastpath may include overhead)")
        lines.append("")
    with open('benchmark_results.txt','w',encoding='utf-8') as f:
        f.write("\\n".join(lines))
    print("Wrote benchmark_results.txt")

if __name__ == '__main__':
    run_benchmarks()
