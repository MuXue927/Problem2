#!/usr/bin/env python3
"""
Benchmark real datasets under datasets/multiple-periods/{small,medium,large}.
Saves results to real_benchmark_results.txt
"""
import os
import time
import statistics
from pathlib import Path

from ALNSCode.InputDataALNS import DataALNS
from ALNSCode import inventory_numba
from ALNSCode.alnsopt import SolutionState

ROOT = "datasets"
MULTI = os.path.join(ROOT, "multiple-periods")

def find_first_dataset(size_dir):
    base = os.path.join(MULTI, size_dir)
    if not os.path.isdir(base):
        return None
    for name in sorted(os.listdir(base)):
        full = os.path.join(base, name)
        if os.path.isdir(full):
            # basic existence check for required files
            req = ["order.csv","product_size.csv","warehouse_production.csv","warehouse_storage.csv","vehicle.csv"]
            present = all(os.path.exists(os.path.join(full, f)) for f in req)
            # some dataset folders may omit some files; accept if production+inventory present
            if os.path.exists(os.path.join(full, "warehouse_production.csv")) and os.path.exists(os.path.join(full, "warehouse_storage.csv")):
                return os.path.join("multiple-periods", size_dir, name)
    return None

def bench_dataset(input_root, dataset_name, repeats=6, warmup=2):
    # load DataALNS
    try:
        # dataset_name may include a path like "multiple-periods/small/dataset_1".
        # DataALNS expects input_file_loc to be the directory that contains model_config_alns.json,
        # and dataset_name to be the subfolder under that directory. Compute parent/base accordingly.
        parent_dir = os.path.dirname(dataset_name)
        base_name = os.path.basename(dataset_name)
        input_loc = os.path.join(input_root, parent_dir) if parent_dir else input_root
        data = DataALNS(input_loc, "outputs", base_name)
        data.load()
    except Exception as e:
        return {"dataset": dataset_name, "error": f"load failed: {e}"}

    # build state
    state = SolutionState(data=data)
    # no vehicles assigned (baseline still measures inventory propagation)
    # prepare arrays
    maps = inventory_numba.build_index_maps(data)
    s_arr, shipments, prod = inventory_numba.state_to_inventory_arrays(state, maps)

    # warmup
    def run_baseline():
        st = SolutionState(data=data)
        st.compute_inventory()

    def run_numpy():
        s_copy = s_arr.copy()
        inventory_numba._compute_inventory_core_numpy(s_copy, shipments, prod, maps.horizons)

    has_numba_kernel = hasattr(inventory_numba, "_compute_inventory_core_numba") and inventory_numba.NUMBA_AVAILABLE
    if has_numba_kernel:
        def run_numba():
            s_copy = s_arr.copy()
            inventory_numba._compute_inventory_core_numba(s_copy, shipments, prod, maps.horizons)
    else:
        def run_numba():
            st = SolutionState(data=data)
            inventory_numba.compute_inventory_fast(st)

    for _ in range(warmup):
        run_baseline()
        run_numpy()
        run_numba()

    def time_fn(fn, repeats=repeats):
        t = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            t.append(t1 - t0)
        return t

    base = time_fn(run_baseline)
    numpy_t = time_fn(run_numpy)
    numba_t = time_fn(run_numba)

    return {
        "dataset": dataset_name,
        "counts": {
            "plants": len(data.plants),
            "skus": len(data.all_skus),
            "horizons": data.horizons
        },
        "baseline": base,
        "numpy": numpy_t,
        "numba": numba_t,
        "numba_kernel_available": has_numba_kernel
    }

def summarize_timings(t):
    return {"mean": statistics.mean(t), "stdev": statistics.stdev(t) if len(t)>1 else 0.0, "min": min(t), "max": max(t)}

def main():
    sizes = ["small","medium","large"]
    results = []
    for size in sizes:
        ds = find_first_dataset(size)
        if ds is None:
            results.append({"dataset": f"{size}/NONE", "error": "no dataset found"})
            continue
        print("Running on dataset:", ds)
        res = bench_dataset("datasets", ds, repeats=6, warmup=2)
        results.append(res)

    # write results
    lines = []
    for r in results:
        lines.append(f"dataset: {r.get('dataset')}")
        if r.get("error"):
            lines.append("  error: " + r["error"])
            continue
        cnts = r["counts"]
        lines.append(f"  counts: plants={cnts['plants']} skus={cnts['skus']} horizons={cnts['horizons']}")
        for k in ("baseline","numpy","numba"):
            stats = summarize_timings(r[k])
            lines.append(f"  {k}: mean={stats['mean']:.6f}s stdev={stats['stdev']:.6f}s min={stats['min']:.6f}s max={stats['max']:.6f}s")
        if r.get("numba_kernel_available"):
            mean_base = statistics.mean(r['baseline'])
            mean_num = statistics.mean(r['numba'])
            mean_np = statistics.mean(r['numpy'])
            lines.append(f"  speedup numpy vs baseline: {mean_base/mean_np:.2f}x")
            lines.append(f"  speedup numba vs baseline: {mean_base/mean_num:.2f}x")
        lines.append("")
    out = "\\n".join(lines)
    with open("real_benchmark_results.txt","w",encoding="utf-8") as f:
        f.write(out)
    print("Wrote real_benchmark_results.txt")

if __name__ == "__main__":
    main()
