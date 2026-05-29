#!/usr/bin/env python3
"""
Large-scale Manhattan quantitative experiment — no visualization.

Samples OD pairs from NYC TLC taxi data (Manhattan zones). --agents controls scale.
Use --methods to choose which methods to run. Reports progress and final results.

Usage:
  python run_manhattan_large_scale.py yellow_tripdata_2024-01.parquet [--agents 10000] [--methods ideal,sp,pibt,glc]
"""

import argparse
import csv
import os
import sys
import time

# Force line-buffered stdout so progress appears immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from manhattan_real import (
    OSMGraphRoad,
    compute_ideal_graph,
    run_sp_graph,
    run_tap_graph,
    run_pibt_graph,
    run_paper_pibt_graph,
    run_guided_pibt_graph,
    run_sp_pibt_graph,
)
from nyc_taxi_real import get_real_od


def make_progress_printer(method_name: str, csv_writer=None, t0: float = None):
    """Print every N steps. If csv_writer and t0, save method, step, completed, accumulated_delay, runtime_s, runtime_per_step_s (cumulative and per-step runtime)."""
    last_time = [t0]  # use list so closure can mutate

    def cb(step: int, max_steps: int, completed: int, n: int, total_delay: int):
        print(f"    {method_name}: step {step:,}/{max_steps:,}, {completed:,}/{n:,} agents completed", flush=True)
        if csv_writer is not None and t0 is not None:
            now = time.perf_counter()
            runtime_s = now - t0
            runtime_segment_s = now - last_time[0] if last_time[0] is not None else runtime_s
            last_time[0] = now
            # Average wall-clock seconds per simulation step so far (when step > 0)
            runtime_per_step_s = round(runtime_s / step, 6) if step > 0 else 0
            csv_writer.writerow([
                method_name, step, completed, total_delay,
                round(runtime_s, 3), round(runtime_segment_s, 3), runtime_per_step_s,
            ])
    return cb


def make_ideal_progress_printer():
    """Return a progress callback for IDEAL (agents processed)."""
    def cb(done: int, total: int):
        print(f"    IDEAL: {done:,}/{total:,} paths computed", flush=True)
    return cb


VALID_METHODS = {"ideal", "sp", "tap", "pibt", "paperpibt", "gpibt", "glc"}
METHOD_NAMES = {"ideal": "IDEAL", "sp": "SP", "tap": "TAP", "pibt": "PIBT", "paperpibt": "Paper-PIBT", "gpibt": "G-PIBT", "glc": "GLC"}


def main():
    ap = argparse.ArgumentParser(description="Large-scale Manhattan quantitative run")
    ap.add_argument("parquet", help="Path to NYC TLC Yellow taxi parquet")
    ap.add_argument("--agents", type=int, default=10000, help="Number of agents")
    ap.add_argument("--methods", type=str, default="ideal,sp,pibt,glc",
                    help="Comma-separated methods: ideal, sp, tap, pibt, paperpibt, gpibt, glc")
    ap.add_argument("--max-steps", type=int, default=50000, help="Max simulation steps")
    ap.add_argument("--cache-dir", default=".manhattan_cache", help="Cache dir for graph/OD")
    ap.add_argument("--progress-interval", type=int, default=1, help="Print progress every N steps (steps elapsed, agents completed)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for OD sampling (also used for cache key)")
    ap.add_argument("--output-csv", default="", help="Save progress to CSV: method, step, completed, accumulated_delay, runtime_s, runtime_segment_s, runtime_per_step_s")
    args = ap.parse_args()

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    invalid = [m for m in methods if m not in VALID_METHODS]
    if invalid:
        print(f"Invalid methods: {invalid}. Valid: {', '.join(sorted(VALID_METHODS))}")
        sys.exit(1)
    if not methods:
        print("At least one method required. Valid: ideal, sp, tap, pibt, paperpibt, gpibt, glc")
        sys.exit(1)

    if not os.path.isfile(args.parquet):
        print(f"Parquet not found: {args.parquet}")
        sys.exit(1)

    cache_dir = args.cache_dir
    graph_path = os.path.join(cache_dir, "manhattan_graph.graphml")

    print("=" * 70, flush=True)
    print("MANHATTAN LARGE-SCALE QUANTITATIVE EXPERIMENT", flush=True)
    print("=" * 70, flush=True)
    print(f"  Parquet: {args.parquet}", flush=True)
    print(f"  Methods: {', '.join(METHOD_NAMES[m] for m in methods)}", flush=True)
    print(f"  Progress interval: every {args.progress_interval:,} steps", flush=True)
    print(flush=True)

    print("Loading road network...", flush=True)
    road = OSMGraphRoad(cache_path=graph_path)
    n_pruned = road.nodes_pruned()
    if n_pruned > 0:
        print(f"  Graph pruning: removed {n_pruned:,} sink nodes (iterative)", flush=True)
    print(f"  Graph: {road.num_nodes():,} nodes", flush=True)
    print("Loading OD pairs (uses cache if available)...", flush=True)
    origins, destinations = get_real_od(road, args.parquet, n_agents=args.agents, seed=args.seed, cache_dir=cache_dir)
    n = len(origins)
    if n == 0:
        print("No valid OD pairs.", flush=True)
        sys.exit(1)
    print(f"Loaded {n:,} OD pairs. max_steps={args.max_steps:,}", flush=True)
    print(flush=True)

    results = {}
    id_val = None
    csv_file = None
    csv_writer = None
    if args.output_csv:
        out_dir = os.path.dirname(args.output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        csv_file = open(args.output_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["method", "step", "completed", "accumulated_delay", "runtime_s", "runtime_segment_s", "runtime_per_step_s"])

    try:
        for idx, m in enumerate(methods, 1):
            label = METHOD_NAMES[m]
            print(f"[{idx}/{len(methods)}] {label}...", flush=True)
            t0 = time.perf_counter()
            if m == "ideal":
                out = compute_ideal_graph(
                    road, origins, destinations,
                    progress_callback=make_ideal_progress_printer(),
                    progress_interval=max(1000, n // 10),
                )
                results[m] = {"total_delay": out["total_delay"], "completion_rate": 1.0, "time": time.perf_counter() - t0}
                id_val = results[m]["total_delay"]
                print(f"      {label} done: delay={results[m]['total_delay']:,}, {results[m]['time']:.1f}s", flush=True)
            elif m == "sp":
                out = run_sp_graph(
                    road, origins, destinations, max_steps=args.max_steps,
                    progress_callback=make_progress_printer("SP", csv_writer, t0),
                    progress_interval=args.progress_interval,
                )
                results[m] = {"total_delay": out["total_delay"], "completion_rate": out["completion_rate"], "time": time.perf_counter() - t0}
                print(f"      {label} done: delay={results[m]['total_delay']:,}, completion={results[m]['completion_rate']*100:.1f}%, {results[m]['time']:.1f}s", flush=True)
            elif m == "tap":
                out = run_tap_graph(
                    road, origins, destinations, max_steps=args.max_steps,
                    progress_callback=make_progress_printer("TAP", csv_writer, t0),
                    progress_interval=args.progress_interval,
                )
                results[m] = {"total_delay": out["total_delay"], "completion_rate": out["completion_rate"], "time": time.perf_counter() - t0}
                print(f"      {label} done: delay={results[m]['total_delay']:,}, completion={results[m]['completion_rate']*100:.1f}%, {results[m]['time']:.1f}s", flush=True)
            elif m == "pibt":
                # PIBT = Paper-PIBT (time-based priority) for paper
                out = run_paper_pibt_graph(
                    road, origins, destinations, max_steps=args.max_steps,
                    progress_callback=make_progress_printer("PIBT", csv_writer, t0),
                    progress_interval=args.progress_interval,
                )
                results[m] = {"total_delay": out["total_delay"], "completion_rate": out["completion_rate"], "time": time.perf_counter() - t0}
                print(f"      {label} done: delay={results[m]['total_delay']:,}, completion={results[m]['completion_rate']*100:.1f}%, {results[m]['time']:.1f}s", flush=True)
            elif m == "paperpibt":
                out = run_paper_pibt_graph(
                    road, origins, destinations, max_steps=args.max_steps,
                    progress_callback=make_progress_printer("Paper-PIBT", csv_writer, t0),
                    progress_interval=args.progress_interval,
                )
                results[m] = {"total_delay": out["total_delay"], "completion_rate": out["completion_rate"], "time": time.perf_counter() - t0}
                print(f"      {label} done: delay={results[m]['total_delay']:,}, completion={results[m]['completion_rate']*100:.1f}%, {results[m]['time']:.1f}s", flush=True)
            elif m == "gpibt":
                out = run_guided_pibt_graph(road, origins, destinations, max_steps=args.max_steps,
                    progress_callback=make_progress_printer("G-PIBT", csv_writer, t0),
                    progress_interval=args.progress_interval)
                results[m] = {"total_delay": out["total_delay"], "completion_rate": out["completion_rate"], "time": time.perf_counter() - t0}
                print(f"      {label} done: delay={results[m]['total_delay']:,}, completion={results[m]['completion_rate']*100:.1f}%, {results[m]['time']:.1f}s", flush=True)
            elif m == "glc":
                out = run_sp_pibt_graph(
                    road, origins, destinations, max_steps=args.max_steps,
                    progress_callback=make_progress_printer("GLC", csv_writer, t0),
                    progress_interval=args.progress_interval,
                )
                results[m] = {"total_delay": out["total_delay"], "completion_rate": out["completion_rate"], "time": time.perf_counter() - t0}
                print(f"      {label} done: delay={results[m]['total_delay']:,}, completion={results[m]['completion_rate']*100:.1f}%, {results[m]['time']:.1f}s", flush=True)
            print()
    finally:
        if csv_file is not None:
            csv_file.close()
            print(f"Saved progress to {args.output_csv}", flush=True)

    # Use IDEAL as baseline for overhead if available
    if id_val is None and "ideal" not in results:
        id_val = results[methods[0]]["total_delay"]  # fallback to first method

    def fmt(d, cr, baseline):
        oh = f"+{(d/baseline-1)*100:.1f}%" if baseline and baseline > 0 and "ideal" in results else "-"
        return f"{d:,} ({oh})" if oh != "-" else f"{d:,}", f"{cr*100:.1f}%"

    print("=" * 90, flush=True)
    print("RESULTS", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Method':<12} {'Total Delay':<28} {'Completion':<12} {'Runtime (s)':<12}", flush=True)
    print("-" * 90, flush=True)
    for m in methods:
        r = results[m]
        label = METHOD_NAMES[m]
        if m == "ideal":
            print(f"{label:<12} {r['total_delay']:,}{'':>15} {'100.0%':<12} {r['time']:<12.2f}", flush=True)
        else:
            d_str, c_str = fmt(r["total_delay"], r["completion_rate"], id_val)
            print(f"{label:<12} {d_str:<28} {c_str:<12} {r['time']:<12.2f}", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
