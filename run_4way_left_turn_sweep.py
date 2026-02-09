#!/usr/bin/env python3
"""
4-way 2-in-1-out: sweep left-turn ratio (0%, 25%, 50%, 75%, 100%).
Paper experiment: compare PIBT (Paper-PIBT) vs Circular w.r.t. % left-turn.
Fixed: 24 agents, 11×11 intersection. Methods: IDEAL, PIBT, Circular.
"""

import numpy as np
from road_network import RoadNetwork4Way
from intersection_lane import (
    run_paper_pibt_lane,
    run_circular_lane,
    compute_ideal_lane,
)
from experiments_summary import generate_lane_agents_4way_left_bias

GRID_SIZE = 64
INTERSECTION_HALF = 5  # 11×11 intersection
NUM_AGENTS = 24
LEFT_TURN_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds to match paper
MAX_STEPS = 5000


def main():
    road = RoadNetwork4Way(grid_size=GRID_SIZE, intersection_half=INTERSECTION_HALF)

    methods = ["IDEAL", "PIBT", "Circular"]

    # results[ratio_str][method] = list of total_delay per seed. PIBT = Paper-PIBT.
    results = {f"{int(r*100)}%": {m: [] for m in methods} for r in LEFT_TURN_RATIOS}

    for left_ratio in LEFT_TURN_RATIOS:
        ratio_str = f"{int(left_ratio*100)}%"
        for seed in SEEDS:
            origins, destinations, _, _ = generate_lane_agents_4way_left_bias(
                road, NUM_AGENTS, seed, left_turn_ratio=left_ratio
            )
            ideal_r = compute_ideal_lane(road, origins, destinations)
            results[ratio_str]["IDEAL"].append(ideal_r["total_agent_time"])
            results[ratio_str]["PIBT"].append(
                run_paper_pibt_lane(road, origins, destinations, MAX_STEPS)["total_agent_time"]
            )
            results[ratio_str]["Circular"].append(
                run_circular_lane(road, origins, destinations, MAX_STEPS)["total_agent_time"]
            )

    # Print table: rows = left-turn ratio, cols = method (PIBT vs Circular)
    print("=" * 80)
    print("4-WAY: PIBT vs CIRCULAR w.r.t. % LEFT-TURN")
    print("24 agents, 11×11 intersection, seeds =", SEEDS)
    print("PIBT = Paper-PIBT. Metric: Total Delay (mean over seeds)")
    print("=" * 80)

    col_width = 14
    header = f"{'Left %':<8}" + "".join(f"{m:<{col_width}}" for m in methods)
    print(header)
    print("-" * len(header))

    for left_ratio in LEFT_TURN_RATIOS:
        ratio_str = f"{int(left_ratio*100)}%"
        row_parts = [f"{ratio_str:<8}"]
        ideal_mean = np.mean(results[ratio_str]["IDEAL"])
        for m in methods:
            mean_d = np.mean(results[ratio_str][m])
            if m == "IDEAL":
                row_parts.append(f"{mean_d:<{col_width}.0f}")
            else:
                oh = (mean_d / ideal_mean - 1) * 100 if ideal_mean > 0 else 0
                row_parts.append(f"{mean_d:.0f}(+{oh:.0f}%)".ljust(col_width))
        print("".join(row_parts))

    print("=" * 100)
    # Winner per row
    print("Winner per row (lowest total delay, excluding IDEAL):")
    for left_ratio in LEFT_TURN_RATIOS:
        ratio_str = f"{int(left_ratio*100)}%"
        candidates = {m: np.mean(results[ratio_str][m]) for m in methods if m != "IDEAL"}
        winner = min(candidates, key=candidates.get)
        print(f"  {ratio_str}: {winner}")
    print("=" * 100)


if __name__ == "__main__":
    main()
