#!/usr/bin/env python3
"""
Plot: Overhead vs % left-turn (100% → 0%).
Two curves: Circular, PIBT. Shows that Circular worsens as left-turn ratio
decreases (hard-coded roundabout wastes time when through traffic dominates);
PIBT stays consistently good (only left-turn vehicles use roundabout).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from road_network import RoadNetwork4Way
from intersection_lane import run_paper_pibt_lane, run_circular_lane, compute_ideal_lane
from experiments_summary import generate_lane_agents_4way_left_bias

GRID_SIZE = 64
INTERSECTION_HALF = 5
NUM_AGENTS = 24
LEFT_TURN_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]  # 0% to 100%
SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds to match paper
MAX_STEPS = 5000


def run_sweep():
    road = RoadNetwork4Way(grid_size=GRID_SIZE, intersection_half=INTERSECTION_HALF)
    x_pct = []       # % left (100 down to 0)
    ideal_mean = []
    circular_mean = []
    pibt_mean = []

    for left_ratio in LEFT_TURN_RATIOS:
        ideal_delays, circular_delays, pibt_delays = [], [], []
        for seed in SEEDS:
            origins, destinations, _, _ = generate_lane_agents_4way_left_bias(
                road, NUM_AGENTS, seed, left_turn_ratio=left_ratio
            )
            ideal_delays.append(compute_ideal_lane(road, origins, destinations)["total_agent_time"])
            circular_delays.append(
                run_circular_lane(road, origins, destinations, MAX_STEPS)["total_agent_time"]
            )
            pibt_delays.append(
                run_paper_pibt_lane(road, origins, destinations, MAX_STEPS)["total_agent_time"]
            )
        x_pct.append(int(left_ratio * 100))
        ideal_mean.append(np.mean(ideal_delays))
        circular_mean.append(np.mean(circular_delays))
        pibt_mean.append(np.mean(pibt_delays))

    return np.array(x_pct), np.array(ideal_mean), np.array(circular_mean), np.array(pibt_mean)


def main():
    print("Running sweep (IDEAL, Circular, PIBT=Paper-PIBT) for left-turn ratios 0%, 25%, 50%, 75%, 100%...")
    x_pct, ideal_mean, circular_mean, pibt_mean = run_sweep()

    # Overhead vs IDEAL (%)
    circular_oh = (circular_mean / ideal_mean - 1) * 100
    pibt_oh = (pibt_mean / ideal_mean - 1) * 100

    # X-axis: 100% left turn (left) to 0% left turn (right)
    x_plot = np.array([100, 75, 50, 25, 0])
    order = [4, 3, 2, 1, 0]
    circular_oh_plot = circular_oh[order]
    pibt_oh_plot = pibt_oh[order]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(x_plot, circular_oh_plot, "o-", color="C1", linewidth=2, markersize=8, label="Circular")
    ax.plot(x_plot, pibt_oh_plot, "s-", color="C0", linewidth=2, markersize=8, label="PIBT")

    ax.set_xlabel("% left-turn (ratio of vehicles that turn left)", fontsize=11)
    ax.set_ylabel("Overhead over IDEAL (%)", fontsize=11)
    ax.set_xticks([100, 75, 50, 25, 0])
    ax.set_xticklabels(["100%", "75%", "50%", "25%", "0%"])
    ax.set_xlim(105, -5)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    plt.tight_layout()
    out_path = "4way_circular_vs_pibt_overhead.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
