#!/usr/bin/env python3
"""
Proof-of-concept: Mixed autonomy on Manhattan (fraction p of "human" agents).

- **AVs (1-p):** Follow SP-PIBT (shortest path + priority inheritance); full compliance.
- **Humans (p):** Follow shortest path but:
  - **Random yielding:** with probability yield_prob they wait even when they could move.
  - **Ignore priority inheritance:** they do not participate in PIBT; they move after AVs
    see remaining capacity (so AVs get priority).
  - Optional **probabilistic gap acceptance:** P(move) = 1 - beta * (congestion at target).

Metrics: completion rate, total delay (travel time), and **constraint violation attempts**
(number of times a human tried to move into a node/edge at capacity — proxy for
near-misses or non-compliance if we did not enforce capacity).
"""

import os
import sys
import argparse
import random
from typing import List, Dict, Set, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

from manhattan_real import OSMGraphRoad, GraphSPPIBTPlanner


def run_mixed_autonomy(
    road: OSMGraphRoad,
    origins: List[int],
    destinations: List[int],
    human_frac: float,
    yield_prob: float = 0.15,
    gap_accept_beta: float = 0.0,
    max_steps: int = 50000,
    seed: int = 42,
    human_model: str = "yield",
) -> dict:
    """
    human_frac: fraction of agents that are "human" (0 = all AV, 1 = all human).
    yield_prob: probability a human yields (waits) even when they could move (human_model='yield').
    gap_accept_beta: for human_model='gap', P(move) = max(0, 1 - beta * (reserved/cap)); 0 = deterministic.
    human_model: 'yield' (random yield) or 'gap' (probabilistic gap acceptance).
    """
    rng = random.Random(seed)
    n = len(origins)
    # Assign types: 0 = AV, 1 = human
    is_human = [rng.random() < human_frac for _ in range(n)]
    av_ids = [i for i in range(n) if not is_human[i]]
    human_ids = [i for i in range(n) if is_human[i]]

    positions = list(origins)
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    paths: Dict[int, List[int]] = {}
    violation_attempts = 0
    planner = GraphSPPIBTPlanner(road)

    for step in range(max_steps):
        if all(reached):
            break

        # Current occupancy (all agents)
        reserved_count: Dict[int, int] = {}
        edge_usage: Dict[Tuple[int, int], int] = {}
        for i in range(n):
            if reached[i]:
                continue
            pos = positions[i]
            reserved_count[pos] = reserved_count.get(pos, 0) + 1

        def can_move_to(pos: int, next_pos: int, rc: Dict, eu: Dict) -> bool:
            if rc.get(next_pos, 0) >= road.node_capacity(next_pos):
                return False
            if next_pos != pos and eu.get((pos, next_pos), 0) >= road.edge_capacity(pos, next_pos):
                return False
            return True

        def reserve(pos: int, next_pos: int, rc: Dict, eu: Dict):
            rc[next_pos] = rc.get(next_pos, 0) + 1
            if next_pos != pos:
                eu[(pos, next_pos)] = eu.get((pos, next_pos), 0) + 1

        # (1) AVs: run PIBT with initial state = current occupancy (so humans are obstacles)
        av_positions = {i: positions[i] for i in av_ids if not reached[i]}
        av_goals = {i: goals[i] for i in av_ids if not reached[i]}
        av_decided: Dict[int, int] = {}
        if av_positions:
            for i in av_ids:
                if reached[i]:
                    continue
                if i not in paths or paths[i] is None:
                    p, _ = road.shortest_path(positions[i], goals[i])
                    paths[i] = p[1:] if p and len(p) > 1 else []
            av_decided = planner.plan_step(av_positions, av_goals, paths=paths)
            for i in av_decided:
                reserve(positions[i], av_decided[i], reserved_count, edge_usage)

        # (2) Humans: try path move (random yield or gap acceptance); no PIBT
        for i in rng.sample(human_ids, len(human_ids)):
            if reached[i]:
                continue
            pos = positions[i]
            goal = goals[i]
            if pos == goal:
                reached[i] = True
                times[i] += 1
                continue
            if i not in paths or paths[i] is None:
                p, _ = road.shortest_path(pos, goal)
                paths[i] = p[1:] if p and len(p) > 1 else []
            if not paths[i]:
                times[i] += 1
                continue
            next_node = paths[i][0]
            e = (pos, next_node)

            # Random yield: with probability yield_prob, wait
            if human_model == "yield" and rng.random() < yield_prob:
                reserve(pos, pos, reserved_count, edge_usage)
                times[i] += 1
                continue

            # Gap acceptance: P(move) = max(0, 1 - beta * congestion_ratio)
            if human_model == "gap" and gap_accept_beta > 0:
                node_cap = road.node_capacity(next_node)
                edge_cap = road.edge_capacity(pos, next_node)
                cong_node = reserved_count.get(next_node, 0) / max(1, node_cap)
                cong_edge = edge_usage.get(e, 0) / max(1, edge_cap)
                cong = max(cong_node, cong_edge)
                if rng.random() > max(0, 1 - gap_accept_beta * cong):
                    reserve(pos, pos, reserved_count, edge_usage)
                    times[i] += 1
                    continue

            if can_move_to(pos, next_node, reserved_count, edge_usage):
                reserve(pos, next_node, reserved_count, edge_usage)
                positions[i] = next_node
                paths[i] = paths[i][1:]
            else:
                # Would have violated capacity; we enforce wait and count
                violation_attempts += 1
                reserve(pos, pos, reserved_count, edge_usage)
            times[i] += 1

        # Apply AV moves (and update paths)
        for i in av_ids:
            if reached[i]:
                continue
            if i in av_decided:
                prev = positions[i]
                positions[i] = av_decided[i]
                if positions[i] == goals[i]:
                    reached[i] = True
                if paths.get(i):
                    if positions[i] == paths[i][0]:
                        paths[i] = paths[i][1:]
                    elif positions[i] != prev:
                        paths[i] = None
            times[i] += 1

    for i in range(n):
        if not reached[i]:
            times[i] = max_steps

    completed = sum(reached)
    return {
        "total_delay": sum(times),
        "completion_rate": completed / n if n else 0,
        "completed": completed,
        "n": n,
        "violation_attempts": violation_attempts,
        "n_human": len(human_ids),
        "n_av": len(av_ids),
    }


def main():
    parser = argparse.ArgumentParser(description="Manhattan mixed autonomy POC (AV + human agents)")
    parser.add_argument("parquet", nargs="?", default="sample_od.parquet", help="OD parquet")
    parser.add_argument("--agents", type=int, default=100, help="Number of agents")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--cache-dir", default=".manhattan_cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yield-prob", type=float, default=0.15, help="Human random yield probability")
    parser.add_argument("--human-model", choices=["yield", "gap"], default="yield")
    parser.add_argument("--gap-beta", type=float, default=0.5, help="Gap acceptance beta (if human_model=gap)")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parquet_path = args.parquet if os.path.isabs(args.parquet) else os.path.join(root, args.parquet)
    if not os.path.isfile(parquet_path):
        print(f"Parquet not found: {parquet_path}")
        sys.exit(1)
    cache_dir = os.path.join(root, args.cache_dir)
    graph_path = os.path.join(cache_dir, "manhattan_graph.graphml")

    from nyc_taxi_real import get_real_od

    print("Loading graph and OD...")
    road = OSMGraphRoad(cache_path=graph_path)
    origins, destinations = get_real_od(road, parquet_path, n_agents=args.agents, seed=args.seed, cache_dir=cache_dir)
    n = len(origins)
    print(f"  Nodes: {road.G.number_of_nodes()}, OD pairs: {n}")
    print(f"  Human model: {args.human_model}, yield_prob={args.yield_prob}, gap_beta={args.gap_beta}")
    print()

    # Sweep over human fraction p
    fractions = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    print(f"{'p (human)':<12} {'Total delay':<14} {'Completion':<12} {'Violations':<12} {'n_AV / n_H':<12}")
    print("-" * 65)
    for p in fractions:
        out = run_mixed_autonomy(
            road, origins, destinations,
            human_frac=p,
            yield_prob=args.yield_prob,
            gap_accept_beta=args.gap_beta if args.human_model == "gap" else 0.0,
            max_steps=args.max_steps,
            seed=args.seed,
            human_model=args.human_model,
        )
        n_av = out["n_av"]
        n_h = out["n_human"]
        print(f"{p:<12.1f} {out['total_delay']:<14,} {out['completion_rate']*100:>6.1f}%      {out['violation_attempts']:<12,} {n_av}/{n_h}")
    print()
    print("Violations = number of times a human tried to move into a node/edge at capacity (constraint violation attempt).")


if __name__ == "__main__":
    main()
