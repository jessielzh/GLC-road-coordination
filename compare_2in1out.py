#!/usr/bin/env python3
"""
Compare IDEAL, SP, Circular, PIBT, G-PIBT, GLC on the 2-in-1-out intersection.

Usage:
  python compare_2in1out.py
  python compare_2in1out.py --agents 48 --left-ratio 0.5 --seeds 5
  python compare_2in1out.py --left-ratios 0.0 0.2 0.4 0.6 0.8 1.0
"""

import argparse
import heapq
import random
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional

from road_network import RoadNetwork2In1Out
from visualize_2in1out import generate_agents, step_gsp, step_circular, _PIBTPlanner, step_pibt


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _goal_arm(road, goal):
    """Return the arm number (0-3) for a goal cell, or -1 if in intersection."""
    gx, gy = int(goal[0]), int(goal[1])
    lo, hi = road.lo, road.hi
    if gy < lo: return 0   # North
    if gx > hi: return 1   # East
    if gy > hi: return 2   # South
    if gx < lo: return 3   # West
    return -1              # inside intersection


def _bfs_path(road, start, goal):
    """BFS shortest path on RoadNetwork2In1Out. Returns path list or None."""
    start = (int(start[0]), int(start[1]))
    goal  = (int(goal[0]),  int(goal[1]))
    if start == goal:
        return [start]
    queue   = deque([(start, [start])])
    visited = {start}
    while queue:
        cur, path = queue.popleft()
        for nc in road.get_valid_moves(cur[0], cur[1]):
            nc = (int(nc[0]), int(nc[1]))
            if nc == cur or nc in visited:
                continue
            np_ = path + [nc]
            if abs(nc[0]-goal[0]) <= 1 and abs(nc[1]-goal[1]) <= 1:
                return np_
            visited.add(nc)
            queue.append((nc, np_))
    return None


def _result(times, reached, n):
    completed = sum(reached)
    return {
        "total_agent_time": sum(times),
        "avg_travel_time":  sum(times) / n if n else 0,
        "completed":        completed,
        "total":            n,
        "completion_rate":  completed / n if n else 0,
    }


# ---------------------------------------------------------------------------
# 1. IDEAL — theoretical lower bound (no collisions)
# ---------------------------------------------------------------------------

def run_ideal(road, origins, destinations):
    n = len(origins)
    total = 0
    for i in range(n):
        path = _bfs_path(road, origins[i], destinations[i])
        total += (len(path) - 1) if path else 9999
    return {"total_agent_time": total, "avg_travel_time": total/n if n else 0,
            "completed": n, "total": n, "completion_rate": 1.0}


# ---------------------------------------------------------------------------
# 2. SP — shortest path, wait if blocked
# ---------------------------------------------------------------------------

def run_sp(road, origins, destinations, max_steps=5000, seed=0):
    """Rigid BFS SP: compute path once at spawn, follow it step-by-step, wait if blocked.
    After STUCK_THRESHOLD consecutive waits, take any valid move toward goal and replan.
    This matches the paper's SP baseline (~+32% overhead).
    """
    STUCK_THRESHOLD = 30
    n          = len(origins)
    positions  = [list(o) for o in origins]
    goals      = list(destinations)
    reached    = [False] * n
    times      = [0] * n
    paths      = {i: None for i in range(n)}
    consec_wait= [0] * n

    for _ in range(max_steps):
        active = [i for i in range(n) if not reached[i]]
        if not active:
            break

        occupied = {(int(positions[i][0]), int(positions[i][1])) for i in range(n) if not reached[i]}
        active.sort(key=lambda i: road.distance(positions[i], goals[i]))
        local_reserved = set()

        for i in active:
            px, py = int(positions[i][0]), int(positions[i][1])
            gx, gy = int(goals[i][0]),    int(goals[i][1])

            if abs(px - gx) <= 1 and abs(py - gy) <= 1:
                reached[i] = True; consec_wait[i] = 0
                local_reserved.add((px, py)); times[i] += 1; continue

            if paths[i] is None:
                path = _bfs_path(road, (px, py), (gx, gy))
                paths[i] = path[1:] if path and len(path) > 1 else []

            blocked = local_reserved | (occupied - {(px, py)})
            moved = False

            if paths[i]:
                nc = paths[i][0]
                if nc not in blocked:
                    positions[i][0], positions[i][1] = nc[0], nc[1]
                    local_reserved.add(nc); paths[i] = paths[i][1:]
                    consec_wait[i] = 0; moved = True

            if not moved and consec_wait[i] >= STUCK_THRESHOLD:
                opts = [m for m in road.get_valid_moves(px, py)
                        if m not in blocked and m != (px, py)]
                if opts:
                    best = min(opts, key=lambda m: road.distance(m, goals[i]))
                    positions[i][0], positions[i][1] = int(best[0]), int(best[1])
                    local_reserved.add((int(best[0]), int(best[1])))
                    paths[i] = None; consec_wait[i] = 0; moved = True

            if not moved:
                local_reserved.add((px, py)); consec_wait[i] += 1

            times[i] += 1

    return _result(times, reached, n)


# ---------------------------------------------------------------------------
# 3. Circular — roundabout policy
# ---------------------------------------------------------------------------

def run_circular(road, origins, destinations, max_steps=5000):
    n         = len(origins)
    positions = [list(o) for o in origins]
    goals     = list(destinations)
    reached   = [False] * n
    times     = [0] * n
    exiting   = [False] * n

    for _ in range(max_steps):
        if all(reached):
            break
        step_circular(road, positions, goals, reached, times, exiting, set())

    return _result(times, reached, n)


# ---------------------------------------------------------------------------
# 4. PIBT — Euclidean-distance priority, matching visualize_2in1out.py
# ---------------------------------------------------------------------------

def run_pibt(road, origins, destinations, max_steps=5000):
    planner   = _PIBTPlanner(road)
    n         = len(origins)
    positions = [list(o) for o in origins]
    goals     = list(destinations)
    reached   = [False] * n
    times     = [0] * n

    for _ in range(max_steps):
        if all(reached):
            break
        step_pibt(road, planner, positions, goals, reached, times)

    return _result(times, reached, n)


# ---------------------------------------------------------------------------
# 5. G-PIBT — congestion-aware guided PIBT (Okumura AAAI 2024)
# ---------------------------------------------------------------------------

def run_guided_pibt(road, origins, destinations, max_steps=5000, seed=0):
    n         = len(origins)
    positions = [list(o) for o in origins]
    goals     = list(destinations)
    reached   = [False] * n
    times     = [0] * n
    _rng      = random.Random(seed)

    # ── Plan congestion-aware paths ──────────────────────────────────
    guide_paths = {}
    path_index  = {}
    flow        = {}

    agents_by_dist = sorted(range(n), key=lambda i: road.distance(origins[i], destinations[i]))

    for i in agents_by_dist:
        start = (int(origins[i][0]), int(origins[i][1]))
        goal  = (int(destinations[i][0]), int(destinations[i][1]))

        def heuristic(pos, g=goal):
            return abs(pos[0]-g[0]) + abs(pos[1]-g[1])

        def edge_cost(v1, v2):
            fwd     = flow.get((v1, v2), 0)
            rev     = flow.get((v2, v1), 0)
            contra  = 2.0 * fwd * rev if fwd and rev else 0
            incoming = sum(flow.get((nb, v2), 0) for nb in road.get_valid_moves(v2[0], v2[1]))
            return 1.0 + contra + 0.3 * incoming

        open_set = [(heuristic(start), 0, start, [start])]
        visited  = {start: 0}
        path     = None
        while open_set:
            f, g, cur, cur_path = heapq.heappop(open_set)
            if abs(cur[0]-goal[0]) <= 1 and abs(cur[1]-goal[1]) <= 1:
                path = cur_path; break
            if g > visited.get(cur, float('inf')):
                continue
            for nb in road.get_valid_moves(cur[0], cur[1]):
                nc = (int(nb[0]), int(nb[1]))
                if nc == cur:
                    continue
                ng = g + edge_cost(cur, nc)
                if nc not in visited or ng < visited[nc]:
                    visited[nc] = ng
                    heapq.heappush(open_set, (ng + heuristic(nc), ng, nc, cur_path + [nc]))

        if path is None:
            path = _bfs_path(road, start, goal) or [start, goal]
        guide_paths[i] = path
        path_index[i]  = 0
        for j in range(len(path)-1):
            e = (path[j], path[j+1])
            flow[e] = flow.get(e, 0) + 1

    # ── Execution with PIBT ──────────────────────────────────────────
    def get_next(aid, pos):
        path = guide_paths.get(aid)
        if not path:
            return None
        cur = (int(pos[0]), int(pos[1]))
        idx = path_index.get(aid, 0)
        search = idx
        while search < len(path) and path[search] != cur:
            search += 1
        if search < len(path):
            path_index[aid] = search
            return path[search+1] if search+1 < len(path) else None
        # Position not on remaining path (displaced by PIBT yield):
        # return path[idx] so the agent navigates back to rejoin its route
        return path[idx] if idx < len(path) else None

    def score(aid, pos, cand, goal):
        cc = (int(cand[0]), int(cand[1]))
        nxt = get_next(aid, pos)
        if nxt and cc == nxt:
            return 0  # best: follow guide path immediately
        path = guide_paths.get(aid, [])
        for k, cell in enumerate(path[path_index.get(aid, 0):]):
            if cell == cc:
                return 1 + k  # on path, k steps ahead
        if cc == (int(pos[0]), int(pos[1])):
            return 10000  # staying is absolute last resort
        return 200 + int(road.distance(cand, goal))  # off-path: distance fallback

    stay_cnt = [0] * n
    STUCK    = 30

    for _ in range(max_steps):
        active = [i for i in range(n) if not reached[i]]
        if not active:
            break
        active.sort(key=lambda i: road.distance(positions[i], goals[i]))
        decided = {}; reserved = set(); in_prog = set()

        def pibt(aid):
            if aid in decided: return True
            if aid in in_prog:  return False
            in_prog.add(aid)
            pos  = positions[aid]; goal = goals[aid]
            px, py = int(pos[0]), int(pos[1])
            cands  = [(px, py)] + road.get_valid_moves(px, py)
            _rng.shuffle(cands)
            # When stuck too long, ignore path guidance — use distance only
            if stay_cnt[aid] >= STUCK:
                cands.sort(key=lambda c: (10000 if c == (px, py) else int(road.distance(c, goal))))
            else:
                cands.sort(key=lambda c: score(aid, pos, c, goal))
            for cand in cands:
                cc = (int(cand[0]), int(cand[1]))
                if cc in reserved: continue
                blocker = next((o for o in active if o != aid and o not in decided
                                and (int(positions[o][0]), int(positions[o][1])) == cc), None)
                if blocker is not None and not pibt(blocker): continue
                decided[aid] = cc; reserved.add(cc); in_prog.discard(aid); return True
            decided[aid] = (px, py); reserved.add((px, py)); in_prog.discard(aid); return False

        old_pos = {i: (int(positions[i][0]), int(positions[i][1])) for i in active}
        for a in active:
            pibt(a)
        for i in active:
            if i in decided:
                nx, ny = decided[i]
                positions[i][0], positions[i][1] = nx, ny
                gx, gy = int(goals[i][0]), int(goals[i][1])
                if abs(nx-gx) <= 1 and abs(ny-gy) <= 1:
                    reached[i] = True
                    stay_cnt[i] = 0
                elif (nx, ny) == old_pos[i]:
                    stay_cnt[i] += 1
                else:
                    stay_cnt[i] = 0
            times[i] += 1

    return _result(times, reached, n)


# ---------------------------------------------------------------------------
# 6. GLC — BFS optimal paths + PIBT collision resolution
# ---------------------------------------------------------------------------

def run_glc(road, origins, destinations, max_steps=5000, seed=0):
    n         = len(origins)
    positions = [list(o) for o in origins]
    goals     = list(destinations)
    reached   = [False] * n
    times     = [0] * n
    _rng      = random.Random(seed)

    guide_paths = {}
    path_index  = {}
    for i in range(n):
        start = (int(origins[i][0]), int(origins[i][1]))
        goal  = (int(destinations[i][0]), int(destinations[i][1]))
        path  = _bfs_path(road, start, goal)
        guide_paths[i] = path if path else [start, goal]
        path_index[i]  = 0

    def get_next(aid, pos):
        path = guide_paths.get(aid)
        if not path:
            return None
        cur = (int(pos[0]), int(pos[1]))
        idx = path_index.get(aid, 0)
        search = idx
        while search < len(path) and path[search] != cur:
            search += 1
        if search < len(path):
            path_index[aid] = search
            return path[search+1] if search+1 < len(path) else None
        # Position not on remaining path (displaced by PIBT yield):
        # return path[idx] so the agent navigates back to rejoin its route
        return path[idx] if idx < len(path) else None

    def score(aid, pos, cand, goal):
        cc = (int(cand[0]), int(cand[1]))
        nxt = get_next(aid, pos)
        if nxt and cc == nxt:
            return 0  # best: follow guide path immediately
        path = guide_paths.get(aid, [])
        for k, cell in enumerate(path[path_index.get(aid, 0):]):
            if cell == cc:
                return 1 + k  # on path, k steps ahead
        if cc == (int(pos[0]), int(pos[1])):
            return 10000  # staying is absolute last resort
        return 200 + int(road.distance(cand, goal))  # off-path: distance fallback

    stay_cnt = [0] * n
    STUCK    = 30

    for _ in range(max_steps):
        active = [i for i in range(n) if not reached[i]]
        if not active:
            break
        active.sort(key=lambda i: road.distance(positions[i], goals[i]))
        decided = {}; reserved = set(); in_prog = set()

        def pibt(aid):
            if aid in decided: return True
            if aid in in_prog:  return False
            in_prog.add(aid)
            pos  = positions[aid]; goal = goals[aid]
            px, py = int(pos[0]), int(pos[1])
            goal_arm = _goal_arm(road, goal)
            raw = [(px, py)] + road.get_valid_moves(px, py)
            # Filter: never enter an outbound lane that leads to the wrong arm
            cands = []
            for c in raw:
                cx, cy = int(c[0]), int(c[1])
                if road.is_outbound_lane(cx, cy) and goal_arm >= 0:
                    if _goal_arm(road, c) != goal_arm:
                        continue
                cands.append(c)
            if not cands:
                cands = [(px, py)]  # fallback: stay
            _rng.shuffle(cands)
            if stay_cnt[aid] >= STUCK:
                cands.sort(key=lambda c: (10000 if c == (px, py) else int(road.distance(c, goal))))
            else:
                cands.sort(key=lambda c: score(aid, pos, c, goal))
            for cand in cands:
                cc = (int(cand[0]), int(cand[1]))
                if cc in reserved: continue
                blocker = next((o for o in active if o != aid and o not in decided
                                and (int(positions[o][0]), int(positions[o][1])) == cc), None)
                if blocker is not None and not pibt(blocker): continue
                decided[aid] = cc; reserved.add(cc); in_prog.discard(aid); return True
            decided[aid] = (px, py); reserved.add((px, py)); in_prog.discard(aid); return False

        old_pos = {i: (int(positions[i][0]), int(positions[i][1])) for i in active}
        for a in active:
            pibt(a)
        for i in active:
            if i in decided:
                nx, ny = decided[i]
                positions[i][0], positions[i][1] = nx, ny
                gx, gy = int(goals[i][0]), int(goals[i][1])
                if abs(nx-gx) <= 1 and abs(ny-gy) <= 1:
                    reached[i] = True
                    stay_cnt[i] = 0
                elif (nx, ny) == old_pos[i]:
                    stay_cnt[i] += 1
                else:
                    stay_cnt[i] = 0
            times[i] += 1

    return _result(times, reached, n)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

METHODS = [
    ("IDEAL",   run_ideal),
    ("SP",      run_sp),
    ("Circular",run_circular),
    ("PIBT",    run_pibt),
    ("G-PIBT",  run_guided_pibt),
    ("GLC",     run_glc),
]

COL_W = 12


def run_comparison(
    num_agents: int       = 24,
    grid_size: int        = 64,
    intersection_half: int= 5,
    left_ratios: List[float] = None,
    right_ratio: float    = 0.35,
    seeds: List[int]      = None,
    max_steps: int        = 5000,
):
    if left_ratios is None:
        left_ratios = [0.3]
    if seeds is None:
        seeds = [85, 196, 410]

    road = RoadNetwork2In1Out(grid_size=grid_size, intersection_half=intersection_half)

    for lr in left_ratios:
        straight_pct = round((1.0 - lr - right_ratio) * 100)
        print(f"\n{'='*70}")
        print(f"  2-in-1-out  |  agents={num_agents}  "
              f"left={int(lr*100)}%  straight={straight_pct}%  right={int(right_ratio*100)}%  "
              f"seeds={seeds}")
        print(f"{'='*70}")

        # Header
        header = f"{'Method':<10}" + "".join(f"{'seed '+str(s):>{COL_W}}" for s in seeds) + f"{'mean':>{COL_W}}"
        print(header)
        print("-" * len(header))

        for name, fn in METHODS:
            row_vals  = []
            row_rates = []
            for s in seeds:
                origins, dests, _ = generate_agents(road, num_agents, s, left_ratio=lr, right_ratio=right_ratio)
                if name == "IDEAL":
                    r = fn(road, origins, dests)
                elif name in ("SP", "G-PIBT", "GLC"):
                    r = fn(road, origins, dests, max_steps=max_steps, seed=s)
                else:
                    r = fn(road, origins, dests, max_steps=max_steps)
                row_vals.append(r["total_agent_time"])
                row_rates.append(r["completion_rate"])

            all_done  = all(rate >= 1.0 for rate in row_rates)
            mean_val  = np.mean(row_vals)

            def fmt_cell(val, rate):
                if rate < 1.0:
                    return f"DNF({int(rate*100)}%)".rjust(COL_W)
                return f"{val:>{COL_W},}"

            cells    = "".join(fmt_cell(v, r) for v, r in zip(row_vals, row_rates))
            mean_str = "  DNF" if not all_done else f"{mean_val:>{COL_W},.1f}"
            print(f"{name:<10}{cells}{mean_str}")

        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare methods on 2-in-1-out intersection")
    p.add_argument("--agents",            type=int,   default=24)
    p.add_argument("--grid",              type=int,   default=64)
    p.add_argument("--intersection-half", type=int,   default=5)
    p.add_argument("--left-ratio",        type=float, default=None,
                   help="Single left-turn ratio (overrides --left-ratios)")
    p.add_argument("--left-ratios",       type=float, nargs="+",
                   default=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    p.add_argument("--right-ratio",       type=float, default=0.35,
                   help="Fraction of right-turn vehicles (paper Table 1: 0.35)")
    p.add_argument("--seeds",             type=int,   nargs="+", default=[85, 196, 410])
    p.add_argument("--max-steps",         type=int,   default=5000)
    args = p.parse_args()

    ratios = [args.left_ratio] if args.left_ratio is not None else args.left_ratios

    run_comparison(
        num_agents=args.agents,
        grid_size=args.grid,
        intersection_half=args.intersection_half,
        left_ratios=ratios,
        right_ratio=args.right_ratio,
        seeds=args.seeds,
        max_steps=args.max_steps,
    )
