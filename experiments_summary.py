#!/usr/bin/env python3
"""
Comprehensive Experiment Summary: Total Delay Comparison

Scenarios:
1. Four-way 2-in-1-out intersection (vary intersection size and number of agents)
2. 3x3 directed grid (vary number of agents)

Methods:
- IDEAL: Theoretical lower bound (no blocking, sum of shortest paths)
- SP: Shortest Path + waiting (BFS paths, wait if blocked)
- PIBT: Priority Inheritance with Backtracking (Euclidean heuristic)
- GSP: Greedy Shortest Path (replans each step)
- Guided-PIBT: Congestion-aware PIBT (AAAI 2024)
- GLC (Ours): BFS optimal paths + PIBT collision resolution. PIBT in experiments = Paper-PIBT (time-based priority).

Metric: Total Delay = sum of arrival timesteps (∑Tᵢ)
"""

import numpy as np
import sys
from typing import List, Dict, Tuple, Optional, Callable
import networkx as nx

# Import 4-way intersection components
from road_network import RoadNetwork4Way
from intersection_lane import (
    generate_lane_agents_4way,
    run_gsp_lane,
    run_pibt_lane,
    run_paper_pibt_lane,
    run_circular_lane,
    run_sp_lane,
    compute_ideal_lane,
)

# Flag to include Circular method
INCLUDE_CIRCULAR = True

# Import 3x3 grid components  
from grid_3x3_network import (
    GridNetwork3x3,
    generate_left_turn_agents,
    step_gsp,
    step_pibt,
    step_sp,
    step_sp_guided_pibt,
    step_guided_pibt,
    PIBTPlanner,
    GraphPIBTPlanner,
    PaperPIBTPlanner,
    SPGuidedPIBT,
    GuidedPIBTCongestion,
)


# =============================================================================
# 4-WAY INTERSECTION: GLC (SP-PIBT) Implementation
# =============================================================================

def run_sp_modified_4way(
    road: RoadNetwork4Way,
    origins: List[Tuple[int, int]],
    destinations: List[Tuple[int, int]],
    max_steps: int = 5000,
) -> Dict:
    """
    Run SP-Modified on 4-way intersection.
    
    Simple improvement over basic SP:
    - Only intervene when an agent has been stuck for 50+ consecutive steps
    - When stuck, try any valid move toward goal
    - Add small randomness to priority ordering to break symmetric deadlocks
    """
    import random
    
    n = len(origins)
    positions = [list(o) for o in origins]
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    
    # SP state
    paths = {i: None for i in range(n)}  # BFS paths
    consecutive_wait = [0] * n  # Consecutive steps stuck
    
    STUCK_THRESHOLD = 50  # Only intervene after 50 consecutive waiting steps
    
    def compute_path(start, goal):
        """Compute BFS shortest path."""
        path, _ = road.shortest_path(start, goal)
        return path
    
    def distance(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    time_steps = 0
    
    for step in range(max_steps):
        time_steps = step + 1
        active = [i for i in range(n) if not reached[i]]
        if not active:
            break
        
        # Sort by distance to goal with small random tie-breaking
        active.sort(key=lambda i: (distance(positions[i], goals[i]), random.random() * 0.01))
        
        # Build occupied set
        occupied = set()
        for i in range(n):
            if not reached[i]:
                occupied.add((int(positions[i][0]), int(positions[i][1])))
        
        next_positions = [None] * n
        local_reserved = set()
        
        for i in active:
            px, py = int(positions[i][0]), int(positions[i][1])
            gx, gy = int(goals[i][0]), int(goals[i][1])
            
            # Check if reached goal
            if abs(px - gx) <= 1 and abs(py - gy) <= 1:
                reached[i] = True
                next_positions[i] = (gx, gy)
                local_reserved.add((gx, gy))
                times[i] += 1
                consecutive_wait[i] = 0
                continue
            
            # Compute path if not available
            if paths.get(i) is None:
                full_path = compute_path((px, py), (gx, gy))
                if full_path and len(full_path) > 1:
                    paths[i] = full_path[1:]
                else:
                    paths[i] = []
            
            moved = False
            blocked_cells = local_reserved | (occupied - {(px, py)})
            
            # Try to follow the planned path
            if paths[i]:
                next_cell = paths[i][0]
                if next_cell not in blocked_cells:
                    next_positions[i] = next_cell
                    local_reserved.add(next_cell)
                    paths[i] = paths[i][1:]
                    consecutive_wait[i] = 0
                    moved = True
            
            # If stuck for too long, try any valid move toward goal
            if not moved and consecutive_wait[i] >= STUCK_THRESHOLD:
                valid_moves = road.get_valid_moves(px, py)
                valid_moves = [m for m in valid_moves if m != (px, py) and m not in blocked_cells]
                if valid_moves:
                    # Pick move closest to goal
                    best_move = min(valid_moves, key=lambda m: distance(m, (gx, gy)))
                    next_positions[i] = best_move
                    local_reserved.add(best_move)
                    paths[i] = None  # Recompute path next step
                    consecutive_wait[i] = 0
                    moved = True
            
            if not moved:
                # Stay in place
                next_positions[i] = (px, py)
                local_reserved.add((px, py))
                consecutive_wait[i] += 1
            
            times[i] += 1
        
        # Apply moves
        for i in range(n):
            if next_positions[i]:
                positions[i][0], positions[i][1] = next_positions[i][0], next_positions[i][1]
    
    completed = sum(reached)
    total_delay = sum(times)
    
    return {
        "total_delay": total_delay,
        "total_agent_time": total_delay,
        "avg_travel_time": total_delay / n if n else float("inf"),
        "completed": completed,
        "total": n,
        "completion_rate": completed / n if n else 0,
        "time_steps": time_steps,
    }


def run_sp_pibt_4way(
    road: RoadNetwork4Way,
    origins: List[Tuple[int, int]],
    destinations: List[Tuple[int, int]],
    max_steps: int = 5000,
    step_callback: Optional[Callable[[list, int], None]] = None,
) -> Dict:
    """
    Run GLC on 4-way intersection (BFS optimal paths + PIBT collision resolution).
    If step_callback is provided, it is called each step as step_callback(positions_list, step).
    positions_list is a list of [x, y] for each agent (copy of current positions).
    """
    from collections import deque
    import random
    
    n = len(origins)
    positions = [list(o) for o in origins]
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    
    # Compute BFS paths once at start
    guide_paths = {}
    path_index = {}
    
    for i in range(n):
        start = (int(origins[i][0]), int(origins[i][1]))
        goal = (int(destinations[i][0]), int(destinations[i][1]))
        path, _ = road.shortest_path(start, goal)
        if path:
            guide_paths[i] = path
            path_index[i] = 0
        else:
            guide_paths[i] = [start, goal]
            path_index[i] = 0
    
    def get_next_on_path(agent_id, current_pos):
        if agent_id not in guide_paths:
            return None
        path = guide_paths[agent_id]
        current = (int(current_pos[0]), int(current_pos[1]))
        idx = path_index.get(agent_id, 0)
        while idx < len(path) and path[idx] != current:
            idx += 1
        if idx < len(path):
            path_index[agent_id] = idx
        if idx + 1 < len(path):
            return path[idx + 1]
        return None
    
    def distance(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def score_move(agent_id, pos, cand, goal):
        cand_cell = (int(cand[0]), int(cand[1]))
        pos_cell = (int(pos[0]), int(pos[1]))
        if cand_cell == pos_cell:
            return 50
        next_on_path = get_next_on_path(agent_id, pos)
        if next_on_path and cand_cell == next_on_path:
            return 0
        if agent_id in guide_paths:
            path = guide_paths[agent_id]
            idx = path_index.get(agent_id, 0)
            for i, cell in enumerate(path[idx:]):
                if cell == cand_cell:
                    return 1 + i
        return 100 + distance(cand, goal)
    
    time_steps = 0
    for step in range(max_steps):
        time_steps = step + 1
        active = [i for i in range(n) if not reached[i]]
        if not active:
            break
        
        # Sort by distance to goal
        active.sort(key=lambda i: distance(positions[i], goals[i]))
        
        decided = {}
        reserved = set()
        in_progress = set()
        
        def pibt(agent_id):
            if agent_id in decided:
                return True
            if agent_id in in_progress:
                return False
            
            in_progress.add(agent_id)
            pos = positions[agent_id]
            goal = goals[agent_id]
            
            # Get valid moves
            px, py = int(pos[0]), int(pos[1])
            candidates = [(px, py)] + road.get_valid_moves(px, py)
            
            # Score and sort candidates
            random.shuffle(candidates)
            candidates.sort(key=lambda c: score_move(agent_id, pos, c, goal))
            
            for cand in candidates:
                cand_cell = (int(cand[0]), int(cand[1]))
                if cand_cell in reserved:
                    continue
                
                blocked_agent = None
                for other in active:
                    if other == agent_id or other in decided:
                        continue
                    other_pos = (int(positions[other][0]), int(positions[other][1]))
                    if other_pos == cand_cell:
                        blocked_agent = other
                        break
                
                if blocked_agent is not None:
                    if not pibt(blocked_agent):
                        continue
                
                decided[agent_id] = cand_cell
                reserved.add(cand_cell)
                in_progress.discard(agent_id)
                return True
            
            decided[agent_id] = (px, py)
            reserved.add((px, py))
            in_progress.discard(agent_id)
            return False
        
        for agent in active:
            pibt(agent)
        
        # Apply moves and check arrivals
        for i in active:
            if i in decided:
                nx, ny = decided[i]
                positions[i][0], positions[i][1] = nx, ny
                
                gx, gy = int(goals[i][0]), int(goals[i][1])
                if abs(nx - gx) <= 1 and abs(ny - gy) <= 1:
                    reached[i] = True
            times[i] += 1

        if step_callback is not None:
            step_callback([[positions[i][0], positions[i][1]] for i in range(n)], time_steps)
    
    completed = sum(reached)
    total_delay = sum(times)
    
    return {
        "total_delay": total_delay,
        "total_agent_time": total_delay,
        "avg_travel_time": total_delay / n if n else float("inf"),
        "completed": completed,
        "total": n,
        "completion_rate": completed / n if n else 0,
        "time_steps": time_steps,
    }


def run_guided_pibt_4way(
    road: RoadNetwork4Way,
    origins: List[Tuple[int, int]],
    destinations: List[Tuple[int, int]],
    max_steps: int = 5000,
) -> Dict:
    """
    Run Guided-PIBT (congestion-aware) on 4-way intersection.
    """
    import heapq
    import random
    
    n = len(origins)
    positions = [list(o) for o in origins]
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    
    # Compute congestion-aware paths
    guide_paths = {}
    path_index = {}
    flow = {}
    
    def distance(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    # Sort agents by distance (shorter paths planned first)
    agents_by_dist = sorted(range(n), key=lambda i: distance(origins[i], destinations[i]))
    
    for i in agents_by_dist:
        start = (int(origins[i][0]), int(origins[i][1]))
        goal = (int(destinations[i][0]), int(destinations[i][1]))
        
        # A* with congestion costs
        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        def get_edge_cost(v1, v2):
            base = 1.0
            forward_flow = flow.get((v1, v2), 0)
            reverse_flow = flow.get((v2, v1), 0)
            contraflow = forward_flow * reverse_flow
            contraflow_penalty = 2.0 * contraflow if contraflow > 0 else 0
            incoming = sum(flow.get((n, v2), 0) for n in road.get_valid_moves(v2[0], v2[1]))
            vertex_penalty = 0.3 * incoming
            return base + contraflow_penalty + vertex_penalty
        
        open_set = [(heuristic(start), 0, start, [start])]
        visited = {start: 0}
        path = None
        
        while open_set:
            f, g, current, curr_path = heapq.heappop(open_set)
            if abs(current[0] - goal[0]) <= 1 and abs(current[1] - goal[1]) <= 1:
                path = curr_path
                break
            if g > visited.get(current, float('inf')):
                continue
            for neighbor in road.get_valid_moves(current[0], current[1]):
                next_cell = (int(neighbor[0]), int(neighbor[1]))
                if next_cell == current:
                    continue
                edge_cost = get_edge_cost(current, next_cell)
                new_g = g + edge_cost
                if next_cell not in visited or new_g < visited[next_cell]:
                    visited[next_cell] = new_g
                    new_f = new_g + heuristic(next_cell)
                    heapq.heappush(open_set, (new_f, new_g, next_cell, curr_path + [next_cell]))
        
        if path is None:
            # Fallback to BFS
            path, _ = road.shortest_path(start, goal)
            if path is None:
                path = [start, goal]
        
        guide_paths[i] = path
        path_index[i] = 0
        
        # Update flow
        for j in range(len(path) - 1):
            edge = (path[j], path[j+1])
            flow[edge] = flow.get(edge, 0) + 1
    
    def get_next_on_path(agent_id, current_pos):
        if agent_id not in guide_paths:
            return None
        path = guide_paths[agent_id]
        current = (int(current_pos[0]), int(current_pos[1]))
        idx = path_index.get(agent_id, 0)
        while idx < len(path) and path[idx] != current:
            idx += 1
        if idx < len(path):
            path_index[agent_id] = idx
        if idx + 1 < len(path):
            return path[idx + 1]
        return None
    
    def score_move(agent_id, pos, cand, goal):
        cand_cell = (int(cand[0]), int(cand[1]))
        pos_cell = (int(pos[0]), int(pos[1]))
        if cand_cell == pos_cell:
            return 50
        next_on_path = get_next_on_path(agent_id, pos)
        if next_on_path and cand_cell == next_on_path:
            return 0
        if agent_id in guide_paths:
            path = guide_paths[agent_id]
            idx = path_index.get(agent_id, 0)
            for i, cell in enumerate(path[idx:]):
                if cell == cand_cell:
                    return 1 + i
        return 100 + distance(cand, goal)
    
    time_steps = 0
    for step in range(max_steps):
        time_steps = step + 1
        active = [i for i in range(n) if not reached[i]]
        if not active:
            break
        
        active.sort(key=lambda i: distance(positions[i], goals[i]))
        
        decided = {}
        reserved = set()
        in_progress = set()
        
        def pibt(agent_id):
            if agent_id in decided:
                return True
            if agent_id in in_progress:
                return False
            
            in_progress.add(agent_id)
            pos = positions[agent_id]
            goal = goals[agent_id]
            
            px, py = int(pos[0]), int(pos[1])
            candidates = [(px, py)] + road.get_valid_moves(px, py)
            
            random.shuffle(candidates)
            candidates.sort(key=lambda c: score_move(agent_id, pos, c, goal))
            
            for cand in candidates:
                cand_cell = (int(cand[0]), int(cand[1]))
                if cand_cell in reserved:
                    continue
                
                blocked_agent = None
                for other in active:
                    if other == agent_id or other in decided:
                        continue
                    other_pos = (int(positions[other][0]), int(positions[other][1]))
                    if other_pos == cand_cell:
                        blocked_agent = other
                        break
                
                if blocked_agent is not None:
                    if not pibt(blocked_agent):
                        continue
                
                decided[agent_id] = cand_cell
                reserved.add(cand_cell)
                in_progress.discard(agent_id)
                return True
            
            decided[agent_id] = (px, py)
            reserved.add((px, py))
            in_progress.discard(agent_id)
            return False
        
        for agent in active:
            pibt(agent)
        
        for i in active:
            if i in decided:
                nx, ny = decided[i]
                positions[i][0], positions[i][1] = nx, ny
                
                gx, gy = int(goals[i][0]), int(goals[i][1])
                if abs(nx - gx) <= 1 and abs(ny - gy) <= 1:
                    reached[i] = True
            times[i] += 1
    
    completed = sum(reached)
    total_delay = sum(times)
    
    return {
        "total_delay": total_delay,
        "total_agent_time": total_delay,
        "avg_travel_time": total_delay / n if n else float("inf"),
        "completed": completed,
        "total": n,
        "completion_rate": completed / n if n else 0,
        "time_steps": time_steps,
    }


# =============================================================================
# 3x3 GRID: Runner Functions
# =============================================================================

def compute_ideal_3x3(road, origins, destinations):
    """Compute ideal total delay for 3x3 grid."""
    total_ideal = 0
    n = len(origins)
    for i in range(n):
        start = (int(origins[i][0]), int(origins[i][1]))
        goal = (int(destinations[i][0]), int(destinations[i][1]))
        path, _ = road.shortest_path(start, goal)
        if path:
            total_ideal += len(path) - 1
        else:
            total_ideal += 1000
    return {"total_delay": total_ideal, "completion_rate": 1.0}


def run_gsp_3x3(road, origins, destinations, max_steps=500):
    """Run GSP on 3x3 grid."""
    n = len(origins)
    positions = [list(o) for o in origins]
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    moves = [0] * n
    prev_moves = {}
    history = {i: [] for i in range(n)}
    
    for step in range(max_steps):
        if all(reached):
            break
        prev_moves, history, moves = step_gsp(road, positions, goals, reached, times, set(), prev_moves, history, moves)
    
    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    
    return {
        "total_delay": sum(times),
        "completion_rate": sum(reached) / n if n else 0,
    }


def run_pibt_3x3(road, origins, destinations, max_steps=500):
    """Run PIBT on 3x3 grid."""
    n = len(origins)
    positions = [list(o) for o in origins]
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    moves = [0] * n
    history = {i: [] for i in range(n)}
    planner = PIBTPlanner(road)
    
    for step in range(max_steps):
        if all(reached):
            break
        history, moves = step_pibt(road, planner, positions, goals, reached, times, history, moves)
    
    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    
    return {
        "total_delay": sum(times),
        "completion_rate": sum(reached) / n if n else 0,
    }


def run_graph_pibt_3x3(road, origins, destinations, max_steps=500):
    """Run Graph-PIBT on 3x3 grid (PIBT with graph/shortest-path distance, like Manhattan)."""
    n = len(origins)
    positions = [list(o) for o in origins]
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    moves = [0] * n
    history = {i: [] for i in range(n)}
    planner = GraphPIBTPlanner(road)
    
    for step in range(max_steps):
        if all(reached):
            break
        history, moves = step_pibt(road, planner, positions, goals, reached, times, history, moves)
    
    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    
    return {
        "total_delay": sum(times),
        "completion_rate": sum(reached) / n if n else 0,
    }


def run_paper_pibt_3x3(road, origins, destinations, max_steps=500):
    """Run Paper-PIBT on 3x3 grid (Okumura et al.: time-based priority + graph distance for moves)."""
    n = len(origins)
    positions = [list(o) for o in origins]
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    moves = [0] * n
    history = {i: [] for i in range(n)}
    planner = PaperPIBTPlanner(road)
    
    for step in range(max_steps):
        if all(reached):
            break
        history, moves = step_pibt(road, planner, positions, goals, reached, times, history, moves)
    
    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    
    return {
        "total_delay": sum(times),
        "completion_rate": sum(reached) / n if n else 0,
    }


def run_sp_3x3(road, origins, destinations, max_steps=500):
    """Run SP on 3x3 grid."""
    n = len(origins)
    positions = [list(o) for o in origins]
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    moves = [0] * n
    paths = {i: None for i in range(n)}
    
    for step in range(max_steps):
        if all(reached):
            break
        paths, moves = step_sp(road, positions, goals, reached, times, set(), paths, moves)
    
    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    
    return {
        "total_delay": sum(times),
        "completion_rate": sum(reached) / n if n else 0,
    }


def run_sp_pibt_3x3(road, origins, destinations, max_steps=500):
    """Run SP-PIBT on 3x3 grid."""
    n = len(origins)
    positions = [list(o) for o in origins]
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    moves = [0] * n
    history = {i: [] for i in range(n)}
    initialized = {'done': False}
    planner = SPGuidedPIBT(road)
    
    for step in range(max_steps):
        if all(reached):
            break
        history, initialized, moves = step_sp_guided_pibt(
            road, planner, positions, goals, reached, times, history, initialized, moves
        )
    
    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    
    return {
        "total_delay": sum(times),
        "completion_rate": sum(reached) / n if n else 0,
    }


def run_guided_pibt_3x3(road, origins, destinations, max_steps=500):
    """Run Guided-PIBT on 3x3 grid."""
    n = len(origins)
    positions = [list(o) for o in origins]
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    moves = [0] * n
    history = {i: [] for i in range(n)}
    initialized = [False]
    planner = GuidedPIBTCongestion(road)
    
    for step in range(max_steps):
        if all(reached):
            break
        history, initialized, moves = step_guided_pibt(
            road, planner, positions, goals, reached, times, history, initialized, moves
        )
    
    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    
    return {
        "total_delay": sum(times),
        "completion_rate": sum(reached) / n if n else 0,
    }


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def generate_lane_agents_4way_left_bias(
    road,
    num_agents: int,
    seed: int,
    left_turn_ratio: float = 0.3,
):
    """
    Generate agents with specified left turn ratio.
    
    Args:
        road: RoadNetwork4Way instance
        num_agents: Number of agents to generate
        seed: Random seed
        left_turn_ratio: Fraction of agents that should turn left (default 30%)
    
    Returns:
        (origins, destinations, entry_arms, turn_types)
    """
    rng = np.random.default_rng(seed)
    origins, destinations = [], []
    entry_arms, turn_types = [], []
    per_arm = max(1, num_agents // 4)
    
    for entry_arm in range(4):
        n = per_arm if entry_arm < 3 else (num_agents - 3 * per_arm)
        for _ in range(n):
            # Determine turn type based on left_turn_ratio
            r = rng.random()
            if r < left_turn_ratio:
                turn_type = 2  # TURN_LEFT
            elif r < left_turn_ratio + (1 - left_turn_ratio) / 2:
                turn_type = 1  # TURN_STRAIGHT
            else:
                turn_type = 0  # TURN_RIGHT
            
            dest_arm = (entry_arm + 1 + turn_type) % 4
            orig_cells = road.get_spawn_cells_inbound(entry_arm, 4)
            dest_cells = road.get_dest_cells_outbound(dest_arm, 4)
            o = tuple(int(x) for x in orig_cells[rng.integers(0, len(orig_cells))])
            d = tuple(int(x) for x in dest_cells[rng.integers(0, len(dest_cells))])
            origins.append(o)
            destinations.append(d)
            entry_arms.append(entry_arm)
            turn_types.append(turn_type)
    
    return origins, destinations, entry_arms, turn_types


def run_4way_experiments(seeds=[42, 123, 456], max_steps=5000):
    """Run 4-way intersection experiments varying intersection size and agent count."""
    
    print("=" * 120)
    print("SCENARIO 1: FOUR-WAY 2-IN-1-OUT INTERSECTION")
    print("=" * 120)
    print()
    
    # Experiment configurations - smaller intersections as requested
    # Small: 5x5 (half=2), Medium: 7x7 (half=3), Large: 11x11 (half=5)
    intersection_configs = [
        (2, "Small (5×5)"),
        (3, "Medium (7×7)"),
        (5, "Large (11×11)"),
    ]
    agent_counts = [24]  # Fixed at 24 agents
    grid_size = 64
    left_turn_ratio = 0.3  # 30% left turns
    
    results = {}
    
    print(f"Configuration: 24 agents, 30% left turns")
    print()
    
    # Paper Table: IDEAL, SP (paper reports SP-Mod for large intersection row), PIBT, G-PIBT, GLC, Circular.
    # Use SP-Mod for the "SP" column to match paper (SP-Mod 1879 (+32%) for 11×11).
    print(f"{'Size':<14} {'IDEAL':<7} {'SP':<12} {'PIBT':<12} {'G-PIBT':<12} {'GLC':<12} {'Circular':<12} {'Winner':<10}")
    print("-" * 95)
    
    for int_half, size_name in intersection_configs:
        road = RoadNetwork4Way(grid_size=grid_size, intersection_half=int_half)
        
        for num_agents in agent_counts:
            ideal_delays = []
            sp_delays = []   # SP-Mod (paper labels as "SP" in table)
            pibt_delays = []   # Paper-PIBT
            gpibt_delays = []
            glc_delays = []
            circular_delays = []
            
            for seed in seeds:
                origins, destinations, _, _ = generate_lane_agents_4way_left_bias(
                    road, num_agents, seed, left_turn_ratio
                )
                ideal_r = compute_ideal_lane(road, origins, destinations)
                spmod_r = run_sp_modified_4way(road, origins, destinations, max_steps)  # Paper: "SP" row = SP-Mod
                pibt_r = run_paper_pibt_lane(road, origins, destinations, max_steps)
                gpibt_r = run_guided_pibt_4way(road, origins, destinations, max_steps)
                glc_r = run_sp_pibt_4way(road, origins, destinations, max_steps)
                circular_r = run_circular_lane(road, origins, destinations, max_steps)
                ideal_delays.append(ideal_r["total_agent_time"])
                sp_delays.append(spmod_r["total_delay"])
                pibt_delays.append(pibt_r["total_agent_time"])
                gpibt_delays.append(gpibt_r["total_delay"])
                glc_delays.append(glc_r["total_delay"])
                circular_delays.append(circular_r["total_agent_time"])
            
            ideal_avg = np.mean(ideal_delays)
            sp_avg = np.mean(sp_delays)
            pibt_avg = np.mean(pibt_delays)
            gpibt_avg = np.mean(gpibt_delays)
            glc_avg = np.mean(glc_delays)
            circular_avg = np.mean(circular_delays)
            
            def fmt_with_overhead(val, ideal):
                oh = (val / ideal - 1) * 100 if ideal > 0 else float('inf')
                return f"{val:.0f}(+{oh:.0f}%)"
            method_delays = {
                "SP": sp_avg, "PIBT": pibt_avg, "G-PIBT": gpibt_avg, "GLC": glc_avg, "Circular": circular_avg
            }
            winner = min(method_delays, key=method_delays.get)
            print(f"{size_name:<14} {ideal_avg:<7.0f} {fmt_with_overhead(sp_avg, ideal_avg):<12} {fmt_with_overhead(pibt_avg, ideal_avg):<12} {fmt_with_overhead(gpibt_avg, ideal_avg):<12} {fmt_with_overhead(glc_avg, ideal_avg):<12} {fmt_with_overhead(circular_avg, ideal_avg):<12} {winner:<10}")
            results[(int_half, num_agents)] = {
                "ideal": ideal_avg, "sp": sp_avg, "pibt": pibt_avg, "gpibt": gpibt_avg, "glc": glc_avg,
                "circular": circular_avg, "winner": winner, "size_name": size_name
            }
    
    return results


def run_sp_modified_3x3(road, origins, destinations, max_steps=500):
    """Run SP-Modified on 3x3 grid with deadlock detection."""
    import random
    
    n = len(origins)
    positions = [list(o) for o in origins]
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    moves = [0] * n
    paths = {i: None for i in range(n)}
    consecutive_wait = [0] * n
    
    STUCK_THRESHOLD = 50  # Only intervene after 50 consecutive waiting steps
    
    def distance(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    for step in range(max_steps):
        if all(reached):
            break
        
        active = [i for i in range(n) if not reached[i]]
        active.sort(key=lambda i: (distance(positions[i], goals[i]), random.random() * 0.01))
        
        # Build occupied set
        occupied = set()
        for i in range(n):
            if not reached[i]:
                occupied.add((int(positions[i][0]), int(positions[i][1])))
        
        next_pos = [None] * n
        local_reserved = set()
        
        for i in active:
            px, py = int(positions[i][0]), int(positions[i][1])
            gx, gy = int(goals[i][0]), int(goals[i][1])
            
            # Check if reached goal
            if (px, py) == (gx, gy):
                reached[i] = True
                next_pos[i] = (gx, gy)
                local_reserved.add((gx, gy))
                times[i] += 1
                consecutive_wait[i] = 0
                continue
            
            # Compute path if not available
            if paths.get(i) is None:
                full_path, _ = road.shortest_path((px, py), (gx, gy))
                if full_path and len(full_path) > 1:
                    paths[i] = full_path[1:]
                else:
                    paths[i] = []
            
            moved = False
            blocked_cells = local_reserved | (occupied - {(px, py)})
            
            # Try to follow the planned path
            if paths[i]:
                next_cell = paths[i][0]
                if next_cell not in blocked_cells:
                    next_pos[i] = next_cell
                    local_reserved.add(next_cell)
                    paths[i] = paths[i][1:]
                    consecutive_wait[i] = 0
                    moves[i] += 1
                    moved = True
            
            # If stuck for too long, try any valid move toward goal
            if not moved and consecutive_wait[i] >= STUCK_THRESHOLD:
                valid_moves = road.get_valid_moves(px, py)
                valid_moves = [m for m in valid_moves if m != (px, py) and m not in blocked_cells]
                if valid_moves:
                    best_move = min(valid_moves, key=lambda m: distance(m, (gx, gy)))
                    next_pos[i] = best_move
                    local_reserved.add(best_move)
                    paths[i] = None
                    consecutive_wait[i] = 0
                    moves[i] += 1
                    moved = True
            
            if not moved:
                next_pos[i] = (px, py)
                local_reserved.add((px, py))
                consecutive_wait[i] += 1
            
            times[i] += 1
        
        # Apply moves
        for i in range(n):
            if next_pos[i]:
                positions[i][0], positions[i][1] = next_pos[i]
    
    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    
    return {
        "total_delay": sum(times),
        "completion_rate": sum(reached) / n if n else 0,
    }


def run_3x3_experiments(seeds=[42, 123, 456], max_steps=500):
    """Run 3x3 grid experiments varying agent count."""
    
    print("\n")
    print("=" * 135)
    print("SCENARIO 2: 3x3 DIRECTED GRID")
    print("=" * 135)
    print()
    
    agent_counts = [8, 12, 16, 20, 24]
    grid_size = 80
    road = GridNetwork3x3(grid_size=grid_size)
    
    print(f"{'Agents':<8} {'IDEAL':<7} {'SP':<12} {'PIBT(E)':<12} {'Graph-PIBT':<12} {'Paper-PIBT':<12} {'GSP':<12} {'G-PIBT':<12} {'SP-PIBT':<12} {'Winner':<10}")
    print("-" * 120)
    
    results = {}
    
    for num_agents in agent_counts:
        ideal_delays = []
        sp_delays = []
        spmod_delays = []
        pibt_delays = []
        graph_pibt_delays = []
        paper_pibt_delays = []
        gsp_delays = []
        gpibt_delays = []
        sppibt_delays = []
        
        for seed in seeds:
            origins, destinations, _ = generate_left_turn_agents(road, num_agents, seed)
            
            ideal_r = compute_ideal_3x3(road, origins, destinations)
            sp_r = run_sp_3x3(road, origins, destinations, max_steps)
            spmod_r = run_sp_modified_3x3(road, origins, destinations, max_steps)
            pibt_r = run_pibt_3x3(road, origins, destinations, max_steps)
            graph_pibt_r = run_graph_pibt_3x3(road, origins, destinations, max_steps)
            paper_pibt_r = run_paper_pibt_3x3(road, origins, destinations, max_steps)
            gsp_r = run_gsp_3x3(road, origins, destinations, max_steps)
            gpibt_r = run_guided_pibt_3x3(road, origins, destinations, max_steps)
            sppibt_r = run_sp_pibt_3x3(road, origins, destinations, max_steps)
            
            ideal_delays.append(ideal_r["total_delay"])
            sp_delays.append(sp_r["total_delay"])
            spmod_delays.append(spmod_r["total_delay"])
            pibt_delays.append(pibt_r["total_delay"])
            graph_pibt_delays.append(graph_pibt_r["total_delay"])
            paper_pibt_delays.append(paper_pibt_r["total_delay"])
            gsp_delays.append(gsp_r["total_delay"])
            gpibt_delays.append(gpibt_r["total_delay"])
            sppibt_delays.append(sppibt_r["total_delay"])
        
        # Compute averages
        ideal_avg = np.mean(ideal_delays)
        sp_avg = np.mean(sp_delays)
        spmod_avg = np.mean(spmod_delays)
        pibt_avg = np.mean(pibt_delays)
        graph_pibt_avg = np.mean(graph_pibt_delays)
        paper_pibt_avg = np.mean(paper_pibt_delays)
        gsp_avg = np.mean(gsp_delays)
        gpibt_avg = np.mean(gpibt_delays)
        sppibt_avg = np.mean(sppibt_delays)
        
        # Compute overheads
        def fmt_with_overhead(val, ideal):
            oh = (val / ideal - 1) * 100 if ideal > 0 else float('inf')
            return f"{val:.0f}(+{oh:.0f}%)"
        
        # Find winner (excluding IDEAL)
        method_delays = {
            "SP": sp_avg, "PIBT(E)": pibt_avg, "Graph-PIBT": graph_pibt_avg, "Paper-PIBT": paper_pibt_avg,
            "GSP": gsp_avg, "G-PIBT": gpibt_avg, "SP-PIBT": sppibt_avg
        }
        winner = min(method_delays, key=method_delays.get)
        
        print(f"{num_agents:<8} {ideal_avg:<7.0f} {fmt_with_overhead(sp_avg, ideal_avg):<12} {fmt_with_overhead(pibt_avg, ideal_avg):<12} {fmt_with_overhead(graph_pibt_avg, ideal_avg):<12} {fmt_with_overhead(paper_pibt_avg, ideal_avg):<12} {fmt_with_overhead(gsp_avg, ideal_avg):<12} {fmt_with_overhead(gpibt_avg, ideal_avg):<12} {fmt_with_overhead(sppibt_avg, ideal_avg):<12} {winner:<10}")
        
        results[num_agents] = {
            "ideal": ideal_avg, "sp": sp_avg, "spmod": spmod_avg, "pibt": pibt_avg,
            "graph_pibt": graph_pibt_avg, "paper_pibt": paper_pibt_avg,
            "gsp": gsp_avg, "gpibt": gpibt_avg, "sppibt": sppibt_avg,
            "winner": winner
        }
    
    return results


def run_3x3_limited(
    agent_counts=None,
    seeds=(42, 123, 456),
    max_steps=500,
):
    """Run 3x3 grid: IDEAL, SP, PIBT (Paper-PIBT), G-PIBT, GLC. Paper scenario 2; Table 2 uses agents 8,16,24,32,40,60."""
    if agent_counts is None:
        agent_counts = [8, 16, 24, 32, 40, 60]
    grid_size = 80
    road = GridNetwork3x3(grid_size=grid_size)
    print("\n")
    print("=" * 95)
    print("SCENARIO 2: 3×3 DIRECTED GRID (IDEAL, SP, PIBT, G-PIBT, GLC)")
    print("=" * 95)
    print()
    print(f"{'Agents':<8} {'IDEAL':<7} {'SP':<14} {'PIBT':<14} {'G-PIBT':<14} {'GLC':<14} {'Winner':<10}")
    print("-" * 95)
    results = {}
    for num_agents in agent_counts:
        ideal_delays, sp_delays, paper_pibt_delays, gpibt_delays, sppibt_delays = [], [], [], [], []
        for seed in seeds:
            origins, destinations, _ = generate_left_turn_agents(road, num_agents, seed)
            ideal_delays.append(compute_ideal_3x3(road, origins, destinations)["total_delay"])
            sp_delays.append(run_sp_3x3(road, origins, destinations, max_steps)["total_delay"])
            paper_pibt_delays.append(run_paper_pibt_3x3(road, origins, destinations, max_steps)["total_delay"])
            gpibt_delays.append(run_guided_pibt_3x3(road, origins, destinations, max_steps)["total_delay"])
            sppibt_delays.append(run_sp_pibt_3x3(road, origins, destinations, max_steps)["total_delay"])
        ideal_avg = np.mean(ideal_delays)
        sp_avg = np.mean(sp_delays)
        paper_pibt_avg = np.mean(paper_pibt_delays)
        gpibt_avg = np.mean(gpibt_delays)
        sppibt_avg = np.mean(sppibt_delays)

        def fmt(val, ideal):
            oh = (val / ideal - 1) * 100 if ideal > 0 else float("inf")
            return f"{val:.0f}(+{oh:.0f}%)"
        method_delays = {"SP": sp_avg, "PIBT": paper_pibt_avg, "G-PIBT": gpibt_avg, "GLC": sppibt_avg}
        winner = min(method_delays, key=method_delays.get)
        print(f"{num_agents:<8} {ideal_avg:<7.0f} {fmt(sp_avg, ideal_avg):<14} {fmt(paper_pibt_avg, ideal_avg):<14} {fmt(gpibt_avg, ideal_avg):<14} {fmt(sppibt_avg, ideal_avg):<14} {winner:<10}")
        results[num_agents] = {"ideal": ideal_avg, "sp": sp_avg, "paper_pibt": paper_pibt_avg, "gpibt": gpibt_avg, "sppibt": sppibt_avg, "winner": winner}
    return results


def print_summary(results_4way, results_3x3):
    """Print summary of all experiments."""
    
    print("\n")
    print("=" * 120)
    print("SUMMARY: TOTAL DELAY COMPARISON")
    print("=" * 120)
    
    print("""
Methods:
  IDEAL:    Theoretical lower bound (no blocking, sum of shortest paths)
  SP:       Shortest Path + waiting (BFS, wait if blocked)
  PIBT:     Paper-PIBT (time-based priority)
  G-PIBT:   Guided-PIBT (congestion-aware paths, AAAI 2024)
  GLC:      [Ours] BFS optimal paths + PIBT collision resolution
  Circular: Roundabout policy (4-way only)

Metric: Total Delay = sum of arrival timesteps (∑Tᵢ). Lower is better.
""")
    
    # Count wins (4-way: SP, PIBT, G-PIBT, GLC, Circular; 3x3 limited: SP, PIBT, G-PIBT, GLC)
    wins_4way = {"SP": 0, "PIBT": 0, "G-PIBT": 0, "GLC": 0, "Circular": 0}
    wins_3x3 = {"SP": 0, "PIBT": 0, "G-PIBT": 0, "GLC": 0}
    
    for key, val in results_4way.items():
        wins_4way[val["winner"]] += 1
    for key, val in results_3x3.items():
        wins_3x3[val["winner"]] += 1
    
    print("Win counts (lowest total delay):")
    print(f"  4-way intersection: {wins_4way}")
    print(f"  3×3 grid:           {wins_3x3}")
    
    all_methods = set(wins_4way) | set(wins_3x3)
    total_wins = {k: wins_4way.get(k, 0) + wins_3x3.get(k, 0) for k in all_methods}
    print(f"  Total:              {total_wins}")
    
    overall_winner = max(total_wins, key=total_wins.get)
    print(f"\nOverall best method: {overall_winner}")


def run_3x3_only():
    """Run only 3×3 grid experiments (paper: IDEAL, SP, PIBT, G-PIBT, GLC)."""
    print("\n" + "=" * 135)
    print("3×3 DIRECTED GRID EXPERIMENT")
    print("Metric: Total Delay (sum of arrival timesteps)")
    print("=" * 135)
    
    seeds = [42, 123, 456, 789, 1024]
    results_3x3 = run_3x3_limited(seeds=seeds, max_steps=500)
    
    print("\n" + "=" * 135)
    print("METHODS: IDEAL, SP, PIBT (Paper-PIBT), G-PIBT, GLC")
    print("=" * 135)
    
    return results_3x3


def run_4way_only():
    """Run only 4-way intersection experiments."""
    print("\n" + "=" * 120)
    print("FOUR-WAY INTERSECTION EXPERIMENT")
    print("Metric: Total Delay (sum of arrival timesteps)")
    print("Settings: 24 agents, 30% left turns")
    print("=" * 120)
    
    seeds = [42, 123, 456, 789, 1024]
    results_4way = run_4way_experiments(seeds=seeds, max_steps=5000)
    
    print("\n" + "=" * 145)
    print("METHODS: IDEAL, SP, PIBT (Paper-PIBT), G-PIBT, GLC, Circular. Default: 24 agents, 30% left, 35% straight, 35% right.")
    print("=" * 145)
    
    return results_4way


def main():
    """Run all paper experiments: 4-way (IDEAL, SP, PIBT, G-PIBT, GLC, Circular) and 3×3 (IDEAL, SP, PIBT, G-PIBT, GLC)."""
    print("\n" + "=" * 120)
    print("PAPER EXPERIMENT SUMMARY")
    print("Metric: Total Delay (sum of arrival timesteps). PIBT = Paper-PIBT. Our method = GLC.")
    print("=" * 120)
    
    # Paper: "mean ± standard deviation over 5 random seeds"
    seeds = [42, 123, 456, 789, 1024]
    
    results_4way = run_4way_experiments(seeds=seeds, max_steps=5000)
    results_3x3 = run_3x3_limited(seeds=seeds, max_steps=500)
    
    print_summary(results_4way, results_3x3)


if __name__ == "__main__":
    main()
