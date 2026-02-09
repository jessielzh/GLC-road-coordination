"""
4-way intersection on a real road network.
- RoadNetwork: one lane per arm.
- RoadNetwork4Way: two lanes per arm (inbound/outbound), left/straight/right turns, right-hand driving.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from road_network import RoadNetwork, RoadNetwork4Way, TURN_LEFT, TURN_RIGHT, TURN_STRAIGHT


def generate_lane_agents(
    road: RoadNetwork,
    num_agents: int,
    seed: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Legacy: one lane per arm. Origins/destinations on any arm (different arm for dest).
    """
    rng = np.random.default_rng(seed)
    spawn_zones = [
        road.get_spawn_cells_north(4),
        road.get_spawn_cells_south(4),
        road.get_spawn_cells_east(4),
        road.get_spawn_cells_west(4),
    ]
    dest_zones = [
        road.get_dest_cells_north(4),
        road.get_dest_cells_south(4),
        road.get_dest_cells_east(4),
        road.get_dest_cells_west(4),
    ]
    origins, destinations = [], []
    per_arm = max(1, num_agents // 4)
    for entry_arm in range(4):
        n = per_arm if entry_arm < 3 else (num_agents - 3 * per_arm)
        for _ in range(n):
            orig_cells = spawn_zones[entry_arm]
            dest_arm = rng.integers(0, 4)
            while dest_arm == entry_arm:
                dest_arm = rng.integers(0, 4)
            dest_cells = dest_zones[dest_arm]
            o = tuple(int(x) for x in orig_cells[rng.integers(0, len(orig_cells))])
            d = tuple(int(x) for x in dest_cells[rng.integers(0, len(dest_cells))])
            origins.append(o)
            destinations.append(d)
    return origins, destinations


def generate_lane_agents_4way(
    road: RoadNetwork4Way,
    num_agents: int,
    seed: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[int], List[int]]:
    """
    Two lanes per arm, right-hand driving. Vehicles spawn on inbound lanes.
    Each vehicle picks turn: left, straight, or right; destination is on the corresponding outbound lane.
    Returns (origins, destinations, entry_arms, turn_types).
    entry_arm in {0=N, 1=E, 2=S, 3=W}, turn_type in {TURN_RIGHT, TURN_STRAIGHT, TURN_LEFT}.
    """
    rng = np.random.default_rng(seed)
    origins, destinations = [], []
    entry_arms, turn_types = [], []
    per_arm = max(1, num_agents // 4)
    for entry_arm in range(4):
        n = per_arm if entry_arm < 3 else (num_agents - 3 * per_arm)
        for _ in range(n):
            turn_type = int(rng.integers(0, 3))  # 0=right, 1=straight, 2=left
            dest_arm = (entry_arm + 1 + turn_type) % 4  # right=(+1), straight=(+2), left=(+3)
            orig_cells = road.get_spawn_cells_inbound(entry_arm, 4)
            dest_cells = road.get_dest_cells_outbound(dest_arm, 4)
            o = tuple(int(x) for x in orig_cells[rng.integers(0, len(orig_cells))])
            d = tuple(int(x) for x in dest_cells[rng.integers(0, len(dest_cells))])
            origins.append(o)
            destinations.append(d)
            entry_arms.append(entry_arm)
            turn_types.append(turn_type)
    return origins, destinations, entry_arms, turn_types


def step_gsp_lane_once(
    road: RoadNetwork,
    positions: List[List[int]],
    goals: List[Tuple[int, int]],
    reached: List[bool],
    times: List[int],
) -> None:
    """Perform one GSP step; updates positions, reached, times in place.
    
    For RoadNetwork4Way, respects right-hand driving:
    - On INBOUND lane: can continue along lane toward intersection, or enter intersection
    - In INTERSECTION: can only exit to OUTBOUND lanes (never to inbound lanes)
    - On OUTBOUND lane: continue along lane away from intersection
    """
    n = len(positions)
    active = [i for i in range(n) if not reached[i]]
    if not active:
        return
    active.sort(key=lambda i: road.distance((positions[i][0], positions[i][1]), goals[i]))
    next_positions = [None] * n
    reserved = set()
    
    # Check if this is a 4-way network with lane direction support
    has_lane_direction = hasattr(road, 'is_outbound_lane') and hasattr(road, 'is_inbound_lane')
    
    for i in active:
        px, py = positions[i][0], positions[i][1]
        gx, gy = goals[i]
        if abs(px - gx) <= 1 and abs(py - gy) <= 1:
            reached[i] = True
            next_positions[i] = (gx, gy)
            reserved.add((gx, gy))
            times[i] += 1
            continue
        raw_neighbors = road.get_road_neighbors(px, py)
        
        if has_lane_direction:
            in_int = road.in_intersection(px, py)
            on_inbound = road.is_inbound_lane(px, py)
            on_outbound = road.is_outbound_lane(px, py)
            
            valid_neighbors = []
            for (nx, ny) in raw_neighbors:
                if (nx, ny) in reserved:
                    continue
                c_in_int = road.in_intersection(nx, ny)
                c_inbound = road.is_inbound_lane(nx, ny)
                c_outbound = road.is_outbound_lane(nx, ny)
                
                if in_int:
                    # From intersection: only allow intersection cells or outbound lanes
                    if c_in_int or c_outbound:
                        valid_neighbors.append((nx, ny))
                elif on_inbound:
                    # From inbound lane: allow same lane (inbound) or intersection
                    if c_inbound or c_in_int:
                        valid_neighbors.append((nx, ny))
                elif on_outbound:
                    # From outbound lane: allow same lane (outbound) or intersection
                    if c_outbound or c_in_int:
                        valid_neighbors.append((nx, ny))
                else:
                    valid_neighbors.append((nx, ny))
        else:
            valid_neighbors = [(nx, ny) for (nx, ny) in raw_neighbors if (nx, ny) not in reserved]
        
        best = None
        best_dist = float("inf")
        for (nx, ny) in valid_neighbors:
            d = road.distance((nx, ny), goals[i])
            if d < best_dist:
                best_dist = d
                best = (nx, ny)
        if best is not None:
            next_positions[i] = best
            reserved.add(best)
        times[i] += 1
    for i in range(n):
        if next_positions[i] is not None:
            positions[i][0], positions[i][1] = next_positions[i][0], next_positions[i][1]


def _angle_from_center(cell: Tuple[int, int], cx: float, cy: float) -> float:
    """Angle from center to cell, in [0, 2*pi). CCW = increasing angle (east=0)."""
    a = math.atan2(cell[1] - cy, cell[0] - cx)
    return (a + 2 * math.pi) % (2 * math.pi)


def step_circular_lane_once(
    road: RoadNetwork4Way,
    positions: List[List[int]],
    goals: List[Tuple[int, int]],
    reached: List[bool],
    times: List[int],
    exiting: List[bool],
) -> None:
    """One step of circular (roundabout) policy: in intersection, circulate CCW then exit toward goal.
    
    Lane rules (right-hand driving):
    - On INBOUND lane: can continue along lane toward intersection, or enter intersection
    - In INTERSECTION: can only exit to OUTBOUND lanes (never to inbound lanes)
    - On OUTBOUND lane: continue along lane away from intersection
    """
    n = len(positions)
    active = [i for i in range(n) if not reached[i]]
    if not active:
        return
    cx, cy = road.center, road.center
    # Priority: closer to goal first (so exiting agents clear the way)
    # Use Euclidean distance for speed (path_distance is too slow with BFS)
    active.sort(key=lambda i: road.distance((positions[i][0], positions[i][1]), goals[i]))
    next_positions = [None] * n
    reserved = set()
    for i in active:
        px, py = positions[i][0], positions[i][1]
        gx, gy = goals[i]
        if abs(px - gx) <= 1 and abs(py - gy) <= 1:
            reached[i] = True
            next_positions[i] = (gx, gy)
            reserved.add((gx, gy))
            times[i] += 1
            exiting[i] = True
            continue
        
        raw_neighbors = road.get_road_neighbors(px, py)
        in_int = road.in_intersection(px, py)
        on_inbound = road.is_inbound_lane(px, py)
        on_outbound = road.is_outbound_lane(px, py)
        
        # Filter neighbors based on current position and lane rules
        neighbors = []
        for c in raw_neighbors:
            if c in reserved:
                continue
            c_in_int = road.in_intersection(c[0], c[1])
            c_inbound = road.is_inbound_lane(c[0], c[1])
            c_outbound = road.is_outbound_lane(c[0], c[1])
            
            if in_int:
                # From intersection: only allow intersection cells or outbound lanes
                if c_in_int or c_outbound:
                    neighbors.append(c)
            elif on_inbound:
                # From inbound lane: allow same lane (inbound) or intersection
                if c_inbound or c_in_int:
                    neighbors.append(c)
            elif on_outbound:
                # From outbound lane: allow same lane (outbound) or intersection (unlikely but ok)
                if c_outbound or c_in_int:
                    neighbors.append(c)
            else:
                # Fallback (shouldn't happen)
                neighbors.append(c)
        
        best = None
        
        if in_int and not exiting[i]:
            cur_angle = _angle_from_center((px, py), cx, cy)
            exit_angle = _angle_from_center((gx, gy), cx, cy)
            # CCW: we've "passed" exit when (cur - exit) mod 2pi is in (0, pi)
            diff = (cur_angle - exit_angle + 2 * math.pi) % (2 * math.pi)
            if diff <= 0.4 or diff >= 2 * math.pi - 0.4:
                exiting[i] = True
            if exiting[i]:
                # When exiting from intersection, only allow outbound lanes or staying in intersection
                valid_exits = [c for c in neighbors if road.in_intersection(c[0], c[1]) or road.is_outbound_lane(c[0], c[1])]
                if valid_exits:
                    best = min(valid_exits, key=lambda c: road.distance(c, goals[i]))
                else:
                    # Fallback: wait in place
                    best = (px, py) if (px, py) not in reserved else None
            else:
                # Circulate CCW: choose neighbor that advances angle the most (stay in intersection)
                best = None
                best_angle = -1
                for c in neighbors:
                    if not road.in_intersection(c[0], c[1]):
                        continue
                    a = _angle_from_center(c, cx, cy)
                    da = (a - cur_angle + 2 * math.pi) % (2 * math.pi)
                    if da > best_angle:
                        best_angle = da
                        best = c
                if best is None:
                    # Can't advance CCW, try to stay in intersection or wait
                    in_int_neighbors = [c for c in neighbors if road.in_intersection(c[0], c[1])]
                    if in_int_neighbors:
                        best = min(in_int_neighbors, key=lambda c: road.distance(c, goals[i]))
                    else:
                        best = (px, py) if (px, py) not in reserved else None
        elif in_int and exiting[i]:
            # Already marked as exiting, continue toward goal but only via outbound lanes
            valid_exits = [c for c in neighbors if road.in_intersection(c[0], c[1]) or road.is_outbound_lane(c[0], c[1])]
            if valid_exits:
                best = min(valid_exits, key=lambda c: road.distance(c, goals[i]))
            else:
                best = (px, py) if (px, py) not in reserved else None
        else:
            # On a lane (inbound or outbound): move toward goal along allowed neighbors
            if neighbors:
                best = min(neighbors, key=lambda c: road.distance(c, goals[i]))
        
        if best is not None:
            next_positions[i] = best
            reserved.add(best)
        times[i] += 1
    for i in range(n):
        if next_positions[i] is not None:
            positions[i][0], positions[i][1] = next_positions[i][0], next_positions[i][1]


def run_circular_lane(
    road: RoadNetwork4Way,
    origins: List[Tuple[int, int]],
    destinations: List[Tuple[int, int]],
    max_steps: int = 5000,
) -> Dict:
    """Run circular (roundabout) policy on 4-way: circulate CCW in intersection, then exit to goal."""
    n = len(origins)
    positions = [list(o) for o in origins]
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    exiting = [False] * n
    time_steps = 0
    for step in range(max_steps):
        time_steps = step + 1
        active = [i for i in range(n) if not reached[i]]
        if not active:
            break
        step_circular_lane_once(road, positions, goals, reached, times, exiting)
    completed = sum(1 for r in reached if r)
    total_time = sum(times)
    return {
        "total_agent_time": total_time,
        "avg_travel_time": total_time / n if n else float("inf"),
        "completed": completed,
        "total": n,
        "completion_rate": completed / n if n else 0,
        "time_steps": time_steps,
        "positions": [tuple(p) for p in positions],
        "reached": reached,
    }


def run_gsp_lane(
    road: RoadNetwork,
    origins: List[Tuple[int, int]],
    destinations: List[Tuple[int, int]],
    max_steps: int = 5000,
) -> Dict:
    """Run GSP (greedy straight-line) constrained to the road. One cell per step."""
    n = len(origins)
    positions = [list(o) for o in origins]
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    time_steps = 0

    for step in range(max_steps):
        time_steps = step + 1
        active = [i for i in range(n) if not reached[i]]
        if not active:
            break
        step_gsp_lane_once(road, positions, goals, reached, times)

    completed = sum(1 for r in reached if r)
    total_time = sum(times)
    return {
        "total_agent_time": total_time,
        "avg_travel_time": total_time / n if n else float("inf"),
        "completed": completed,
        "total": n,
        "completion_rate": completed / n if n else 0,
        "time_steps": time_steps,
        "positions": [tuple(p) for p in positions],
        "reached": reached,
    }


def run_pibt_lane(
    road: RoadNetwork,
    origins: List[Tuple[int, int]],
    destinations: List[Tuple[int, int]],
    max_steps: int = 5000,
) -> Dict:
    """Run PIBT constrained to the road."""
    planner = road.create_pibt_planner(collision_radius=0.6)
    if planner is None:
        return run_gsp_lane(road, origins, destinations, max_steps)
    n = len(origins)
    positions = {i: (float(origins[i][0]), float(origins[i][1])) for i in range(n)}
    goals = {i: (float(destinations[i][0]), float(destinations[i][1])) for i in range(n)}
    reached = {i: False for i in range(n)}
    times = {i: 0 for i in range(n)}
    time_steps = 0
    for step in range(max_steps):
        time_steps = step + 1
        active = {i for i in range(n) if not reached[i]}
        if not active:
            break
        active_pos = {i: positions[i] for i in active}
        active_goals = {i: goals[i] for i in active}
        next_pos = planner.plan_step(active_pos, active_goals)
        for i in active:
            positions[i] = next_pos.get(i, positions[i])
            times[i] += 1
            # Use threshold of 2.0 to handle cases where agent ends up on parallel lane
            # (inbound vs outbound are 1 cell apart)
            if planner.distance(positions[i], goals[i]) < 2.0:
                reached[i] = True
    completed = sum(1 for r in reached.values() if r)
    total_time = sum(times.values())
    return {
        "total_agent_time": total_time,
        "avg_travel_time": total_time / n if n else float("inf"),
        "completed": completed,
        "total": n,
        "completion_rate": completed / n if n else 0,
        "time_steps": time_steps,
        "positions": [positions[i] for i in range(n)],
        "reached": [reached[i] for i in range(n)],
    }


def run_paper_pibt_lane(
    road: RoadNetwork,
    origins: List[Tuple[int, int]],
    destinations: List[Tuple[int, int]],
    max_steps: int = 5000,
) -> Dict:
    """Run Paper-PIBT (Okumura et al.: time-based priority) constrained to the road. 4-way only if create_paper_pibt_planner exists."""
    planner = getattr(road, "create_paper_pibt_planner", lambda **kw: None)(collision_radius=0.6)
    if planner is None:
        return run_pibt_lane(road, origins, destinations, max_steps)
    n = len(origins)
    positions = {i: (float(origins[i][0]), float(origins[i][1])) for i in range(n)}
    goals = {i: (float(destinations[i][0]), float(destinations[i][1])) for i in range(n)}
    reached = {i: False for i in range(n)}
    times = {i: 0 for i in range(n)}
    time_steps = 0
    for step in range(max_steps):
        time_steps = step + 1
        active = {i for i in range(n) if not reached[i]}
        if not active:
            break
        active_pos = {i: positions[i] for i in active}
        active_goals = {i: goals[i] for i in active}
        next_pos = planner.plan_step(active_pos, active_goals)
        for i in active:
            positions[i] = next_pos.get(i, positions[i])
            times[i] += 1
            if planner.distance(positions[i], goals[i]) < 2.0:
                reached[i] = True
    completed = sum(1 for r in reached.values() if r)
    total_time = sum(times.values())
    return {
        "total_agent_time": total_time,
        "avg_travel_time": total_time / n if n else float("inf"),
        "completed": completed,
        "total": n,
        "completion_rate": completed / n if n else 0,
        "time_steps": time_steps,
        "positions": [positions[i] for i in range(n)],
        "reached": [reached[i] for i in range(n)],
    }


def compute_ideal_lane(
    road: RoadNetwork4Way,
    origins: List[Tuple[int, int]],
    destinations: List[Tuple[int, int]],
) -> Dict:
    """Compute ideal total time: sum of shortest path lengths assuming no blocking.
    
    This is the theoretical lower bound - the best any algorithm could achieve
    if there were no other agents on the road.
    """
    total_ideal_time = 0
    n = len(origins)
    
    for i in range(n):
        start = (int(origins[i][0]), int(origins[i][1]))
        goal = (int(destinations[i][0]), int(destinations[i][1]))
        
        path, _ = road.shortest_path(start, goal)
        if path:
            # Path length - 1 = number of moves needed (path includes start)
            total_ideal_time += len(path) - 1
        else:
            # No path exists - count as large value
            total_ideal_time += 1000
    
    return {
        "total_agent_time": total_ideal_time,
        "avg_travel_time": total_ideal_time / n if n else 0,
        "completed": n,  # Ideal assumes all complete
        "total": n,
        "completion_rate": 1.0,
        "time_steps": 0,  # Not applicable
    }


def run_sp_lane(
    road: RoadNetwork4Way,
    origins: List[Tuple[int, int]],
    destinations: List[Tuple[int, int]],
    max_steps: int = 5000,
) -> Dict:
    """Run SP (true shortest path with BFS, computed once per agent) on 4-way intersection.
    
    Each agent computes its shortest path once at the start, then follows it.
    If blocked by another agent, waits in place instead of replanning.
    """
    n = len(origins)
    positions = [list(o) for o in origins]
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    paths = {i: None for i in range(n)}  # Paths computed once per agent
    
    time_steps = 0
    for step in range(max_steps):
        time_steps = step + 1
        active = [i for i in range(n) if not reached[i]]
        if not active:
            break
        
        # Sort by distance to goal (closer agents get priority)
        active.sort(key=lambda i: road.distance((positions[i][0], positions[i][1]), goals[i]))
        
        # Build set of currently occupied cells
        occupied = set()
        for i in range(n):
            if not reached[i]:
                occupied.add((int(positions[i][0]), int(positions[i][1])))
        
        next_positions = [None] * n
        local_reserved = set()
        
        for i in active:
            px, py = int(positions[i][0]), int(positions[i][1])
            gx, gy = int(goals[i][0]), int(goals[i][1])
            
            # Check if reached goal (within 1 cell)
            if abs(px - gx) <= 1 and abs(py - gy) <= 1:
                reached[i] = True
                next_positions[i] = (gx, gy)
                local_reserved.add((gx, gy))
                times[i] += 1
                continue
            
            # Compute path if not already computed
            if paths.get(i) is None:
                full_path, _ = road.shortest_path((px, py), (gx, gy))
                if full_path and len(full_path) > 1:
                    # Store path excluding current position
                    paths[i] = full_path[1:]
                else:
                    paths[i] = []
            
            # Get next cell from path
            if paths[i]:
                next_cell = paths[i][0]
                
                # Check if next cell is blocked
                if next_cell not in local_reserved and next_cell not in occupied:
                    # Move to next cell
                    next_positions[i] = next_cell
                    local_reserved.add(next_cell)
                    paths[i] = paths[i][1:]  # Remove from path
                else:
                    # Blocked - wait in place
                    next_positions[i] = (px, py)
                    local_reserved.add((px, py))
            else:
                # No path - stay in place
                next_positions[i] = (px, py)
                local_reserved.add((px, py))
            
            times[i] += 1
        
        # Apply moves
        for i in range(n):
            if next_positions[i]:
                positions[i][0], positions[i][1] = next_positions[i][0], next_positions[i][1]
    
    completed = sum(1 for r in reached if r)
    total_time = sum(times)
    return {
        "total_agent_time": total_time,
        "avg_travel_time": total_time / n if n else float("inf"),
        "completed": completed,
        "total": n,
        "completion_rate": completed / n if n else 0,
        "time_steps": time_steps,
        "positions": [tuple(p) for p in positions],
        "reached": reached,
    }
