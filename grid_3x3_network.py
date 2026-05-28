#!/usr/bin/env python3
"""
3x3 Grid of Tiny Intersections - Can methods learn to detour?

Layout: 3x3 grid of intersections connected by roads.
Each intersection is tiny (3x3 cells) - no room for roundabout.

Key question: Can a method learn that 3 right turns = 1 left turn?
(Michigan left / jughandle concept)

  N       N       N
  |       |       |
W-+--E W-+--E W-+--E
  |       |       |
  S       S       S
  |       |       |
W-+--E W-+--E W-+--E
  |       |       |
  S       S       S
  |       |       |
W-+--E W-+--E W-+--E
  |       |       |
  S       S       S
"""

import pygame
import numpy as np
from typing import List, Tuple, Dict, Set

# Colors
COLORS = {
    "background": (18, 20, 26),
    "road": (70, 75, 85),
    "intersection": (90, 95, 105),
    "vehicle_left": (255, 100, 100),   # Left turn = RED
    "vehicle_right": (100, 255, 150),  # Right turn = GREEN
    "vehicle_through": (100, 180, 255),# Through = BLUE
    "reached": (80, 80, 80),
    "text": (220, 220, 232),
    "panel": (26, 28, 36),
}


class GridNetwork3x3:
    """3x3 grid of tiny intersections connected by 2-lane roads (1 lane each direction)."""
    
    def __init__(self, grid_size=80, intersection_size=3, road_length=10):
        self.grid_size = grid_size
        self.intersection_size = intersection_size  # 3x3 tiny intersection
        self.road_length = road_length  # cells between intersections
        
        # Calculate intersection centers (3x3 grid)
        # Total width: 3 intersections + 2 roads
        total_width = 3 * intersection_size + 2 * road_length
        start = (grid_size - total_width) // 2
        
        self.intersection_centers = []
        for row in range(3):
            for col in range(3):
                cx = start + col * (intersection_size + road_length) + intersection_size // 2
                cy = start + row * (intersection_size + road_length) + intersection_size // 2
                self.intersection_centers.append((cx, cy))
        
        # Store intersection bounds
        self.intersections = []
        half = intersection_size // 2
        for cx, cy in self.intersection_centers:
            self.intersections.append({
                'center': (cx, cy),
                'lo_x': cx - half, 'hi_x': cx + half,
                'lo_y': cy - half, 'hi_y': cy + half,
            })
        
        # Build road cells set
        self._build_roads()
    
    def _build_roads(self):
        """Build 2-lane roads (1 lane per direction) connecting intersections."""
        self.road_cells = set()
        
        # Add intersection cells
        for inter in self.intersections:
            for x in range(inter['lo_x'], inter['hi_x'] + 1):
                for y in range(inter['lo_y'], inter['hi_y'] + 1):
                    self.road_cells.add((x, y))
        
        # Add horizontal roads (2 lanes: right-hand driving)
        # When going EAST, you're on the RIGHT (south) side of road = higher y
        # When going WEST, you're on the RIGHT (north) side of road = lower y
        for row in range(3):
            for col in range(2):
                inter1 = self.intersections[row * 3 + col]
                inter2 = self.intersections[row * 3 + col + 1]
                x_start = inter1['hi_x'] + 1
                x_end = inter2['lo_x']
                cy = inter1['center'][1]
                # 2 lanes: cy-1 (west-bound, north side) and cy (east-bound, south side)
                for x in range(x_start, x_end):
                    self.road_cells.add((x, cy - 1))  # West-bound lane (north side)
                    self.road_cells.add((x, cy))      # East-bound lane (south side)
        
        # Add vertical roads (2 lanes: left=south, right=north)
        for row in range(2):
            for col in range(3):
                inter1 = self.intersections[row * 3 + col]
                inter2 = self.intersections[(row + 1) * 3 + col]
                y_start = inter1['hi_y'] + 1
                y_end = inter2['lo_y']
                cx = inter1['center'][0]
                # 2 lanes: cx-1 (south-bound) and cx (north-bound)
                for y in range(y_start, y_end):
                    self.road_cells.add((cx - 1, y))  # South-bound lane
                    self.road_cells.add((cx, y))      # North-bound lane
    
    def is_on_road(self, x, y):
        return (x, y) in self.road_cells
    
    def in_intersection(self, x, y):
        for inter in self.intersections:
            if inter['lo_x'] <= x <= inter['hi_x'] and inter['lo_y'] <= y <= inter['hi_y']:
                return True
        return False
    
    def get_intersection_id(self, x, y):
        """Return which intersection (0-8) the cell is in, or -1 if not in any."""
        for i, inter in enumerate(self.intersections):
            if inter['lo_x'] <= x <= inter['hi_x'] and inter['lo_y'] <= y <= inter['hi_y']:
                return i
        return -1
    
    def is_horizontal_road(self, x, y):
        """Check if cell is on a horizontal road (not intersection)."""
        if self.in_intersection(x, y):
            return False
        if not self.is_on_road(x, y):
            return False
        # Check if there are horizontal neighbors on road
        return self.is_on_road(x-1, y) or self.is_on_road(x+1, y)
    
    def is_vertical_road(self, x, y):
        """Check if cell is on a vertical road (not intersection)."""
        if self.in_intersection(x, y):
            return False
        if not self.is_on_road(x, y):
            return False
        # Vertical roads have vertical extent
        return self.is_on_road(x, y-1) or self.is_on_road(x, y+1)
    
    def get_lane_direction(self, x, y):
        """
        Get the allowed direction of travel for a cell (right-hand driving).
        
        Right-hand driving means you stay on the RIGHT side of the road:
        - Horizontal: y=cy-1 (north side) goes WEST, y=cy (south side) goes EAST
        - Vertical: x=cx-1 (west side) goes SOUTH, x=cx (east side) goes NORTH
        """
        if self.in_intersection(x, y):
            return 'any'
        
        # Find nearest intersection to determine road type
        for inter in self.intersections:
            cx, cy = inter['center']
            half = self.intersection_size // 2
            
            # Check if on horizontal road near this intersection
            if y == cy - 1 or y == cy:  # On horizontal road level
                if x < inter['lo_x'] or x > inter['hi_x']:  # Outside intersection
                    if y == cy - 1:
                        return 'west'  # North side lane goes west
                    else:
                        return 'east'  # South side lane goes east
            
            # Check if on vertical road near this intersection
            if x == cx - 1 or x == cx:  # On vertical road level
                if y < inter['lo_y'] or y > inter['hi_y']:  # Outside intersection
                    if x == cx - 1:
                        return 'south'  # West side lane goes south
                    else:
                        return 'north'  # East side lane goes north
        
        return 'any'
    
    def get_road_neighbors(self, x, y):
        """Get neighboring cells that are on the road."""
        candidates = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        return [c for c in candidates if self.is_on_road(c[0], c[1])]
    
    def get_valid_moves(self, px, py, reserved=None):
        """Get valid moves respecting STRICT lane discipline (right-hand driving, no backward).
        
        STRICT RULES:
        1. Right-side driving: agents must use correct lane for their direction
        2. No backward driving: agents cannot move backwards on a lane
        
        In intersection: can exit to any lane (intersection allows direction changes)
        On road lane: can ONLY move forward in lane direction OR enter intersection
        """
        if reserved is None:
            reserved = set()
        
        valid = []
        direction = self.get_lane_direction(px, py)
        in_int = self.in_intersection(px, py)
        
        for c in self.get_road_neighbors(px, py):
            if c in reserved:
                continue
            cx, cy = c
            c_in_int = self.in_intersection(cx, cy)
            
            if in_int:
                # In intersection: can move to adjacent intersection cells or exit to road
                # When exiting to road, that lane's direction becomes the agent's new direction
                valid.append(c)
            else:
                # On road: STRICT RULES - only forward movement or enter intersection
                if c_in_int:
                    # Allow entering intersection (for turns)
                    valid.append(c)
                else:
                    # On road to road: MUST follow lane direction (no backward)
                    if direction == 'east' and cx > px and cy == py:
                        valid.append(c)
                    elif direction == 'west' and cx < px and cy == py:
                        valid.append(c)
                    elif direction == 'north' and cy < py and cx == px:
                        valid.append(c)
                    elif direction == 'south' and cy > py and cx == px:
                        valid.append(c)
                    # Note: diagonal moves and backward moves are NOT allowed
        
        return valid
    
    def distance(self, a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    def manhattan_distance(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    
    def shortest_path(self, start, goal, reserved=None):
        """
        Find shortest path from start to goal using BFS.
        Respects lane direction restrictions.
        
        Returns:
            path: list of (x, y) tuples from start to goal, or None if no path
            next_move: the next cell to move to, or None if at goal or no path
        """
        from collections import deque
        
        if reserved is None:
            reserved = set()
        
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        if start == goal:
            return [start], None
        
        if not self.is_on_road(start[0], start[1]):
            return None, None
        if not self.is_on_road(goal[0], goal[1]):
            return None, None
        
        # BFS
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            # Get valid moves from current position (ignore reserved for pathfinding)
            valid_moves = self.get_valid_moves(current[0], current[1])
            
            for next_cell in valid_moves:
                next_cell = (int(next_cell[0]), int(next_cell[1]))
                
                if next_cell in visited:
                    continue
                
                new_path = path + [next_cell]
                
                if next_cell == goal:
                    # Found the goal
                    next_move = new_path[1] if len(new_path) > 1 else None
                    return new_path, next_move
                
                visited.add(next_cell)
                queue.append((next_cell, new_path))
        
        # No path found
        return None, None
    
    def graph_distance(self, a, b):
        """
        Shortest-path (hop) distance from a to b respecting lane directions.
        Same idea as Manhattan road network: distance = number of steps to goal.
        Returns float('inf') if no path exists.
        """
        path, _ = self.shortest_path(a, b)
        if path is None:
            return float('inf')
        return len(path) - 1  # steps = len(path) - 1
    
    def get_next_move_toward_goal(self, start, goal, reserved=None):
        """
        Get the next move toward goal using shortest path.
        If direct path is blocked by reserved cells, find alternative.
        
        Returns:
            next_cell: (x, y) tuple for next move, or None if stuck
        """
        if reserved is None:
            reserved = set()
        
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        if start == goal:
            return None
        
        # First, find the ideal path (ignoring reserved)
        path, next_move = self.shortest_path(start, goal)
        
        if path is None:
            return None
        
        # If next move is not reserved, use it
        if next_move and next_move not in reserved:
            return next_move
        
        # Next move is blocked - find alternative valid move that's on ANY shortest path
        valid_moves = self.get_valid_moves(start[0], start[1], reserved)
        
        if not valid_moves:
            return None
        
        # Score each valid move by how much it helps reach the goal
        best_move = None
        best_path_len = float('inf')
        
        for move in valid_moves:
            move = (int(move[0]), int(move[1]))
            # Find shortest path from this move to goal
            path_from_move, _ = self.shortest_path(move, goal)
            if path_from_move:
                if len(path_from_move) < best_path_len:
                    best_path_len = len(path_from_move)
                    best_move = move
        
        return best_move


def generate_left_turn_agents(road, n_agents, seed):
    """Generate agents for the 3x3 grid network with 2-lane roads.
    
    Spawns on inbound lanes, destinations on outbound lanes.
    Supports many agents by using multiple positions along each road segment.
    """
    rng = np.random.default_rng(seed)
    
    origins = []
    destinations = []
    turn_types = []
    
    spawn_points = []
    dest_points = []
    
    # Generate spawn/dest points along entire road segments (not just near intersections)
    # Horizontal roads (right-hand driving: y=cy-1 west, y=cy east)
    for row in range(3):
        for col in range(2):
            inter1 = road.intersections[row * 3 + col]
            inter2 = road.intersections[row * 3 + col + 1]
            cy = inter1['center'][1]
            x_start = inter1['hi_x'] + 1
            x_end = inter2['lo_x'] - 1
            
            # Add multiple spawn points along each road segment
            for x in range(x_start, x_end + 1, 2):  # Every 2 cells
                # West-bound lane (y = cy - 1)
                if road.is_on_road(x, cy - 1):
                    spawn_points.append({
                        'pos': (x, cy - 1),
                        'dir': 'west',
                        'inter': row * 3 + col + 1 if x > (x_start + x_end) // 2 else row * 3 + col
                    })
                    dest_points.append({
                        'pos': (x, cy - 1),
                        'dir': 'west',
                        'inter': row * 3 + col if x < (x_start + x_end) // 2 else row * 3 + col + 1
                    })
                
                # East-bound lane (y = cy)
                if road.is_on_road(x, cy):
                    spawn_points.append({
                        'pos': (x, cy),
                        'dir': 'east',
                        'inter': row * 3 + col if x < (x_start + x_end) // 2 else row * 3 + col + 1
                    })
                    dest_points.append({
                        'pos': (x, cy),
                        'dir': 'east',
                        'inter': row * 3 + col + 1 if x > (x_start + x_end) // 2 else row * 3 + col
                    })
    
    # Vertical roads (2 lanes: x=cx-1 south, x=cx north)
    for row in range(2):
        for col in range(3):
            inter1 = road.intersections[row * 3 + col]
            inter2 = road.intersections[(row + 1) * 3 + col]
            cx = inter1['center'][0]
            y_start = inter1['hi_y'] + 1
            y_end = inter2['lo_y'] - 1
            
            # Add multiple spawn points along each road segment
            for y in range(y_start, y_end + 1, 2):  # Every 2 cells
                # South-bound lane (x = cx - 1)
                if road.is_on_road(cx - 1, y):
                    spawn_points.append({
                        'pos': (cx - 1, y),
                        'dir': 'south',
                        'inter': row * 3 + col if y < (y_start + y_end) // 2 else (row + 1) * 3 + col
                    })
                    dest_points.append({
                        'pos': (cx - 1, y),
                        'dir': 'south',
                        'inter': (row + 1) * 3 + col if y > (y_start + y_end) // 2 else row * 3 + col
                    })
                
                # North-bound lane (x = cx)
                if road.is_on_road(cx, y):
                    spawn_points.append({
                        'pos': (cx, y),
                        'dir': 'north',
                        'inter': (row + 1) * 3 + col if y > (y_start + y_end) // 2 else row * 3 + col
                    })
                    dest_points.append({
                        'pos': (cx, y),
                        'dir': 'north',
                        'inter': row * 3 + col if y < (y_start + y_end) // 2 else (row + 1) * 3 + col
                    })
    
    # Track used spawn points to avoid duplicates
    used_spawns = set()
    
    # Generate agents - random spawn/destination pairs (no strict compatibility)
    for _ in range(n_agents * 10):
        if not spawn_points or not dest_points:
            break
        if len(origins) >= n_agents:
            break
        
        spawn_info = spawn_points[rng.integers(len(spawn_points))]
        dest_info = dest_points[rng.integers(len(dest_points))]
        
        spawn = spawn_info['pos']
        dest = dest_info['pos']
        
        # Skip if spawn already used (avoid duplicate spawns)
        if spawn in used_spawns:
            continue
        
        # Skip if same or too close
        if spawn == dest or road.manhattan_distance(spawn, dest) < 8:
            continue
        
        # Determine turn type based on direction change
        spawn_inter = spawn_info['inter']
        dest_inter = dest_info['inter']
        spawn_row, spawn_col = spawn_inter // 3, spawn_inter % 3
        dest_row, dest_col = dest_inter // 3, dest_inter % 3
        
        is_diagonal = (spawn_row != dest_row) and (spawn_col != dest_col)
        
        used_spawns.add(spawn)  # Mark spawn as used
        origins.append(list(spawn))
        destinations.append(dest)
        turn_types.append('left' if is_diagonal else 'through')
    
    return origins, destinations, turn_types


def step_gsp(road, positions, goals, reached, times, reserved, prev_moves=None, history=None, moves=None):
    """GSP: Greedy shortest path with anti-oscillation.
    
    Key features:
    - Smart intersection exit selection
    - Oscillation detection and breaking
    - History-aware to avoid revisiting recent positions
    
    Metrics:
        - times[i]: arrival timestep (total delay) - incremented every step for active agents
        - moves[i]: distance traveled - incremented only when agent actually moves
    """
    n = len(positions)
    if prev_moves is None:
        prev_moves = {}
    if history is None:
        history = {i: [] for i in range(n)}
    if moves is None:
        moves = [0] * n
    
    active = sorted([i for i in range(n) if not reached[i]], 
                    key=lambda i: road.distance(positions[i], goals[i]))
    next_pos = [None] * n
    
    def get_exit_type(road, cx, cy, gx, gy):
        """Check what type of exits an intersection cell has.
        
        Returns: 'primary', 'secondary', or None
        A good exit requires BOTH:
        1. Lane direction matches goal direction
        2. Movement to reach the lane goes toward goal
        """
        dx, dy = gx - cx, gy - cy
        primary_is_vertical = abs(dy) > abs(dx)
        
        has_primary = False
        has_secondary = False
        
        for nc in road.get_road_neighbors(cx, cy):
            nx, ny = nc
            if not road.in_intersection(nx, ny):
                lane_dir = road.get_lane_direction(nx, ny)
                move_dx, move_dy = nx - cx, ny - cy
                is_vertical = lane_dir in ('north', 'south')
                
                is_good = False
                if lane_dir == 'east' and dx > 0 and move_dx > 0:
                    is_good = True
                elif lane_dir == 'west' and dx < 0 and move_dx < 0:
                    is_good = True
                elif lane_dir == 'north' and dy < 0 and move_dy < 0:
                    is_good = True
                elif lane_dir == 'south' and dy > 0 and move_dy > 0:
                    is_good = True
                
                if is_good:
                    if is_vertical == primary_is_vertical:
                        has_primary = True
                    else:
                        has_secondary = True
        
        if has_primary:
            return 'primary'
        elif has_secondary:
            return 'secondary'
        return None
    
    for i in active:
        px, py = positions[i]
        gx, gy = goals[i]
        
        # Track history for oscillation/stuck detection
        history[i].append((px, py))
        if len(history[i]) > 20:
            history[i] = history[i][-20:]
        
        # Detect stuck patterns
        is_oscillating = False
        is_severely_stuck = False
        
        if len(history[i]) >= 6:
            recent = history[i][-6:]
            if len(set(recent)) <= 2:
                is_oscillating = True
        
        # Severely stuck: made no progress in last 15 steps
        if len(history[i]) >= 15:
            recent15 = history[i][-15:]
            if len(set(recent15)) <= 4:
                is_severely_stuck = True
        
        # Check if reached goal - must be at EXACT destination
        if (px, py) == (gx, gy):
            reached[i] = True
            next_pos[i] = (gx, gy)
            reserved.add((gx, gy))
            times[i] += 1
            continue
        
        valid = road.get_valid_moves(px, py, reserved)
        in_int = road.in_intersection(px, py)
        
        if valid:
            best = None
            
            # Helper to categorize moves by quality (for smart exploration)
            def categorize_move(c, px, py, gx, gy):
                """Return priority: 0=best (correct direction), 1=neutral, 2=bad (wrong direction)"""
                cx, cy = c
                c_in_int = road.in_intersection(cx, cy)
                
                if c_in_int:
                    return 1  # Intersection moves are neutral
                
                lane_dir = road.get_lane_direction(cx, cy)
                dx, dy = gx - px, gy - py
                move_dx, move_dy = cx - px, cy - py
                
                # Check if exit leads in correct direction
                is_good = False
                if lane_dir == 'east' and dx > 0:
                    is_good = True
                elif lane_dir == 'west' and dx < 0:
                    is_good = True
                elif lane_dir == 'north' and dy < 0:
                    is_good = True
                elif lane_dir == 'south' and dy > 0:
                    is_good = True
                
                return 0 if is_good else 2
            
            # Smart exploration: prefer good moves even when stuck
            if is_severely_stuck or is_oscillating:
                import random
                recent_set = set(history[i][-10:] if is_severely_stuck else history[i][-4:])
                
                # Categorize moves
                good_moves = []
                neutral_moves = []
                bad_moves = []
                
                for c in valid:
                    cat = categorize_move(c, px, py, gx, gy)
                    if cat == 0:
                        good_moves.append(c)
                    elif cat == 1:
                        neutral_moves.append(c)
                    else:
                        bad_moves.append(c)
                
                # Filter out recently visited, but prioritize by category
                def filter_recent(moves):
                    fresh = [c for c in moves if c not in recent_set]
                    return fresh if fresh else moves
                
                good_fresh = filter_recent(good_moves)
                neutral_fresh = filter_recent(neutral_moves)
                
                if good_fresh:
                    best = random.choice(good_fresh)
                elif neutral_fresh:
                    best = random.choice(neutral_fresh)
                elif good_moves:
                    best = random.choice(good_moves)
                elif neutral_moves:
                    best = random.choice(neutral_moves)
                else:
                    # Only use bad moves as last resort
                    best = random.choice(valid)
                
                if best:
                    prev_moves[i] = (px, py)
                    next_pos[i] = best
                    reserved.add(best)
                    times[i] += 1
                    continue
            
            if in_int:
                # In intersection: prioritize actual road exits, then intersection cells with exits
                dx, dy = gx - px, gy - py
                
                # Determine which direction is primary (larger distance component)
                primary_is_vertical = abs(dy) > abs(dx)
                
                # Categories for moves (in priority order)
                primary_exits = []
                int_with_primary = []
                secondary_exits = []
                int_with_secondary = []
                int_corners = []
                
                for c in valid:
                    cx, cy = c
                    c_in_int = road.in_intersection(cx, cy)
                    
                    if not c_in_int:
                        # Road exit - check if MOVEMENT direction AND lane direction both align with goal
                        lane_dir = road.get_lane_direction(cx, cy)
                        move_dx, move_dy = cx - px, cy - py
                        
                        is_good = False
                        is_vertical = lane_dir in ('north', 'south')
                        
                        if lane_dir == 'east' and dx > 0 and move_dx > 0:
                            is_good = True
                        elif lane_dir == 'west' and dx < 0 and move_dx < 0:
                            is_good = True
                        elif lane_dir == 'north' and dy < 0 and move_dy < 0:
                            is_good = True
                        elif lane_dir == 'south' and dy > 0 and move_dy > 0:
                            is_good = True
                        elif lane_dir == 'any':
                            is_good = True
                        
                        if is_good:
                            if is_vertical == primary_is_vertical:
                                primary_exits.append(c)
                            else:
                                secondary_exits.append(c)
                    else:
                        exit_type = get_exit_type(road, cx, cy, gx, gy)
                        if exit_type == 'primary':
                            int_with_primary.append(c)
                        elif exit_type == 'secondary':
                            int_with_secondary.append(c)
                        else:
                            int_corners.append(c)
                
                # Priority selection with anti-oscillation
                def pick_best(candidates):
                    if not candidates:
                        return None
                    if is_oscillating:
                        # When oscillating, add randomness to break ties
                        import random
                        candidates = sorted(candidates, key=lambda c: road.distance(c, goals[i]))
                        # Pick from top candidates with some randomness
                        top_n = min(3, len(candidates))
                        return random.choice(candidates[:top_n])
                    return min(candidates, key=lambda c: road.distance(c, goals[i]))
                
                if primary_exits:
                    best = pick_best(primary_exits)
                elif int_with_primary:
                    best = pick_best(int_with_primary)
                elif secondary_exits:
                    best = pick_best(secondary_exits)
                elif int_with_secondary:
                    best = pick_best(int_with_secondary)
                elif int_corners:
                    best = pick_best(int_corners)
                else:
                    # Fallback: take ANY valid move (even "wrong" direction exits)
                    # This ensures agents don't get permanently stuck in intersections
                    best = pick_best(valid)
                
                # If still no best and oscillating, just pick randomly from all valid
                if best is None and valid and is_oscillating:
                    import random
                    best = random.choice(valid)
            else:
                # On road: follow the lane, but if oscillating try alternatives
                if is_oscillating and len(valid) > 1:
                    import random
                    best = random.choice(valid)
                else:
                    best = min(valid, key=lambda c: road.distance(c, goals[i]))
            
            if best:
                prev_moves[i] = (px, py)
                next_pos[i] = best
                reserved.add(best)
        
        times[i] += 1
    
    for i in range(n):
        if next_pos[i]:
            old_pos = (int(positions[i][0]), int(positions[i][1]))
            new_pos = (int(next_pos[i][0]), int(next_pos[i][1]))
            positions[i][0], positions[i][1] = next_pos[i]
            # Track moves (only if actually moved)
            if new_pos != old_pos:
                moves[i] += 1
    
    return prev_moves, history, moves


def step_sp(road, positions, goals, reached, times, reserved, paths=None, moves=None):
    """
    SP: True Shortest Path algorithm using BFS.
    
    Unlike GSP (greedy), this computes the shortest path ONCE and follows it.
    If blocked by another agent, it waits instead of replanning.
    
    Features:
    - Uses BFS to find true shortest path (computed once per agent)
    - Respects lane directions (no backward movement)
    - Waits if next cell is blocked (no replanning)
    - Path is fixed once computed
    
    Args:
        paths: dict mapping agent_id -> list of cells to visit (excluding current pos)
               If None or missing for an agent, path will be computed
    
    Metrics:
        - times[i]: arrival timestep (total delay) - incremented every step for active agents
        - moves[i]: distance traveled - incremented only when agent actually moves
    """
    n = len(positions)
    if paths is None:
        paths = {i: None for i in range(n)}
    if moves is None:
        moves = [0] * n
    
    # Build set of currently occupied cells
    occupied = set()
    for i in range(n):
        if not reached[i]:
            occupied.add((int(positions[i][0]), int(positions[i][1])))
    occupied.update(reserved)
    
    # Sort by distance to goal (closer agents get priority)
    active = sorted([i for i in range(n) if not reached[i]], 
                    key=lambda i: road.distance(positions[i], goals[i]))
    
    next_pos = [None] * n
    local_reserved = set(reserved)
    
    for i in active:
        px, py = int(positions[i][0]), int(positions[i][1])
        gx, gy = int(goals[i][0]), int(goals[i][1])
        old_pos = (px, py)
        
        # Check if reached goal
        if (px, py) == (gx, gy):
            reached[i] = True
            next_pos[i] = (gx, gy)
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
            if next_cell not in local_reserved:
                # Move to next cell
                next_pos[i] = next_cell
                local_reserved.add(next_cell)
                paths[i] = paths[i][1:]  # Remove from path
                moves[i] += 1  # Actual movement
            else:
                # Blocked - wait in place
                next_pos[i] = (px, py)
                local_reserved.add((px, py))
                # No move increment - just waiting
        else:
            # No path - stay in place
            next_pos[i] = (px, py)
            local_reserved.add((px, py))
        
        times[i] += 1
    
    # Apply moves
    for i in range(n):
        if next_pos[i]:
            positions[i][0], positions[i][1] = next_pos[i]
    
    return paths, moves


def step_detour(road, positions, goals, reached, times, reserved, detour_state=None):
    """
    Detour strategy: When needing left turn, do 3 right turns instead.
    
    State machine per agent:
    - 'direct': Try to go directly
    - 'detour_1': First right turn taken
    - 'detour_2': Second right turn taken
    - 'detour_3': Third right turn (completing the block)
    """
    n = len(positions)
    if detour_state is None:
        detour_state = {i: {'mode': 'direct', 'waypoints': []} for i in range(n)}
    
    active = sorted([i for i in range(n) if not reached[i]], 
                    key=lambda i: road.distance(positions[i], goals[i]))
    next_pos = [None] * n
    
    for i in active:
        px, py = positions[i]
        gx, gy = goals[i]
        
        # Check if reached goal - must be at EXACT destination
        if (px, py) == (gx, gy):
            reached[i] = True
            next_pos[i] = (gx, gy)
            reserved.add((gx, gy))
            times[i] += 1
            continue
        
        valid = road.get_valid_moves(px, py, reserved)
        
        if valid:
            # Simple greedy for now - detour logic would be more complex
            # For now, just prefer right turns when available
            best = min(valid, key=lambda c: road.distance(c, goals[i]))
            next_pos[i] = best
            reserved.add(best)
        
        times[i] += 1
    
    for i in range(n):
        if next_pos[i]:
            positions[i][0], positions[i][1] = next_pos[i]
    
    return detour_state


class PIBTPlanner:
    """PIBT with centralized lane discipline and smart intersection navigation.
    
    Uses Euclidean distance for priority and move scoring (our baseline).
    """
    
    def __init__(self, road):
        self.road = road
    
    def distance(self, p1, p2):
        """Euclidean distance (geometric)."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def score_move(self, pos, cand, goal):
        """Score a candidate move - lower is better.
        
        Smart scoring with intersection navigation:
        - Primary metric: distance to goal
        - Penalize wrong-direction lane exits
        - Prioritize primary direction (larger distance component)
        - Small penalty for staying in place
        """
        px, py = pos
        cx, cy = cand
        gx, gy = goal
        
        base_dist = self.distance(cand, goal)
        
        # Small penalty for staying in place
        if (cx, cy) == (px, py):
            return base_dist + 0.1
        
        dx, dy = gx - px, gy - py
        move_dx, move_dy = cx - px, cy - py
        in_int = self.road.in_intersection(px, py)
        cand_in_int = self.road.in_intersection(cx, cy)
        
        # Smart intersection exit selection
        if in_int and not cand_in_int:
            # Exiting intersection to a lane
            lane_dir = self.road.get_lane_direction(cx, cy)
            is_vertical = lane_dir in ('north', 'south')
            primary_is_vertical = abs(dy) > abs(dx)
            
            # Check if BOTH lane direction and movement match goal direction
            is_good_exit = False
            if lane_dir == 'east' and dx > 0 and move_dx > 0:
                is_good_exit = True
            elif lane_dir == 'west' and dx < 0 and move_dx < 0:
                is_good_exit = True
            elif lane_dir == 'north' and dy < 0 and move_dy < 0:
                is_good_exit = True
            elif lane_dir == 'south' and dy > 0 and move_dy > 0:
                is_good_exit = True
            elif lane_dir == 'any':
                is_good_exit = True
            
            if not is_good_exit:
                # Penalize wrong-direction exits heavily
                return base_dist + 50
            else:
                # Good exit - bonus for primary direction
                if is_vertical == primary_is_vertical:
                    return base_dist - 5  # Primary direction
                else:
                    return base_dist - 2  # Secondary direction
        
        # When in intersection, prefer moves toward primary exit
        if in_int and cand_in_int:
            primary_is_vertical = abs(dy) > abs(dx)
            # Check if this intersection cell has primary direction exits
            for nc in self.road.get_road_neighbors(cx, cy):
                nx, ny = nc
                if not self.road.in_intersection(nx, ny):
                    lane_dir = self.road.get_lane_direction(nx, ny)
                    is_vertical = lane_dir in ('north', 'south')
                    nmove_dx, nmove_dy = nx - cx, ny - cy
                    
                    is_good = False
                    if lane_dir == 'east' and dx > 0 and nmove_dx > 0:
                        is_good = True
                    elif lane_dir == 'west' and dx < 0 and nmove_dx < 0:
                        is_good = True
                    elif lane_dir == 'north' and dy < 0 and nmove_dy < 0:
                        is_good = True
                    elif lane_dir == 'south' and dy > 0 and nmove_dy > 0:
                        is_good = True
                    
                    if is_good and is_vertical == primary_is_vertical:
                        return base_dist - 3  # Leads to primary exit
        
        return base_dist
    
    def plan_step(self, positions, goals, oscillating_agents=None, history=None, rng_seed=None):
        """Plan one step for all agents using PIBT.
        
        Tie-breaking: Seeded random among equal-score candidates.
        """
        import random
        
        if oscillating_agents is None:
            oscillating_agents = set()
        if history is None:
            history = {}
        
        # Use seeded RNG for reproducible tie-breaking
        if rng_seed is not None:
            rng = random.Random(rng_seed)
        else:
            rng = random.Random()
        
        agents = list(positions.keys())
        # Prioritize oscillating agents to give them better choices
        agents.sort(key=lambda a: (0 if a in oscillating_agents else 1, self.distance(positions[a], goals[a])))
        
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
            
            candidates = [tuple(pos)] + self.road.get_valid_moves(pos[0], pos[1])
            
            # For oscillating agents, penalize recently visited positions
            is_oscillating = agent_id in oscillating_agents
            recent_set = set(history.get(agent_id, [])[-6:]) if is_oscillating else set()
            
            def adjusted_score(c):
                base = self.score_move(pos, c, goal)
                if is_oscillating and tuple(c) in recent_set:
                    base += 20  # Penalize recently visited
                return base
            
            # Principled tie-breaking: shuffle first, then stable sort by score
            rng.shuffle(candidates)
            candidates.sort(key=adjusted_score)
            
            for cand in candidates:
                cand_cell = (int(cand[0]), int(cand[1]))
                if cand_cell in reserved:
                    continue
                
                blocked_agent = None
                for other in agents:
                    if other == agent_id or other in decided:
                        continue
                    other_pos = tuple(positions[other])
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
            
            decided[agent_id] = tuple(pos)
            reserved.add(tuple(pos))
            in_progress.discard(agent_id)
            return False
        
        for agent in agents:
            pibt(agent)
        
        return decided


class GraphPIBTPlanner(PIBTPlanner):
    """
    PIBT using graph (shortest-path) distance, like Manhattan's GraphPIBTPlanner.
    
    Priority and move scoring use road.graph_distance(a, b) = BFS hop count
    instead of Euclidean/Manhattan. This matches the true cost to reach the goal
    on the directed lane graph and should improve 3×3 performance.
    """
    
    def distance(self, p1, p2):
        """Graph distance = number of steps on shortest path (BFS). inf if no path."""
        a = (int(p1[0]), int(p1[1]))
        b = (int(p2[0]), int(p2[1]))
        return self.road.graph_distance(a, b)


class PaperPIBTPlanner(PIBTPlanner):
    """
    PIBT as in the original paper (Okumura et al., IJCAI-19 / AIJ-22).
    
    - Agent priority: time-based. pi = epsilon_i if at goal, else pi = pi + 1 each step.
      Agents sorted in decreasing order of pi (longest waiting decides first).
    - Move choice: candidates sorted by graph distance to goal only (dist(u, gi)).
    """
    
    def __init__(self, road):
        super().__init__(road)
        self._priority = {}  # agent_id -> pi (persisted across steps)
    
    def _epsilon(self, agent_id, n):
        """Tie-breaker in [0, 1), distinct per agent."""
        return (agent_id + 1) / (n + 2)
    
    def score_move(self, pos, cand, goal):
        """Paper: sort C in increasing order of dist(u, gi). Tie-break: staying +0.1."""
        a = (int(cand[0]), int(cand[1]))
        b = (int(goal[0]), int(goal[1]))
        d = self.road.graph_distance(a, b)
        if (int(pos[0]), int(pos[1])) == a:
            return d + 0.1  # prefer moving when distance is equal
        return d
    
    def plan_step(self, positions, goals, oscillating_agents=None, history=None, rng_seed=None):
        """One timestep: update priorities (paper rule), then PIBT with graph-distance move scoring."""
        import random
        if oscillating_agents is None:
            oscillating_agents = set()
        if history is None:
            history = {}
        if rng_seed is not None:
            rng = random.Random(rng_seed)
        else:
            rng = random.Random()
        
        agents = list(positions.keys())
        n = len(agents)
        # Initialize or update priority (paper Line 3)
        for a in agents:
            if a not in self._priority:
                self._priority[a] = self._epsilon(a, n)
            pos = tuple(positions[a])
            goal = tuple(goals[a])
            if pos == goal:
                self._priority[a] = self._epsilon(a, n)
            else:
                self._priority[a] = self._priority[a] + 1
        # Sort in decreasing order of priority (paper Line 4: high priority first)
        agents.sort(key=lambda a: -self._priority[a])
        
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
            candidates = [tuple(pos)] + self.road.get_valid_moves(pos[0], pos[1])
            is_oscillating = agent_id in oscillating_agents
            recent_set = set(history.get(agent_id, [])[-6:]) if is_oscillating else set()
            
            def adjusted_score(c):
                base = self.score_move(pos, c, goal)
                if is_oscillating and tuple(c) in recent_set:
                    base += 20
                return base
            
            rng.shuffle(candidates)
            candidates.sort(key=adjusted_score)
            
            for cand in candidates:
                cand_cell = (int(cand[0]), int(cand[1]))
                if cand_cell in reserved:
                    continue
                blocked_agent = None
                for other in agents:
                    if other == agent_id or other in decided:
                        continue
                    if tuple(positions[other]) == cand_cell:
                        blocked_agent = other
                        break
                if blocked_agent is not None:
                    if not pibt(blocked_agent):
                        continue
                decided[agent_id] = cand_cell
                reserved.add(cand_cell)
                in_progress.discard(agent_id)
                return True
            decided[agent_id] = tuple(pos)
            reserved.add(tuple(pos))
            in_progress.discard(agent_id)
            return False
        
        for agent in agents:
            pibt(agent)
        return decided


class SPGuidedPIBT:
    """
    SP-Guided PIBT: Combines optimal shortest paths with PIBT collision resolution.
    
    Key insight: SP achieves near-ideal performance (3-4% overhead) because it uses
    BFS to find true shortest paths. PIBT uses Euclidean distance which is suboptimal.
    
    This hybrid approach:
    1. Computes true shortest paths using BFS (like SP)
    2. Uses PIBT's priority inheritance for collision resolution
    3. When blocked, PIBT can push agents out of the way instead of just waiting
    
    Expected: Better than SP in high-contention scenarios (PIBT coordination),
              Better than PIBT in all scenarios (optimal paths).
    """
    
    def __init__(self, road):
        self.road = road
        self.guide_paths = {}  # {agent_id: [path_cells]}
        self.path_index = {}   # {agent_id: current_index_in_path}
    
    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def compute_guide_paths(self, positions, goals):
        """Compute BFS shortest paths for all agents."""
        self.guide_paths = {}
        self.path_index = {}
        
        for agent_id in positions.keys():
            pos = positions[agent_id]
            goal = goals[agent_id]
            start = (int(pos[0]), int(pos[1]))
            end = (int(goal[0]), int(goal[1]))
            
            path, _ = self.road.shortest_path(start, end)
            if path:
                self.guide_paths[agent_id] = path
                self.path_index[agent_id] = 0
            else:
                self.guide_paths[agent_id] = [start, end]
                self.path_index[agent_id] = 0
    
    def get_next_on_path(self, agent_id, current_pos):
        """Get the next cell on the guide path for this agent."""
        if agent_id not in self.guide_paths:
            return None

        path = self.guide_paths[agent_id]
        current = (int(current_pos[0]), int(current_pos[1]))
        idx = self.path_index.get(agent_id, 0)
        search = idx
        while search < len(path) and path[search] != current:
            search += 1
        if search < len(path):
            self.path_index[agent_id] = search
            return path[search + 1] if search + 1 < len(path) else None
        # Displaced off remaining path: guide back to path[idx] to rejoin
        return path[idx] if idx < len(path) else None

    def score_move(self, agent_id, pos, cand, goal):
        """
        Score a candidate move using guide path.
        
        Priority:
        1. Next cell on guide path (score = 0)
        2. Any cell on guide path ahead (score = 1 + steps_away)
        3. Other valid moves (score = 100 + euclidean_distance)
        """
        cand_cell = (int(cand[0]), int(cand[1]))
        pos_cell = (int(pos[0]), int(pos[1]))
        
        # Check if this is the next cell on guide path
        next_on_path = self.get_next_on_path(agent_id, pos)
        if next_on_path and cand_cell == next_on_path:
            return 0  # Best score - exactly on path

        # Check if this cell is anywhere on the remaining path
        if agent_id in self.guide_paths:
            path = self.guide_paths[agent_id]
            idx = self.path_index.get(agent_id, 0)

            for i, cell in enumerate(path[idx:]):
                if cell == cand_cell:
                    return 1 + i  # On path, but not next

        if cand_cell == pos_cell:
            return 10000  # staying is absolute last resort
        return 200 + self.distance(cand, goal)
    
    def plan_step(self, positions, goals, oscillating_agents=None, history=None, rng_seed=None):
        """
        Plan one step for all agents using SP-guided PIBT.
        
        Uses PIBT's priority inheritance but with BFS-optimal path guidance.
        
        Tie-breaking: Seeded random among equal-score candidates.
        This removes arbitrary directional bias (West→East→North→South)
        and demonstrates robustness when averaged over seeds.
        """
        import random
        
        if oscillating_agents is None:
            oscillating_agents = set()
        if history is None:
            history = {}
        
        # Use seeded RNG for reproducible tie-breaking
        if rng_seed is not None:
            rng = random.Random(rng_seed)
        else:
            rng = random.Random()
        
        agents = list(positions.keys())
        # Sort by distance to goal (closer = higher priority)
        agents.sort(key=lambda a: (0 if a in oscillating_agents else 1, 
                                   self.distance(positions[a], goals[a])))
        
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
            candidates = [tuple(pos)] + self.road.get_valid_moves(pos[0], pos[1])
            
            # Score candidates
            is_oscillating = agent_id in oscillating_agents
            recent_set = set(history.get(agent_id, [])[-6:]) if is_oscillating else set()
            
            def adjusted_score(c):
                base = self.score_move(agent_id, pos, c, goal)
                if is_oscillating and tuple(c) in recent_set:
                    base += 200  # Heavy penalty for recently visited
                return base
            
            # Principled tie-breaking: shuffle first, then stable sort by score
            # This ensures random order among equal-score candidates
            rng.shuffle(candidates)
            candidates.sort(key=adjusted_score)
            
            # Try candidates in order
            for cand in candidates:
                cand_cell = (int(cand[0]), int(cand[1]))
                if cand_cell in reserved:
                    continue
                
                # Check if another agent is blocking
                blocked_agent = None
                for other in agents:
                    if other == agent_id or other in decided:
                        continue
                    other_pos = tuple(positions[other])
                    if other_pos == cand_cell:
                        blocked_agent = other
                        break
                
                # PIBT: recursively ask blocking agent to move
                if blocked_agent is not None:
                    if not pibt(blocked_agent):
                        continue
                
                decided[agent_id] = cand_cell
                reserved.add(cand_cell)
                in_progress.discard(agent_id)
                return True
            
            # No valid move - stay in place
            decided[agent_id] = tuple(pos)
            reserved.add(tuple(pos))
            in_progress.discard(agent_id)
            return False
        
        for agent in agents:
            pibt(agent)
        
        return decided


class GuidedPIBTCongestion:
    """
    Guided PIBT with Congestion-Aware Path Planning.
    
    Based on: "Traffic Flow Optimisation for Lifelong Multi-Agent Path Finding" (AAAI 2024)
    
    Key differences from SP-PIBT:
    1. Tracks flow on edges (how many agents use each edge)
    2. Adds contraflow congestion penalty (agents crossing in opposite directions)
    3. Adds vertex congestion penalty (many agents entering same vertex)
    4. Computes paths using A* with congestion costs instead of BFS
    """
    
    def __init__(self, road):
        self.road = road
        self.guide_paths = {}  # {agent_id: [path_cells]}
        self.path_index = {}   # {agent_id: current_index_in_path}
        self.flow = {}         # {(v1, v2): count} - edge flow counts
    
    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def compute_guide_paths(self, positions, goals):
        """
        Compute congestion-aware guide paths for all agents.
        
        Key insight: Plan agents sequentially, updating edge costs after each.
        Earlier agents get "free" paths; later agents avoid congested edges.
        """
        import heapq
        
        self.guide_paths = {}
        self.path_index = {}
        self.flow = {}  # Reset flow counts
        
        # Sort agents by distance (shorter paths planned first)
        agents = sorted(positions.keys(), 
                       key=lambda a: self.distance(positions[a], goals[a]))
        
        for agent_id in agents:
            pos = positions[agent_id]
            goal = goals[agent_id]
            start = (int(pos[0]), int(pos[1]))
            end = (int(goal[0]), int(goal[1]))
            
            # A* with congestion costs
            path = self._astar_congestion(start, end)
            
            if path:
                self.guide_paths[agent_id] = path
                self.path_index[agent_id] = 0
                
                # Update flow counts
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    self.flow[edge] = self.flow.get(edge, 0) + 1
            else:
                # Fallback to BFS if A* fails
                path, _ = self.road.shortest_path(start, end)
                if path:
                    self.guide_paths[agent_id] = path
                    self.path_index[agent_id] = 0
                else:
                    self.guide_paths[agent_id] = [start, end]
                    self.path_index[agent_id] = 0
    
    def _astar_congestion(self, start, end):
        """A* search with congestion-aware edge costs."""
        import heapq
        
        def heuristic(pos):
            return abs(pos[0] - end[0]) + abs(pos[1] - end[1])
        
        def get_edge_cost(v1, v2):
            """
            Edge cost from paper:
            - Base cost: 1
            - Contraflow penalty: flow(v2→v1) * flow(v1→v2) * weight
            - Vertex congestion: sum of incoming flows to v2
            """
            base = 1.0
            
            # Contraflow congestion (critical for narrow corridors)
            forward_flow = self.flow.get((v1, v2), 0)
            reverse_flow = self.flow.get((v2, v1), 0)
            contraflow = forward_flow * reverse_flow
            contraflow_penalty = 2.0 * contraflow if contraflow > 0 else 0
            
            # Vertex congestion (how crowded is the destination?)
            incoming = 0
            for neighbor in self.road.get_valid_moves(v2[0], v2[1]):
                n = (int(neighbor[0]), int(neighbor[1]))
                incoming += self.flow.get((n, v2), 0)
            vertex_penalty = 0.3 * incoming
            
            return base + contraflow_penalty + vertex_penalty
        
        # A* search
        open_set = [(heuristic(start), 0, start, [start])]
        visited = {start: 0}
        
        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            
            if current == end:
                return path
            
            if g > visited.get(current, float('inf')):
                continue
            
            for neighbor in self.road.get_valid_moves(current[0], current[1]):
                next_cell = (int(neighbor[0]), int(neighbor[1]))
                edge_cost = get_edge_cost(current, next_cell)
                new_g = g + edge_cost
                
                if next_cell not in visited or new_g < visited[next_cell]:
                    visited[next_cell] = new_g
                    new_f = new_g + heuristic(next_cell)
                    heapq.heappush(open_set, (new_f, new_g, next_cell, path + [next_cell]))
        
        return None  # No path found
    
    def get_next_on_path(self, agent_id, current_pos):
        """Get the next cell on the guide path for this agent."""
        if agent_id not in self.guide_paths:
            return None

        path = self.guide_paths[agent_id]
        current = (int(current_pos[0]), int(current_pos[1]))
        idx = self.path_index.get(agent_id, 0)
        search = idx
        while search < len(path) and path[search] != current:
            search += 1
        if search < len(path):
            self.path_index[agent_id] = search
            return path[search + 1] if search + 1 < len(path) else None
        # Displaced off remaining path: guide back to path[idx] to rejoin
        return path[idx] if idx < len(path) else None

    def score_move(self, agent_id, pos, cand, goal):
        """Score a candidate move using guide path (same as SP-PIBT)."""
        cand_cell = (int(cand[0]), int(cand[1]))
        pos_cell = (int(pos[0]), int(pos[1]))

        next_on_path = self.get_next_on_path(agent_id, pos)
        if next_on_path and cand_cell == next_on_path:
            return 0

        if agent_id in self.guide_paths:
            path = self.guide_paths[agent_id]
            idx = self.path_index.get(agent_id, 0)
            for i, cell in enumerate(path[idx:]):
                if cell == cand_cell:
                    return 1 + i

        if cand_cell == pos_cell:
            return 10000  # staying is absolute last resort
        return 200 + self.distance(cand, goal)
    
    def plan_step(self, positions, goals, oscillating_agents=None, history=None, rng_seed=None):
        """Plan one step for all agents using Guided PIBT.
        
        Tie-breaking: Seeded random among equal-score candidates.
        """
        import random
        
        if oscillating_agents is None:
            oscillating_agents = set()
        if history is None:
            history = {}
        
        # Use seeded RNG for reproducible tie-breaking
        if rng_seed is not None:
            rng = random.Random(rng_seed)
        else:
            rng = random.Random()
        
        agents = list(positions.keys())
        agents.sort(key=lambda a: (0 if a in oscillating_agents else 1, 
                                   self.distance(positions[a], goals[a])))
        
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
            
            candidates = [tuple(pos)] + self.road.get_valid_moves(pos[0], pos[1])
            
            is_oscillating = agent_id in oscillating_agents
            recent_set = set(history.get(agent_id, [])[-6:]) if is_oscillating else set()
            
            def adjusted_score(c):
                base = self.score_move(agent_id, pos, c, goal)
                if is_oscillating and tuple(c) in recent_set:
                    base += 200
                return base
            
            # Principled tie-breaking: shuffle first, then stable sort by score
            rng.shuffle(candidates)
            candidates.sort(key=adjusted_score)
            
            for cand in candidates:
                cand_cell = (int(cand[0]), int(cand[1]))
                if cand_cell in reserved:
                    continue
                
                blocked_agent = None
                for other in agents:
                    if other == agent_id or other in decided:
                        continue
                    other_pos = tuple(positions[other])
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
            
            decided[agent_id] = tuple(pos)
            reserved.add(tuple(pos))
            in_progress.discard(agent_id)
            return False
        
        for agent in agents:
            pibt(agent)
        
        return decided


def step_guided_pibt(road, planner, positions, goals, reached, times, history=None, initialized=None, moves=None):
    """Guided PIBT step function with congestion-aware paths.
    
    Metrics:
        - times[i]: arrival timestep (total delay) - incremented every step for active agents
        - moves[i]: distance traveled - incremented only when agent actually moves
    """
    n = len(positions)
    active = {i for i in range(n) if not reached[i]}
    
    if history is None:
        history = {i: [] for i in range(n)}
    if initialized is None:
        initialized = [False]
    if moves is None:
        moves = [0] * n
    
    if not active:
        return history, initialized, moves
    
    # Compute guide paths once at start
    if not initialized[0]:
        pos_dict = {i: tuple(positions[i]) for i in range(n)}
        goal_dict = {i: tuple(goals[i]) for i in range(n)}
        planner.compute_guide_paths(pos_dict, goal_dict)
        initialized[0] = True
    
    pos_dict = {i: tuple(positions[i]) for i in active}
    goal_dict = {i: tuple(goals[i]) for i in active}
    
    for i in active:
        px, py = positions[i]
        history[i].append((px, py))
        history[i] = history[i][-10:]
    
    oscillating_agents = set()
    for i in active:
        if len(history[i]) >= 6:
            recent = history[i][-6:]
            if len(set(recent)) <= 2:
                oscillating_agents.add(i)
    
    next_pos = planner.plan_step(pos_dict, goal_dict, oscillating_agents, history)
    
    for i in active:
        old_pos = (int(positions[i][0]), int(positions[i][1]))
        if i in next_pos:
            nx, ny = next_pos[i]
            new_pos = (nx, ny)
            positions[i][0], positions[i][1] = nx, ny
            
            # Track moves (only if actually moved)
            if new_pos != old_pos:
                moves[i] += 1
            
            gx, gy = goals[i]
            if (nx, ny) == (gx, gy):
                reached[i] = True
        
        # Track time (every timestep for active agents)
        times[i] += 1
    
    return history, initialized, moves


def step_pibt(road, planner, positions, goals, reached, times, history=None, moves=None):
    """PIBT step function with anti-oscillation.
    
    Metrics:
        - times[i]: arrival timestep (total delay) - incremented every step for active agents
        - moves[i]: distance traveled - incremented only when agent actually moves
    """
    import random
    n = len(positions)
    active = {i for i in range(n) if not reached[i]}
    
    if history is None:
        history = {i: [] for i in range(n)}
    if moves is None:
        moves = [0] * n
    
    if not active:
        return history, moves
    
    pos_dict = {i: tuple(positions[i]) for i in active}
    goal_dict = {i: tuple(goals[i]) for i in active}
    
    # Track history before planning
    for i in active:
        px, py = positions[i]
        history[i].append((px, py))
        history[i] = history[i][-10:]  # Keep last 10 positions
    
    # Detect oscillating agents
    oscillating_agents = set()
    for i in active:
        if len(history[i]) >= 6:
            recent = history[i][-6:]
            if len(set(recent)) <= 2:
                oscillating_agents.add(i)
    
    next_pos = planner.plan_step(pos_dict, goal_dict, oscillating_agents, history)
    
    for i in active:
        old_pos = (int(positions[i][0]), int(positions[i][1]))
        if i in next_pos:
            nx, ny = next_pos[i]
            new_pos = (nx, ny)
            positions[i][0], positions[i][1] = nx, ny
            
            # Track moves (only if actually moved)
            if new_pos != old_pos:
                moves[i] += 1
            
            gx, gy = goals[i]
            # Check if reached goal - must be at EXACT destination
            if (nx, ny) == (gx, gy):
                reached[i] = True
        
        # Track time (every timestep for active agents)
        times[i] += 1
    
    return history, moves


def step_sp_guided_pibt(road, planner, positions, goals, reached, times, history=None, initialized=None, moves=None):
    """
    SP-Guided PIBT step function.
    
    Combines:
    - BFS optimal paths (from SP) for guidance
    - PIBT collision resolution (priority inheritance)
    
    Args:
        planner: SPGuidedPIBT instance
        initialized: dict tracking if guide paths are computed
        times: list tracking arrival timestep for each agent (incremented every step)
        moves: list tracking actual moves for each agent (incremented only when moving)
    
    Metrics:
        - times[i]: arrival timestep (total delay) - incremented every step for active agents
        - moves[i]: distance traveled - incremented only when agent actually moves
    """
    n = len(positions)
    active = {i for i in range(n) if not reached[i]}
    
    if history is None:
        history = {i: [] for i in range(n)}
    if initialized is None:
        initialized = {'done': False}
    if moves is None:
        moves = [0] * n
    
    if not active:
        return history, initialized, moves
    
    # Compute guide paths once at the start
    if not initialized.get('done', False):
        pos_dict = {i: tuple(positions[i]) for i in range(n)}
        goal_dict = {i: tuple(goals[i]) for i in range(n)}
        planner.compute_guide_paths(pos_dict, goal_dict)
        initialized['done'] = True
    
    pos_dict = {i: tuple(positions[i]) for i in active}
    goal_dict = {i: tuple(goals[i]) for i in active}
    
    # Track history before planning
    for i in active:
        px, py = positions[i]
        history[i].append((px, py))
        history[i] = history[i][-10:]
    
    # Detect oscillating agents
    oscillating_agents = set()
    for i in active:
        if len(history[i]) >= 6:
            recent = history[i][-6:]
            if len(set(recent)) <= 2:
                oscillating_agents.add(i)
    
    next_pos = planner.plan_step(pos_dict, goal_dict, oscillating_agents, history)
    
    for i in active:
        old_pos = (int(positions[i][0]), int(positions[i][1]))
        if i in next_pos:
            nx, ny = next_pos[i]
            new_pos = (nx, ny)
            positions[i][0], positions[i][1] = nx, ny
            
            # Track moves (only if actually moved)
            if new_pos != old_pos:
                moves[i] += 1
            
            gx, gy = goals[i]
            if (nx, ny) == (gx, gy):
                reached[i] = True
        
        # Track time (every timestep for active agents)
        times[i] += 1
    
    return history, initialized, moves


def visualize(num_agents=12, grid_size=80, cell_size=8, seed=42):
    pygame.init()
    pygame.font.init()
    
    road = GridNetwork3x3(grid_size=grid_size)
    
    panel_w = 220
    win_w = grid_size * cell_size
    screen = pygame.display.set_mode((win_w + panel_w, win_w))
    pygame.display.set_caption("3x3 Grid Network - Left Turn Challenge")
    
    font = pygame.font.SysFont("Monaco", 11)
    font_large = pygame.font.SysFont("Monaco", 13, bold=True)
    
    origins, destinations, turn_types = generate_left_turn_agents(road, num_agents, seed)
    n = len(origins)
    
    method = "gsp"
    paused = True
    clock = pygame.time.Clock()
    speed = 8
    current_seed = seed
    sim_step = 0
    
    # GSP state
    gsp_positions = [list(o) for o in origins]
    gsp_goals = list(destinations)
    gsp_reached = [False] * n
    gsp_times = [0] * n
    gsp_prev_moves = {}
    gsp_history = {i: [] for i in range(n)}
    
    # PIBT state
    pibt_planner = PIBTPlanner(road)
    pibt_positions = [list(o) for o in origins]
    pibt_goals = list(destinations)
    pibt_reached = [False] * n
    pibt_times = [0] * n
    pibt_history = {i: [] for i in range(n)}
    
    # SP (Shortest Path) state
    sp_positions = [list(o) for o in origins]
    sp_goals = list(destinations)
    sp_reached = [False] * n
    sp_times = [0] * n
    sp_paths = {i: None for i in range(n)}  # Paths computed once per agent
    
    # SP-PIBT (SP-Guided PIBT) state
    sp_pibt_planner = SPGuidedPIBT(road)
    sp_pibt_positions = [list(o) for o in origins]
    sp_pibt_goals = list(destinations)
    sp_pibt_reached = [False] * n
    sp_pibt_times = [0] * n
    sp_pibt_history = {i: [] for i in range(n)}
    sp_pibt_initialized = {'done': False}
    
    def reset_simulation(new_seed):
        nonlocal origins, destinations, turn_types, n, sim_step
        nonlocal gsp_positions, gsp_goals, gsp_reached, gsp_times, gsp_prev_moves, gsp_history
        nonlocal pibt_positions, pibt_goals, pibt_reached, pibt_times, pibt_history
        nonlocal sp_positions, sp_goals, sp_reached, sp_times, sp_paths
        nonlocal sp_pibt_planner, sp_pibt_positions, sp_pibt_goals, sp_pibt_reached, sp_pibt_times, sp_pibt_history, sp_pibt_initialized
        
        origins, destinations, turn_types = generate_left_turn_agents(road, num_agents, new_seed)
        n = len(origins)
        sim_step = 0
        
        gsp_positions = [list(o) for o in origins]
        gsp_goals = list(destinations)
        gsp_reached = [False] * n
        gsp_times = [0] * n
        gsp_prev_moves = {}
        gsp_history = {i: [] for i in range(n)}
        
        pibt_positions = [list(o) for o in origins]
        pibt_goals = list(destinations)
        pibt_reached = [False] * n
        pibt_times = [0] * n
        pibt_history = {i: [] for i in range(n)}
        
        sp_positions = [list(o) for o in origins]
        sp_goals = list(destinations)
        sp_reached = [False] * n
        sp_times = [0] * n
        sp_paths = {i: None for i in range(n)}
        
        sp_pibt_planner = SPGuidedPIBT(road)
        sp_pibt_positions = [list(o) for o in origins]
        sp_pibt_goals = list(destinations)
        sp_pibt_reached = [False] * n
        sp_pibt_times = [0] * n
        sp_pibt_history = {i: [] for i in range(n)}
        sp_pibt_initialized = {'done': False}
    
    def cell_to_screen(x, y):
        return (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2)
    
    def draw_road():
        screen.fill(COLORS["background"])
        
        for x in range(grid_size):
            for y in range(grid_size):
                if road.is_on_road(x, y):
                    if road.in_intersection(x, y):
                        color = COLORS["intersection"]
                    else:
                        color = COLORS["road"]
                    rect = (x * cell_size, y * cell_size, cell_size, cell_size)
                    pygame.draw.rect(screen, color, rect)
        
        # Draw intersection numbers
        for i, inter in enumerate(road.intersections):
            cx, cy = inter['center']
            sx, sy = cell_to_screen(cx, cy)
            text = font.render(str(i), True, (150, 150, 150))
            screen.blit(text, (sx - 4, sy - 5))
    
    def draw_vehicles():
        if method == "gsp":
            positions = gsp_positions
            reached = gsp_reached
            goals = gsp_goals
        elif method == "sp":
            positions = sp_positions
            reached = sp_reached
            goals = sp_goals
        elif method == "sp_pibt":
            positions = sp_pibt_positions
            reached = sp_pibt_reached
            goals = sp_pibt_goals
        else:  # pibt
            positions = pibt_positions
            reached = pibt_reached
            goals = pibt_goals
        
        for i in range(n):
            px, py = int(positions[i][0]), int(positions[i][1])
            sx, sy = cell_to_screen(px, py)
            
            color = COLORS["vehicle_left"]  # All are left turns
            
            if reached[i]:
                pygame.draw.circle(screen, COLORS["reached"], (sx, sy), 3)
            else:
                pygame.draw.circle(screen, color, (sx, sy), 4)
                gx, gy = goals[i]
                dx, dy = cell_to_screen(gx, gy)
                pygame.draw.circle(screen, color, (dx, dy), 2, 1)
    
    def draw_panel():
        pygame.draw.rect(screen, COLORS["panel"], (win_w, 0, panel_w, win_w))
        pygame.draw.line(screen, (55, 58, 70), (win_w, 0), (win_w, win_w), 2)
        
        px, y = win_w + 12, 15
        
        screen.blit(font_large.render("3x3 GRID NETWORK", True, COLORS["text"]), (px, y))
        y += 20
        screen.blit(font.render("Left Turn Challenge", True, (180, 180, 180)), (px, y))
        y += 25
        
        if method == "gsp":
            display_name = "GSP (Greedy)"
            method_color = (100, 255, 150)
        elif method == "sp":
            display_name = "SP (Shortest Path)"
            method_color = (255, 200, 100)
        elif method == "sp_pibt":
            display_name = "SP-PIBT (Best!)"
            method_color = (255, 100, 200)
        else:  # pibt
            display_name = "PIBT (Priority)"
            method_color = (180, 150, 255)
        screen.blit(font_large.render(f"Method: {display_name}", True, method_color), (px, y))
        y += 25
        
        if method == "gsp":
            completed = sum(gsp_reached)
            total_time = sum(gsp_times)
        elif method == "sp":
            completed = sum(sp_reached)
            total_time = sum(sp_times)
        elif method == "sp_pibt":
            completed = sum(sp_pibt_reached)
            total_time = sum(sp_pibt_times)
        else:  # pibt
            completed = sum(pibt_reached)
            total_time = sum(pibt_times)
        
        screen.blit(font.render(f"Completed: {completed}/{n}", True, COLORS["text"]), (px, y))
        y += 18
        screen.blit(font.render(f"Total time: {total_time}", True, COLORS["text"]), (px, y))
        y += 18
        screen.blit(font.render(f"Step: {sim_step}", True, COLORS["text"]), (px, y))
        y += 25
        
        # Controls
        screen.blit(font.render("Controls:", True, COLORS["text"]), (px, y))
        y += 18
        for line in [
            "1 = GSP (greedy)",
            "2 = PIBT (priority)",
            "3 = SP (shortest path)",
            "4 = SP-PIBT (best!)",
            "SPACE = Pause/Resume",
            "R = Reset",
            "+/- = Speed",
            "Q = Quit",
        ]:
            screen.blit(font.render(line, True, (150, 150, 160)), (px, y))
            y += 16
        
        y = win_w - 35
        status = "PAUSED" if paused else "RUNNING"
        status_color = (255, 180, 80) if paused else (100, 255, 120)
        screen.blit(font_large.render(status, True, status_color), (px, y))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    current_seed = np.random.randint(10000)
                    reset_simulation(current_seed)
                elif event.key == pygame.K_1:
                    method = "gsp"
                    reset_simulation(current_seed)
                elif event.key == pygame.K_2:
                    method = "pibt"
                    reset_simulation(current_seed)
                elif event.key == pygame.K_3:
                    method = "sp"
                    reset_simulation(current_seed)
                elif event.key == pygame.K_4:
                    method = "sp_pibt"
                    reset_simulation(current_seed)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(60, speed + 2)
                elif event.key == pygame.K_MINUS:
                    speed = max(1, speed - 2)
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
        
        if not paused:
            if method == "gsp":
                if not all(gsp_reached):
                    gsp_prev_moves, gsp_history = step_gsp(road, gsp_positions, gsp_goals, 
                                              gsp_reached, gsp_times, set(), gsp_prev_moves, gsp_history)
                    sim_step += 1
            elif method == "sp":
                if not all(sp_reached):
                    sp_paths = step_sp(road, sp_positions, sp_goals,
                              sp_reached, sp_times, set(), sp_paths)
                    sim_step += 1
            elif method == "sp_pibt":
                if not all(sp_pibt_reached):
                    sp_pibt_history, sp_pibt_initialized = step_sp_guided_pibt(
                        road, sp_pibt_planner, sp_pibt_positions, sp_pibt_goals,
                        sp_pibt_reached, sp_pibt_times, sp_pibt_history, sp_pibt_initialized)
                    sim_step += 1
            else:  # pibt
                if not all(pibt_reached):
                    pibt_history = step_pibt(road, pibt_planner, pibt_positions, pibt_goals,
                              pibt_reached, pibt_times, pibt_history)
                    sim_step += 1
        
        draw_road()
        draw_vehicles()
        draw_panel()
        
        pygame.display.flip()
        clock.tick(speed)
    
    pygame.quit()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="3x3 Grid Network - Left Turn Challenge")
    p.add_argument("--agents", type=int, default=12)
    p.add_argument("--grid", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    visualize(num_agents=args.agents, grid_size=args.grid, seed=args.seed)
