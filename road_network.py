"""
Realistic 4-way road network intersection.

- RoadNetwork: legacy one-lane-per-arm layout.
- RoadNetwork4Way: two lanes per arm (inbound + outbound), right-hand driving,
  vehicles can turn left, straight, or right. Agents only on the road.
"""

import numpy as np
from typing import List, Tuple, Set, Optional

# (x, y) or (int, int)
Cell = Tuple[int, int]


# ---------------------------------------------------------------------------
# 4-way with 2 lanes per arm (inbound / outbound), left/straight/right turns
# ---------------------------------------------------------------------------

# Arm index: 0=North (top), 1=East (right), 2=South (bottom), 3=West (left). Clockwise.
# From arm i: right=(i+1)%4, straight=(i+2)%4, left=(i+3)%4.
TURN_RIGHT = 0
TURN_STRAIGHT = 1
TURN_LEFT = 2


class RoadNetwork4Way:
    """
    4-way intersection: each arm has two lanes (inbound toward center, outbound away).
    Right-hand driving. Vehicles spawn on inbound lanes, exit on outbound lanes.
    Turn: left, straight, or right.
    """

    def __init__(
        self,
        grid_size: int = 64,
        intersection_half: int = 10,
    ):
        self.grid_size = grid_size
        self.center = grid_size // 2
        self.half = intersection_half
        self.lo = self.center - self.half
        self.hi = self.center + self.half
        self._road_cells: Optional[Set[Cell]] = None

        # US right-hand driving: each direction uses the lane on the driver's RIGHT.
        # North arm (vertical, top): Southbound (inbound) → right is WEST = (center, y). Northbound (outbound) → right is EAST = (center+1, y).
        # South arm: Northbound (inbound) → right is EAST = (center+1, y). Southbound (outbound) → right is WEST = (center, y).
        # East arm (horizontal): Westbound (inbound) → right is NORTH = (x, center). Eastbound (outbound) → right is SOUTH = (x, center+1).
        # West arm: Eastbound (inbound) → right is SOUTH = (x, center+1). Westbound (outbound) → right is NORTH = (x, center).
        self._inbound_north: List[Cell] = [(self.center, y) for y in range(0, self.lo)]       # southbound = right lane is west
        self._outbound_north: List[Cell] = [(self.center + 1, y) for y in range(0, self.lo)]  # northbound = right lane is east
        self._inbound_south: List[Cell] = [(self.center + 1, y) for y in range(self.hi + 1, self.grid_size)]  # northbound
        self._outbound_south: List[Cell] = [(self.center, y) for y in range(self.hi + 1, self.grid_size)]     # southbound
        self._inbound_east: List[Cell] = [(x, self.center) for x in range(self.hi + 1, self.grid_size)]        # westbound = right is north
        self._outbound_east: List[Cell] = [(x, self.center + 1) for x in range(self.hi + 1, self.grid_size)]  # eastbound = right is south
        self._inbound_west: List[Cell] = [(x, self.center + 1) for x in range(0, self.lo)]    # W→E = south-sided
        self._outbound_west: List[Cell] = [(x, self.center) for x in range(0, self.lo)]      # E→W = north-sided

    def _other_lane(self, x: int, y: int) -> Optional[Cell]:
        """The other lane on the same arm (for same arm only); None if in intersection or no other lane."""
        if self.lo <= x <= self.hi and self.lo <= y <= self.hi:
            return None
        if 0 <= y < self.lo or self.hi < y < self.grid_size:
            if x == self.center:
                return (self.center + 1, y)
            if x == self.center + 1:
                return (self.center, y)
        if 0 <= x < self.lo or self.hi < x < self.grid_size:
            if y == self.center:
                return (x, self.center + 1)
            if y == self.center + 1:
                return (x, self.center)
        return None

    def is_on_road(self, x: int, y: int) -> bool:
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        if self.lo <= x <= self.hi and self.lo <= y <= self.hi:
            return True
        # North arm (2 lanes)
        if 0 <= y < self.lo and x in (self.center, self.center + 1):
            return True
        # South arm
        if self.hi < y < self.grid_size and x in (self.center, self.center + 1):
            return True
        # East arm
        if self.hi < x < self.grid_size and y in (self.center, self.center + 1):
            return True
        # West arm
        if 0 <= x < self.lo and y in (self.center, self.center + 1):
            return True
        return False

    def get_road_cells(self) -> Set[Cell]:
        if self._road_cells is not None:
            return self._road_cells
        out: Set[Cell] = set()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.is_on_road(x, y):
                    out.add((x, y))
        self._road_cells = out
        return out

    def get_road_neighbors(self, x: int, y: int) -> List[Cell]:
        """4-connected neighbors on the road. Excludes the other lane on the same arm so vehicles stay in their lane."""
        candidates = [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        other = self._other_lane(x, y)
        return [c for c in candidates if self.is_on_road(c[0], c[1]) and c != other]

    def is_outbound_lane(self, x: int, y: int) -> bool:
        """True if (x, y) is on an outbound lane (not intersection, not inbound)."""
        if self.in_intersection(x, y):
            return False
        # North arm outbound: (center+1, y) for y < lo
        if 0 <= y < self.lo and x == self.center + 1:
            return True
        # South arm outbound: (center, y) for y > hi
        if self.hi < y < self.grid_size and x == self.center:
            return True
        # East arm outbound: (x, center+1) for x > hi
        if self.hi < x < self.grid_size and y == self.center + 1:
            return True
        # West arm outbound: (x, center) for x < lo
        if 0 <= x < self.lo and y == self.center:
            return True
        return False

    def is_inbound_lane(self, x: int, y: int) -> bool:
        """True if (x, y) is on an inbound lane."""
        if self.in_intersection(x, y):
            return False
        # North arm inbound: (center, y) for y < lo
        if 0 <= y < self.lo and x == self.center:
            return True
        # South arm inbound: (center+1, y) for y > hi
        if self.hi < y < self.grid_size and x == self.center + 1:
            return True
        # East arm inbound: (x, center) for x > hi
        if self.hi < x < self.grid_size and y == self.center:
            return True
        # West arm inbound: (x, center+1) for x < lo
        if 0 <= x < self.lo and y == self.center + 1:
            return True
        return False

    def get_outbound_entry_cell(self, arm: int) -> Cell:
        """Get the first cell of the outbound lane for the given arm (cell adjacent to intersection)."""
        if arm == 0:  # North
            return (self.center + 1, self.lo - 1)
        if arm == 1:  # East
            return (self.hi + 1, self.center + 1)
        if arm == 2:  # South
            return (self.center, self.hi + 1)
        # arm == 3: West
        return (self.lo - 1, self.center)

    def get_road_neighbors_float(self, pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        ix, iy = int(round(pos[0])), int(round(pos[1]))
        nb = self.get_road_neighbors(ix, iy)
        return [(float(a), float(b)) for a, b in nb]

    def distance(self, a: Cell, b: Cell) -> float:
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def distance_float(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def path_distance(self, start: Cell, goal: Cell, max_steps: int = 500) -> float:
        """BFS shortest path length (number of steps) on the road graph. Returns max_steps if unreachable."""
        if start == goal:
            return 0.0
        from collections import deque
        gx, gy = int(goal[0]), int(goal[1])
        if not self.is_on_road(gx, gy):
            return float(max_steps)
        queue: deque = deque([(int(start[0]), int(start[1]), 0)])
        visited = {start}
        while queue:
            x, y, d = queue.popleft()
            if (x, y) == (gx, gy):
                return float(d)
            if d >= max_steps:
                break
            for n in self.get_road_neighbors(x, y):
                if n not in visited:
                    visited.add(n)
                    queue.append((n[0], n[1], d + 1))
        return float(max_steps)

    def get_valid_moves(self, x: int, y: int, reserved: Set[Cell] = None) -> List[Cell]:
        """Get valid moves from (x, y) respecting lane direction rules and reserved cells.
        
        Lane rules (right-hand driving, enforcing DIRECTION):
        - North inbound (x=center, y<lo): can only move SOUTH (y+1) toward intersection
        - South inbound (x=center+1, y>hi): can only move NORTH (y-1) toward intersection
        - East inbound (x>hi, y=center): can only move WEST (x-1) toward intersection
        - West inbound (x<lo, y=center+1): can only move EAST (x+1) toward intersection
        - In INTERSECTION: can move within or exit to OUTBOUND lanes only
        - North outbound (x=center+1, y<lo): can only move NORTH (y-1) away from intersection
        - South outbound (x=center, y>hi): can only move SOUTH (y+1) away from intersection
        - East outbound (x>hi, y=center+1): can only move EAST (x+1) away from intersection
        - West outbound (x<lo, y=center): can only move WEST (x-1) away from intersection
        """
        if reserved is None:
            reserved = set()
        
        raw_neighbors = self.get_road_neighbors(x, y)
        in_int = self.in_intersection(x, y)
        
        valid = []
        for (nx, ny) in raw_neighbors:
            if (nx, ny) in reserved:
                continue
            if (nx, ny) == (x, y):  # Waiting is always ok
                valid.append((nx, ny))
                continue
            
            c_in_int = self.in_intersection(nx, ny)
            c_outbound = self.is_outbound_lane(nx, ny)
            
            if in_int:
                # From intersection: only allow intersection cells or outbound lanes
                if c_in_int or c_outbound:
                    valid.append((nx, ny))
            else:
                # On a lane - check direction is correct
                dx, dy = nx - x, ny - y
                
                # North inbound (x=center, y < lo): must go south (dy > 0)
                if x == self.center and y < self.lo:
                    if dy > 0 or c_in_int:  # toward intersection or into intersection
                        valid.append((nx, ny))
                # South inbound (x=center+1, y > hi): must go north (dy < 0)
                elif x == self.center + 1 and y > self.hi:
                    if dy < 0 or c_in_int:
                        valid.append((nx, ny))
                # East inbound (y=center, x > hi): must go west (dx < 0)
                elif y == self.center and x > self.hi:
                    if dx < 0 or c_in_int:
                        valid.append((nx, ny))
                # West inbound (y=center+1, x < lo): must go east (dx > 0)
                elif y == self.center + 1 and x < self.lo:
                    if dx > 0 or c_in_int:
                        valid.append((nx, ny))
                # North outbound (x=center+1, y < lo): must go north (dy < 0)
                elif x == self.center + 1 and y < self.lo:
                    if dy < 0:
                        valid.append((nx, ny))
                # South outbound (x=center, y > hi): must go south (dy > 0)
                elif x == self.center and y > self.hi:
                    if dy > 0:
                        valid.append((nx, ny))
                # East outbound (y=center+1, x > hi): must go east (dx > 0)
                elif y == self.center + 1 and x > self.hi:
                    if dx > 0:
                        valid.append((nx, ny))
                # West outbound (y=center, x < lo): must go west (dx < 0)
                elif y == self.center and x < self.lo:
                    if dx < 0:
                        valid.append((nx, ny))
        
        return valid

    def shortest_path(self, start: Cell, goal: Cell, reserved: Set[Cell] = None) -> Tuple[Optional[List[Cell]], Optional[Cell]]:
        """BFS shortest path respecting lane direction rules.
        
        Returns (path, next_move) where path is the full path from start to goal,
        and next_move is the first step to take (path[1] if path exists).
        Returns (None, None) if no path exists.
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
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            # Get valid moves respecting lane direction rules (no reservation check for path finding)
            valid_moves = self.get_valid_moves(current[0], current[1])
            
            for next_cell in valid_moves:
                next_cell = (int(next_cell[0]), int(next_cell[1]))
                if next_cell == current:  # Skip waiting
                    continue
                if next_cell in visited:
                    continue
                
                new_path = path + [next_cell]
                
                # Check if close enough to goal (within 1 cell)
                if abs(next_cell[0] - goal[0]) <= 1 and abs(next_cell[1] - goal[1]) <= 1:
                    next_move = new_path[1] if len(new_path) > 1 else None
                    return new_path, next_move
                
                visited.add(next_cell)
                queue.append((next_cell, new_path))
        
        return None, None

    def in_intersection(self, x: int, y: int) -> bool:
        return self.lo <= x <= self.hi and self.lo <= y <= self.hi

    def get_intersection_bounds(self) -> Tuple[int, int, int, int]:
        return (self.lo, self.lo, self.hi, self.hi)

    # Spawn on inbound lanes (far end). Dest on outbound lanes by turn.
    def get_spawn_cells_inbound(self, arm: int, num_cells: int = 4) -> List[Cell]:
        if arm == 0:
            lane = self._inbound_north
            return lane[:num_cells] if len(lane) >= num_cells else lane
        if arm == 1:
            lane = self._inbound_east
            return lane[-num_cells:] if len(lane) >= num_cells else lane
        if arm == 2:
            lane = self._inbound_south
            return lane[-num_cells:] if len(lane) >= num_cells else lane
        lane = self._inbound_west
        return lane[:num_cells] if len(lane) >= num_cells else lane

    def get_dest_cells_outbound(self, arm: int, num_cells: int = 4) -> List[Cell]:
        if arm == 0:
            lane = self._outbound_north
            return lane[:num_cells] if len(lane) >= num_cells else lane
        if arm == 1:
            lane = self._outbound_east
            return lane[-num_cells:] if len(lane) >= num_cells else lane
        if arm == 2:
            lane = self._outbound_south
            return lane[-num_cells:] if len(lane) >= num_cells else lane
        lane = self._outbound_west
        return lane[:num_cells] if len(lane) >= num_cells else lane

    def create_pibt_planner(self, collision_radius: float = 0.6):
        """Create a PIBT planner that enforces right-hand driving rules."""
        try:
            from pibt_guided import PIBTPlanner
        except ImportError:
            return None
        road = self
        class PIBTRoadPlanner(PIBTPlanner):
            def get_neighbors(self, pos):
                ix, iy = int(round(pos[0])), int(round(pos[1]))
                # Use get_valid_moves to enforce lane direction rules (right-hand driving)
                nb = road.get_valid_moves(ix, iy)
                return [(float(a), float(b)) for a, b in nb]
        return PIBTRoadPlanner(self.grid_size, collision_radius)

    def create_paper_pibt_planner(self, collision_radius: float = 0.6):
        """Create a Paper-PIBT planner (time-based priority, Okumura et al.) with right-hand driving."""
        try:
            from pibt_guided import PaperPIBTPlanner
        except ImportError:
            return None
        road = self
        class PaperPIBTRoadPlanner(PaperPIBTPlanner):
            def get_neighbors(self, pos):
                ix, iy = int(round(pos[0])), int(round(pos[1]))
                nb = road.get_valid_moves(ix, iy)
                return [(float(a), float(b)) for a, b in nb]
        return PaperPIBTRoadPlanner(self.grid_size, collision_radius)


# ---------------------------------------------------------------------------
# Legacy one-lane-per-arm (kept for backward compatibility)
# ---------------------------------------------------------------------------


class RoadNetwork:
    """
    Grid-based 4-way intersection: 4 one-lane arms + central intersection.
    
    Layout (example grid_size=60, center=30, half=10):
        North lane: (center, 0) .. (center, lo-1)   [vertical]
        South lane: (center, hi+1) .. (center, gs-1)
        East lane:  (hi+1, center) .. (gs-1, center)
        West lane:  (0, center) .. (lo-1, center)
        Intersection: square [lo, hi] x [lo, hi]  (large central area)
    """

    def __init__(
        self,
        grid_size: int = 60,
        intersection_half: int = 10,
    ):
        self.grid_size = grid_size
        self.center = grid_size // 2
        self.half = intersection_half  # intersection is [c-half, c+half] inclusive
        self.lo = self.center - self.half   # e.g. 20
        self.hi = self.center + self.half   # e.g. 40
        self._road_cells: Optional[Set[Cell]] = None

    def is_on_road(self, x: int, y: int) -> bool:
        """True if (x, y) is a valid road cell (lane or intersection)."""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        # Intersection (large central square)
        if self.lo <= x <= self.hi and self.lo <= y <= self.hi:
            return True
        # North lane (one column, top to just before intersection)
        if x == self.center and 0 <= y < self.lo:
            return True
        # South lane
        if x == self.center and self.hi < y < self.grid_size:
            return True
        # East lane
        if self.hi < x < self.grid_size and y == self.center:
            return True
        # West lane
        if 0 <= x < self.lo and y == self.center:
            return True
        return False

    def get_road_cells(self) -> Set[Cell]:
        """Set of all road cells (cached)."""
        if self._road_cells is not None:
            return self._road_cells
        out: Set[Cell] = set()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.is_on_road(x, y):
                    out.add((x, y))
        self._road_cells = out
        return out

    def get_road_neighbors(self, x: int, y: int) -> List[Cell]:
        """4-connected neighbors that lie on the road (includes wait: same cell)."""
        candidates = [
            (x, y),      # wait
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
        ]
        return [c for c in candidates if self.is_on_road(c[0], c[1])]

    def get_road_neighbors_float(self, pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Same but for float positions (snap to cell for check)."""
        ix, iy = int(round(pos[0])), int(round(pos[1]))
        neighbors = self.get_road_neighbors(ix, iy)
        return [(float(nx), float(ny)) for nx, ny in neighbors]

    def distance(self, a: Cell, b: Cell) -> float:
        """Euclidean distance between two cells."""
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def distance_float(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # ---------- Spawn / destination zones (one per arm) ----------
    # Each arm has an "entry" end (far from center) and connects to intersection at the near end.

    def get_north_lane_cells(self) -> List[Cell]:
        return [(self.center, y) for y in range(0, self.lo)]
    def get_south_lane_cells(self) -> List[Cell]:
        return [(self.center, y) for y in range(self.hi + 1, self.grid_size)]
    def get_east_lane_cells(self) -> List[Cell]:
        return [(x, self.center) for x in range(self.hi + 1, self.grid_size)]
    def get_west_lane_cells(self) -> List[Cell]:
        return [(x, self.center) for x in range(0, self.lo)]

    def get_spawn_cells_north(self, num_cells: int = 3) -> List[Cell]:
        """Cells at the north end (top of grid) for spawning agents entering from north."""
        lane = self.get_north_lane_cells()
        return lane[:num_cells] if len(lane) >= num_cells else lane
    def get_spawn_cells_south(self, num_cells: int = 3) -> List[Cell]:
        lane = self.get_south_lane_cells()
        return lane[-num_cells:] if len(lane) >= num_cells else lane
    def get_spawn_cells_east(self, num_cells: int = 3) -> List[Cell]:
        lane = self.get_east_lane_cells()
        return lane[-num_cells:] if len(lane) >= num_cells else lane
    def get_spawn_cells_west(self, num_cells: int = 3) -> List[Cell]:
        lane = self.get_west_lane_cells()
        return lane[:num_cells] if len(lane) >= num_cells else lane

    def get_dest_cells_north(self, num_cells: int = 3) -> List[Cell]:
        return self.get_spawn_cells_north(num_cells)
    def get_dest_cells_south(self, num_cells: int = 3) -> List[Cell]:
        return self.get_spawn_cells_south(num_cells)
    def get_dest_cells_east(self, num_cells: int = 3) -> List[Cell]:
        return self.get_spawn_cells_east(num_cells)
    def get_dest_cells_west(self, num_cells: int = 3) -> List[Cell]:
        return self.get_spawn_cells_west(num_cells)

    def get_intersection_bounds(self) -> Tuple[int, int, int, int]:
        """(x_min, y_min, x_max, y_max) for the intersection rectangle."""
        return (self.lo, self.lo, self.hi, self.hi)

    def in_intersection(self, x: int, y: int) -> bool:
        return self.lo <= x <= self.hi and self.lo <= y <= self.hi

    def create_pibt_planner(self, collision_radius: float = 0.6):
        """Create a PIBT planner that only moves on the road (one agent per cell: radius < 1)."""
        try:
            from pibt_guided import PIBTPlanner
        except ImportError:
            return None
        road = self

        class PIBTRoadPlanner(PIBTPlanner):
            def get_neighbors(self, pos):
                ix, iy = int(round(pos[0])), int(round(pos[1]))
                nb = road.get_road_neighbors(ix, iy)
                return [(float(a), float(b)) for a, b in nb]

        return PIBTRoadPlanner(self.grid_size, collision_radius)
