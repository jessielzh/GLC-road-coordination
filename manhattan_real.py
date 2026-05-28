#!/usr/bin/env python3
"""
Real Manhattan road network from OpenStreetMap via OSMnx.

Graph-based: nodes = OSM nodes, edges = road segments.
Same conceptual interface as grid networks: shortest_path, get_valid_moves, distance.

Simulation time model:
  - Discrete timesteps. Each step, an agent either moves to an adjacent node (1 edge) or waits.
  - Traversing one edge = 1 time unit (no multi-step edge travel).
  - node_capacity(n): max agents that may occupy node n at once (intersection size; 2--12).
  - edge_capacity(u,v): max agents that may traverse (u,v) in one timestep (from OSM lanes).
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Callable
from collections import deque
import networkx as nx

try:
    import osmnx as ox
    _HAS_OSMNX = True
except ImportError:
    _HAS_OSMNX = False


def _prune_sinks_iterative(G) -> int:
    """
    Iteratively remove sink nodes (out-degree 0) and nodes that become sinks.
    Returns total number of nodes removed.
    """
    total_removed = 0
    while True:
        sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
        if not sinks:
            break
        G.remove_nodes_from(sinks)
        total_removed += len(sinks)
    return total_removed


class OSMGraphRoad:
    """
    Real Manhattan street network from OSMnx.
    
    Positions are node IDs (integers). Graph is directed (respects one-way streets).
    Sink nodes (no outlet) are pruned iteratively for fair simulation.
    """

    def __init__(self, cache_path: Optional[str] = None):
        """
        Load Manhattan road network from OSM.
        
        Args:
            cache_path: Optional path to save/load graph (speeds up repeated runs).
        """
        import os
        if not _HAS_OSMNX:
            raise ImportError("osmnx is required. pip install osmnx")

        if cache_path:
            try:
                if os.path.isfile(cache_path):
                    self.G = ox.load_graphml(cache_path)
                    n_before = self.G.number_of_nodes()
                    self._nodes_pruned = _prune_sinks_iterative(self.G)
                    self._build_index()
                    if cache_path and self._nodes_pruned > 0:
                        try:
                            ox.save_graphml(self.G, cache_path)
                        except Exception:
                            pass
                    return
            except Exception:
                pass

        self.G = ox.graph_from_place(
            "Manhattan, New York, NY, USA",
            network_type="drive",
            simplify=True,
        )
        self.G = ox.project_graph(self.G)
        self._nodes_pruned = _prune_sinks_iterative(self.G)
        if cache_path:
            try:
                ox.save_graphml(self.G, cache_path)
            except Exception:
                pass
        self._build_index()

    def nodes_pruned(self) -> int:
        """Number of sink nodes removed by iterative pruning."""
        return getattr(self, "_nodes_pruned", 0)

    def valid_nodes(self) -> Set[int]:
        """Set of node IDs in the pruned graph."""
        return set(self.G.nodes())

    def _parse_lanes(self, val) -> int:
        """Parse OSM lanes attribute to int. Handles '2', ['3','4'], etc."""
        if val is None:
            return 1
        if isinstance(val, (list, tuple)):
            val = val[0] if val else 1
        try:
            return max(1, int(float(str(val).strip())))
        except (ValueError, TypeError):
            return 1

    def _build_index(self):
        """Build node position index and capacity indices from OSM."""
        self._node_to_xy: Dict[int, Tuple[float, float]] = {}
        for node, data in self.G.nodes(data=True):
            x = data.get("x", data.get("lon", 0))
            y = data.get("y", data.get("lat", 0))
            self._node_to_xy[node] = (x, y)

        # Edge capacity from OSM 'lanes' (lanes per direction)
        self._edge_capacity: Dict[Tuple[int, int], int] = {}
        for u, v, key, data in self.G.edges(keys=True, data=True):
            cap = self._parse_lanes(data.get("lanes"))
            self._edge_capacity[(u, v)] = cap

        # Node capacity: infer from incident edge lanes (sum of incoming lanes, capped)
        self._node_capacity: Dict[int, int] = {}
        for node in self.G.nodes():
            total_lanes = 0
            for _, _, key, data in self.G.in_edges(node, keys=True, data=True):
                total_lanes += self._parse_lanes(data.get("lanes"))
            # Heuristic: cap between 2 and 12, use max(2, total_lanes//2) as base
            street_count = int(self.G.nodes[node].get("street_count", 2) or 2)
            cap = min(12, max(2, max(total_lanes // 2, street_count)))
            self._node_capacity[node] = cap

    def node_capacity(self, node_id: int) -> int:
        """Max agents allowed at this node (intersection) at the same time. Range 2--12 from OSM; default 2."""
        return self._node_capacity.get(node_id, 2)

    def edge_capacity(self, u: int, v: int) -> int:
        """Max agents that can traverse edge (u,v) in one timestep (from OSM lanes per direction). One edge = 1 time unit."""
        return self._edge_capacity.get((u, v), 1)

    def get_neighbors(self, node_id: int) -> List[int]:
        """Out-neighbors (successors) for directed graph."""
        return list(self.G.successors(node_id))

    def get_valid_moves(self, node_id: int, reserved: Set[int] = None) -> List[int]:
        """Neighbors (legacy: reserved as set for capacity=1). Use get_valid_moves_capacity for capacity."""
        if reserved is None:
            reserved = set()
        return [n for n in self.get_neighbors(node_id) if n not in reserved]

    def get_valid_moves_capacity(
        self, pos: int, reserved_count: Dict[int, int], edge_usage: Dict[Tuple[int, int], int]
    ) -> List[int]:
        """Neighbors reachable respecting node and edge capacity. Includes wait (pos)."""
        candidates = [pos] + self.get_neighbors(pos)
        valid = []
        for n in candidates:
            if n == pos:
                if reserved_count.get(pos, 0) < self.node_capacity(pos):
                    valid.append(n)
            else:
                if reserved_count.get(n, 0) < self.node_capacity(n) and edge_usage.get((pos, n), 0) < self.edge_capacity(pos, n):
                    valid.append(n)
        return valid

    def distance(self, a: int, b: int) -> float:
        """Graph shortest path length (number of edges)."""
        try:
            return nx.shortest_path_length(self.G, a, b)
        except nx.NetworkXNoPath:
            return float("inf")

    def shortest_path(self, start: int, goal: int, reserved: Set[int] = None) -> Tuple[Optional[List[int]], Optional[int]]:
        """BFS shortest path. Returns (path, next_move) or (None, None)."""
        if reserved is None:
            reserved = set()
        if start == goal:
            return [start], None
        try:
            path = nx.shortest_path(self.G, start, goal)
            if len(path) < 2:
                return path, None
            return path, path[1]
        except nx.NetworkXNoPath:
            return None, None

    def in_intersection(self, node_id: int) -> bool:
        """Graph nodes don't have intersection concept; all nodes are valid."""
        return True

    def node_position(self, node_id: int) -> Tuple[float, float]:
        """Get (x, y) for a node."""
        return self._node_to_xy.get(node_id, (0.0, 0.0))

    def nearest_node(self, x: float, y: float) -> int:
        """Find nearest graph node to (x, y) in projected coords."""
        try:
            return ox.distance.nearest_nodes(self.G, x, y)
        except Exception:
            pass
        if not hasattr(self, "_nodes_xy"):
            self._nodes_xy = np.array([self._node_to_xy[n] for n in self.G.nodes()])
            self._node_list = list(self.G.nodes())
        xy = np.array([x, y])
        dists = np.linalg.norm(self._nodes_xy - xy, axis=1)
        return self._node_list[int(np.argmin(dists))]

    def num_nodes(self) -> int:
        return self.G.number_of_nodes()


def compute_ideal_graph(
    road: OSMGraphRoad, origins: List[int], destinations: List[int],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    progress_interval: int = 2000,
) -> dict:
    """IDEAL = sum of shortest path lengths (no blocking)."""
    total = 0
    n = len(origins)
    for i in range(n):
        path, _ = road.shortest_path(origins[i], destinations[i])
        if path:
            total += len(path) - 1
        else:
            total += 100000
        if progress_callback and (i + 1) % progress_interval == 0:
            progress_callback(i + 1, n)
    return {"total_delay": total, "completion_rate": 1.0}


def run_sp_graph(
    road: OSMGraphRoad, origins: List[int], destinations: List[int],
    max_steps: int = 50000,
    progress_callback: Optional[Callable[[int, int, int, int, int], None]] = None,
    progress_interval: int = 1000,
) -> dict:
    """Shortest path: BFS path once, wait if blocked. Respects node and edge capacity."""
    n = len(origins)
    positions = list(origins)
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    paths = {i: None for i in range(n)}

    for step in range(max_steps):
        if progress_callback and step > 0 and step % progress_interval == 0:
            progress_callback(step, max_steps, sum(reached), n, sum(times))
        if all(reached):
            break
        next_positions = [None] * n
        reserved_count: Dict[int, int] = {}
        edge_usage: Dict[Tuple[int, int], int] = {}

        def can_move_to(pos: int, next_pos: int) -> bool:
            if reserved_count.get(next_pos, 0) >= road.node_capacity(next_pos):
                return False
            if next_pos != pos and edge_usage.get((pos, next_pos), 0) >= road.edge_capacity(pos, next_pos):
                return False
            return True

        def reserve(pos: int, next_pos: int):
            reserved_count[next_pos] = reserved_count.get(next_pos, 0) + 1
            if next_pos != pos:
                edge_usage[(pos, next_pos)] = edge_usage.get((pos, next_pos), 0) + 1

        active = sorted([i for i in range(n) if not reached[i]], key=lambda i: road.distance(positions[i], goals[i]))

        for i in active:
            pos = positions[i]
            goal = goals[i]
            if pos == goal:
                reached[i] = True
                next_positions[i] = goal
                reserve(pos, goal)
                times[i] += 1
                continue
            if paths[i] is None:
                path, _ = road.shortest_path(pos, goal)
                paths[i] = path[1:] if path and len(path) > 1 else []
            next_cand = paths[i][0] if paths[i] else pos
            if paths[i] and can_move_to(pos, next_cand):
                next_positions[i] = next_cand
                reserve(pos, next_cand)
                paths[i] = paths[i][1:]
            else:
                next_positions[i] = pos
                reserve(pos, pos)
            times[i] += 1

        for i in range(n):
            if next_positions[i] is not None:
                positions[i] = next_positions[i]

    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    return {"total_delay": sum(times), "completion_rate": sum(reached) / n if n else 0, "completed": sum(reached), "n": n}


def _tap_msa_paths(
    road: "OSMGraphRoad",
    origins: List[int],
    destinations: List[int],
    max_iter: int = 50,
    bpr_a: float = 0.15,
    bpr_b: float = 4.0,
) -> List[List[int]]:
    """
    Traffic Assignment (User Equilibrium) via Method of Successive Averages (MSA)
    with BPR link cost: t_e = 1 + a * (flow_e / cap_e)^b.
    Returns one path per agent (list of nodes from origin to destination).
    """
    n = len(origins)
    G = road.G
    # Collect all (u,v) edges and capacities
    edges: List[Tuple[int, int]] = []
    for u in G:
        for v in G.successors(u):
            edges.append((u, v))
    cap = lambda u, v: max(1, road.edge_capacity(u, v))
    flow: Dict[Tuple[int, int], float] = {e: 0.0 for e in edges}

    # Last iteration paths (one path per agent)
    paths: List[Optional[List[int]]] = [None] * n

    for k in range(1, max_iter + 1):
        # BPR cost per edge
        cost = {}
        for (u, v) in edges:
            x, c = flow[(u, v)], cap(u, v)
            cost[(u, v)] = 1.0 + bpr_a * (x / c) ** bpr_b
        # Weighted graph for this iteration
        H = nx.DiGraph()
        for (u, v) in edges:
            H.add_edge(u, v, weight=cost[(u, v)])
        # All-or-nothing: shortest path per agent
        flow_new: Dict[Tuple[int, int], float] = {e: 0.0 for e in edges}
        for i in range(n):
            o, d = origins[i], destinations[i]
            if o == d:
                paths[i] = [o]
                continue
            try:
                path_i = nx.shortest_path(H, o, d, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                path_i = nx.shortest_path(G, o, d)
            paths[i] = path_i
            for j in range(len(path_i) - 1):
                e = (path_i[j], path_i[j + 1])
                if e in flow_new:
                    flow_new[e] += 1.0
        # MSA update
        for e in edges:
            flow[e] = (1.0 - 1.0 / k) * flow[e] + (1.0 / k) * flow_new[e]

    # Fallback to graph shortest path if any path is missing
    for i in range(n):
        if paths[i] is None or len(paths[i]) < 2 and origins[i] != destinations[i]:
            try:
                paths[i] = nx.shortest_path(G, origins[i], destinations[i])
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                paths[i] = [origins[i], destinations[i]] if origins[i] != destinations[i] else [origins[i]]
    return [p if p is not None else [origins[i], destinations[i]] for i, p in enumerate(paths)]


def run_tap_graph(
    road: OSMGraphRoad,
    origins: List[int],
    destinations: List[int],
    max_steps: int = 50000,
    progress_callback: Optional[Callable[[int, int, int, int, int], None]] = None,
    progress_interval: int = 1000,
    tap_iter: int = 50,
) -> dict:
    """TAP baseline: User Equilibrium (MSA + BPR) path assignment, then simulate with capacity; wait if blocked."""
    n = len(origins)
    assigned_paths = _tap_msa_paths(road, origins, destinations, max_iter=tap_iter)
    positions = list(origins)
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    # paths[i] = remaining nodes to visit (excluding current position)
    paths = {}
    for i in range(n):
        full = assigned_paths[i]
        paths[i] = full[1:] if len(full) > 1 else []

    for step in range(max_steps):
        if progress_callback and step > 0 and step % progress_interval == 0:
            progress_callback(step, max_steps, sum(reached), n, sum(times))
        if all(reached):
            break
        next_positions = [None] * n
        reserved_count: Dict[int, int] = {}
        edge_usage: Dict[Tuple[int, int], int] = {}

        def can_move_to(pos: int, next_pos: int) -> bool:
            if reserved_count.get(next_pos, 0) >= road.node_capacity(next_pos):
                return False
            if next_pos != pos and edge_usage.get((pos, next_pos), 0) >= road.edge_capacity(pos, next_pos):
                return False
            return True

        def reserve(pos: int, next_pos: int):
            reserved_count[next_pos] = reserved_count.get(next_pos, 0) + 1
            if next_pos != pos:
                edge_usage[(pos, next_pos)] = edge_usage.get((pos, next_pos), 0) + 1

        active = sorted([i for i in range(n) if not reached[i]], key=lambda i: road.distance(positions[i], goals[i]))

        for i in active:
            pos = positions[i]
            goal = goals[i]
            if pos == goal:
                reached[i] = True
                next_positions[i] = goal
                reserve(pos, goal)
                times[i] += 1
                continue
            next_cand = paths[i][0] if paths[i] else pos
            if paths[i] and can_move_to(pos, next_cand):
                next_positions[i] = next_cand
                reserve(pos, next_cand)
                paths[i] = paths[i][1:]
            else:
                next_positions[i] = pos
                reserve(pos, pos)
            times[i] += 1

        for i in range(n):
            if next_positions[i] is not None:
                positions[i] = next_positions[i]

    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    return {"total_delay": sum(times), "completion_rate": sum(reached) / n if n else 0, "completed": sum(reached), "n": n}


def run_sp_graph_with_trajectory(
    road: OSMGraphRoad, origins: List[int], destinations: List[int], max_steps: int = 50000
) -> Tuple[dict, List[Dict[int, int]], List[int]]:
    """SP (shortest path, wait if blocked) on graph. Returns (results, trajectory, total_delay_per_step). Can deadlock."""
    n = len(origins)
    positions = list(origins)
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    paths = {i: None for i in range(n)}
    trajectory: List[Dict[int, int]] = []
    total_delay_per_step: List[int] = []

    for step in range(max_steps):
        total_delay_per_step.append(sum(times))
        trajectory.append(dict(enumerate(positions)))
        if all(reached):
            break
        next_positions = [None] * n
        reserved_count: Dict[int, int] = {}
        edge_usage: Dict[Tuple[int, int], int] = {}

        def can_move_to(pos: int, next_pos: int) -> bool:
            if reserved_count.get(next_pos, 0) >= road.node_capacity(next_pos):
                return False
            if next_pos != pos and edge_usage.get((pos, next_pos), 0) >= road.edge_capacity(pos, next_pos):
                return False
            return True

        def reserve(pos: int, next_pos: int):
            reserved_count[next_pos] = reserved_count.get(next_pos, 0) + 1
            if next_pos != pos:
                edge_usage[(pos, next_pos)] = edge_usage.get((pos, next_pos), 0) + 1

        active = sorted([i for i in range(n) if not reached[i]], key=lambda i: road.distance(positions[i], goals[i]))

        for i in active:
            pos = positions[i]
            goal = goals[i]
            if pos == goal:
                reached[i] = True
                next_positions[i] = goal
                reserve(pos, goal)
                times[i] += 1
                continue
            if paths[i] is None:
                path, _ = road.shortest_path(pos, goal)
                paths[i] = path[1:] if path and len(path) > 1 else []
            next_cand = paths[i][0] if paths[i] else pos
            if paths[i] and can_move_to(pos, next_cand):
                next_positions[i] = next_cand
                reserve(pos, next_cand)
                paths[i] = paths[i][1:]
            else:
                next_positions[i] = pos
                reserve(pos, pos)
            times[i] += 1

        for i in range(n):
            if next_positions[i] is not None:
                positions[i] = next_positions[i]

    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    results = {"total_delay": sum(times), "completion_rate": sum(reached) / n if n else 0, "completed": sum(reached), "n": n}
    return results, trajectory, total_delay_per_step


def run_sp_mod_graph(
    road: OSMGraphRoad,
    origins: List[int],
    destinations: List[int],
    max_steps: int = 50000,
    stuck_threshold: int = 50,
) -> dict:
    """
    SP-Modified: like SP but intervenes when stuck for stuck_threshold steps.
    - When stuck 50+ steps, try any valid move toward goal (breaks deadlocks)
    - Random tie-breaking in priority order to break symmetric deadlocks
    """
    import random
    n = len(origins)
    positions = list(origins)
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    paths = {i: None for i in range(n)}
    consecutive_wait = [0] * n

    for step in range(max_steps):
        if all(reached):
            break
        next_positions = [None] * n
        reserved_count: Dict[int, int] = {}
        edge_usage: Dict[Tuple[int, int], int] = {}

        def can_move_to(pos: int, next_pos: int) -> bool:
            if reserved_count.get(next_pos, 0) >= road.node_capacity(next_pos):
                return False
            if next_pos != pos and edge_usage.get((pos, next_pos), 0) >= road.edge_capacity(pos, next_pos):
                return False
            return True

        def reserve(pos: int, next_pos: int):
            reserved_count[next_pos] = reserved_count.get(next_pos, 0) + 1
            if next_pos != pos:
                edge_usage[(pos, next_pos)] = edge_usage.get((pos, next_pos), 0) + 1

        active = [i for i in range(n) if not reached[i]]
        active.sort(key=lambda i: (road.distance(positions[i], goals[i]), random.random() * 0.01))

        for i in active:
            pos = positions[i]
            goal = goals[i]
            if pos == goal:
                reached[i] = True
                next_positions[i] = goal
                reserve(pos, goal)
                times[i] += 1
                consecutive_wait[i] = 0
                continue
            if paths[i] is None:
                path, _ = road.shortest_path(pos, goal)
                paths[i] = path[1:] if path and len(path) > 1 else []
            moved = False
            if paths[i] and can_move_to(pos, paths[i][0]):
                next_positions[i] = paths[i][0]
                reserve(pos, paths[i][0])
                paths[i] = paths[i][1:]
                consecutive_wait[i] = 0
                moved = True
            elif consecutive_wait[i] >= stuck_threshold:
                valid = [m for m in road.get_valid_moves_capacity(pos, reserved_count, edge_usage) if m != pos]
                if valid:
                    best = min(valid, key=lambda m: road.distance(m, goal))
                    if can_move_to(pos, best):
                        next_positions[i] = best
                        reserve(pos, best)
                        paths[i] = None
                        consecutive_wait[i] = 0
                        moved = True
            if not moved:
                next_positions[i] = pos
                reserve(pos, pos)
                consecutive_wait[i] += 1
            times[i] += 1

        for i in range(n):
            if next_positions[i] is not None:
                positions[i] = next_positions[i]

    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    return {"total_delay": sum(times), "completion_rate": sum(reached) / n if n else 0, "completed": sum(reached), "n": n}


def run_gsp_graph(road: OSMGraphRoad, origins: List[int], destinations: List[int], max_steps: int = 50000) -> dict:
    """GSP: Greedy shortest path - at each step pick neighbor that minimizes distance to goal. Respects capacity."""
    n = len(origins)
    positions = list(origins)
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n

    for step in range(max_steps):
        if all(reached):
            break
        next_positions = [None] * n
        reserved_count: Dict[int, int] = {}
        edge_usage: Dict[Tuple[int, int], int] = {}

        def can_move_to(pos: int, next_pos: int) -> bool:
            if reserved_count.get(next_pos, 0) >= road.node_capacity(next_pos):
                return False
            if next_pos != pos and edge_usage.get((pos, next_pos), 0) >= road.edge_capacity(pos, next_pos):
                return False
            return True

        def reserve(pos: int, next_pos: int):
            reserved_count[next_pos] = reserved_count.get(next_pos, 0) + 1
            if next_pos != pos:
                edge_usage[(pos, next_pos)] = edge_usage.get((pos, next_pos), 0) + 1

        active = sorted([i for i in range(n) if not reached[i]], key=lambda i: road.distance(positions[i], goals[i]))

        for i in active:
            pos = positions[i]
            goal = goals[i]
            if pos == goal:
                reached[i] = True
                next_positions[i] = goal
                reserve(pos, goal)
                times[i] += 1
                continue
            candidates = road.get_valid_moves_capacity(pos, reserved_count, edge_usage)
            best = min(candidates, key=lambda c: road.distance(c, goal))
            if can_move_to(pos, best):
                next_positions[i] = best
                reserve(pos, best)
            else:
                next_positions[i] = pos
                reserve(pos, pos)
            times[i] += 1

        for i in range(n):
            if next_positions[i] is not None:
                positions[i] = next_positions[i]

    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    return {"total_delay": sum(times), "completion_rate": sum(reached) / n if n else 0, "completed": sum(reached), "n": n}


class GraphPIBTPlanner:
    """PIBT for graph: priority inheritance with backtracking. Respects node and edge capacity."""

    def __init__(self, road: OSMGraphRoad):
        self.road = road

    def distance(self, a: int, b: int) -> float:
        return self.road.distance(a, b)

    def get_neighbors(self, node_id: int) -> List[int]:
        neighbors = self.road.get_neighbors(node_id)
        return [node_id] + neighbors  # Include wait in place

    def plan_step(self, positions: Dict[int, int], goals: Dict[int, int], oscillating_agents=None, history=None, rng_seed=None) -> Dict[int, int]:
        agents = list(positions.keys())
        priorities = {a: self.distance(positions[a], goals[a]) for a in agents}
        agents.sort(key=lambda a: priorities[a])
        decided: Dict[int, int] = {}
        reserved_count: Dict[int, int] = {}
        edge_usage: Dict[Tuple[int, int], int] = {}

        def can_move_to(pos: int, next_pos: int) -> bool:
            if reserved_count.get(next_pos, 0) >= self.road.node_capacity(next_pos):
                return False
            if next_pos != pos and edge_usage.get((pos, next_pos), 0) >= self.road.edge_capacity(pos, next_pos):
                return False
            return True

        def reserve(pos: int, next_pos: int):
            reserved_count[next_pos] = reserved_count.get(next_pos, 0) + 1
            if next_pos != pos:
                edge_usage[(pos, next_pos)] = edge_usage.get((pos, next_pos), 0) + 1

        def unreserve(pos: int, next_pos: int):
            reserved_count[next_pos] = max(0, reserved_count.get(next_pos, 1) - 1)
            if next_pos != pos:
                k = (pos, next_pos)
                edge_usage[k] = max(0, edge_usage.get(k, 1) - 1)

        def pibt(agent_id: int, blocker: Optional[int] = None) -> bool:
            if agent_id in decided:
                return True
            pos = positions[agent_id]
            goal = goals[agent_id]
            candidates = self.get_neighbors(pos)
            candidates.sort(key=lambda c: self.distance(c, goal))
            for next_pos in candidates:
                if not can_move_to(pos, next_pos):
                    continue
                if blocker is not None:
                    blocker_pos = positions[blocker]
                    if next_pos == blocker_pos:
                        continue
                other_agent = None
                for oid, opos in positions.items():
                    if oid != agent_id and oid not in decided and opos == next_pos:
                        other_agent = oid
                        break
                decided[agent_id] = next_pos
                reserve(pos, next_pos)
                if other_agent is not None:
                    if not pibt(other_agent, agent_id):
                        del decided[agent_id]
                        unreserve(pos, next_pos)
                        continue
                return True
            decided[agent_id] = pos
            reserve(pos, pos)
            return False

        for a in agents:
            if a not in decided:
                pibt(a)
        return decided


class GraphPaperPIBTPlanner(GraphPIBTPlanner):
    """
    PIBT as in the original paper (Okumura et al.): time-based priority + graph distance for moves.
    Agent priority: pi = epsilon_i if at goal, else pi + 1 each step (sort decreasing).
    Move choice: candidates by graph distance to goal (unchanged).
    """
    def __init__(self, road: OSMGraphRoad):
        super().__init__(road)
        self._priority: Dict[int, float] = {}

    def _epsilon(self, agent_id: int, n: int) -> float:
        return (agent_id + 1) / (n + 2)

    def plan_step(self, positions: Dict[int, int], goals: Dict[int, int], oscillating_agents=None, history=None, rng_seed=None) -> Dict[int, int]:
        agents = list(positions.keys())
        n = len(agents)
        for a in agents:
            if a not in self._priority:
                self._priority[a] = self._epsilon(a, n)
            if positions[a] == goals[a]:
                self._priority[a] = self._epsilon(a, n)
            else:
                self._priority[a] = self._priority[a] + 1
        agents.sort(key=lambda a: -self._priority[a])
        decided: Dict[int, int] = {}
        reserved_count: Dict[int, int] = {}
        edge_usage: Dict[Tuple[int, int], int] = {}

        def can_move_to(pos: int, next_pos: int) -> bool:
            if reserved_count.get(next_pos, 0) >= self.road.node_capacity(next_pos):
                return False
            if next_pos != pos and edge_usage.get((pos, next_pos), 0) >= self.road.edge_capacity(pos, next_pos):
                return False
            return True

        def reserve(pos: int, next_pos: int):
            reserved_count[next_pos] = reserved_count.get(next_pos, 0) + 1
            if next_pos != pos:
                edge_usage[(pos, next_pos)] = edge_usage.get((pos, next_pos), 0) + 1

        def unreserve(pos: int, next_pos: int):
            reserved_count[next_pos] = max(0, reserved_count.get(next_pos, 1) - 1)
            if next_pos != pos:
                k = (pos, next_pos)
                edge_usage[k] = max(0, edge_usage.get(k, 1) - 1)

        def pibt(agent_id: int, blocker: Optional[int] = None) -> bool:
            if agent_id in decided:
                return True
            pos = positions[agent_id]
            goal = goals[agent_id]
            candidates = self.get_neighbors(pos)
            candidates.sort(key=lambda c: self.distance(c, goal))
            for next_pos in candidates:
                if not can_move_to(pos, next_pos):
                    continue
                if blocker is not None and next_pos == positions[blocker]:
                    continue
                other_agent = None
                for oid, opos in positions.items():
                    if oid != agent_id and oid not in decided and opos == next_pos:
                        other_agent = oid
                        break
                decided[agent_id] = next_pos
                reserve(pos, next_pos)
                if other_agent is not None:
                    if not pibt(other_agent, agent_id):
                        del decided[agent_id]
                        unreserve(pos, next_pos)
                        continue
                return True
            decided[agent_id] = pos
            reserve(pos, pos)
            return False

        for a in agents:
            if a not in decided:
                pibt(a)
        return decided


class GraphSPPIBTPlanner(GraphPIBTPlanner):
    """SP-PIBT: BFS optimal paths + PIBT collision resolution. Path-guided candidate ordering."""

    def plan_step(
        self, positions: Dict[int, int], goals: Dict[int, int],
        oscillating_agents=None, history=None, rng_seed=None,
        paths: Optional[Dict[int, List[int]]] = None,
    ) -> Dict[int, int]:
        if paths is None:
            paths = {}
        for aid in positions:
            if aid not in paths or paths[aid] is None:
                pos, goal = positions[aid], goals[aid]
                p, _ = self.road.shortest_path(pos, goal)
                paths[aid] = (p[1:] if p and len(p) > 1 else [])

        agents = list(positions.keys())
        priorities = {a: self.distance(positions[a], goals[a]) for a in agents}
        agents.sort(key=lambda a: priorities[a])
        decided: Dict[int, int] = {}
        reserved_count: Dict[int, int] = {}
        edge_usage: Dict[Tuple[int, int], int] = {}

        def can_move_to(pos: int, next_pos: int) -> bool:
            if reserved_count.get(next_pos, 0) >= self.road.node_capacity(next_pos):
                return False
            if next_pos != pos and edge_usage.get((pos, next_pos), 0) >= self.road.edge_capacity(pos, next_pos):
                return False
            return True

        def reserve(pos: int, next_pos: int):
            reserved_count[next_pos] = reserved_count.get(next_pos, 0) + 1
            if next_pos != pos:
                edge_usage[(pos, next_pos)] = edge_usage.get((pos, next_pos), 0) + 1

        def unreserve(pos: int, next_pos: int):
            reserved_count[next_pos] = max(0, reserved_count.get(next_pos, 1) - 1)
            if next_pos != pos:
                k = (pos, next_pos)
                edge_usage[k] = max(0, edge_usage.get(k, 1) - 1)

        def score(aid: int, pos: int, cand: int, goal: int) -> float:
            path = paths.get(aid) or []
            next_on_path = path[0] if path else None
            if cand == pos:
                return 1e9
            if cand == next_on_path:
                return -1
            if cand in path:
                return path.index(cand)
            return 1000 + self.distance(cand, goal)

        def pibt(agent_id: int, blocker: Optional[int] = None) -> bool:
            if agent_id in decided:
                return True
            pos = positions[agent_id]
            goal = goals[agent_id]
            candidates = self.get_neighbors(pos)
            candidates.sort(key=lambda c: score(agent_id, pos, c, goal))
            for next_pos in candidates:
                if not can_move_to(pos, next_pos):
                    continue
                if blocker is not None and next_pos == positions[blocker]:
                    continue
                other_agent = None
                for oid, opos in positions.items():
                    if oid != agent_id and oid not in decided and opos == next_pos:
                        other_agent = oid
                        break
                decided[agent_id] = next_pos
                reserve(pos, next_pos)
                if other_agent is not None:
                    if not pibt(other_agent, agent_id):
                        del decided[agent_id]
                        unreserve(pos, next_pos)
                        continue
                return True
            decided[agent_id] = pos
            reserve(pos, pos)
            return False

        for a in agents:
            if a not in decided:
                pibt(a)
        return decided


class GraphGuidedPIBTPlanner(GraphPIBTPlanner):
    """G-PIBT: Guided PIBT with congestion-aware A* paths (Chen et al. AAAI 2024)."""

    def plan_step(
        self, positions: Dict[int, int], goals: Dict[int, int],
        oscillating_agents=None, history=None, rng_seed=None,
    ) -> Dict[int, int]:
        import heapq
        flow: Dict[Tuple[int, int], int] = {}
        paths: Dict[int, List[int]] = {}

        def astar_congestion(start: int, goal: int) -> Optional[List[int]]:
            def h(v):
                return self.road.distance(v, goal)
            def edge_cost(u: int, v: int) -> float:
                base = 1.0
                fwd = flow.get((u, v), 0)
                rev = flow.get((v, u), 0)
                contraflow = 2.0 * fwd * rev if fwd and rev else 0
                incoming = sum(flow.get((w, v), 0) for w in self.road.G.predecessors(v))
                return base + contraflow + 0.3 * incoming
            open_set = [(h(start), 0, start, [start])]
            visited = {start: 0}
            while open_set:
                _, g, cur, path = heapq.heappop(open_set)
                if cur == goal:
                    return path
                if g > visited.get(cur, float("inf")):
                    continue
                for nxt in self.road.get_neighbors(cur):
                    c = edge_cost(cur, nxt)
                    ng = g + c
                    if ng < visited.get(nxt, float("inf")):
                        visited[nxt] = ng
                        heapq.heappush(open_set, (ng + h(nxt), ng, nxt, path + [nxt]))
            return None

        agents_sorted = sorted(positions.keys(), key=lambda a: self.distance(positions[a], goals[a]))
        for aid in agents_sorted:
            pos, goal = positions[aid], goals[aid]
            path = astar_congestion(pos, goal)
            if path and len(path) > 1:
                paths[aid] = path[1:]
                for j in range(1, len(path)):
                    flow[(path[j - 1], path[j])] = flow.get((path[j - 1], path[j]), 0) + 1
            else:
                p, _ = self.road.shortest_path(pos, goal)
                paths[aid] = (p[1:] if p and len(p) > 1 else [])
                if p and len(p) > 1:
                    for j in range(1, len(p)):
                        flow[(p[j - 1], p[j])] = flow.get((p[j - 1], p[j]), 0) + 1

        return GraphSPPIBTPlanner(self.road).plan_step(positions, goals, paths=paths)


def run_sp_pibt_graph(
    road: OSMGraphRoad, origins: List[int], destinations: List[int],
    max_steps: int = 50000,
    progress_callback: Optional[Callable[[int, int, int, int, int], None]] = None,
    progress_interval: int = 1000,
    stuck_check_interval: int = 100,
    stuck_history_len: int = 20,
) -> dict:
    """SP-PIBT: BFS optimal paths + PIBT collision resolution. Early-terminates stuck agents."""
    n = len(origins)
    positions = list(origins)
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    failed: Set[int] = set()
    pos_history: Dict[int, deque] = {i: deque(maxlen=stuck_history_len) for i in range(n)}
    planner = GraphSPPIBTPlanner(road)
    paths: Dict[int, List[int]] = {}

    for step in range(max_steps):
        if progress_callback and step > 0 and step % progress_interval == 0:
            progress_callback(step, max_steps, sum(reached), n, sum(times))
        active = {i for i in range(n) if not reached[i] and i not in failed}
        if not active:
            break

        if step > 0 and step % stuck_check_interval == 0:
            stuck = _check_stuck_agents(road, active, positions, goals, pos_history, step)
            for i in stuck:
                failed.add(i)
                times[i] = step
            active -= stuck

        if not active:
            break
        pos_dict = {i: positions[i] for i in active}
        goal_dict = {i: goals[i] for i in active}
        next_pos = planner.plan_step(pos_dict, goal_dict, paths=paths)
        for i in active:
            prev = positions[i]
            positions[i] = next_pos[i]
            pos_history[i].append(positions[i])
            if positions[i] == goals[i]:
                reached[i] = True
            if i in paths and paths[i]:
                if positions[i] == paths[i][0]:
                    paths[i] = paths[i][1:]
                elif positions[i] != prev:
                    paths[i] = None
            times[i] += 1

    for i in range(n):
        if not reached[i] and i not in failed:
            times[i] = max_steps
    completed = sum(reached)
    return {"total_delay": sum(times), "completion_rate": completed / n if n else 0, "completed": completed, "n": n, "failed": len(failed)}


def run_guided_pibt_graph(road: OSMGraphRoad, origins: List[int], destinations: List[int], max_steps: int = 50000,
                          progress_callback=None, progress_interval: int = 1) -> dict:
    """G-PIBT: Guided PIBT with congestion-aware paths."""
    n = len(origins)
    positions = list(origins)
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    planner = GraphGuidedPIBTPlanner(road)

    for step in range(max_steps):
        if all(reached):
            break
        active = {i for i in range(n) if not reached[i]}
        pos_dict = {i: positions[i] for i in active}
        goal_dict = {i: goals[i] for i in active}
        next_pos = planner.plan_step(pos_dict, goal_dict)
        for i in active:
            positions[i] = next_pos[i]
            if positions[i] == goals[i]:
                reached[i] = True
            times[i] += 1
        if progress_callback and (step + 1) % progress_interval == 0:
            progress_callback(step + 1, max_steps, sum(reached), n, sum(times))

    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    return {"total_delay": sum(times), "completion_rate": sum(reached) / n if n else 0, "completed": sum(reached), "n": n}


def _detect_cycle(positions_history: List[int], max_period: int = 10) -> Optional[int]:
    """Detect if last positions form a cycle. Returns cycle length or None."""
    if len(positions_history) < 6:
        return None
    tail = positions_history[-max_period * 2:]
    for period in range(2, min(max_period, len(tail) // 2) + 1):
        if len(tail) >= 2 * period and tail[-period:] == tail[-2*period:-period]:
            return period
    return None


def _check_stuck_agents(
    road: OSMGraphRoad,
    active: Set[int],
    positions: List[int],
    goals: List[int],
    pos_history: Dict[int, deque],
    step: int,
) -> Set[int]:
    """Return set of agent IDs that are stuck (unreachable goal only).
    Oscillation-based failure disabled: temporary A↔B swaps can resolve on their own."""
    failed = set()
    for i in active:
        pos, goal = positions[i], goals[i]
        if road.distance(pos, goal) == float("inf"):
            failed.add(i)
    return failed


def run_pibt_graph(
    road: OSMGraphRoad, origins: List[int], destinations: List[int],
    max_steps: int = 50000,
    progress_callback: Optional[Callable[[int, int, int, int, int], None]] = None,
    progress_interval: int = 1000,
    stuck_check_interval: int = 100,
    stuck_history_len: int = 20,
) -> dict:
    """PIBT on graph. Early-terminates agents that are unreachable or oscillating."""
    n = len(origins)
    positions = list(origins)
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    failed: Set[int] = set()
    pos_history: Dict[int, deque] = {i: deque(maxlen=stuck_history_len) for i in range(n)}
    planner = GraphPIBTPlanner(road)

    for step in range(max_steps):
        if progress_callback and step > 0 and step % progress_interval == 0:
            progress_callback(step, max_steps, sum(reached), n, sum(times))
        active = {i for i in range(n) if not reached[i] and i not in failed}
        if not active:
            break

        # Oscillation / unreachable check
        if step > 0 and step % stuck_check_interval == 0:
            stuck = _check_stuck_agents(road, active, positions, goals, pos_history, step)
            for i in stuck:
                failed.add(i)
                times[i] = step
            active -= stuck

        if not active:
            break
        pos_dict = {i: positions[i] for i in active}
        goal_dict = {i: goals[i] for i in active}
        next_pos = planner.plan_step(pos_dict, goal_dict)
        for i in active:
            positions[i] = next_pos[i]
            pos_history[i].append(positions[i])
            if positions[i] == goals[i]:
                reached[i] = True
            times[i] += 1

    for i in range(n):
        if not reached[i] and i not in failed:
            times[i] = max_steps
    completed = sum(reached)
    return {"total_delay": sum(times), "completion_rate": completed / n if n else 0, "completed": completed, "n": n, "failed": len(failed)}


def run_paper_pibt_graph(
    road: OSMGraphRoad, origins: List[int], destinations: List[int],
    max_steps: int = 50000,
    progress_callback: Optional[Callable[[int, int, int, int, int], None]] = None,
    progress_interval: int = 1000,
    stuck_check_interval: int = 100,
    stuck_history_len: int = 20,
) -> dict:
    """Paper-PIBT on graph (time-based priority + graph distance for moves). Same interface as run_pibt_graph."""
    n = len(origins)
    positions = list(origins)
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    failed: Set[int] = set()
    pos_history: Dict[int, deque] = {i: deque(maxlen=stuck_history_len) for i in range(n)}
    planner = GraphPaperPIBTPlanner(road)

    for step in range(max_steps):
        if progress_callback and step > 0 and step % progress_interval == 0:
            progress_callback(step, max_steps, sum(reached), n, sum(times))
        active = {i for i in range(n) if not reached[i] and i not in failed}
        if not active:
            break
        if step > 0 and step % stuck_check_interval == 0:
            stuck = _check_stuck_agents(road, active, positions, goals, pos_history, step)
            for i in stuck:
                failed.add(i)
                times[i] = step
            active -= stuck
        if not active:
            break
        pos_dict = {i: positions[i] for i in active}
        goal_dict = {i: goals[i] for i in active}
        next_pos = planner.plan_step(pos_dict, goal_dict)
        for i in active:
            positions[i] = next_pos[i]
            pos_history[i].append(positions[i])
            if positions[i] == goals[i]:
                reached[i] = True
            times[i] += 1

    for i in range(n):
        if not reached[i] and i not in failed:
            times[i] = max_steps
    completed = sum(reached)
    return {"total_delay": sum(times), "completion_rate": completed / n if n else 0, "completed": completed, "n": n, "failed": len(failed)}


def run_pibt_graph_with_trajectory(
    road: OSMGraphRoad, origins: List[int], destinations: List[int], max_steps: int = 50000
) -> Tuple[dict, List[Dict[int, int]], List[int]]:
    """PIBT on graph. Returns (results, trajectory, total_delay_per_step)."""
    n = len(origins)
    positions = list(origins)
    goals = list(destinations)
    reached = [False] * n
    times = [0] * n
    planner = GraphPIBTPlanner(road)
    trajectory: List[Dict[int, int]] = []
    total_delay_per_step: List[int] = []

    for step in range(max_steps):
        total_delay_per_step.append(sum(times))
        trajectory.append(dict(enumerate(positions)))
        if all(reached):
            break
        active = {i for i in range(n) if not reached[i]}
        pos_dict = {i: positions[i] for i in active}
        goal_dict = {i: goals[i] for i in active}
        next_pos = planner.plan_step(pos_dict, goal_dict)
        for i in active:
            positions[i] = next_pos[i]
            if positions[i] == goals[i]:
                reached[i] = True
            times[i] += 1

    for i in range(n):
        if not reached[i]:
            times[i] = max_steps
    results = {"total_delay": sum(times), "completion_rate": sum(reached) / n if n else 0, "completed": sum(reached), "n": n}
    return results, trajectory, total_delay_per_step
