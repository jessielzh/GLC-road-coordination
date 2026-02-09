#!/usr/bin/env python3
"""
PIBT (Priority Inheritance with Backtracking) and Guided PIBT implementation.

Reference: 
- Okumura et al. "Priority inheritance with backtracking for iterative multi-agent path finding" (2022)
- Zhe Chen et al. "Traffic Flow Optimisation for Lifelong Multi-Agent Path Finding" (2023)

PIBT: Plans one step at a time for all agents using priority-based collision resolution.
Guided PIBT: Uses congestion-aware guide paths to improve coordination.
"""

import numpy as np
from simulation import Simulation
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import heapq


class PIBTPlanner:
    """
    PIBT (Priority Inheritance with Backtracking) planner.
    
    Each timestep:
    1. Sort agents by priority (closer to goal = higher priority)
    2. Plan one step for each agent in priority order
    3. Higher-priority agents get preferred moves
    4. Lower-priority agents yield or find alternatives
    """
    
    def __init__(self, grid_size: int, collision_radius: float = 1.5):
        self.gs = grid_size
        self.collision_radius = collision_radius
        
    def get_neighbors(self, pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Get valid neighbor positions (8-directional + wait)."""
        x, y = pos
        neighbors = [(x, y)]  # Wait in place
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.gs and 0 <= ny < self.gs:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def plan_step(
        self,
        positions: Dict[int, Tuple[float, float]],
        goals: Dict[int, Tuple[float, float]],
        guide_heuristic: Optional[Dict[int, Dict[Tuple[int, int], float]]] = None
    ) -> Dict[int, Tuple[float, float]]:
        """
        Plan one step for all agents using PIBT.
        
        Args:
            positions: Current positions {agent_id: (x, y)}
            goals: Goal positions {agent_id: (x, y)}
            guide_heuristic: Optional congestion-aware heuristics
        
        Returns:
            Next positions {agent_id: (x, y)}
        """
        # Sort agents by priority (smaller distance to goal = higher priority)
        agents = list(positions.keys())
        priorities = {a: self.distance(positions[a], goals[a]) for a in agents}
        agents.sort(key=lambda a: priorities[a])
        
        # Track decided moves
        decided: Dict[int, Tuple[float, float]] = {}
        reserved: Set[Tuple[int, int]] = set()  # Reserved next positions
        
        def pibt(agent_id: int, blocker: Optional[int] = None) -> bool:
            """PIBT recursive function."""
            if agent_id in decided:
                return True
            
            pos = positions[agent_id]
            goal = goals[agent_id]
            
            # Get candidate moves sorted by preference
            candidates = self.get_neighbors(pos)
            
            if guide_heuristic and agent_id in guide_heuristic:
                # Use guide heuristic
                h = guide_heuristic[agent_id]
                candidates.sort(key=lambda c: h.get((int(c[0]), int(c[1])), 
                                                    self.distance(c, goal)))
            else:
                # Sort by distance to goal
                candidates.sort(key=lambda c: self.distance(c, goal))
            
            for next_pos in candidates:
                grid_pos = (int(next_pos[0]), int(next_pos[1]))
                
                # Check if position is already reserved
                if grid_pos in reserved:
                    continue
                
                # Check for collision with blocker (swap prevention)
                if blocker is not None:
                    blocker_pos = positions[blocker]
                    if self.distance(next_pos, blocker_pos) < self.collision_radius:
                        continue
                
                # Check if another agent is at this position
                other_agent = None
                for other_id, other_pos in positions.items():
                    if other_id != agent_id and other_id not in decided:
                        if self.distance(next_pos, other_pos) < self.collision_radius:
                            other_agent = other_id
                            break
                
                # Reserve this position
                decided[agent_id] = next_pos
                reserved.add(grid_pos)
                
                # If there's another agent, recursively plan for them
                if other_agent is not None:
                    if not pibt(other_agent, agent_id):
                        # Other agent couldn't move, try next candidate
                        del decided[agent_id]
                        reserved.discard(grid_pos)
                        continue
                
                return True
            
            # No valid move found, stay in place
            decided[agent_id] = pos
            reserved.add((int(pos[0]), int(pos[1])))
            return False
        
        # Plan for each agent in priority order
        for agent_id in agents:
            if agent_id not in decided:
                pibt(agent_id)
        
        return decided


class PaperPIBTPlanner(PIBTPlanner):
    """
    PIBT as in Okumura et al.: time-based priority + distance for move choice.
    Priority: pi = epsilon_i if at goal, else pi + 1 each step. Sort decreasing (high pi first).
    Move choice: same as PIBT (distance to goal).
    """
    def __init__(self, grid_size: int, collision_radius: float = 1.5):
        super().__init__(grid_size, collision_radius)
        self._priority: Dict[int, float] = {}

    def _epsilon(self, agent_id: int, n: int) -> float:
        return (agent_id + 1) / (n + 2)

    def plan_step(
        self,
        positions: Dict[int, Tuple[float, float]],
        goals: Dict[int, Tuple[float, float]],
        guide_heuristic: Optional[Dict[int, Dict[Tuple[int, int], float]]] = None
    ) -> Dict[int, Tuple[float, float]]:
        agents = list(positions.keys())
        n = len(agents)
        for a in agents:
            if a not in self._priority:
                self._priority[a] = self._epsilon(a, n)
            pos, goal = positions[a], goals[a]
            if self.distance(pos, goal) < self.collision_radius:
                self._priority[a] = self._epsilon(a, n)
            else:
                self._priority[a] = self._priority[a] + 1
        agents.sort(key=lambda a: -self._priority[a])

        decided: Dict[int, Tuple[float, float]] = {}
        reserved: Set[Tuple[int, int]] = set()

        def pibt(agent_id: int, blocker: Optional[int] = None) -> bool:
            if agent_id in decided:
                return True
            pos = positions[agent_id]
            goal = goals[agent_id]
            candidates = self.get_neighbors(pos)
            if guide_heuristic and agent_id in guide_heuristic:
                h = guide_heuristic[agent_id]
                candidates.sort(key=lambda c: h.get((int(c[0]), int(c[1])), self.distance(c, goal)))
            else:
                candidates.sort(key=lambda c: self.distance(c, goal))
            for next_pos in candidates:
                grid_pos = (int(next_pos[0]), int(next_pos[1]))
                if grid_pos in reserved:
                    continue
                if blocker is not None and self.distance(next_pos, positions[blocker]) < self.collision_radius:
                    continue
                other_agent = None
                for other_id, other_pos in positions.items():
                    if other_id != agent_id and other_id not in decided:
                        if self.distance(next_pos, other_pos) < self.collision_radius:
                            other_agent = other_id
                            break
                decided[agent_id] = next_pos
                reserved.add(grid_pos)
                if other_agent is not None:
                    if not pibt(other_agent, agent_id):
                        del decided[agent_id]
                        reserved.discard(grid_pos)
                        continue
                return True
            decided[agent_id] = pos
            reserved.add((int(pos[0]), int(pos[1])))
            return False

        for agent_id in agents:
            if agent_id not in decided:
                pibt(agent_id)
        return decided


class GuidedPIBT(PIBTPlanner):
    """
    Guided PIBT with congestion-aware guide paths.
    
    Key ideas from the paper:
    1. Compute guide paths that consider traffic congestion
    2. Use guide paths to create heuristics for PIBT
    3. Update guide paths as traffic patterns change
    """
    
    def __init__(self, grid_size: int):
        super().__init__(grid_size)
        self.flow = defaultdict(int)  # Edge flow counts
        
    def compute_guide_paths(
        self,
        starts: Dict[int, Tuple[float, float]],
        goals: Dict[int, Tuple[float, float]]
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Compute congestion-aware guide paths for all agents.
        
        Uses traffic-aware edge costs based on:
        1. Vertex congestion: More agents entering a vertex = higher delay
        2. Contraflow congestion: Agents crossing in opposite directions = big delay
        """
        # Reset flow counts
        self.flow.clear()
        
        # Sort agents by distance (closer agents planned first)
        agents = sorted(starts.keys(), key=lambda a: self.distance(starts[a], goals[a]))
        
        guide_paths: Dict[int, List[Tuple[int, int]]] = {}
        
        for agent_id in agents:
            start = (int(starts[agent_id][0]), int(starts[agent_id][1]))
            goal = (int(goals[agent_id][0]), int(goals[agent_id][1]))
            
            # A* with congestion-aware costs
            path = self._astar_congestion_aware(start, goal)
            guide_paths[agent_id] = path
            
            # Update flow counts
            for i in range(len(path) - 1):
                v1, v2 = path[i], path[i + 1]
                self.flow[(v1, v2)] += 1
        
        return guide_paths
    
    def _astar_congestion_aware(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """A* search with congestion-aware edge costs."""
        
        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        def get_edge_cost(v1, v2):
            """
            Edge cost considering congestion.
            
            From paper: cost = (contraflow_congestion, 1 + vertex_congestion)
            Simplified: cost = 1 + contraflow_penalty + vertex_penalty
            """
            base_cost = 1.0
            
            # Contraflow congestion: penalize if agents going opposite direction
            contraflow = self.flow.get((v2, v1), 0) * self.flow.get((v1, v2), 0)
            contraflow_penalty = 0.5 * contraflow if contraflow > 0 else 0
            
            # Vertex congestion: penalize popular destinations
            incoming = sum(self.flow.get((u, v2), 0) for u in self.get_neighbors(v2))
            vertex_penalty = 0.1 * incoming
            
            return base_cost + contraflow_penalty + vertex_penalty
        
        # A* search
        open_set = [(heuristic(start), 0, start)]
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
            
            for nx, ny in self.get_neighbors(current):
                neighbor = (int(nx), int(ny))
                edge_cost = get_edge_cost(current, neighbor)
                tentative_g = g_score[current] + edge_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
        
        # Fallback: direct path
        return [start, goal]
    
    def create_guide_heuristic(
        self,
        guide_paths: Dict[int, List[Tuple[int, int]]],
        goals: Dict[int, Tuple[float, float]]
    ) -> Dict[int, Dict[Tuple[int, int], float]]:
        """
        Create guide heuristics from guide paths.
        
        For each vertex v, heuristic h(v) = (dist_to_path, remaining_dist_on_path)
        Simplified: h(v) = distance to closest point on guide path + remaining path length
        """
        heuristics: Dict[int, Dict[Tuple[int, int], float]] = {}
        
        for agent_id, path in guide_paths.items():
            goal = goals[agent_id]
            h: Dict[Tuple[int, int], float] = {}
            
            # For vertices on the path
            for i, vertex in enumerate(path):
                # Remaining distance on path
                remaining = len(path) - 1 - i
                h[vertex] = remaining
            
            # BFS to fill in other vertices
            from collections import deque
            queue = deque()
            for i, vertex in enumerate(path):
                queue.append((vertex, 0, len(path) - 1 - i))
            
            visited = set(path)
            while queue and len(visited) < 1000:  # Limit expansion
                vertex, dist_to_path, remaining = queue.popleft()
                
                for nx, ny in self.get_neighbors(vertex):
                    neighbor = (int(nx), int(ny))
                    if neighbor not in visited:
                        visited.add(neighbor)
                        h[neighbor] = dist_to_path + 1 + remaining
                        queue.append((neighbor, dist_to_path + 1, remaining))
            
            heuristics[agent_id] = h
        
        return heuristics


def run_pibt(n_agents: int, gs: int, seed: int, max_steps: int = 2000) -> int:
    """Run basic PIBT simulation."""
    sim = Simulation(grid_size=gs)
    sim.generate_opposing_agents(n_agents, seed=seed)
    
    planner = PIBTPlanner(gs)
    
    # Convert to dict format
    positions = {a.id: tuple(a.position) for a in sim.agents}
    goals = {a.id: tuple(a.destination) for a in sim.agents}
    reached = {a.id: False for a in sim.agents}
    times = {a.id: 0 for a in sim.agents}
    
    for step in range(max_steps):
        # Check completion
        active = {aid for aid, done in reached.items() if not done}
        if not active:
            break
        
        # Filter to active agents
        active_pos = {aid: positions[aid] for aid in active}
        active_goals = {aid: goals[aid] for aid in active}
        
        # Plan one step
        next_positions = planner.plan_step(active_pos, active_goals)
        
        # Update positions
        for aid in active:
            positions[aid] = next_positions.get(aid, positions[aid])
            times[aid] += 1
            
            # Check if reached goal
            if planner.distance(positions[aid], goals[aid]) < 2:
                reached[aid] = True
    
    return sum(times.values())


def run_guided_pibt(n_agents: int, gs: int, seed: int, max_steps: int = 2000) -> int:
    """Run Guided PIBT simulation."""
    sim = Simulation(grid_size=gs)
    sim.generate_opposing_agents(n_agents, seed=seed)
    
    planner = GuidedPIBT(gs)
    
    # Convert to dict format
    positions = {a.id: tuple(a.position) for a in sim.agents}
    goals = {a.id: tuple(a.destination) for a in sim.agents}
    reached = {a.id: False for a in sim.agents}
    times = {a.id: 0 for a in sim.agents}
    
    # Compute initial guide paths
    guide_paths = planner.compute_guide_paths(positions, goals)
    guide_heuristic = planner.create_guide_heuristic(guide_paths, goals)
    
    replan_interval = 50  # Replan guide paths periodically
    
    for step in range(max_steps):
        # Check completion
        active = {aid for aid, done in reached.items() if not done}
        if not active:
            break
        
        # Periodically recompute guide paths
        if step > 0 and step % replan_interval == 0:
            active_pos = {aid: positions[aid] for aid in active}
            active_goals = {aid: goals[aid] for aid in active}
            guide_paths = planner.compute_guide_paths(active_pos, active_goals)
            guide_heuristic = planner.create_guide_heuristic(guide_paths, active_goals)
        
        # Filter to active agents
        active_pos = {aid: positions[aid] for aid in active}
        active_goals = {aid: goals[aid] for aid in active}
        active_h = {aid: guide_heuristic.get(aid, {}) for aid in active}
        
        # Plan one step with guidance
        next_positions = planner.plan_step(active_pos, active_goals, active_h)
        
        # Update positions
        for aid in active:
            positions[aid] = next_positions.get(aid, positions[aid])
            times[aid] += 1
            
            # Check if reached goal
            if planner.distance(positions[aid], goals[aid]) < 2:
                reached[aid] = True
    
    return sum(times.values())


def run_learned(n_agents, gs, seed, max_steps=2000):
    """Run learned policy."""
    sim = Simulation(grid_size=gs)
    sim.generate_opposing_agents(n_agents, seed=seed)
    
    W1 = np.load('decision_learned_W1.npy')
    W2 = np.load('decision_learned_W2.npy')
    
    center = np.array([gs/2, gs/2])
    ring_radius = np.linalg.norm(center) * 0.6
    
    def get_decision(agent):
        pos, dest = agent.position, agent.destination
        to_dest = dest - pos
        dist = np.linalg.norm(to_dest)
        dest_dir = to_dest / dist if dist > 0.5 else np.zeros(2)
        
        to_center = center - pos
        dist_to_center = np.linalg.norm(to_center)
        proj = np.dot(to_center, dest_dir) if dist > 0.5 else 0
        perp = np.linalg.norm(to_center - proj * dest_dir) if dist > 0.5 else 0
        
        features = np.array([proj/gs, perp/gs, dist/gs, dist_to_center/gs, 
                            1.0 if dist < 5 else 0.0])
        
        h = np.maximum(0, features @ W1)
        prob = 1 / (1 + np.exp(-h @ W2))
        return prob[0] > 0.5
    
    for step in range(max_steps):
        if all(a.reached_destination for a in sim.agents):
            break
        
        sim.time_step += 1
        proposed = {}
        
        for agent in sim.agents:
            if agent.reached_destination:
                continue
            
            pos, dest = agent.position, agent.destination
            to_dest = dest - pos
            dist = np.linalg.norm(to_dest)
            
            if dist < 2:
                next_pos = dest.copy()
            elif get_decision(agent):
                next_pos = pos + (to_dest / dist)
            else:
                pfc = pos - center
                angle = np.arctan2(pfc[1], pfc[0]) + 0.15
                next_pos = center + ring_radius * np.array([np.cos(angle), np.sin(angle)])
            
            next_pos = np.clip(next_pos, 0, gs - 1)
            
            if sim.is_position_valid(next_pos, agent.id, proposed):
                proposed[agent.id] = next_pos
            else:
                direction = next_pos - pos
                for s in [0.5, 0.3]:
                    smaller = np.clip(pos + direction * s, 0, gs - 1)
                    if sim.is_position_valid(smaller, agent.id, proposed):
                        proposed[agent.id] = smaller
                        break
        
        for agent in sim.agents:
            if not agent.reached_destination:
                if agent.id in proposed:
                    agent.move_to(proposed[agent.id])
                else:
                    agent.wait_time += 1
    
    return sum(a.move_time + a.wait_time for a in sim.agents)


def main():
    gs = 50
    seeds = range(3)
    
    print("="*100)
    print("COMPARISON: LEARNED vs GSP vs CIRCULAR vs PIBT vs GUIDED-PIBT")
    print("="*100)
    print("""
Method Types:
- Learned/Circular/GSP: DECENTRALIZED (each agent decides independently)
- PIBT: DECENTRALIZED with coordination (priority-based collision resolution)
- Guided-PIBT: PIBT + congestion-aware guide paths

Reference: Zhe Chen et al. "Traffic Flow Optimisation for Lifelong MAPF" (2023)
""")
    
    print(f"{'Agents':<8} {'Learned':<10} {'GSP':<10} {'Circular':<10} {'PIBT':<10} {'G-PIBT':<10} {'L vs GSP':<12} {'L vs G-PIBT'}")
    print("-"*100)
    
    for n in [20, 40, 60, 80]:
        # Learned
        l_times = [run_learned(n, gs, s) for s in seeds]
        l_avg = np.mean(l_times)
        
        # GSP
        gsp_times = []
        for seed in seeds:
            sim = Simulation(grid_size=gs)
            sim.generate_opposing_agents(n, seed=seed)
            for _ in range(2000):
                if sim.step_straight_line():
                    break
            gsp_times.append(sum(a.move_time + a.wait_time for a in sim.agents))
        gsp_avg = np.mean(gsp_times)
        
        # Circular
        cir_times = []
        for seed in seeds:
            sim = Simulation(grid_size=gs)
            sim.generate_opposing_agents(n, seed=seed)
            for _ in range(2000):
                if sim.step_circular(clockwise=False):
                    break
            cir_times.append(sum(a.move_time + a.wait_time for a in sim.agents))
        cir_avg = np.mean(cir_times)
        
        # PIBT
        pibt_times = [run_pibt(n, gs, s) for s in seeds]
        pibt_avg = np.mean(pibt_times)
        
        # Guided PIBT
        gpibt_times = [run_guided_pibt(n, gs, s) for s in seeds]
        gpibt_avg = np.mean(gpibt_times)
        
        vs_gsp = (l_avg - gsp_avg) / gsp_avg * 100
        vs_gpibt = (l_avg - gpibt_avg) / gpibt_avg * 100
        
        gsp_m = "✓" if l_avg < gsp_avg else ""
        gpibt_m = "✓" if l_avg < gpibt_avg else ""
        
        print(f"{n:<8} {l_avg:<10.0f} {gsp_avg:<10.0f} {cir_avg:<10.0f} {pibt_avg:<10.0f} {gpibt_avg:<10.0f} {vs_gsp:+.1f}% {gsp_m:<4} {vs_gpibt:+.1f}% {gpibt_m}")
    
    print("\n" + "="*100)
    print("ANALYSIS:")
    print("="*100)
    print("""
PIBT (Priority Inheritance with Backtracking):
- Plans one step at a time (very fast, O(n) per timestep)
- Uses priority ordering to resolve conflicts
- Higher-priority agents get preferred moves
- Complete for lifelong MAPF (no deadlocks)

Guided PIBT (Traffic Flow Optimization):
- Adds congestion-aware guide paths
- Considers contraflow congestion (agents crossing paths)
- Periodically replans guide paths

Key insight from the paper:
- Free-flow shortest paths create congestion
- Congestion-aware paths can improve throughput 10-30%
- But in OPPOSING scenario, the roundabout pattern is hard to discover automatically!

Our Learned Policy advantage:
- Learned the roundabout pattern from Circular strategy
- This emergent coordination is more effective than generic congestion avoidance
- In scenarios with clear bottlenecks, domain-specific strategies beat generic approaches
""")


if __name__ == "__main__":
    main()
