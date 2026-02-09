"""
Simulation environment for multi-agent movement.
"""
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from agent import Agent
from enum import Enum


class MovementStrategy(Enum):
    STRAIGHT_LINE = "straight_line"
    CIRCULAR = "circular"


class Simulation:
    """
    Simulation environment managing multiple agents.
    Agents cannot occupy the same grid cell simultaneously.
    """
    
    def __init__(self, grid_size: int = 50, collision_radius: float = 0.8):
        self.grid_size = grid_size
        self.collision_radius = collision_radius
        self.agents: List[Agent] = []
        self.center = np.array([grid_size / 2, grid_size / 2])
        self.time_step = 0
        self.max_time_steps = 10000  # Prevent infinite loops
        
    def add_agent(self, agent: Agent):
        """Add an agent to the simulation."""
        self.agents.append(agent)
    
    def clear_agents(self):
        """Remove all agents."""
        self.agents = []
        self.time_step = 0
    
    def generate_random_agents(self, num_agents: int, seed: Optional[int] = None):
        """Generate random agents with random origins and destinations."""
        if seed is not None:
            np.random.seed(seed)
        
        self.clear_agents()
        
        # Generate unique positions for origins and destinations
        margin = 2
        positions = set()
        
        for i in range(num_agents):
            # Generate unique origin
            while True:
                origin = (
                    np.random.randint(margin, self.grid_size - margin),
                    np.random.randint(margin, self.grid_size - margin)
                )
                if origin not in positions:
                    positions.add(origin)
                    break
            
            # Generate unique destination (different from origin)
            while True:
                destination = (
                    np.random.randint(margin, self.grid_size - margin),
                    np.random.randint(margin, self.grid_size - margin)
                )
                if destination != origin:
                    break
            
            agent = Agent(i, origin, destination)
            self.add_agent(agent)
    
    def generate_opposing_agents(self, num_agents: int, seed: Optional[int] = None):
        """
        Generate agents that need to cross paths through the CENTER (maximum conflict).
        Creates a 4-way intersection scenario where paths cross at the center.
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.clear_agents()
        
        margin = 3
        quarter = num_agents // 4
        center = self.grid_size // 2
        spread = self.grid_size // 4  # How spread out agents are on each side
        
        agent_id = 0
        
        # North to South (top to bottom)
        for i in range(quarter):
            x = center + np.random.randint(-spread//2, spread//2)
            origin = (x, margin + np.random.randint(0, 3))
            destination = (center + np.random.randint(-spread//2, spread//2), self.grid_size - margin - np.random.randint(0, 3))
            self.add_agent(Agent(agent_id, origin, destination))
            agent_id += 1
        
        # South to North (bottom to top)
        for i in range(quarter):
            x = center + np.random.randint(-spread//2, spread//2)
            origin = (x, self.grid_size - margin - np.random.randint(0, 3))
            destination = (center + np.random.randint(-spread//2, spread//2), margin + np.random.randint(0, 3))
            self.add_agent(Agent(agent_id, origin, destination))
            agent_id += 1
        
        # East to West (right to left)
        for i in range(quarter):
            y = center + np.random.randint(-spread//2, spread//2)
            origin = (self.grid_size - margin - np.random.randint(0, 3), y)
            destination = (margin + np.random.randint(0, 3), center + np.random.randint(-spread//2, spread//2))
            self.add_agent(Agent(agent_id, origin, destination))
            agent_id += 1
        
        # West to East (left to right)
        for i in range(num_agents - 3*quarter):
            y = center + np.random.randint(-spread//2, spread//2)
            origin = (margin + np.random.randint(0, 3), y)
            destination = (self.grid_size - margin - np.random.randint(0, 3), center + np.random.randint(-spread//2, spread//2))
            self.add_agent(Agent(agent_id, origin, destination))
            agent_id += 1
    
    def generate_commute_agents(self, num_agents: int, seed: Optional[int] = None,
                                 num_residential: int = 4, num_work: int = 3):
        """
        Generate agents simulating daily commute patterns.
        
        Layout:
        - Randomly placed residential clusters (where people live)
        - Randomly placed work clusters (where people work)
        - Agents go from a random home to a random workplace
        
        This creates realistic crossing traffic patterns.
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.clear_agents()
        
        cluster_radius = self.grid_size // 10
        margin = cluster_radius + 3
        
        # Generate random residential cluster centers
        residential_clusters = []
        for _ in range(num_residential):
            cx = np.random.randint(margin, self.grid_size - margin)
            cy = np.random.randint(margin, self.grid_size - margin)
            residential_clusters.append((cx, cy))
        
        # Generate random work cluster centers (try to keep some distance from residential)
        work_clusters = []
        for _ in range(num_work):
            for attempt in range(20):  # Try to find a spot not too close to residential
                cx = np.random.randint(margin, self.grid_size - margin)
                cy = np.random.randint(margin, self.grid_size - margin)
                # Check distance from all residential clusters
                min_dist = min(np.sqrt((cx - r[0])**2 + (cy - r[1])**2) for r in residential_clusters)
                if min_dist > self.grid_size // 4 or attempt == 19:  # Accept if far enough or last attempt
                    work_clusters.append((cx, cy))
                    break
        
        for i in range(num_agents):
            # Pick random residential cluster for origin
            home_cluster = residential_clusters[np.random.randint(len(residential_clusters))]
            # Pick random work cluster for destination
            work_cluster = work_clusters[np.random.randint(len(work_clusters))]
            
            # Random position within cluster
            origin = (
                int(np.clip(home_cluster[0] + np.random.randint(-cluster_radius, cluster_radius + 1), 
                           2, self.grid_size - 3)),
                int(np.clip(home_cluster[1] + np.random.randint(-cluster_radius, cluster_radius + 1), 
                           2, self.grid_size - 3))
            )
            destination = (
                int(np.clip(work_cluster[0] + np.random.randint(-cluster_radius, cluster_radius + 1), 
                           2, self.grid_size - 3)),
                int(np.clip(work_cluster[1] + np.random.randint(-cluster_radius, cluster_radius + 1), 
                           2, self.grid_size - 3))
            )
            
            self.add_agent(Agent(i, origin, destination))
    
    def generate_fourway_intersection_agents(
        self,
        num_agents: int,
        seed: Optional[int] = None,
        road_width: int = 10,
    ):
        """
        Generate agents for a 4-way road network intersection.
        
        Layout: Four arms (North, South, East, West) meet at the center.
        Agents spawn on the outer end of each arm and have destinations on
        one of the other three arms (straight, left, or right). All paths
        pass through or near the center intersection.
        
        Args:
            num_agents: Total number of agents (split evenly across 4 entry arms).
            seed: Random seed for reproducibility.
            road_width: Width of each road arm in cells (strip through center).
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.clear_agents()
        
        gs = self.grid_size
        center = gs // 2
        margin = 2
        lo = center - road_width // 2
        hi = center + road_width // 2
        
        # Entry arm index: 0=North (y=0), 1=South (y=gs-1), 2=East (x=gs-1), 3=West (x=0)
        def sample_origin(arm: int) -> Tuple[int, int]:
            x = int(np.clip(np.random.uniform(lo, hi + 0.99), 0, gs - 1))
            y = int(np.clip(np.random.uniform(lo, hi + 0.99), 0, gs - 1))
            if arm == 0:   # North: spawn near top
                return (x, margin + np.random.randint(0, 3))
            elif arm == 1: # South
                return (x, gs - margin - np.random.randint(0, 3))
            elif arm == 2: # East
                return (gs - margin - np.random.randint(0, 3), y)
            else:          # West
                return (margin + np.random.randint(0, 3), y)
        
        def sample_destination(exit_arm: int) -> Tuple[int, int]:
            x = int(np.clip(np.random.uniform(lo, hi + 0.99), 0, gs - 1))
            y = int(np.clip(np.random.uniform(lo, hi + 0.99), 0, gs - 1))
            if exit_arm == 0:   # North
                return (x, margin + np.random.randint(0, 3))
            elif exit_arm == 1:
                return (x, gs - margin - np.random.randint(0, 3))
            elif exit_arm == 2:
                return (gs - margin - np.random.randint(0, 3), y)
            else:
                return (margin + np.random.randint(0, 3), y)
        
        per_arm = num_agents // 4
        agent_id = 0
        
        for entry_arm in range(4):
            n_this_arm = per_arm if entry_arm < 3 else (num_agents - 3 * per_arm)
            for _ in range(n_this_arm):
                origin = sample_origin(entry_arm)
                # Destination: one of the other 3 arms (random)
                exit_arms = [a for a in range(4) if a != entry_arm]
                exit_arm = np.random.choice(exit_arms)
                destination = sample_destination(exit_arm)
                # Avoid origin == destination (same arm corner)
                if origin[0] == destination[0] and origin[1] == destination[1]:
                    destination = sample_destination(exit_arm)
                self.add_agent(Agent(agent_id, np.array(origin, dtype=float), np.array(destination, dtype=float)))
                agent_id += 1
    
    def check_collision(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """Check if two positions collide."""
        return np.linalg.norm(pos1 - pos2) < self.collision_radius
    
    def get_occupied_positions(self) -> Dict[int, np.ndarray]:
        """Get positions of all agents that haven't reached destination."""
        return {
            agent.id: agent.position.copy() 
            for agent in self.agents 
            if not agent.reached_destination
        }
    
    def is_position_valid(self, position: np.ndarray, exclude_agent_id: int, 
                          proposed_positions: Dict[int, np.ndarray]) -> bool:
        """Check if a position is valid (no collision with others)."""
        # Check bounds
        if (position[0] < 0 or position[0] >= self.grid_size or
            position[1] < 0 or position[1] >= self.grid_size):
            return False
        
        # Check collision with other agents' current and proposed positions
        for agent in self.agents:
            if agent.id == exclude_agent_id or agent.reached_destination:
                continue
            
            # Check against proposed position if available, otherwise current position
            other_pos = proposed_positions.get(agent.id, agent.position)
            if self.check_collision(position, other_pos):
                return False
        
        return True
    
    def step_straight_line(self) -> bool:
        """
        Execute one time step with straight line movement.
        Simple baseline: agents go straight toward destination.
        Basic deadlock escape: sidestep when blocked.
        Returns True if all agents have reached destination.
        """
        self.time_step += 1
        
        # Collect all proposed moves
        proposed_moves: Dict[int, np.ndarray] = {}
        
        # Process agents in random order to avoid bias
        agent_order = list(range(len(self.agents)))
        np.random.shuffle(agent_order)
        
        for idx in agent_order:
            agent = self.agents[idx]
            if agent.reached_destination:
                continue
            
            # Get proposed next position (straight toward destination)
            next_pos = agent.get_straight_line_next_position(speed=1.0)
            
            # Check if move is valid
            if self.is_position_valid(next_pos, agent.id, proposed_moves):
                proposed_moves[agent.id] = next_pos
                agent.move_to(next_pos)
            else:
                # BASIC DEADLOCK ESCAPE: try to sidestep when blocked
                moved = False
                
                # First try smaller direct steps
                for scale in [0.5, 0.25]:
                    direction = next_pos - agent.position
                    smaller_step = agent.position + direction * scale
                    if self.is_position_valid(smaller_step, agent.id, proposed_moves):
                        proposed_moves[agent.id] = smaller_step
                        agent.move_to(smaller_step)
                        moved = True
                        break
                
                # If blocked, try perpendicular sidestep (simple deadlock escape)
                if not moved and agent.wait_time >= 2:
                    direction = agent.destination - agent.position
                    if np.linalg.norm(direction) > 0.1:
                        direction = direction / np.linalg.norm(direction)
                        # Try both perpendicular directions
                        perp1 = np.array([-direction[1], direction[0]])
                        perp2 = -perp1
                        
                        # Alternate which direction to try first based on wait time
                        if agent.wait_time % 2 == 0:
                            perps = [perp1, perp2]
                        else:
                            perps = [perp2, perp1]
                        
                        for perp in perps:
                            side_step = agent.position + perp * 0.8
                            side_step = np.clip(side_step, 0, self.grid_size - 1)
                            if self.is_position_valid(side_step, agent.id, proposed_moves):
                                proposed_moves[agent.id] = side_step
                                agent.move_to(side_step)
                                moved = True
                                break
                
                # If still stuck, try diagonal movements (more options to escape)
                if not moved and agent.wait_time >= 5:
                    direction = agent.destination - agent.position
                    if np.linalg.norm(direction) > 0.1:
                        direction = direction / np.linalg.norm(direction)
                        perp = np.array([-direction[1], direction[0]])
                        
                        # Try 4 diagonal directions (forward-left, forward-right, back-left, back-right)
                        diagonals = [
                            direction * 0.5 + perp * 0.5,
                            direction * 0.5 - perp * 0.5,
                            -direction * 0.3 + perp * 0.5,
                            -direction * 0.3 - perp * 0.5,
                        ]
                        np.random.shuffle(diagonals)
                        
                        for diag in diagonals:
                            diag_step = agent.position + diag
                            diag_step = np.clip(diag_step, 0, self.grid_size - 1)
                            if self.is_position_valid(diag_step, agent.id, proposed_moves):
                                proposed_moves[agent.id] = diag_step
                                agent.move_to(diag_step)
                                moved = True
                                break
                
                # Last resort: random small movement to break symmetry
                if not moved and agent.wait_time >= 10:
                    for _ in range(4):
                        random_dir = np.random.randn(2)
                        random_dir = random_dir / np.linalg.norm(random_dir) * 0.5
                        random_step = agent.position + random_dir
                        random_step = np.clip(random_step, 0, self.grid_size - 1)
                        if self.is_position_valid(random_step, agent.id, proposed_moves):
                            proposed_moves[agent.id] = random_step
                            agent.move_to(random_step)
                            moved = True
                            break
                
                if not moved:
                    proposed_moves[agent.id] = agent.position
                    agent.wait()
        
        # Check if all done
        return all(agent.reached_destination for agent in self.agents)
    
    def step_circular(self, clockwise: bool = True) -> bool:
        """
        Execute one time step with circular movement.
        Agents rotate around the center until aligned with destination, then move straight.
        Returns True if all agents have reached destination.
        """
        self.time_step += 1
        
        proposed_moves: Dict[int, np.ndarray] = {}
        
        # Sort agents by distance to destination - closer agents get priority
        # This helps prevent gridlock by letting almost-done agents finish first
        active_agents = [(i, self.agents[i]) for i in range(len(self.agents)) 
                         if not self.agents[i].reached_destination]
        active_agents.sort(key=lambda x: x[1].distance_to_destination())
        
        for idx, agent in active_agents:
            # Get proposed next position using circular movement
            next_pos = agent.get_circular_next_position(
                center=self.center,
                clockwise=clockwise,
                angular_speed=0.15,
                speed=1.0
            )
            
            # Ensure within bounds
            next_pos = np.clip(next_pos, 0, self.grid_size - 1)
            
            # Check if move is valid
            if self.is_position_valid(next_pos, agent.id, proposed_moves):
                proposed_moves[agent.id] = next_pos
                agent.move_to(next_pos)
            else:
                # Try multiple alternative movements to escape congestion
                moved = False
                
                # Try smaller circular step
                for scale in [0.5, 0.3]:
                    smaller_pos = agent.position + (next_pos - agent.position) * scale
                    smaller_pos = np.clip(smaller_pos, 0, self.grid_size - 1)
                    if self.is_position_valid(smaller_pos, agent.id, proposed_moves):
                        proposed_moves[agent.id] = smaller_pos
                        agent.move_to(smaller_pos)
                        moved = True
                        break
                
                # Try straight line toward destination
                if not moved:
                    straight_pos = agent.get_straight_line_next_position(speed=0.5)
                    straight_pos = np.clip(straight_pos, 0, self.grid_size - 1)
                    if self.is_position_valid(straight_pos, agent.id, proposed_moves):
                        proposed_moves[agent.id] = straight_pos
                        agent.move_to(straight_pos)
                        moved = True
                
                # Try perpendicular movement to escape gridlock
                if not moved and agent.wait_time > 2:
                    to_dest = agent.destination - agent.position
                    if np.linalg.norm(to_dest) > 0.1:
                        to_dest_norm = to_dest / np.linalg.norm(to_dest)
                        # Try both perpendicular directions
                        for sign in [1, -1]:
                            perp = np.array([-to_dest_norm[1], to_dest_norm[0]]) * sign * 0.5
                            perp_pos = agent.position + perp
                            perp_pos = np.clip(perp_pos, 0, self.grid_size - 1)
                            if self.is_position_valid(perp_pos, agent.id, proposed_moves):
                                proposed_moves[agent.id] = perp_pos
                                agent.move_to(perp_pos)
                                moved = True
                                break
                
                # Try moving away from congestion center
                if not moved and agent.wait_time > 5:
                    # Find nearest blocking agent and move away
                    for other in self.agents:
                        if other.id != agent.id and not other.reached_destination:
                            dist = np.linalg.norm(other.position - agent.position)
                            if dist < 2.0:
                                away = agent.position - other.position
                                if np.linalg.norm(away) > 0.1:
                                    away = away / np.linalg.norm(away) * 0.5
                                    away_pos = agent.position + away
                                    away_pos = np.clip(away_pos, 0, self.grid_size - 1)
                                    if self.is_position_valid(away_pos, agent.id, proposed_moves):
                                        proposed_moves[agent.id] = away_pos
                                        agent.move_to(away_pos)
                                        moved = True
                                        break
                
                if not moved:
                    proposed_moves[agent.id] = agent.position
                    agent.wait()
        
        return all(agent.reached_destination for agent in self.agents)
    
    def run_simulation(self, strategy: MovementStrategy, max_steps: Optional[int] = None) -> Dict:
        """
        Run simulation until all agents reach destination or max steps reached.
        Returns statistics.
        """
        if max_steps is None:
            max_steps = self.max_time_steps
        
        self.time_step = 0
        
        # Reset all agents
        for agent in self.agents:
            agent.reset()
        
        # Run simulation
        while self.time_step < max_steps:
            if strategy == MovementStrategy.STRAIGHT_LINE:
                done = self.step_straight_line()
            else:
                done = self.step_circular()
            
            if done:
                break
        
        # Collect statistics
        completed = sum(1 for a in self.agents if a.reached_destination)
        total_time = sum(a.get_total_time() for a in self.agents)
        total_wait = sum(a.wait_time for a in self.agents)
        total_distance = sum(a.total_distance_traveled for a in self.agents)
        
        return {
            "strategy": strategy.value,
            "num_agents": len(self.agents),
            "completed": completed,
            "total_time_steps": self.time_step,
            "total_agent_time": total_time,
            "total_wait_time": total_wait,
            "total_distance": total_distance,
            "avg_time_per_agent": total_time / len(self.agents) if self.agents else 0,
            "avg_wait_per_agent": total_wait / len(self.agents) if self.agents else 0,
            "completion_rate": completed / len(self.agents) if self.agents else 0,
        }
