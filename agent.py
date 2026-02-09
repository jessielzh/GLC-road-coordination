"""
Agent class representing a dot that moves from origin to destination.
"""
import numpy as np
from typing import Tuple, Optional, List


class Agent:
    """A dot that needs to move from origin to destination."""
    
    def __init__(self, agent_id: int, origin: Tuple[int, int], destination: Tuple[int, int]):
        self.id = agent_id
        self.origin = np.array(origin, dtype=float)
        self.destination = np.array(destination, dtype=float)
        self.position = np.array(origin, dtype=float)
        self.reached_destination = False
        self.wait_time = 0
        self.move_time = 0
        self.total_distance_traveled = 0.0
        self._prev_positions = []  # Track recent positions to detect oscillation
        self._circulation_count = 0  # Track how long we've been on the ring
        self._exiting_roundabout = False  # Once we decide to exit, commit to it
        
    @property
    def grid_position(self) -> Tuple[int, int]:
        """Get current position as grid coordinates."""
        return (int(round(self.position[0])), int(round(self.position[1])))
    
    def distance_to_destination(self) -> float:
        """Calculate Euclidean distance to destination."""
        return np.linalg.norm(self.destination - self.position)
    
    def has_reached_destination(self, tolerance: float = 0.5) -> bool:
        """Check if agent has reached its destination."""
        return self.distance_to_destination() < tolerance
    
    def get_straight_line_next_position(self, speed: float = 1.0) -> np.ndarray:
        """Calculate next position moving in straight line toward destination."""
        if self.has_reached_destination():
            return self.position.copy()
        
        direction = self.destination - self.position
        distance = np.linalg.norm(direction)
        
        if distance < speed:
            return self.destination.copy()
        
        unit_direction = direction / distance
        return self.position + unit_direction * speed
    
    def get_circular_next_position(self, center: np.ndarray, clockwise: bool = True, 
                                    angular_speed: float = 0.15, speed: float = 1.0) -> np.ndarray:
        """
        IMPROVED ROUNDABOUT movement strategy with deadlock prevention:
        
        Key insight: The roundabout's purpose is to avoid head-on collisions by
        having everyone flow the same direction. Once an agent has rotated past
        the "danger zone" (where they'd collide with oncoming traffic), they
        should exit - they don't need perfect angle alignment.
        
        Rules:
        1. COMMIT TO EXIT: Once we decide to exit, don't re-enter circulation
        2. CIRCULATION LIMIT: Max time on ring before forced exit
        3. SMART EXIT: Exit when path to destination is "clear" (won't cross center)
        4. DISTANCE PRIORITY: If getting farther from destination, exit
        """
        if self.has_reached_destination():
            return self.position.copy()
        
        to_dest = self.destination - self.position
        dist_to_dest = np.linalg.norm(to_dest)
        
        if dist_to_dest < speed:
            return self.destination.copy()
        
        # Once committed to exiting, keep going straight
        if self._exiting_roundabout:
            return self.get_straight_line_next_position(speed)
        
        # Close to destination - go straight and commit to exit
        if dist_to_dest < 5.0:
            self._exiting_roundabout = True
            return self.get_straight_line_next_position(speed)
        
        # Calculate positions relative to center
        pos_from_center = self.position - center
        dest_from_center = self.destination - center
        
        current_radius = np.linalg.norm(pos_from_center)
        dest_radius = np.linalg.norm(dest_from_center)
        
        # Define roundabout ring radius
        ring_radius = np.linalg.norm(center) * 0.6
        ring_tolerance = 3.0
        
        # Calculate angles
        current_angle = np.arctan2(pos_from_center[1], pos_from_center[0])
        dest_angle = np.arctan2(dest_from_center[1], dest_from_center[0])
        
        # Angle difference (normalized to [-pi, pi])
        angle_diff = dest_angle - current_angle
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        # Rotation direction: clockwise = -1, counter-clockwise = +1
        rot_sign = -1 if clockwise else 1
        
        # SMART EXIT CONDITIONS:
        # 1. Direct path to destination doesn't cross near center (safe to exit)
        # 2. We've been circulating too long
        # 3. We're now on the "correct side" to reach destination
        
        # Check if path to destination crosses center zone
        # Use cross product to determine if center is to the left or right of our path
        # and dot product to see if center is ahead or behind
        path_dir = to_dest / dist_to_dest
        to_center_from_pos = center - self.position
        
        # Project center onto our path
        proj_length = np.dot(to_center_from_pos, path_dir)
        # Perpendicular distance from center to our path
        perp_dist = np.linalg.norm(to_center_from_pos - proj_length * path_dir)
        
        # Path is "safe" if:
        # - Center is not in front of us (proj_length <= 0), OR
        # - Center is far from our path (perp_dist > threshold), OR
        # - Center is beyond our destination (proj_length > dist_to_dest)
        center_clearance = ring_radius * 0.5
        path_is_safe = (proj_length <= 0 or 
                        perp_dist > center_clearance or 
                        proj_length > dist_to_dest)
        
        # Check if we're roughly aligned (within 90 degrees)
        roughly_aligned = abs(angle_diff) < np.pi / 2
        
        # Force exit conditions
        max_circulation = 50  # Maximum steps on the ring
        force_exit = self._circulation_count > max_circulation
        
        # Determine if we should exit
        should_exit = path_is_safe or force_exit or (roughly_aligned and self._circulation_count > 10)
        
        # Check if we're on/near the ring
        on_ring = abs(current_radius - ring_radius) < ring_tolerance
        
        if on_ring:
            self._circulation_count += 1
            
            if should_exit:
                # EXIT: Commit to leaving and go toward destination
                self._exiting_roundabout = True
                direction = to_dest / dist_to_dest
                return self.position + direction * speed
            else:
                # CIRCULATE: Keep rotating on the ring
                new_angle = current_angle + rot_sign * angular_speed
                new_pos = center + ring_radius * np.array([np.cos(new_angle), np.sin(new_angle)])
                return new_pos
        
        # Outside the ring - move toward ring
        elif current_radius > ring_radius + ring_tolerance:
            # Check if we should skip the roundabout entirely
            if path_is_safe and roughly_aligned:
                self._exiting_roundabout = True
                return self.get_straight_line_next_position(speed)
            
            # Move toward ring with rotational bias to merge smoothly
            target_on_ring = center + ring_radius * (pos_from_center / current_radius)
            direction = target_on_ring - self.position
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 0.1:
                direction = direction / dir_norm
            else:
                direction = np.array([1.0, 0.0])
            
            tangent = np.array([rot_sign * pos_from_center[1], -rot_sign * pos_from_center[0]])
            tang_norm = np.linalg.norm(tangent)
            if tang_norm > 0.1:
                tangent = tangent / tang_norm
            else:
                tangent = np.array([0.0, rot_sign])
            
            movement = direction * 0.7 + tangent * 0.3
            movement = movement / np.linalg.norm(movement) * speed
            return self.position + movement
        
        # Inside the ring - move outward or toward destination
        else:
            if should_exit or roughly_aligned:
                self._exiting_roundabout = True
                direction = to_dest / dist_to_dest
                return self.position + direction * speed
            else:
                # Move outward to ring while rotating
                target_on_ring = center + ring_radius * (pos_from_center / max(current_radius, 0.1))
                direction = target_on_ring - self.position
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 0.1:
                    direction = direction / dir_norm
                else:
                    direction = np.array([1.0, 0.0])
                
                tangent = np.array([rot_sign * pos_from_center[1], -rot_sign * pos_from_center[0]])
                tang_norm = np.linalg.norm(tangent)
                if tang_norm > 0.1:
                    tangent = tangent / tang_norm
                else:
                    tangent = np.array([0.0, rot_sign])
                
                movement = direction * 0.5 + tangent * 0.5
                movement = movement / np.linalg.norm(movement) * speed
                return self.position + movement
    
    def move_to(self, new_position: np.ndarray):
        """Move agent to new position."""
        distance = np.linalg.norm(new_position - self.position)
        self.total_distance_traveled += distance
        self.position = new_position
        self.move_time += 1
        
        # Track positions for oscillation detection
        self._prev_positions.append(self.position.copy())
        if len(self._prev_positions) > 20:
            self._prev_positions.pop(0)
        
        if self.has_reached_destination():
            self.reached_destination = True
            self.position = self.destination.copy()
    
    def wait(self):
        """Agent waits (blocked by collision)."""
        self.wait_time += 1
    
    def get_total_time(self) -> int:
        """Get total time spent (moving + waiting)."""
        return self.move_time + self.wait_time
    
    def reset(self):
        """Reset agent to origin."""
        self.position = self.origin.copy()
        self.reached_destination = False
        self.wait_time = 0
        self.move_time = 0
        self.total_distance_traveled = 0.0
        self._prev_positions = []
        self._circulation_count = 0
        self._exiting_roundabout = False
