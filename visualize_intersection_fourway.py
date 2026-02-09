#!/usr/bin/env python3
"""
Visualize the 4-way road network: two lanes per arm (inbound/outbound), right-hand driving.
Vehicles can turn left, straight, or right. Methods: GSP, PIBT (Paper-PIBT), Circular, SP, GLC.
Supports --method and --save-gif for paper figures.
"""

import pygame
import numpy as np
import sys
from road_network import RoadNetwork4Way
from intersection_lane import (
    generate_lane_agents_4way,
    step_gsp_lane_once,
    step_circular_lane_once,
)
from experiments_summary import run_sp_pibt_4way, generate_lane_agents_4way_left_bias

# Colors: by turn (right=green, straight=yellow, left=red) and entry arm
COLORS = {
    "background": (18, 20, 26),
    "road": (70, 75, 85),
    "intersection": (55, 60, 72),
    "center": (90, 95, 108),
    "turn_right": (90, 220, 110),
    "turn_straight": (255, 200, 80),
    "turn_left": (255, 90, 90),
    "reached": (100, 180, 100),
    "text": (220, 220, 232),
    "panel": (26, 28, 36),
}


def visualize(
    num_agents: int = 32,
    grid_size: int = 64,
    intersection_half: int = 10,
    cell_size: int = 9,
    seed: int = 42,
    initial_method: str = "gsp",
):
    pygame.init()
    pygame.font.init()
    road = RoadNetwork4Way(grid_size=grid_size, intersection_half=intersection_half)
    win_w = grid_size * cell_size
    panel_w = 220
    screen = pygame.display.set_mode((win_w + panel_w, win_w))
    pygame.display.set_caption("4-Way (2 Lanes/Arm, Left/Straight/Right) — GSP vs PIBT vs Circular")
    font = pygame.font.SysFont("Monaco", 14)
    font_large = pygame.font.SysFont("Monaco", 16)

    origins, destinations, entry_arms, turn_types = generate_lane_agents_4way(road, num_agents, seed)
    n = len(origins)
    center = road.center

    method = initial_method if initial_method in ("gsp", "pibt", "circular", "sp") else "gsp"
    paused = True
    clock = pygame.time.Clock()
    speed = 6
    current_seed = seed

    gsp_positions = [list(o) for o in origins]
    gsp_goals = list(destinations)
    gsp_reached = [False] * n
    gsp_times = [0] * n

    # PIBT = Paper-PIBT (time-based priority) for paper
    planner = road.create_paper_pibt_planner(collision_radius=0.6)
    HAS_PIBT = planner is not None
    pibt_positions = {i: (float(origins[i][0]), float(origins[i][1])) for i in range(n)}
    pibt_goals = {i: (float(destinations[i][0]), float(destinations[i][1])) for i in range(n)}
    pibt_reached = {i: False for i in range(n)}
    pibt_times = {i: 0 for i in range(n)}
    sim_step = 0

    # Circular (roundabout) state
    circular_positions = [list(o) for o in origins]
    circular_goals = list(destinations)
    circular_reached = [False] * n
    circular_times = [0] * n
    circular_exiting = [False] * n

    # SP (Shortest Path) state
    sp_positions = [list(o) for o in origins]
    sp_goals = list(destinations)
    sp_reached = [False] * n
    sp_times = [0] * n
    sp_paths = {i: None for i in range(n)}  # Paths computed once per agent

    def reset_simulation(new_seed: int):
        nonlocal origins, destinations, entry_arms, turn_types
        nonlocal gsp_positions, gsp_goals, gsp_reached, gsp_times
        nonlocal pibt_positions, pibt_goals, pibt_reached, pibt_times, sim_step
        nonlocal circular_positions, circular_goals, circular_reached, circular_times, circular_exiting
        nonlocal sp_positions, sp_goals, sp_reached, sp_times, sp_paths
        origins, destinations, entry_arms, turn_types = generate_lane_agents_4way(road, num_agents, new_seed)
        gsp_positions = [list(o) for o in origins]
        gsp_goals = list(destinations)
        gsp_reached = [False] * n
        gsp_times = [0] * n
        pibt_positions = {i: (float(origins[i][0]), float(origins[i][1])) for i in range(n)}
        pibt_goals = {i: (float(destinations[i][0]), float(destinations[i][1])) for i in range(n)}
        pibt_reached = {i: False for i in range(n)}
        pibt_times = {i: 0 for i in range(n)}
        circular_positions = [list(o) for o in origins]
        circular_goals = list(destinations)
        circular_reached = [False] * n
        circular_times = [0] * n
        circular_exiting = [False] * n
        sp_positions = [list(o) for o in origins]
        sp_goals = list(destinations)
        sp_reached = [False] * n
        sp_times = [0] * n
        sp_paths = {i: None for i in range(n)}
        sim_step = 0

    def cell_to_screen(x: int, y: int):
        return (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2)

    def draw_road():
        screen.fill(COLORS["background"])
        lo, hi = road.lo, road.hi
        # Large central intersection (for future roundabout)
        pygame.draw.rect(
            screen, COLORS["intersection"],
            (lo * cell_size, lo * cell_size, (hi - lo + 1) * cell_size, (hi - lo + 1) * cell_size),
        )
        cx, cy = cell_to_screen(road.center, road.center)
        pygame.draw.circle(screen, COLORS["center"], (cx, cy), 6)
        # One lane per direction: draw road cells (lanes = lines of cells, intersection = block)
        r = max(2, cell_size // 2)
        for (x, y) in road.get_road_cells():
            px, py = cell_to_screen(x, y)
            if road.in_intersection(x, y):
                pygame.draw.circle(screen, COLORS["intersection"], (px, py), r)
            else:
                pygame.draw.circle(screen, COLORS["road"], (px, py), r)

    def step_pibt():
        nonlocal sim_step
        active = {i for i in range(n) if not pibt_reached[i]}
        if not active:
            return
        active_pos = {i: pibt_positions[i] for i in active}
        active_goals = {i: pibt_goals[i] for i in active}
        next_pos = planner.plan_step(active_pos, active_goals)
        for i in active:
            pibt_positions[i] = next_pos.get(i, pibt_positions[i])
            pibt_times[i] += 1
            # Use threshold of 2.0 to handle parallel lane cases
            if planner.distance(pibt_positions[i], pibt_goals[i]) < 2.0:
                pibt_reached[i] = True
        sim_step += 1

    def step_circular():
        nonlocal sim_step
        active = [i for i in range(n) if not circular_reached[i]]
        if not active:
            return
        step_circular_lane_once(
            road, circular_positions, circular_goals,
            circular_reached, circular_times, circular_exiting,
        )
        sim_step += 1

    def step_sp():
        """SP: Compute shortest path once, wait if blocked."""
        nonlocal sim_step
        active = [i for i in range(n) if not sp_reached[i]]
        if not active:
            return
        
        # Build occupied set (excluding agents that have reached)
        occupied = set()
        for i in range(n):
            if not sp_reached[i]:
                occupied.add((int(sp_positions[i][0]), int(sp_positions[i][1])))
        
        # Compute paths for agents that don't have one yet
        for i in active:
            if sp_paths[i] is None:
                start = (int(sp_positions[i][0]), int(sp_positions[i][1]))
                goal = (int(sp_goals[i][0]), int(sp_goals[i][1]))
                path, _ = road.shortest_path(start, goal)
                if path:
                    sp_paths[i] = list(path)
                else:
                    sp_paths[i] = [start]  # No path, just stay
        
        # Move agents along their fixed paths
        moves = {}
        for i in active:
            path = sp_paths[i]
            if not path or len(path) <= 1:
                continue
            
            curr = (int(sp_positions[i][0]), int(sp_positions[i][1]))
            
            # Find current position in path and get next step
            path_idx = 0
            for idx, p in enumerate(path):
                if p == curr:
                    path_idx = idx
                    break
            
            if path_idx + 1 < len(path):
                next_pos = path[path_idx + 1]
                # Check if next position is blocked
                if next_pos not in occupied or next_pos == curr:
                    moves[i] = next_pos
                # else: wait (no move)
        
        # Apply moves
        for i, next_pos in moves.items():
            occupied.discard((int(sp_positions[i][0]), int(sp_positions[i][1])))
            sp_positions[i] = list(next_pos)
            occupied.add(next_pos)
            sp_times[i] += 1
            
            goal = (int(sp_goals[i][0]), int(sp_goals[i][1]))
            if next_pos == goal or (abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])) <= 1:
                sp_reached[i] = True
        
        # Increment time for waiting agents
        for i in active:
            if i not in moves and not sp_reached[i]:
                sp_times[i] += 1
        
        sim_step += 1

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
                elif event.key == pygame.K_2 and HAS_PIBT:
                    method = "pibt"
                    reset_simulation(current_seed)
                elif event.key == pygame.K_3:
                    method = "circular"
                    reset_simulation(current_seed)
                elif event.key == pygame.K_4:
                    method = "sp"
                    reset_simulation(current_seed)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(40, speed + 2)
                elif event.key == pygame.K_MINUS:
                    speed = max(1, speed - 2)
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        if not paused:
            if method == "gsp":
                step_gsp_lane_once(road, gsp_positions, gsp_goals, gsp_reached, gsp_times)
                sim_step += 1
            elif method == "circular":
                if not all(circular_reached):
                    step_circular()
            elif method == "sp":
                if not all(sp_reached):
                    step_sp()
            else:
                if HAS_PIBT and not all(pibt_reached.values()):
                    step_pibt()

        draw_road()

        # Draw agents (color by turn: right=green, straight=yellow, left=red)
        turn_color = [COLORS["turn_right"], COLORS["turn_straight"], COLORS["turn_left"]]
        if method == "gsp":
            for i in range(n):
                pos = (gsp_positions[i][0], gsp_positions[i][1])
                color = COLORS["reached"] if gsp_reached[i] else turn_color[turn_types[i] if i < len(turn_types) else 1]
                px, py = cell_to_screen(pos[0], pos[1])
                pygame.draw.circle(screen, color, (px, py), 4)
                if not gsp_reached[i]:
                    gx, gy = cell_to_screen(gsp_goals[i][0], gsp_goals[i][1])
                    pygame.draw.circle(screen, color, (gx, gy), 2, 1)
        elif method == "circular":
            for i in range(n):
                pos = (circular_positions[i][0], circular_positions[i][1])
                color = COLORS["reached"] if circular_reached[i] else turn_color[turn_types[i] if i < len(turn_types) else 1]
                px, py = cell_to_screen(pos[0], pos[1])
                pygame.draw.circle(screen, color, (px, py), 4)
                if not circular_reached[i]:
                    gx, gy = cell_to_screen(circular_goals[i][0], circular_goals[i][1])
                    pygame.draw.circle(screen, color, (gx, gy), 2, 1)
        elif method == "sp":
            for i in range(n):
                pos = (sp_positions[i][0], sp_positions[i][1])
                color = COLORS["reached"] if sp_reached[i] else turn_color[turn_types[i] if i < len(turn_types) else 1]
                px, py = cell_to_screen(pos[0], pos[1])
                pygame.draw.circle(screen, color, (px, py), 4)
                if not sp_reached[i]:
                    gx, gy = cell_to_screen(sp_goals[i][0], sp_goals[i][1])
                    pygame.draw.circle(screen, color, (gx, gy), 2, 1)
        else:
            for i in range(n):
                pos = pibt_positions[i]
                ix, iy = int(round(pos[0])), int(round(pos[1]))
                color = COLORS["reached"] if pibt_reached[i] else turn_color[turn_types[i] if i < len(turn_types) else 1]
                px, py = cell_to_screen(ix, iy)
                pygame.draw.circle(screen, color, (px, py), 4)
                if not pibt_reached[i]:
                    gx, gy = cell_to_screen(int(pibt_goals[i][0]), int(pibt_goals[i][1]))
                    pygame.draw.circle(screen, color, (gx, gy), 2, 1)

        # Panel
        pygame.draw.rect(screen, COLORS["panel"], (win_w, 0, panel_w, win_w))
        pygame.draw.line(screen, (55, 58, 70), (win_w, 0), (win_w, win_w), 2)
        px, y = win_w + 12, 20
        screen.blit(font_large.render("4-Way (2 Lanes/Arm)", True, COLORS["text"]), (px, y))
        y += 26
        screen.blit(font.render(f"Method: {method.upper()}", True, (140, 220, 140)), (px, y))
        y += 18
        screen.blit(font.render("Green=R  Yel=S  Red=L", True, COLORS["text"]), (px, y))
        y += 22
        if method == "gsp":
            completed = sum(1 for r in gsp_reached if r)
            total_time = sum(gsp_times)
        elif method == "circular":
            completed = sum(1 for r in circular_reached if r)
            total_time = sum(circular_times)
        elif method == "sp":
            completed = sum(1 for r in sp_reached if r)
            total_time = sum(sp_times)
        else:
            completed = sum(1 for r in pibt_reached.values() if r)
            total_time = sum(pibt_times.values())
        screen.blit(font.render(f"Completed: {completed}/{n}", True, COLORS["text"]), (px, y))
        y += 18
        screen.blit(font.render(f"Total agent time: {total_time}", True, COLORS["text"]), (px, y))
        y += 18
        screen.blit(font.render(f"Step: {sim_step}", True, COLORS["text"]), (px, y))
        y += 26
        for line in [
            "1 = GSP",
            "2 = PIBT" if HAS_PIBT else "(PIBT N/A)",
            "3 = Circular",
            "4 = SP",
            "SPACE = Pause",
            "R = Reset",
            "+/- = Speed",
            "Q = Quit",
        ]:
            screen.blit(font.render(line, True, COLORS["text"]), (px, y))
            y += 18
        y = win_w - 28
        status = "PAUSED" if paused else "RUNNING"
        screen.blit(font_large.render(status, True, (255, 180, 80) if paused else (100, 255, 120)), (px, y))

        pygame.display.flip()
        clock.tick(speed)

    pygame.quit()


def save_4way_gif(
    method: str,
    num_agents: int,
    grid_size: int,
    intersection_half: int,
    seed: int,
    out_path: str,
    max_steps: int = 2000,
    cell_size: int = 9,
    fps: int = 12,
):
    """Run 4-way simulation for the given method and save frames as GIF. Supports GLC (and others if added)."""
    pygame.init()
    road = RoadNetwork4Way(grid_size=grid_size, intersection_half=intersection_half)
    win_w = grid_size * cell_size
    # Paper default: 30% left, 35% straight, 35% right
    origins, destinations, _, turn_types = generate_lane_agents_4way_left_bias(
        road, num_agents, seed, left_turn_ratio=0.3
    )
    n = len(origins)
    turn_color = [COLORS["turn_right"], COLORS["turn_straight"], COLORS["turn_left"]]

    if method.lower() != "glc":
        raise ValueError("save_4way_gif currently supports method=glc only. Use --save-gif with method glc.")

    frames_positions = []
    def step_callback(positions_list, step):
        frames_positions.append([list(p) for p in positions_list])

    run_sp_pibt_4way(road, origins, destinations, max_steps=max_steps, step_callback=step_callback)
    if not frames_positions:
        frames_positions = [[list(origins[i]) for i in range(n)]]

    # Render each frame to a surface and collect as RGB arrays for GIF
    surface = pygame.Surface((win_w, win_w))
    lo, hi = road.lo, road.hi

    def cell_to_screen(x, y):
        return (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2)

    try:
        import imageio
    except ImportError:
        try:
            from PIL import Image
            HAS_IMAGEIO = False
        except ImportError:
            raise ImportError("Need imageio or Pillow (PIL) to save GIF. Install with: pip install imageio")
    else:
        HAS_IMAGEIO = True

    images = []
    for positions_list in frames_positions:
        surface.fill(COLORS["background"])
        pygame.draw.rect(
            surface, COLORS["intersection"],
            (lo * cell_size, lo * cell_size, (hi - lo + 1) * cell_size, (hi - lo + 1) * cell_size),
        )
        cx, cy = cell_to_screen(road.center, road.center)
        pygame.draw.circle(surface, COLORS["center"], (cx, cy), 6)
        r = max(2, cell_size // 2)
        for (x, y) in road.get_road_cells():
            px, py = cell_to_screen(x, y)
            if road.in_intersection(x, y):
                pygame.draw.circle(surface, COLORS["intersection"], (px, py), r)
            else:
                pygame.draw.circle(surface, COLORS["road"], (px, py), r)
        for i in range(n):
            pos = positions_list[i] if i < len(positions_list) else origins[i]
            color = turn_color[turn_types[i] if i < len(turn_types) else 1]
            px, py = cell_to_screen(pos[0], pos[1])
            pygame.draw.circle(surface, color, (px, py), 4)

        if HAS_IMAGEIO:
            img = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
            images.append(img)
        else:
            img_str = pygame.image.tostring(surface, "RGB", False)
            img = Image.frombytes("RGB", (win_w, win_w), img_str)
            images.append(img)

    if HAS_IMAGEIO:
        imageio.mimsave(out_path, images, fps=fps, loop=0)
    else:
        images[0].save(out_path, save_all=True, append_images=images[1:], duration=1000//fps, loop=0)
    pygame.quit()
    print(f"Saved {len(images)} frames to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Visualize 4-way intersection. Methods: gsp, pibt, circular, sp, glc.")
    p.add_argument("--agents", type=int, default=24, help="Number of agents (paper default 24)")
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--intersection-half", type=int, default=5,
                   help="Half-size of intersection (paper: 5 = 11×11)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--method", type=str, default="gsp",
                   choices=["gsp", "pibt", "circular", "sp", "glc"],
                   help="Method to visualize (PIBT = Paper-PIBT)")
    p.add_argument("--save-gif", type=str, default="",
                   help="Save animation to GIF (e.g. 4way_glc_24agents.gif). Supports method=glc.")
    args = p.parse_args()

    if args.save_gif:
        method_for_gif = args.method if args.method else "glc"
        if method_for_gif.lower() != "glc":
            print("Warning: --save-gif currently only records GLC. Running GLC for the GIF.")
            method_for_gif = "glc"
        save_4way_gif(
            method=method_for_gif,
            num_agents=args.agents,
            grid_size=args.grid,
            intersection_half=args.intersection_half,
            seed=args.seed,
            out_path=args.save_gif,
        )
    else:
        visualize(
            num_agents=args.agents,
            grid_size=args.grid,
            intersection_half=args.intersection_half,
            seed=args.seed,
            initial_method=args.method,
        )
