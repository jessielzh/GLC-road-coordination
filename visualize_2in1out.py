#!/usr/bin/env python3
"""
Visualize 2-in-1-out intersection: 2 inbound lanes, 1 outbound lane per arm.
Compare SP (rigid BFS), Circular (roundabout), and PIBT (priority) methods.

Controls:
  1 = SP (rigid BFS, same as compare script)
  2 = Circular (roundabout)
  3 = PIBT (priority)
  SPACE = Pause/Resume
  R = Reset (new seed)
  UP/DOWN = Left turn ratio (+/- 10%)
  +/- = Speed
  Q = Quit
"""

import pygame
import numpy as np
import math
from collections import deque
from road_network import RoadNetwork2In1Out

COLORS = {
    "background":     (18,  20,  26),
    "road":           (70,  75,  85),
    "intersection":   (60,  65,  78),
    "inbound_left":   (70,  75,  85),
    "inbound_right":  (70,  75,  85),
    "outbound":       (90,  95, 105),
    "vehicle_left":   (255, 100, 100),   # RED    — left-turn vehicles
    "vehicle_through":(100, 180, 255),   # BLUE   — straight/through vehicles
    "vehicle_right":  (255, 180,  50),   # ORANGE — right-turn vehicles
    "reached":        (80,  80,  80),    # GRAY   — completed
    "text":           (220, 220, 232),
    "panel":          (26,  28,  36),
}


# ---------------------------------------------------------------------------
# Agent generation
# ---------------------------------------------------------------------------

def generate_agents(road, n_agents, seed, left_ratio=0.5, right_ratio=0.0):
    """Generate agents with configurable turn ratios.

    Turn directions (right-hand driving):
      left     = (entry_arm + 1) % 4
      straight = (entry_arm + 2) % 4
      right    = (entry_arm + 3) % 4

    Args:
        left_ratio:  fraction making left turns
        right_ratio: fraction making right turns (remainder go straight)

    Returns (origins, destinations, turn_types) where turn_types[i] in {'left','through','right'}.
    """
    rng = np.random.default_rng(seed)
    origins, destinations, turn_types = [], [], []
    per_arm = max(1, n_agents // 4)
    for entry_arm in range(4):
        n = per_arm if entry_arm < 3 else (n_agents - 3 * per_arm)
        for _ in range(n):
            r = rng.random()
            if r < left_ratio:
                turn = 'left'
                dest_arm = (entry_arm + 1) % 4
            elif r < left_ratio + right_ratio:
                turn = 'right'
                dest_arm = (entry_arm + 3) % 4
            else:
                turn = 'through'
                dest_arm = (entry_arm + 2) % 4
            spawn = road.get_spawn_cells_inbound(entry_arm, 8)
            dest  = road.get_dest_cells_outbound(dest_arm, 4)
            origins.append(list(spawn[rng.integers(len(spawn))]))
            destinations.append(dest[rng.integers(len(dest))])
            turn_types.append(turn)
    return origins, destinations, turn_types


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def step_gsp(road, positions, goals, reached, times, reserved, prev_moves=None):
    """GSP: greedy shortest path with diagonal zig-zag inside the intersection."""
    n = len(positions)
    if prev_moves is None:
        prev_moves = {}

    active = sorted([i for i in range(n) if not reached[i]],
                    key=lambda i: road.distance(positions[i], goals[i]))
    next_pos = [None] * n

    for i in active:
        px, py = positions[i]
        gx, gy = goals[i]
        if abs(px - gx) <= 1 and abs(py - gy) <= 1:
            reached[i] = True
            next_pos[i] = (gx, gy)
            reserved.add((gx, gy))
            times[i] += 1
            continue

        valid = road.get_valid_moves(px, py, reserved)
        best = None

        if road.in_intersection(px, py) and valid:
            x_moves = [c for c in valid
                       if abs(c[0] - gx) + abs(c[1] - gy) < abs(px - gx) + abs(py - gy)
                       and c[0] != px]
            y_moves = [c for c in valid
                       if abs(c[0] - gx) + abs(c[1] - gy) < abs(px - gx) + abs(py - gy)
                       and c[1] != py]
            if x_moves and y_moves:
                prev_dir = prev_moves.get(i)
                if prev_dir == 'x':
                    best = y_moves[0]; prev_moves[i] = 'y'
                elif prev_dir == 'y':
                    best = x_moves[0]; prev_moves[i] = 'x'
                else:
                    if np.random.random() < 0.5:
                        best = x_moves[0]; prev_moves[i] = 'x'
                    else:
                        best = y_moves[0]; prev_moves[i] = 'y'
            elif x_moves:
                best = x_moves[0]; prev_moves[i] = 'x'
            elif y_moves:
                best = y_moves[0]; prev_moves[i] = 'y'
            else:
                best = min(valid, key=lambda c: road.distance(c, goals[i]))
        elif valid:
            best = min(valid, key=lambda c: road.distance(c, goals[i]))

        if best:
            next_pos[i] = best
            reserved.add(best)
        times[i] += 1

    for i in range(n):
        if next_pos[i]:
            positions[i][0], positions[i][1] = next_pos[i]

    return prev_moves


def _bfs_path(road, start, goal):
    """BFS shortest path on RoadNetwork2In1Out."""
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


def step_sp_rigid(road, positions, goals, reached, times, paths, consec_wait,
                  stuck_threshold=30):
    """Rigid BFS SP: compute path once at spawn, follow it, wait if blocked.
    After stuck_threshold consecutive waits, take any greedy move and replan.
    Matches compare_2in1out.py run_sp and grid_3x3_network.py step_sp.
    """
    n = len(positions)
    occupied = {(int(positions[i][0]), int(positions[i][1])) for i in range(n) if not reached[i]}
    active = sorted([i for i in range(n) if not reached[i]],
                    key=lambda i: road.distance(positions[i], goals[i]))
    local_reserved = set()

    for i in active:
        px, py = int(positions[i][0]), int(positions[i][1])
        gx, gy = int(goals[i][0]),    int(goals[i][1])

        if abs(px-gx) <= 1 and abs(py-gy) <= 1:
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

        if not moved and consec_wait[i] >= stuck_threshold:
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


def step_circular(road, positions, goals, reached, times, exiting, reserved):
    """Circular (roundabout): vehicles follow the intersection perimeter clockwise then exit."""
    lo, hi = road.lo, road.hi

    def is_on_edge(x, y):
        return road.in_intersection(x, y) and (x == lo or x == hi or y == lo or y == hi)

    def cw_dir(x, y):
        if x == hi and y == lo: return (0,  1)
        if x == hi and y == hi: return (-1, 0)
        if x == lo and y == hi: return (0, -1)
        if x == lo and y == lo: return (1,  0)
        if y == lo:  return (1,  0)
        if x == hi:  return (0,  1)
        if y == hi:  return (-1, 0)
        if x == lo:  return (0, -1)
        return (0, 0)

    def should_exit(x, y, gx, gy):
        if gy < lo and y == lo:  return True
        if gy > hi and y == hi:  return True
        if gx > hi and x == hi:  return True
        if gx < lo and x == lo:  return True
        return False

    n = len(positions)
    active = sorted([i for i in range(n) if not reached[i]],
                    key=lambda i: road.distance(positions[i], goals[i]))
    next_pos = [None] * n

    for i in active:
        px, py = positions[i]
        gx, gy = goals[i]
        if abs(px - gx) <= 1 and abs(py - gy) <= 1:
            reached[i] = True
            next_pos[i] = (gx, gy)
            reserved.add((gx, gy))
            times[i] += 1
            exiting[i] = True
            continue

        valid = road.get_valid_moves(px, py, reserved)
        best = None

        if road.in_intersection(px, py):
            if should_exit(px, py, gx, gy) or exiting[i]:
                exiting[i] = True
                exits = [c for c in valid if road.in_intersection(c[0], c[1]) or road.is_outbound_lane(c[0], c[1])]
                if exits:
                    best = min(exits, key=lambda c: road.distance(c, goals[i]))
                elif valid:
                    best = min(valid, key=lambda c: road.distance(c, goals[i]))
            elif is_on_edge(px, py):
                dx, dy = cw_dir(px, py)
                cw_next = (px + dx, py + dy)
                if cw_next in valid:
                    best = cw_next
                else:
                    edge_nb = [c for c in valid if is_on_edge(c[0], c[1])]
                    best = edge_nb[0] if edge_nb else (min(valid, key=lambda c: road.distance(c, goals[i])) if valid else None)
            else:
                edge_nb = [c for c in valid if is_on_edge(c[0], c[1])]
                if edge_nb:
                    best = edge_nb[0]
                elif valid:
                    best = min(valid, key=lambda c: min(abs(c[0]-lo), abs(c[0]-hi), abs(c[1]-lo), abs(c[1]-hi)))
        elif valid:
            best = min(valid, key=lambda c: road.distance(c, goals[i]))

        if best:
            next_pos[i] = best
            reserved.add(best)
        times[i] += 1

    for i in range(n):
        if next_pos[i]:
            positions[i][0], positions[i][1] = next_pos[i]


class _PIBTPlanner:
    """PIBT collision resolver using road.get_valid_moves for lane discipline."""

    def __init__(self, road):
        self.road = road

    def distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def plan_step(self, positions, goals):
        agents = sorted(positions.keys(), key=lambda a: self.distance(positions[a], goals[a]))
        decided = {}
        reserved = set()
        in_progress = set()

        def pibt(aid):
            if aid in decided: return True
            if aid in in_progress: return False
            in_progress.add(aid)
            pos  = positions[aid]
            goal = goals[aid]
            candidates = [pos] + self.road.get_valid_moves(pos[0], pos[1])
            candidates.sort(key=lambda c: self.distance(c, goal))
            for cand in candidates:
                cell = (int(cand[0]), int(cand[1]))
                if cell in reserved: continue
                blocker = next((o for o in agents if o != aid and o not in decided
                                and positions[o] == cell), None)
                if blocker is not None and not pibt(blocker):
                    continue
                decided[aid] = cell
                reserved.add(cell)
                in_progress.discard(aid)
                return True
            decided[aid] = pos
            reserved.add(pos)
            in_progress.discard(aid)
            return False

        for a in agents:
            pibt(a)
        return decided


def step_pibt(road, planner, positions, goals, reached, times):
    """PIBT step: priority-based collision resolution."""
    n = len(positions)
    active = {i for i in range(n) if not reached[i]}
    if not active:
        return
    pos_dict  = {i: tuple(positions[i]) for i in active}
    goal_dict = {i: tuple(goals[i])     for i in active}
    next_pos  = planner.plan_step(pos_dict, goal_dict)
    for i in active:
        if i in next_pos:
            nx, ny = next_pos[i]
            positions[i][0], positions[i][1] = nx, ny
            gx, gy = goals[i]
            if abs(nx - gx) <= 1 and abs(ny - gy) <= 1:
                reached[i] = True
        times[i] += 1


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

def visualize(
    num_agents: int = 24,
    grid_size: int = 64,
    intersection_half: int = 5,
    cell_size: int = 10,
    seed: int = 42,
    left_ratio: float = 0.3,
    right_ratio: float = 0.35,
    initial_method: str = "gsp",
):
    pygame.init()
    pygame.font.init()

    road = RoadNetwork2In1Out(grid_size=grid_size, intersection_half=intersection_half)
    win_w   = grid_size * cell_size
    panel_w = 280
    screen  = pygame.display.set_mode((win_w + panel_w, win_w))
    pygame.display.set_caption(
        f"2-in-1-out: SP vs Circular ({int(left_ratio*100)}% Left Turns)"
    )
    # Font hierarchy: title → section → body → hint
    _mono      = "Monaco"
    font_title  = pygame.font.SysFont(_mono, 22, bold=True)
    font_section= pygame.font.SysFont(_mono, 13, bold=True)
    font        = pygame.font.SysFont(_mono, 13)
    font_large  = pygame.font.SysFont(_mono, 15)
    font_small  = pygame.font.SysFont(_mono, 11)

    origins, destinations, turn_types = generate_agents(road, num_agents, seed, left_ratio, right_ratio)
    n = len(origins)

    method             = initial_method if initial_method in ("gsp", "circular", "pibt") else "gsp"
    paused             = True
    clock              = pygame.time.Clock()
    speed              = 8
    current_seed       = seed
    sim_step           = 0
    current_left_ratio = left_ratio

    # --- SP (rigid BFS) state ---
    gsp_positions  = [list(o) for o in origins]
    gsp_goals      = list(destinations)
    gsp_reached    = [False] * n
    gsp_times      = [0] * n
    gsp_paths      = {i: None for i in range(n)}
    gsp_consec_wait= [0] * n

    # --- Circular state ---
    circ_positions = [list(o) for o in origins]
    circ_goals     = list(destinations)
    circ_reached   = [False] * n
    circ_times     = [0] * n
    circ_exiting   = [False] * n

    # --- PIBT state ---
    pibt_planner   = _PIBTPlanner(road)
    pibt_positions = [list(o) for o in origins]
    pibt_goals     = list(destinations)
    pibt_reached   = [False] * n
    pibt_times     = [0] * n

    # --- Comparison record: final total time per method, None = not yet completed ---
    record = {"gsp": None, "circular": None, "pibt": None}

    def reset_simulation(new_seed, reset_record=True):
        nonlocal origins, destinations, turn_types, n, sim_step, record
        nonlocal gsp_positions, gsp_goals, gsp_reached, gsp_times, gsp_paths, gsp_consec_wait
        nonlocal circ_positions, circ_goals, circ_reached, circ_times, circ_exiting
        nonlocal pibt_positions, pibt_goals, pibt_reached, pibt_times
        origins, destinations, turn_types = generate_agents(
            road, num_agents, new_seed, left_ratio=current_left_ratio, right_ratio=right_ratio
        )
        n        = len(origins)
        sim_step = 0
        gsp_positions   = [list(o) for o in origins]
        gsp_goals       = list(destinations)
        gsp_reached     = [False] * n
        gsp_times       = [0] * n
        gsp_paths       = {i: None for i in range(n)}
        gsp_consec_wait = [0] * n
        circ_positions = [list(o) for o in origins]
        circ_goals     = list(destinations)
        circ_reached   = [False] * n
        circ_times     = [0] * n
        circ_exiting   = [False] * n
        pibt_positions = [list(o) for o in origins]
        pibt_goals     = list(destinations)
        pibt_reached   = [False] * n
        pibt_times     = [0] * n
        if reset_record:
            record = {"gsp": None, "circular": None, "pibt": None}
        pygame.display.set_caption(
            f"2-in-1-out: GSP vs Circular ({int(current_left_ratio*100)}% Left Turns)"
        )

    def cell_to_screen(x, y):
        return (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2)

    def draw_road():
        screen.fill(COLORS["background"])
        for x in range(grid_size):
            for y in range(grid_size):
                if road.is_on_road(x, y):
                    ltype = road.get_lane_type(x, y)
                    color = COLORS.get(ltype, COLORS["road"])
                    px, py = cell_to_screen(x, y)
                    pygame.draw.circle(screen, color, (px, py), cell_size // 2 - 1)
        cx, cy = cell_to_screen(road.center, road.center)
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 4)

    def draw_vehicles():
        if method == "gsp":
            pos, rch, gls = gsp_positions, gsp_reached, gsp_goals
        elif method == "circular":
            pos, rch, gls = circ_positions, circ_reached, circ_goals
        else:
            pos, rch, gls = pibt_positions, pibt_reached, pibt_goals
        for i in range(n):
            sx, sy = cell_to_screen(int(pos[i][0]), int(pos[i][1]))
            if turn_types[i] == 'left':
                color = COLORS["vehicle_left"]
            elif turn_types[i] == 'right':
                color = COLORS["vehicle_right"]
            else:
                color = COLORS["vehicle_through"]
            if rch[i]:
                pygame.draw.circle(screen, COLORS["reached"], (sx, sy), 4)
            else:
                pygame.draw.circle(screen, color, (sx, sy), 5)
                dx, dy = cell_to_screen(gls[i][0], gls[i][1])
                pygame.draw.circle(screen, color, (dx, dy), 2, 1)

    def _divider(y_pos):
        """Draw a subtle horizontal rule across the panel."""
        pygame.draw.line(screen, (48, 52, 64),
                         (win_w + 10, y_pos), (win_w + panel_w - 10, y_pos))

    def draw_panel():
        pygame.draw.rect(screen, COLORS["panel"], (win_w, 0, panel_w, win_w))
        pygame.draw.line(screen, (55, 58, 70), (win_w, 0), (win_w, win_w), 2)

        C_DIM  = (120, 122, 135)   # dimmed secondary text
        C_MID  = (170, 172, 185)   # medium-weight text
        px = win_w + 16
        y  = 16

        # ── 1. SCENE LABEL (small, top) ────────────────────────────
        screen.blit(font_small.render("2-IN-1-OUT INTERSECTION", True, C_DIM), (px, y)); y += 20

        _divider(y); y += 12

        # ── 2. ACTIVE METHOD (hero element) ────────────────────────
        # Method colours chosen to be distinct from status (green/amber) and vehicle (red/blue)
        if method == "gsp":
            m_name, m_sub = "SP", "shortest path"
            mcolor = (55, 195, 210)    # cyan-teal
        elif method == "circular":
            m_name, m_sub = "Circular", "roundabout"
            mcolor = (70, 185, 145)    # seafoam / mid-teal
        else:
            m_name, m_sub = "PIBT", "priority"
            mcolor = (165, 135, 255)   # soft purple

        screen.blit(font_title.render(m_name, True, mcolor), (px, y)); y += 30
        screen.blit(font.render(m_sub, True, mcolor), (px, y)); y += 20

        # Switch-method hint — two short lines so they fit the panel
        screen.blit(font_small.render("switch:  1=SP  2=Circular", True, C_DIM), (px, y)); y += 14
        screen.blit(font_small.render("         3=PIBT", True, C_DIM), (px, y)); y += 20

        _divider(y); y += 12

        # ── 3. RUN STATUS ──────────────────────────────────────────
        # Status colours: green (run) and cool blue (pause) — neither matches a method colour
        if paused:
            status_text  = "PAUSED"
            status_hint  = "press SPACE to run"
            status_color = (100, 155, 240)   # steel blue
        else:
            status_text  = "RUNNING"
            status_hint  = "press SPACE to pause"
            status_color = (80, 220, 110)    # green

        screen.blit(font_large.render(status_text, True, status_color), (px, y)); y += 22
        screen.blit(font_small.render(status_hint, True, C_DIM), (px, y)); y += 20

        _divider(y); y += 12

        # ── 4. KEY METRIC: TOTAL TIME ──────────────────────────────
        if method == "gsp":
            completed  = sum(gsp_reached)
            total_time = sum(gsp_times)
        elif method == "circular":
            completed  = sum(circ_reached)
            total_time = sum(circ_times)
        else:
            completed  = sum(pibt_reached)
            total_time = sum(pibt_times)

        all_done = (completed == n)

        # Total time — live during run, highlighted gold when all done
        if all_done:
            tt_color   = (255, 215, 60)
            num_font   = font_title
            box_bg     = (46, 40, 10)
            box_border = (110, 90, 20)
        else:
            tt_color   = C_MID
            num_font   = font_title
            box_bg     = (38, 40, 48)
            box_border = (55, 58, 70)

        label_surf = font.render("Total time", True, tt_color)
        tt_surf    = num_font.render(f"{total_time}", True, tt_color)
        box_w = max(tt_surf.get_width(), label_surf.get_width()) + 20
        box_h = label_surf.get_height() + tt_surf.get_height() + 16
        box_rect = pygame.Rect(px - 6, y - 4, box_w, box_h)
        pygame.draw.rect(screen, box_bg,     box_rect, border_radius=5)
        pygame.draw.rect(screen, box_border, box_rect, 1, border_radius=5)
        screen.blit(label_surf, (px, y)); y += label_surf.get_height() + 4
        screen.blit(tt_surf,    (px, y)); y += tt_surf.get_height() + 10

        # Completed / step (smaller, secondary)
        done_color = (255, 215, 60) if all_done else C_MID   # gold when done, not green
        screen.blit(font.render(f"Completed: {completed}/{n}", True, done_color), (px, y)); y += 18
        screen.blit(font.render(f"Step: {sim_step}", True, C_DIM), (px, y)); y += 18

        _divider(y); y += 12

        # ── 5. LEFT-TURN RATIO (interactive, secondary) ────────────
        left_pct = int(current_left_ratio * 100)
        screen.blit(font_section.render(f"Left turns: {left_pct}%", True, COLORS["vehicle_left"]), (px, y)); y += 20
        screen.blit(font_small.render("UP / DN  to adjust (+-10%)", True, C_DIM), (px, y)); y += 16
        screen.blit(font_small.render(f"# agents: {n}   R = new seed", True, C_DIM), (px, y)); y += 20

        _divider(y); y += 12

        # ── 6. VEHICLE LEGEND ──────────────────────────────────────
        screen.blit(font_small.render("Vehicle colors:", True, C_DIM), (px, y)); y += 15
        for dot_color, label in [
            (COLORS["vehicle_left"],    "Left turn"),
            (COLORS["vehicle_through"], "Straight"),
            (COLORS["vehicle_right"],   "Right turn"),
            (COLORS["reached"],         "Completed"),
        ]:
            pygame.draw.circle(screen, dot_color, (px + 5, y + 5), 4)
            screen.blit(font_small.render(f"  {label}", True, C_DIM), (px + 10, y)); y += 14

        _divider(y); y += 10

        # ── 7. METHOD COMPARISON ───────────────────────────────────
        METHOD_COLORS = {
            "gsp":      (55,  195, 210),
            "circular": ( 70, 185, 145),
            "pibt":     (165, 135, 255),
        }
        METHOD_LABELS = {"gsp": "SP", "circular": "Circular", "pibt": "PIBT"}

        screen.blit(font_section.render("Comparison", True, C_MID), (px, y)); y += 16

        # Determine best (lowest) among completed runs
        done_vals = {k: v for k, v in record.items() if v is not None}
        best_key  = min(done_vals, key=done_vals.get) if done_vals else None

        for key in ("gsp", "circular", "pibt"):
            mc   = METHOD_COLORS[key]
            lbl  = METHOD_LABELS[key]
            val  = record[key]
            is_best = (key == best_key and len(done_vals) > 1)

            # Dim the method label; bright for the number
            name_surf = font_small.render(f"{lbl:<9}", True, mc)
            if val is None:
                val_surf  = font_small.render("—", True, C_DIM)
                star_surf = None
            else:
                val_color = (255, 215, 60) if is_best else mc
                val_surf  = font.render(str(val), True, val_color)
                star_surf = font_small.render(" best", True, (255, 215, 60)) if is_best else None

            screen.blit(name_surf, (px, y))
            screen.blit(val_surf,  (px + name_surf.get_width(), y))
            if star_surf:
                screen.blit(star_surf, (px + name_surf.get_width() + val_surf.get_width(), y))
            y += 16

        _divider(y); y += 10

        # ── 8. REMAINING CONTROLS (compact) ────────────────────────
        screen.blit(font_small.render("+/- = speed    Q = quit", True, C_DIM), (px, y))

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
                    method = "gsp";      reset_simulation(current_seed, reset_record=False)
                elif event.key == pygame.K_2:
                    method = "circular"; reset_simulation(current_seed, reset_record=False)
                elif event.key == pygame.K_3:
                    method = "pibt";     reset_simulation(current_seed, reset_record=False)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(60, speed + 2)
                elif event.key == pygame.K_MINUS:
                    speed = max(1,  speed - 2)
                elif event.key == pygame.K_UP:
                    current_left_ratio = min(1.0, current_left_ratio + 0.1)
                    reset_simulation(current_seed)
                elif event.key == pygame.K_DOWN:
                    current_left_ratio = max(0.0, current_left_ratio - 0.1)
                    reset_simulation(current_seed)
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        if not paused:
            if method == "gsp":
                if not all(gsp_reached):
                    step_sp_rigid(
                        road, gsp_positions, gsp_goals,
                        gsp_reached, gsp_times, gsp_paths, gsp_consec_wait,
                    )
                    sim_step += 1
                if all(gsp_reached) and record["gsp"] is None:
                    record["gsp"] = sum(gsp_times)
            elif method == "circular":
                if not all(circ_reached):
                    step_circular(
                        road, circ_positions, circ_goals,
                        circ_reached, circ_times, circ_exiting, set(),
                    )
                    sim_step += 1
                if all(circ_reached) and record["circular"] is None:
                    record["circular"] = sum(circ_times)
            else:
                if not all(pibt_reached):
                    step_pibt(road, pibt_planner, pibt_positions, pibt_goals,
                              pibt_reached, pibt_times)
                    sim_step += 1
                if all(pibt_reached) and record["pibt"] is None:
                    record["pibt"] = sum(pibt_times)

        draw_road()
        draw_vehicles()
        draw_panel()
        pygame.display.flip()
        clock.tick(speed)

    pygame.quit()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Visualize 2-in-1-out intersection (GSP vs Circular vs PIBT)")
    p.add_argument("--agents",            type=int,   default=24)
    p.add_argument("--grid",              type=int,   default=64)
    p.add_argument("--intersection-half", type=int,   default=5)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--left-ratio",        type=float, default=0.3,
                   help="Fraction of vehicles making left turns (0.0–1.0)")
    p.add_argument("--right-ratio",       type=float, default=0.35,
                   help="Fraction of vehicles making right turns (0.0–1.0)")
    p.add_argument("--method",            type=str,   default="gsp",
                   choices=["gsp", "circular", "pibt"])
    args = p.parse_args()
    visualize(
        num_agents=args.agents,
        grid_size=args.grid,
        intersection_half=args.intersection_half,
        seed=args.seed,
        left_ratio=args.left_ratio,
        right_ratio=args.right_ratio,
        initial_method=args.method,
    )
