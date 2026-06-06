"""
Phase 2: pygame simulation window.
Animates the computed parking trajectory.
Controls: SPACE pause/resume  |  R restart  |  G grid overlay  |  T track overlay
          ESC/Q quit          |  S back to settings
"""
import math
import os

import pygame

from config import CarConfig, ParkingConfig
from hybrid_astar import OccupancyGrid
from parking_lot import ParkingLot, Rect
from scenarios import obstacles_for
from trajectory import TrajectoryResult, plan_trajectory

WIN_W, WIN_H = 1080, 720

# Colours
C_BG         = (30,  30,  30)
C_LANE       = (110, 110, 110)
C_SPOT_LINE  = (255, 255, 255)
C_SPOT_FILL  = (55,  55,  55)
C_CAR_BODY   = (25, 118, 210)
C_CAR_FRONT  = (13,  71, 161)
C_TEXT       = (240, 240, 240)
C_DIM        = (160, 160, 160)
C_PATH_PLAN  = (80,  160, 255)   # full planned path
C_PATH_DONE  = (80,  220, 120)   # already-travelled portion
C_PATH_EXEC  = (250, 200,  80)   # executed (closed-loop) path
C_GRID_FREE  = (70,   85, 110)
C_GRID_BLOCK = (200,  90,  90)
C_WARN       = (255,  80,  80)
C_OBSTACLE   = (190,  45,  45)
C_OBS_LINE   = (255, 190, 190)

# Animation speed: waypoints advanced per frame (float; supports sub-1 for slow-mo)
SPEED         = 2.0
SPEED_MIN     = 0.05
SPEED_MAX     = 10.0
SPEED_STEP_UP = 2.0   # multiply on each ↑ press
SPEED_STEP_DN = 0.5   # multiply on each ↓ press


class Simulation:
    def __init__(self, parking_config: ParkingConfig, car_config: CarConfig):
        self.pc  = parking_config
        self.cc  = car_config
        self.lot = ParkingLot(parking_config, car_config)
        self._compute_scale()
        # Resolve the requested planner: env var > Settings UI choice. We
        # normalise the legacy "baseline" alias to the concrete MPC variant
        # so plan_trajectory dispatches deterministically and the HUD label
        # matches what actually ran. AUTOPARK_PLANNER=baseline means
        # "single-step MPC" (the original baseline); to fall back to the
        # Settings UI choice, leave AUTOPARK_PLANNER unset.
        env_planner = os.environ.get("AUTOPARK_PLANNER")
        if env_planner:
            requested_planner = env_planner
            if requested_planner == "baseline":
                requested_planner = "single"
        else:
            requested_planner = parking_config.planner or "single"
        self.planner_name = self._effective_planner_name(requested_planner)

        track_env = os.environ.get("AUTOPARK_TRACK", "")
        self.track_enabled = track_env.lower() in ("1", "true", "yes", "on")

        self.result: TrajectoryResult = plan_trajectory(
            parking_config,
            car_config,
            planner=requested_planner,
            track=self.track_enabled,
        )
        self.animation_speed: float = 1.0 if parking_config.parking_type == "parallel" else SPEED

        # UI toggles
        self.show_grid = False
        self.show_executed = bool(self.result.executed_waypoints)

        # Build a lightweight occupancy grid for visualization regardless of
        # planner. Cache the obstacle list so per-frame drawing does not
        # re-run scenario dispatch + list allocation 60 times per second.
        # User-placed obstacle from the settings window: (x, y, w, h) or None.
        # The planner already accounts for it (via pc.obstacle in plan_hybrid_astar);
        # here we include it in the visualization grid so the G overlay matches.
        self._user_obstacle = self.pc.obstacle

        self._obstacles = obstacles_for(self.lot)
        viz_obstacles = list(self._obstacles)
        if self._user_obstacle is not None:
            x, y, w, h = self._user_obstacle
            viz_obstacles.append(Rect(x, y, w, h))
        self._viz_grid = OccupancyGrid(
            self.lot,
            resolution=0.5,
            obstacles=viz_obstacles,
        )

    def _effective_planner_name(self, requested_planner: str) -> str:
        if requested_planner == "hybrid_astar":
            return "hybrid_astar"
        if requested_planner == "qlearn":
            return "qlearn (RL)"
        if self.pc.parking_type == "parallel" or self.pc.obstacle_scenario != "none":
            return "hybrid_astar"
        return requested_planner

    # ------------------------------------------------------------------
    # Coordinate transform
    # ------------------------------------------------------------------
    def _compute_scale(self):
        margin = 0.14
        self.scale = min(
            WIN_W * (1 - margin * 2) / self.lot.scene_w,
            WIN_H * (1 - margin * 2) / self.lot.scene_h,
        )
        scene_px_w = self.lot.scene_w * self.scale
        scene_px_h = self.lot.scene_h * self.scale
        self.ox = (WIN_W - scene_px_w) / 2
        self.oy = WIN_H - (WIN_H - scene_px_h) / 2

    def w2s(self, wx: float, wy: float):
        return (int(self.ox + wx * self.scale),
                int(self.oy - wy * self.scale))

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _dashed_rect(self, surf, color, rect, lw=2, dash=12, gap=7):
        x, y, w, h = rect
        for (ax, ay), (bx, by) in [
            ((x, y),     (x+w, y)),
            ((x+w, y),   (x+w, y+h)),
            ((x+w, y+h), (x,   y+h)),
            ((x,   y+h), (x,   y)),
        ]:
            length = math.hypot(bx-ax, by-ay)
            if length == 0:
                continue
            dx, dy = (bx-ax)/length, (by-ay)/length
            pos, draw = 0.0, True
            while pos < length:
                seg = min(dash if draw else gap, length - pos)
                if draw:
                    pygame.draw.line(
                        surf, color,
                        (int(ax + dx*pos),       int(ay + dy*pos)),
                        (int(ax + dx*(pos+seg)),  int(ay + dy*(pos+seg))), lw)
                pos += seg
                draw = not draw

    def _draw_car(self, surf, pose, collision: bool = False):
        corners = self.lot.car_corners(pose)
        pts = [self.w2s(wx, wy) for wx, wy in corners]
        body_color  = C_WARN       if collision else C_CAR_BODY
        front_color = (180, 40, 40) if collision else C_CAR_FRONT
        pygame.draw.polygon(surf, body_color, pts)
        pygame.draw.polygon(surf, (200, 200, 200), pts, 1)
        pygame.draw.line(surf, front_color, pts[1], pts[2], 4)

    def _draw_path(self, surf, step: int):
        wps = self.result.waypoints
        if len(wps) < 2:
            return
        # Full planned path (thin)
        all_pts = [self.w2s(w.x, w.y) for w in wps]
        pygame.draw.lines(surf, C_PATH_PLAN, False, all_pts, 1)
        # Travelled portion (thicker green)
        done_pts = all_pts[:step + 1]
        if len(done_pts) > 1:
            pygame.draw.lines(surf, C_PATH_DONE, False, done_pts, 3)
        # Executed (closed-loop) trajectory overlay
        if self.show_executed and self.result.executed_waypoints:
            exec_pts = [
                self.w2s(w.x, w.y) for w in self.result.executed_waypoints
            ]
            if len(exec_pts) > 1:
                pygame.draw.lines(surf, C_PATH_EXEC, False, exec_pts, 2)

    def _draw_grid(self, surf):
        """Overlay the occupancy grid.

        Free cells are rendered once into a cached background surface (built
        on first toggle) and blitted thereafter. Blocked cells are drawn on
        top each frame so changes (if obstacles ever become dynamic) update
        live without rebuilding the cache.
        """
        if not self.show_grid:
            return
        g = self._viz_grid
        res = g.resolution
        if getattr(self, "_grid_cache", None) is None:
            cache = pygame.Surface((WIN_W, WIN_H), flags=pygame.SRCALPHA)
            for ix in range(g.width):
                for iy in range(g.height):
                    x = g.min_x + ix * res
                    y = g.min_y + iy * res
                    x1, y1 = self.w2s(x, y + res)
                    x2, y2 = self.w2s(x + res, y)
                    rect = pygame.Rect(min(x1, x2), min(y1, y2),
                                       abs(x2 - x1), abs(y2 - y1))
                    pygame.draw.rect(cache, C_GRID_FREE, rect, 1)
            self._grid_cache = cache
        surf.blit(self._grid_cache, (0, 0))
        for ix, iy in g.blocked:
            x = g.min_x + ix * res
            y = g.min_y + iy * res
            x1, y1 = self.w2s(x, y + res)
            x2, y2 = self.w2s(x + res, y)
            rect = pygame.Rect(min(x1, x2), min(y1, y2),
                               abs(x2 - x1), abs(y2 - y1))
            pygame.draw.rect(surf, C_GRID_BLOCK, rect, 1)

    def _draw_scene(self, surf, step: int):
        surf.fill(C_BG)

        # Occupancy-grid overlay (toggled by G)
        self._draw_grid(surf)

        # Lane
        r = self.lot.lane_rect
        lx1, ly1 = self.w2s(r.x,     r.top)
        lx2, ly2 = self.w2s(r.right, r.y)
        pygame.draw.rect(surf, C_LANE,
                         (min(lx1,lx2), min(ly1,ly2),
                          abs(lx2-lx1), abs(ly2-ly1)))

        # Parking spot
        sr = self.lot.spot_rect
        sx1, sy1 = self.w2s(sr.x,     sr.top)
        sx2, sy2 = self.w2s(sr.right, sr.y)
        spot_r = pygame.Rect(min(sx1,sx2), min(sy1,sy2),
                             abs(sx2-sx1), abs(sy2-sy1))
        pygame.draw.rect(surf, C_SPOT_FILL, spot_r)
        self._dashed_rect(surf, C_SPOT_LINE, spot_r)
        font_s = pygame.font.SysFont("Arial", 15)
        lbl = font_s.render("Spot", True, C_SPOT_LINE)
        surf.blit(lbl, lbl.get_rect(center=spot_r.center))

        for obs in self._obstacles:
            ox1, oy1 = self.w2s(obs.x, obs.top)
            ox2, oy2 = self.w2s(obs.right, obs.y)
            obs_r = pygame.Rect(min(ox1, ox2), min(oy1, oy2),
                                abs(ox2 - ox1), abs(oy2 - oy1))
            pygame.draw.rect(surf, C_OBSTACLE, obs_r)
            pygame.draw.rect(surf, C_OBS_LINE, obs_r, 1)

        # User-placed obstacle (settings window) — reference overlay only
        if self._user_obstacle is not None:
            ux, uy, uw, uh = self._user_obstacle
            px1, py1 = self.w2s(ux, uy + uh)
            px2, py2 = self.w2s(ux + uw, uy)
            u_r = pygame.Rect(min(px1, px2), min(py1, py2),
                              abs(px2 - px1), abs(py2 - py1))
            pygame.draw.rect(surf, C_OBSTACLE, u_r)
            pygame.draw.rect(surf, C_OBS_LINE, u_r, 2)

        # Planned path + travelled path
        self._draw_path(surf, step)

        # Car at current waypoint
        wps = self.result.waypoints
        if wps:
            wp = wps[min(step, len(wps) - 1)]
            pose = (wp.x, wp.y, wp.theta)
        else:
            pose = self.lot.car_start_pose
        collision_frame = (not self.result.feasible
                           and bool(wps)
                           and step >= len(wps) - 1)
        self._draw_car(surf, pose, collision=collision_frame)

        self._draw_hud(surf, step)

    def _draw_hud(self, surf, step: int):
        font   = pygame.font.SysFont("Arial", 18)
        font_b = pygame.font.SysFont("Arial", 20, bold=True)

        wps    = self.result.waypoints
        total  = max(len(wps) - 1, 1)

        # Determine current phase name. Guard against any planner that
        # returns mismatched-length phase_starts/phase_names so the per-frame
        # HUD draw cannot raise IndexError and kill the pygame loop.
        phase_name = ""
        names = self.result.phase_names
        for i, ps in enumerate(self.result.phase_starts):
            if step >= ps and i < len(names):
                phase_name = names[i]

        type_label = ("Reverse into Spot" if self.pc.parking_type == "perpendicular"
                      else "Parallel Parking")

        lines = [
            (font_b, f"Parking type: {type_label}",                 C_TEXT),
            (font,   f"Planner: {self.planner_name}",                C_DIM),
            (font,   f"Scenario: {self.pc.obstacle_scenario}",        C_DIM),
            (font,   f"Lane: {self.pc.lane_width:.1f} m  |  "
                     f"Spot: {self.pc.spot_length:.1f}x{self.pc.spot_width:.1f} m  |  "
                     f"Car: {self.cc.length:.1f}x{self.cc.width:.1f} m",  C_DIM),
            (font,   f"Min turn radius: {self.cc.min_turn_radius:.2f} m", C_DIM),
        ]

        collision_frame = (not self.result.feasible
                           and bool(wps)
                           and step >= total)
        metrics = self.result.metrics
        status = "SUCCESS" if self.result.feasible else "FAILED"
        status_color = C_PATH_DONE if self.result.feasible else C_WARN
        full_spot = metrics.get("fully_in_spot", False)

        lines.append((font_b, f"Status: {status}", status_color))
        lines.append((font, f"Phase: {phase_name}", C_TEXT))
        lines.append((font, f"Step: {min(step, total)} / {total}", C_DIM))
        if metrics:
            lines.append((
                font,
                f"Path: {metrics.get('path_length_m', 0.0):.1f} m  |  "
                f"Plan: {metrics.get('planning_time_s', 0.0):.2f}s  |  "
                f"Final err: {metrics.get('final_pos_error_m', 0.0):.2f} m",
                C_DIM,
            ))
            lines.append((
                font,
                f"Heading err: {metrics.get('final_heading_error_deg', 0.0):.1f} deg  |  "
                f"Fully in spot: {full_spot}",
                C_DIM,
            ))
            if metrics.get("used_analytic_shot"):
                lines.append((
                    font,
                    f"Reeds-Shepp shot: {metrics.get('rs_shot_successes', 0)} "
                    f"/ {metrics.get('rs_shot_attempts', 0)} attempts",
                    C_PATH_EXEC,
                ))
            if metrics.get("planner_kind") == "qlearn":
                lines.append((
                    font,
                    f"RL: trained {metrics.get('training_time_s', 0):.1f}s, "
                    f"{metrics.get('successful_episodes', 0)} success eps, "
                    f"{metrics.get('expanded_states', 0)} states",
                    C_PATH_EXEC,
                ))

        tm = self.result.tracking_metrics or {}
        if tm:
            lines.append((font_b, "Pure Pursuit (closed loop):", C_PATH_EXEC))
            lines.append((
                font,
                f"  CTE mean {tm.get('mean_cte_m', 0):.3f} m  |  "
                f"max {tm.get('max_cte_m', 0):.3f} m  |  "
                f"cusps {tm.get('cusps', 0)}",
                C_DIM,
            ))
            lines.append((
                font,
                f"  exec err {tm.get('exec_final_pos_error_m', 0):.3f} m  |  "
                f"in spot: {tm.get('exec_fully_in_spot')}",
                C_DIM,
            ))

        if not self.result.feasible:
            lines.append((font_b, f"[!] {self.result.message}", C_WARN))
            if collision_frame:
                font_big = pygame.font.SysFont("Arial", 36, bold=True)
                label = font_big.render("FAILED", True, C_WARN)
                surf.blit(label, label.get_rect(
                    center=(WIN_W // 2, WIN_H // 2)))

        lines += [
            (font, f"Speed: {self.animation_speed:.2g}x  (↑↓ to adjust)", C_DIM),
            (font, "SPACE  pause / resume",  (100, 100, 100)),
            (font, "R      restart",         (100, 100, 100)),
            (font, "↑ ↓    speed up / down", (100, 100, 100)),
            (font, "G      toggle grid",     (100, 100, 100)),
            (font, "T      toggle executed", (100, 100, 100)),
            (font, "S      back to settings",(100, 100, 100)),
            (font, "ESC    quit",            (100, 100, 100)),
        ]

        y = 12
        for f, text, color in lines:
            s = f.render(text, True, color)
            surf.blit(s, (14, y))
            y += s.get_height() + 3

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        pygame.init()
        pygame.display.set_caption("Smart Parking Simulator")
        screen = pygame.display.set_mode((WIN_W, WIN_H))
        clock  = pygame.time.Clock()

        step      = 0
        step_acc  = 0.0   # fractional accumulator for sub-1 speeds
        paused    = False
        total     = max(len(self.result.waypoints) - 1, 0)

        go_back = False
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        step     = 0
                        step_acc = 0.0
                        paused   = False
                    elif event.key == pygame.K_s:
                        go_back = True
                        running = False
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    elif event.key == pygame.K_t:
                        self.show_executed = not self.show_executed
                    elif event.key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
                        self.animation_speed = min(
                            self.animation_speed * SPEED_STEP_UP, SPEED_MAX)
                    elif event.key in (pygame.K_DOWN, pygame.K_MINUS):
                        self.animation_speed = max(
                            self.animation_speed * SPEED_STEP_DN, SPEED_MIN)

            if not paused and step < total:
                step_acc += self.animation_speed
                step      = min(int(step_acc), total)

            self._draw_scene(screen, step)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        return go_back
