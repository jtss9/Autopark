"""
Phase 2: pygame simulation window.
Animates the computed parking trajectory.
Controls: SPACE pause/resume  |  R restart  |  ESC/Q quit
"""
import math
import os

import pygame

from config import CarConfig, ParkingConfig
from parking_lot import ParkingLot
from scenarios import obstacles_for
from trajectory import TrajectoryResult, plan_trajectory

WIN_W, WIN_H = 960, 640

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
C_WARN       = (255,  80,  80)
C_OBSTACLE   = (190,  45,  45)
C_OBS_LINE   = (255, 190, 190)

# Waypoints advanced per frame (controls animation speed)
SPEED = 2


class Simulation:
    def __init__(self, parking_config: ParkingConfig, car_config: CarConfig):
        self.pc  = parking_config
        self.cc  = car_config
        self.lot = ParkingLot(parking_config, car_config)
        self._compute_scale()
        requested_planner = os.environ.get(
            "AUTOPARK_PLANNER",
            "hybrid_astar" if parking_config.planner == "hybrid_astar" else "baseline",
        )
        self.planner_name = self._effective_planner_name(requested_planner)
        self.result: TrajectoryResult = plan_trajectory(
            parking_config,
            car_config,
            planner=requested_planner,
        )
        self.animation_speed = 1 if parking_config.parking_type == "parallel" else SPEED

    def _effective_planner_name(self, requested_planner: str) -> str:
        if requested_planner == "hybrid_astar":
            return "hybrid_astar"
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
        # Full planned path (thin, semi-transparent look via colour)
        all_pts = [self.w2s(w.x, w.y) for w in wps]
        pygame.draw.lines(surf, C_PATH_PLAN, False, all_pts, 1)
        # Travelled portion (thicker green)
        done_pts = all_pts[:step + 1]
        if len(done_pts) > 1:
            pygame.draw.lines(surf, C_PATH_DONE, False, done_pts, 3)

    def _draw_scene(self, surf, step: int):
        surf.fill(C_BG)

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

        for obs in obstacles_for(self.lot):
            ox1, oy1 = self.w2s(obs.x, obs.top)
            ox2, oy2 = self.w2s(obs.right, obs.y)
            obs_r = pygame.Rect(min(ox1, ox2), min(oy1, oy2),
                                abs(ox2 - ox1), abs(oy2 - oy1))
            pygame.draw.rect(surf, C_OBSTACLE, obs_r)
            pygame.draw.rect(surf, C_OBS_LINE, obs_r, 1)

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

        # Determine current phase name
        phase_name = ""
        for i, ps in enumerate(self.result.phase_starts):
            if step >= ps:
                phase_name = self.result.phase_names[i]

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
        if not self.result.feasible:
            lines.append((font_b, f"[!] {self.result.message}", C_WARN))
            if collision_frame:
                font_big = pygame.font.SysFont("Arial", 36, bold=True)
                label = font_big.render("COLLISION", True, C_WARN)
                surf.blit(label, label.get_rect(
                    center=(WIN_W // 2, WIN_H // 2)))
        else:
            lines.append((font, f"Phase: {phase_name}", C_TEXT))
            lines.append((font, f"Step: {min(step, total)} / {total}", C_DIM))
            if self.result.metrics:
                metrics = self.result.metrics
                lines.append((
                    font,
                    f"Path: {metrics['path_length_m']:.1f} m  |  "
                    f"Plan: {metrics['planning_time_s']:.2f}s  |  "
                    f"Final err: {metrics['final_pos_error_m']:.2f} m",
                    C_DIM,
                ))

        lines += [
            (font, "SPACE  pause / resume", (100, 100, 100)),
            (font, "R      restart",        (100, 100, 100)),
            (font, "S      back to settings",(100, 100, 100)),
            (font, "ESC    quit",           (100, 100, 100)),
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

        step   = 0
        paused = False
        total  = max(len(self.result.waypoints) - 1, 0)

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
                        step   = 0
                        paused = False
                    elif event.key == pygame.K_s:
                        go_back = True
                        running = False

            if not paused and step < total:
                step = min(step + self.animation_speed, total)

            self._draw_scene(screen, step)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        return go_back
