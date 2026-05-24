"""
Phase 2: pygame window — currently shows the initial static scene.
Press ESC or Q to close.
"""
import math
import os
import sys

import pygame

from config import CarConfig, ParkingConfig
from parking_lot import ParkingLot

# Window dimensions
WIN_W, WIN_H = 960, 640

# Colors (RGB)
C_BG         = (30,  30,  30)
C_LANE       = (110, 110, 110)
C_LANE_LINE  = (255, 214,  0)
C_SPOT_LINE  = (255, 255, 255)
C_SPOT_FILL  = (55,  55,  55)
C_CAR_BODY   = (25, 118, 210)
C_CAR_FRONT  = (13,  71, 161)
C_CAR_WHEEL  = (200, 200, 200)
C_TEXT       = (240, 240, 240)
C_DIM        = (160, 160, 160)
C_ARROW      = (255, 235,  59)


class Simulation:
    def __init__(self, parking_config: ParkingConfig, car_config: CarConfig):
        self.pc = parking_config
        self.cc = car_config
        self.lot = ParkingLot(parking_config, car_config)
        self._compute_scale()

    # ------------------------------------------------------------------
    # Scale / coordinate transform
    # ------------------------------------------------------------------
    def _compute_scale(self):
        margin = 0.14
        self.scale = min(
            WIN_W * (1 - margin * 2) / self.lot.scene_w,
            WIN_H * (1 - margin * 2) / self.lot.scene_h,
        )
        # Origin: world (0,0) maps to lower-left of the scene, centered
        scene_px_w = self.lot.scene_w * self.scale
        scene_px_h = self.lot.scene_h * self.scale
        self.ox = (WIN_W - scene_px_w) / 2
        self.oy = WIN_H - (WIN_H - scene_px_h) / 2

    def w2s(self, wx: float, wy: float):
        """World (m) → screen (px). Y axis is flipped."""
        return (
            int(self.ox + wx * self.scale),
            int(self.oy - wy * self.scale),
        )

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_rect_world(self, surface, rect_w, color, fill=True, width=2,
                          dash=False):
        """Draw a world-space Rect. fill=True fills it, else outlines."""
        x1, y1 = self.w2s(rect_w.x, rect_w.top)
        x2, y2 = self.w2s(rect_w.right, rect_w.y)
        r = pygame.Rect(min(x1, x2), min(y1, y2),
                        abs(x2 - x1), abs(y2 - y1))
        if fill:
            pygame.draw.rect(surface, color, r)
        if dash:
            self._draw_dashed_rect(surface, color, r, width)
        elif not fill:
            pygame.draw.rect(surface, color, r, width)

    def _draw_dashed_rect(self, surface, color, rect, width=2, dash=12, gap=8):
        """Draw a dashed rectangle outline."""
        x, y, w, h = rect
        edges = [
            ((x, y), (x + w, y)),
            ((x + w, y), (x + w, y + h)),
            ((x + w, y + h), (x, y + h)),
            ((x, y + h), (x, y)),
        ]
        for (x1, y1), (x2, y2) in edges:
            length = math.hypot(x2 - x1, y2 - y1)
            if length == 0:
                continue
            dx = (x2 - x1) / length
            dy = (y2 - y1) / length
            pos = 0
            drawing = True
            while pos < length:
                seg = min(dash if drawing else gap, length - pos)
                if drawing:
                    sx1 = x1 + dx * pos
                    sy1 = y1 + dy * pos
                    sx2 = x1 + dx * (pos + seg)
                    sy2 = y1 + dy * (pos + seg)
                    pygame.draw.line(surface, color,
                                     (int(sx1), int(sy1)),
                                     (int(sx2), int(sy2)), width)
                pos += seg
                drawing = not drawing

    def _draw_car(self, surface, pose):
        corners = self.lot.car_corners(pose)
        pts = [self.w2s(wx, wy) for wx, wy in corners]
        pygame.draw.polygon(surface, C_CAR_BODY, pts)
        pygame.draw.polygon(surface, (200, 200, 200), pts, 1)
        # Front edge (darker) — corners[1] and corners[2]
        pygame.draw.line(surface, C_CAR_FRONT, pts[1], pts[2], 4)

    def _draw_dashed_line_world(self, surface, wx1, wy1, wx2, wy2,
                                 color, width=2, dash=10, gap=8):
        sx1, sy1 = self.w2s(wx1, wy1)
        sx2, sy2 = self.w2s(wx2, wy2)
        length = math.hypot(sx2 - sx1, sy2 - sy1)
        if length == 0:
            return
        dx = (sx2 - sx1) / length
        dy = (sy2 - sy1) / length
        pos = 0
        drawing = True
        while pos < length:
            seg = min(dash if drawing else gap, length - pos)
            if drawing:
                ax = sx1 + dx * pos
                ay = sy1 + dy * pos
                bx = sx1 + dx * (pos + seg)
                by = sy1 + dy * (pos + seg)
                pygame.draw.line(surface, color,
                                 (int(ax), int(ay)), (int(bx), int(by)),
                                 width)
            pos += seg
            drawing = not drawing

    # ------------------------------------------------------------------
    # Main scene draw
    # ------------------------------------------------------------------
    def _draw_scene(self, surface):
        lot = self.lot
        surface.fill(C_BG)

        # Lane
        self._draw_rect_world(surface, lot.lane_rect, C_LANE, fill=True)


        # Parking spot (dashed outline)
        sr = lot.spot_rect
        sx1, sy1 = self.w2s(sr.x, sr.top)
        sx2, sy2 = self.w2s(sr.right, sr.y)
        spot_screen = pygame.Rect(min(sx1, sx2), min(sy1, sy2),
                                   abs(sx2 - sx1), abs(sy2 - sy1))
        pygame.draw.rect(surface, C_SPOT_FILL, spot_screen)
        self._draw_dashed_rect(surface, C_SPOT_LINE, spot_screen)

        # Spot label
        font_small = pygame.font.SysFont("Arial", 16)
        label = font_small.render("Spot", True, C_SPOT_LINE)
        surface.blit(label, label.get_rect(center=spot_screen.center))

        # Car at start pose
        self._draw_car(surface, lot.car_start_pose)

        # HUD
        self._draw_hud(surface)

    def _draw_hud(self, surface):
        font = pygame.font.SysFont("Arial", 18)
        font_b = pygame.font.SysFont("Arial", 20, bold=True)

        type_label = ("Reverse into Spot" if self.pc.parking_type == "perpendicular"
                      else "Parallel Parking")
        lines = [
            (font_b, f"Parking type: {type_label}", C_TEXT),
            (font, f"Lane width:  {self.pc.lane_width:.1f} m", C_DIM),
            (font, f"Spot:  {self.pc.spot_length:.1f} x {self.pc.spot_width:.1f} m", C_DIM),
            (font, f"Car:   {self.cc.length:.1f} x {self.cc.width:.1f} m", C_DIM),
            (font, f"Min turn radius: {self.cc.min_turn_radius:.2f} m", C_DIM),
            (font, "ESC to quit", (100, 100, 100)),
        ]
        y = 14
        for f, text, color in lines:
            surf = f.render(text, True, color)
            surface.blit(surf, (14, y))
            y += surf.get_height() + 4

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        pygame.init()
        pygame.display.set_caption("Smart Parking Simulator")
        screen = pygame.display.set_mode((WIN_W, WIN_H))
        clock = pygame.time.Clock()


        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False

            self._draw_scene(screen)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
