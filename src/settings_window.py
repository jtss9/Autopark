"""
Phase 1: tkinter settings window with sliders and live canvas preview.
Returns (ParkingConfig, CarConfig) after the user clicks Start Simulation.
"""
import math
import tkinter as tk
from tkinter import messagebox
from typing import Optional, Tuple

from config import CarConfig, ParkingConfig
from parking_lot import ParkingLot

# Palette + typography matched to the Truck-Simulator settings screen:
# Consolas monospace, dark-grey tone, light-blue section headers,
# yellow value readouts, red Reset / green Start buttons.
FONT_NAME       = "Consolas"

COLOR_BG        = "#1c1c1c"   # screen background        (28,28,28)
COLOR_PANEL     = "#22222a"   # cards / preview panel    (34,34,42)
COLOR_PANEL_2   = "#34343c"   # radio select indicator
COLOR_TEXT      = "#cdcdcd"   # body / label text        (205,205,205)
COLOR_TITLE     = "#ffffff"   # title                    (255,255,255)
COLOR_SECTION   = "#82beff"   # section headers          (130,190,255)
COLOR_VALUE     = "#ffde4b"   # slider value readouts    (255,222,75)
COLOR_MUTED     = "#7d7d7d"   # hints / dim text         (125,125,125)
COLOR_BORDER    = "#484848"   # panel borders            (72,72,72)
COLOR_TROUGH    = "#414141"   # slider trough            (65,65,65)
COLOR_ACCENT    = "#3a7dd7"   # slider active / accent   (58,125,215)
COLOR_BTN_GO    = "#265f3a"   # Start button             (38,95,58)
COLOR_BTN_GO_H  = "#379458"   # Start button hover       (55,148,88)

COLOR_LANE      = "#414141"   # canvas lane fill
COLOR_SPOT_OK   = "#82beff"   # canvas spot outline (ok)
COLOR_SPOT_ERR  = "#ff5252"   # canvas spot outline (error)
COLOR_CAR       = "#376ec8"   # canvas car body          (55,110,200)
COLOR_CAR_FRONT = "#96bcff"   # canvas car front edge    (150,188,255)

# (display label, var attribute name, slider min, slider max, default)
# Defaults match config.py (the scene the RL models are strongest on:
# 100% fixed-start success for both parking types); the lane slider must
# reach 6.0 m or that scene is unreachable from the UI.
SLIDER_DEFS = [
    ("Lane Width (m)",  "var_lane_w",   3.5, 6.0,  6.0),
    ("Spot Length (m)", "var_spot_len", 5.0, 6.0,  6.0),
    ("Spot Width (m)",  "var_spot_w",   2.0, 3.0,  2.5),
    ("Car Length (m)",  "var_car_len",  3.5, 5.0,  4.5),
    ("Car Width (m)",   "var_car_w",    1.6, 2.2,  1.8),
]


class SettingsWindow:
    OBS_W = 0.9   # fixed user-obstacle size (m) — not resizable
    OBS_H = 0.9

    def __init__(self,
                 initial_parking: Optional[ParkingConfig] = None,
                 initial_car:     Optional[CarConfig]     = None):
        self.result: Optional[Tuple[ParkingConfig, CarConfig]] = None
        self._initial_parking = initial_parking
        self._initial_car     = initial_car

        self.root = tk.Tk()
        self.root.title("Smart Parking Simulator - Settings")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(True, True)
        self.root.geometry("1100x680")
        self.root.bind("<Return>", lambda _: self._on_confirm())
        self.root.minsize(800, 520)

        self._init_vars()
        self._build_ui()
        # Defer first draw until canvas is actually rendered and has a size
        self.root.after(50, self._update_preview)

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    def _init_vars(self):
        pc = self._initial_parking
        cc = self._initial_car

        self.var_lane_w   = tk.DoubleVar(value=pc.lane_width   if pc else 4.4)
        self.var_spot_len = tk.DoubleVar(value=pc.spot_length  if pc else 5.5)
        self.var_spot_w   = tk.DoubleVar(value=pc.spot_width   if pc else 2.5)
        self.var_car_len  = tk.DoubleVar(value=cc.length       if cc else 4.2)
        self.var_car_w    = tk.DoubleVar(value=cc.width        if cc else 1.8)
        self.var_type     = tk.StringVar(value=pc.parking_type if pc else "perpendicular")
        self.var_planner  = tk.StringVar(value=pc.planner      if pc else "single")
        self.var_show_obstacle = tk.BooleanVar(
            value=bool(pc.obstacle) if pc else False)

        # Fixed-size user obstacle; centre stored in world coords (lazy default).
        self._obs_center = None
        self._drag_obs   = False
        if pc and pc.obstacle:
            ox, oy, ow, oh = pc.obstacle
            self.OBS_W, self.OBS_H = ow, oh
            self._obs_center = (ox + ow / 2, oy + oh / 2)

        for *_, attr, _min, _max, _def in SLIDER_DEFS:
            getattr(self, attr).trace_add("write", self._on_change)
        self.var_type.trace_add("write", self._on_change)
        self.var_planner.trace_add("write", self._on_change)
        self.var_show_obstacle.trace_add("write", self._on_change)

    # ------------------------------------------------------------------
    # Widget factories
    # ------------------------------------------------------------------
    def _seg_radio(self, parent, text, var, value):
        """Truck-style segmented toggle button: flat, dark-grey when idle,
        accent-blue when selected (indicatoron=0 renders it as a button)."""
        return tk.Radiobutton(
            parent, text=text, variable=var, value=value,
            indicatoron=0,
            font=(FONT_NAME, 10, "bold"),
            bg=COLOR_PANEL_2, fg=COLOR_TITLE,
            selectcolor=COLOR_ACCENT,
            activebackground=COLOR_ACCENT, activeforeground=COLOR_TITLE,
            relief="flat", bd=0,
            highlightthickness=1, highlightbackground=COLOR_BORDER,
            padx=10, pady=5, cursor="hand2",
        )

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = self.root

        # Allow column 1 (canvas side) and row 1 (main content) to expand
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        tk.Label(root, text="Smart Parking Simulator  —  Setup",
                 font=(FONT_NAME, 18, "bold"),
                 bg=COLOR_BG, fg=COLOR_TITLE).grid(
            row=0, column=0, columnspan=2, pady=(12, 8))

        # ---- Left panel: sliders (fixed width) ----
        left = tk.Frame(root, bg=COLOR_PANEL, padx=14, pady=8,
                        highlightthickness=1, highlightbackground=COLOR_BORDER)
        left.grid(row=1, column=0, sticky="nw", padx=(14, 8))

        for i, (label, attr, from_, to, _) in enumerate(SLIDER_DEFS):
            var = getattr(self, attr)
            tk.Label(left, text=label, bg=COLOR_PANEL, fg=COLOR_TEXT,
                     font=(FONT_NAME, 11), anchor="w").grid(
                row=i * 2, column=0, sticky="w", pady=(10, 0))
            row_frame = tk.Frame(left, bg=COLOR_PANEL)
            row_frame.grid(row=i * 2 + 1, column=0, sticky="ew")
            tk.Scale(row_frame, variable=var, from_=from_, to=to,
                     resolution=0.1, orient=tk.HORIZONTAL,
                     length=260, showvalue=False,
                     bg=COLOR_PANEL_2, fg=COLOR_TEXT,
                     highlightthickness=0,
                     width=12, sliderlength=22,
                     borderwidth=1, sliderrelief="flat",
                     troughcolor=COLOR_TROUGH,
                     activebackground=COLOR_ACCENT).pack(side="left")
            tk.Label(row_frame, textvariable=var, width=5,
                     bg=COLOR_PANEL, fg=COLOR_VALUE,
                     font=(FONT_NAME, 11)).pack(
                side="left", padx=(6, 0))

        n = len(SLIDER_DEFS)
        tk.Label(left, text="▸ Parking Type", bg=COLOR_PANEL, fg=COLOR_SECTION,
                 font=(FONT_NAME, 12, "bold")).grid(
            row=n * 2, column=0, sticky="w", pady=(14, 2))
        type_frame = tk.Frame(left, bg=COLOR_PANEL)
        type_frame.grid(row=n * 2 + 1, column=0, sticky="w")
        for text, value in (("Reverse into Spot", "perpendicular"),
                            ("Parallel Parking",  "parallel")):
            self._seg_radio(type_frame, text, self.var_type, value).pack(
                side="left", padx=3)

        tk.Label(left, text="▸ Planner", bg=COLOR_PANEL, fg=COLOR_SECTION,
                 font=(FONT_NAME, 12, "bold")).grid(
            row=n * 2 + 2, column=0, sticky="w", pady=(14, 2))
        planner_frame = tk.Frame(left, bg=COLOR_PANEL)
        planner_frame.grid(row=n * 2 + 3, column=0, sticky="w")
        for text, value in (("Single-step MPC", "single"),
                            ("Hybrid A*",        "hybrid_astar"),
                            ("SAC (RL)",         "sac"),
                            ("Q-learning (RL)",  "qlearn")):
            self._seg_radio(planner_frame, text, self.var_planner, value).pack(
                side="left", padx=3)

        # Obstacle toggle — only shown for Hybrid A*. When on, a fixed-size
        # obstacle appears in the preview and can be dragged (off the car).
        self.obstacle_check = tk.Checkbutton(
            left, text="Add obstacle  (drag it in the preview)",
            variable=self.var_show_obstacle,
            bg=COLOR_PANEL, fg=COLOR_TEXT,
            activebackground=COLOR_PANEL, activeforeground=COLOR_TEXT,
            selectcolor=COLOR_PANEL_2, font=(FONT_NAME, 11),
            anchor="w", cursor="hand2")
        self.obstacle_check.grid(row=n * 2 + 4, column=0, sticky="w",
                                 pady=(14, 2))
        self._sync_obstacle_controls()

        # ---- Right panel: expandable canvas ----
        right = tk.Frame(root, bg=COLOR_PANEL, padx=10, pady=8,
                         highlightthickness=1, highlightbackground=COLOR_BORDER)
        right.grid(row=1, column=1, sticky="nsew", padx=(0, 14))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)   # canvas row expands

        tk.Label(right, text="Live Preview", font=(FONT_NAME, 12, "bold"),
                 bg=COLOR_PANEL, fg=COLOR_SECTION).grid(row=0, column=0, sticky="w")

        self.canvas = tk.Canvas(right, bg="#11151b",
                                highlightthickness=1,
                                highlightbackground=COLOR_BORDER)
        self.canvas.grid(row=1, column=0, sticky="nsew",
                         padx=4, pady=(4, 2))
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>",   self._on_canvas_press)
        self.canvas.bind("<B1-Motion>",       self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        self.status_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.status_var, fg="#e53935",
                 bg=COLOR_PANEL, font=(FONT_NAME, 9)).grid(
            row=2, column=0, sticky="w")

        # ---- Start button: full width at the bottom ----
        tk.Button(root, text="Start Simulation ▶", command=self._on_confirm,
                  font=(FONT_NAME, 13, "bold"), bg=COLOR_BTN_GO, fg=COLOR_TITLE,
                  activebackground=COLOR_BTN_GO_H, activeforeground=COLOR_TITLE,
                  relief="flat", padx=16, pady=8).grid(
            row=2, column=0, columnspan=2, pady=(16, 14))

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _on_change(self, *_):
        self._sync_obstacle_controls()
        self._update_preview()

    def _on_canvas_resize(self, event):
        self._update_preview()

    # ------------------------------------------------------------------
    # Obstacle placement (planners that consume the user obstacle)
    # ------------------------------------------------------------------
    OBSTACLE_PLANNERS = ("hybrid_astar", "sac")

    def _obstacle_enabled(self) -> bool:
        return (self.var_planner.get() in self.OBSTACLE_PLANNERS
                and self.var_show_obstacle.get())

    def _sync_obstacle_controls(self):
        """Show the obstacle checkbox only for planners that handle it."""
        if not hasattr(self, "obstacle_check"):
            return
        if self.var_planner.get() in self.OBSTACLE_PLANNERS:
            self.obstacle_check.grid()
        else:
            self.obstacle_check.grid_remove()

    def _current_lot(self) -> ParkingLot:
        """Build a ParkingLot from the current sliders (without the obstacle,
        to avoid recursion with _read_parking_config)."""
        pc = ParkingConfig(
            lane_width=round(self.var_lane_w.get(), 1),
            spot_length=round(self.var_spot_len.get(), 1),
            spot_width=round(self.var_spot_w.get(), 1),
            parking_type=self.var_type.get(),
        )
        return ParkingLot(pc, self._read_car_config())

    def _car_aabb(self, lot: ParkingLot):
        corners = lot.car_corners(lot.car_start_pose)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)

    def _clamp_to_bounds(self, cx, cy, lot: ParkingLot):
        hw, hh = self.OBS_W / 2, self.OBS_H / 2
        lane, spot = lot.lane_rect, lot.spot_rect
        ymin = min(lane.y, spot.y)
        ymax = max(lane.top, spot.top)
        cx = min(max(cx, lane.x + hw), lane.right - hw)
        cy = min(max(cy, ymin + hh), ymax - hh)
        return cx, cy

    def _obstacle_overlaps_car(self, cx, cy, lot: ParkingLot) -> bool:
        hw, hh = self.OBS_W / 2, self.OBS_H / 2
        ax0, ay0, ax1, ay1 = self._car_aabb(lot)
        return (ax0 - hw < cx < ax1 + hw) and (ay0 - hh < cy < ay1 + hh)

    def _obstacle_overlaps_spot(self, cx, cy, lot: ParkingLot) -> bool:
        """Obstacle on the spot would invalidate the goal pose — forbid it."""
        hw, hh = self.OBS_W / 2, self.OBS_H / 2
        spot = lot.spot_rect
        return (spot.x - hw < cx < spot.right + hw) and \
               (spot.y - hh < cy < spot.top + hh)

    def _obstacle_blocked(self, cx, cy, lot: ParkingLot) -> bool:
        return (self._obstacle_overlaps_car(cx, cy, lot)
                or self._obstacle_overlaps_spot(cx, cy, lot))

    def _default_obs_center(self, lot: ParkingLot):
        """A starting spot for the obstacle: right of the parking bay, clear
        of the car. Nudged right until it no longer overlaps the car."""
        lane, spot = lot.lane_rect, lot.spot_rect
        cx = min(spot.right + 1.0, lane.right - self.OBS_W / 2 - 0.3)
        cy = lane.y + lane.h / 2
        cx, cy = self._clamp_to_bounds(cx, cy, lot)
        tries = 0
        while self._obstacle_blocked(cx, cy, lot) and tries < 120:
            cx, cy = self._clamp_to_bounds(cx + 0.2, cy, lot)
            tries += 1
        return cx, cy

    def _c2w(self, cx, cy):
        """Canvas pixel → world coords (inverse of the draw transform)."""
        return ((cx - self._ox) / self._scale,
                (self._oy - cy) / self._scale)

    def _on_canvas_press(self, ev):
        if not self._obstacle_enabled() or self._obs_center is None:
            return
        if not hasattr(self, "_scale"):
            return
        wx, wy = self._c2w(ev.x, ev.y)
        cx, cy = self._obs_center
        if abs(wx - cx) <= self.OBS_W / 2 and abs(wy - cy) <= self.OBS_H / 2:
            self._drag_obs = True
            self._grab_dx = cx - wx
            self._grab_dy = cy - wy

    def _on_canvas_drag(self, ev):
        if not self._drag_obs or not self._obstacle_enabled():
            return
        lot = self._current_lot()
        wx, wy = self._c2w(ev.x, ev.y)
        cx, cy = self._clamp_to_bounds(wx + self._grab_dx,
                                       wy + self._grab_dy, lot)
        # Reject positions on the car (start) or the spot (goal) — both would
        # make planning impossible.
        if not self._obstacle_blocked(cx, cy, lot):
            self._obs_center = (cx, cy)
            self._update_preview()

    def _on_canvas_release(self, ev):
        self._drag_obs = False

    def _update_preview(self):
        pc = self._read_parking_config()
        cc = self._read_car_config()
        lot = ParkingLot(pc, cc)
        fits = lot.car_fits()

        self.canvas.delete("all")
        self._draw_preview(lot, fits)
        self.status_var.set(
            "[!] Car does not fit in the spot — adjust dimensions."
            if not fits else "")

    def _max_scene_dims(self, parking_type: str):
        """Scene dimensions at maximum slider values — used for a stable scale."""
        slider_map = {d[1]: d[3] for d in SLIDER_DEFS}  # attr → max value
        pc = ParkingConfig(
            lane_width=slider_map["var_lane_w"],
            spot_length=slider_map["var_spot_len"],
            spot_width=slider_map["var_spot_w"],
            parking_type=parking_type,
        )
        cc = CarConfig(
            length=slider_map["var_car_len"],
            width=slider_map["var_car_w"],
        )
        lot = ParkingLot(pc, cc)
        return lot.scene_w, lot.scene_h

    def _draw_preview(self, lot: ParkingLot, fits: bool):
        c = self.canvas
        cw = c.winfo_width()
        ch = c.winfo_height()
        if cw < 10 or ch < 10:   # widget not yet rendered
            return

        # Use max-possible scene dims for a stable scale that never changes
        # as sliders move — only the drawn objects change size.
        margin = 0.10
        max_w, max_h = self._max_scene_dims(lot.pc.parking_type)
        scale = min(
            cw * (1 - margin * 2) / max_w,
            ch * (1 - margin * 2) / max_h,
        )
        # Centre the actual (current) scene in the canvas
        scene_px_w = lot.scene_w * scale
        scene_px_h = lot.scene_h * scale
        ox = (cw - scene_px_w) / 2
        oy = ch - (ch - scene_px_h) / 2

        # Remember the transform so mouse handlers can map canvas → world.
        self._scale, self._ox, self._oy = scale, ox, oy

        def w2c(wx, wy):
            return ox + wx * scale, oy - wy * scale

        # Lane fill
        r = lot.lane_rect
        x1, y1 = w2c(r.x, r.top)
        x2, y2 = w2c(r.right, r.y)
        c.create_rectangle(x1, y1, x2, y2, fill=COLOR_LANE, outline="")

        # Parking spot
        sr = lot.spot_rect
        sx1, sy1 = w2c(sr.x, sr.top)
        sx2, sy2 = w2c(sr.right, sr.y)
        spot_color = COLOR_SPOT_OK if fits else COLOR_SPOT_ERR
        c.create_rectangle(sx1, sy1, sx2, sy2,
                            fill="#2b323c", outline=spot_color,
                            width=2, dash=(6, 4))
        c.create_text((sx1 + sx2) / 2, (sy1 + sy2) / 2,
                      text="Spot", fill=spot_color, font=(FONT_NAME, 10))

        # User-placed obstacle (Hybrid A* only) — draggable, fixed size,
        # kept off the car and inside the scene.
        if self._obstacle_enabled():
            if self._obs_center is None:
                self._obs_center = self._default_obs_center(lot)
            else:
                cx, cy = self._clamp_to_bounds(*self._obs_center, lot)
                if self._obstacle_blocked(cx, cy, lot):
                    cx, cy = self._default_obs_center(lot)
                self._obs_center = (cx, cy)
            cx, cy = self._obs_center
            hw, hh = self.OBS_W / 2, self.OBS_H / 2
            ox1, oy1 = w2c(cx - hw, cy + hh)
            ox2, oy2 = w2c(cx + hw, cy - hh)
            c.create_rectangle(ox1, oy1, ox2, oy2,
                               fill="#dc2626", outline="#fecaca", width=2)
            c.create_text((ox1 + ox2) / 2, (oy1 + oy2) / 2,
                          text="Obs", fill="#fecaca", font=(FONT_NAME, 9))

        # Car body (blue rectangle)
        corners = lot.car_corners(lot.car_start_pose)
        pts = [w2c(wx, wy) for wx, wy in corners]
        flat = [coord for pt in pts for coord in pt]
        c.create_polygon(flat, fill=COLOR_CAR, outline="#dbeafe", width=1)
        # Front edge (darker) — corners[1] and corners[2]
        c.create_line(*pts[1], *pts[2], fill=COLOR_CAR_FRONT, width=3)

        # ── Dimension annotations ─────────────────────────────────────────
        def _hdim(xa, xb, y_screen, text, below=True):
            """Horizontal double-arrow between screen-x xa..xb at y_screen."""
            c.create_line(xa, y_screen, xb, y_screen, fill=COLOR_MUTED,
                          width=1, arrow=tk.BOTH, arrowshape=(5, 7, 3))
            c.create_text((xa + xb) / 2, y_screen + (9 if below else -9),
                          text=text, fill=COLOR_MUTED, font=(FONT_NAME, 9))

        def _vdim(x_screen, ya, yb, text, left=True):
            """Vertical double-arrow between screen-y ya..yb at x_screen."""
            c.create_line(x_screen, ya, x_screen, yb, fill=COLOR_MUTED,
                          width=1, arrow=tk.BOTH, arrowshape=(5, 7, 3))
            c.create_text(x_screen + (-14 if left else 14), (ya + yb) / 2,
                          text=text, fill=COLOR_MUTED, font=(FONT_NAME, 9),
                          angle=90)

        pc, cc = lot.pc, lot.cc

        # Lane width — vertical arrow at the lane's left edge
        lx, _ = w2c(r.x, 0)
        _, ly_bot = w2c(0, r.y)
        _, ly_top = w2c(0, r.top)
        _vdim(lx - 8, ly_bot, ly_top, f"{pc.lane_width:.1f}m", left=True)

        # Spot dimensions — orientation depends on parking type
        if pc.parking_type == "perpendicular":
            # spot width = horizontal (top edge), length = vertical (right edge)
            _hdim(sx1, sx2, sy1 - 10, f"W={pc.spot_width:.1f}m", below=False)
            _vdim(sx2 + 12, sy1, sy2, f"L={pc.spot_length:.1f}m", left=False)
        else:
            # parallel: spot length = horizontal (bottom edge), width = vertical
            _hdim(sx1, sx2, sy2 + 10, f"L={pc.spot_length:.1f}m", below=True)
            _vdim(sx2 + 12, sy1, sy2, f"W={pc.spot_width:.1f}m", left=False)

        # Car dimensions (start pose, heading +x → axis-aligned)
        # pts = [rear-left, front-left, front-right, rear-right]
        _hdim(pts[3][0], pts[2][0], pts[3][1] + 11,
              f"L={cc.length:.1f}m", below=True)
        _vdim(pts[0][0] - 12, pts[0][1], pts[3][1],
              f"W={cc.width:.1f}m", left=True)

    # ------------------------------------------------------------------
    # Confirm
    # ------------------------------------------------------------------
    def _on_confirm(self):
        pc = self._read_parking_config()
        cc = self._read_car_config()
        if not ParkingLot(pc, cc).car_fits():
            messagebox.showerror(
                "Invalid Dimensions",
                "The car does not fit in the parking spot.\n"
                "Please adjust the car or spot dimensions.")
            return
        self.result = (pc, cc)
        self.root.quit()
        self.root.destroy()

    def _read_parking_config(self) -> ParkingConfig:
        obstacle = None
        if self._obstacle_enabled() and self._obs_center is not None:
            cx, cy = self._obs_center
            obstacle = (cx - self.OBS_W / 2, cy - self.OBS_H / 2,
                        self.OBS_W, self.OBS_H)
        return ParkingConfig(
            lane_width=round(self.var_lane_w.get(), 1),
            spot_length=round(self.var_spot_len.get(), 1),
            spot_width=round(self.var_spot_w.get(), 1),
            parking_type=self.var_type.get(),
            obstacle_scenario="none",
            planner=self.var_planner.get(),
            obstacle=obstacle,
        )

    def _read_car_config(self) -> CarConfig:
        return CarConfig(
            length=round(self.var_car_len.get(), 1),
            width=round(self.var_car_w.get(), 1),
        )

    def run(self) -> Optional[Tuple[ParkingConfig, CarConfig]]:
        self.root.mainloop()
        return self.result
