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
from scenarios import obstacles_for

COLOR_LANE      = "#737a84"
COLOR_SPOT_OK   = "#7dd3fc"
COLOR_SPOT_ERR  = "#ff5252"
COLOR_CAR       = "#3b82f6"
COLOR_CAR_FRONT = "#1d4ed8"
COLOR_BG        = "#161a20"
COLOR_PANEL     = "#20262e"
COLOR_PANEL_2   = "#252c35"
COLOR_TEXT      = "#f3f6fb"
COLOR_MUTED     = "#a8b3c2"
COLOR_BORDER    = "#3a4552"
COLOR_ACCENT    = "#3b82f6"

# (display label, var attribute name, slider min, slider max, default)
SLIDER_DEFS = [
    ("Lane Width (m)",  "var_lane_w",   3.5, 5.5,  4.4),
    ("Spot Length (m)", "var_spot_len", 5.0, 6.0,  5.5),
    ("Spot Width (m)",  "var_spot_w",   2.0, 3.0,  2.5),
    ("Car Length (m)",  "var_car_len",  3.5, 5.0,  4.2),
    ("Car Width (m)",   "var_car_w",    1.6, 2.2,  1.8),
]


class SettingsWindow:
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
        self.var_obstacle = tk.StringVar(value=pc.obstacle_scenario if pc else "none")
        self.var_planner  = tk.StringVar(value=pc.planner      if pc else "single")

        for *_, attr, _min, _max, _def in SLIDER_DEFS:
            getattr(self, attr).trace_add("write", self._on_change)
        self.var_type.trace_add("write", self._on_change)
        self.var_obstacle.trace_add("write", self._on_change)
        self.var_planner.trace_add("write", self._on_change)

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = self.root

        # Allow column 1 (canvas side) and row 1 (main content) to expand
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        tk.Label(root, text="Smart Parking Simulator",
                 font=("Arial", 15, "bold"),
                 bg=COLOR_BG, fg=COLOR_TEXT).grid(
            row=0, column=0, columnspan=2, pady=(12, 8))

        # ---- Left panel: sliders (fixed width) ----
        left = tk.Frame(root, bg=COLOR_PANEL, padx=14, pady=8,
                        highlightthickness=1, highlightbackground=COLOR_BORDER)
        left.grid(row=1, column=0, sticky="nw", padx=(14, 8))

        for i, (label, attr, from_, to, _) in enumerate(SLIDER_DEFS):
            var = getattr(self, attr)
            tk.Label(left, text=label, bg=COLOR_PANEL, fg=COLOR_TEXT,
                     font=("Arial", 11), anchor="w").grid(
                row=i * 2, column=0, sticky="w", pady=(10, 0))
            row_frame = tk.Frame(left, bg=COLOR_PANEL)
            row_frame.grid(row=i * 2 + 1, column=0, sticky="ew")
            tk.Scale(row_frame, variable=var, from_=from_, to=to,
                     resolution=0.1, orient=tk.HORIZONTAL,
                     length=260, showvalue=False,
                     bg=COLOR_PANEL, fg=COLOR_TEXT,
                     highlightthickness=0,
                     troughcolor="#384250",
                     activebackground=COLOR_ACCENT).pack(side="left")
            tk.Label(row_frame, textvariable=var, width=5,
                     bg=COLOR_PANEL, fg=COLOR_MUTED,
                     font=("Arial", 11)).pack(
                side="left", padx=(6, 0))

        n = len(SLIDER_DEFS)
        tk.Label(left, text="Parking Type", bg=COLOR_PANEL, fg=COLOR_TEXT,
                 font=("Arial", 11)).grid(
            row=n * 2, column=0, sticky="w", pady=(14, 2))
        type_frame = tk.Frame(left, bg=COLOR_PANEL)
        type_frame.grid(row=n * 2 + 1, column=0, sticky="w")
        tk.Radiobutton(type_frame, text="Reverse into Spot",
                       variable=self.var_type, value="perpendicular",
                       bg=COLOR_PANEL, fg=COLOR_TEXT,
                       activebackground=COLOR_PANEL,
                       activeforeground=COLOR_TEXT,
                       selectcolor=COLOR_PANEL_2,
                       font=("Arial", 11)).pack(
            side="left", padx=4)
        tk.Radiobutton(type_frame, text="Parallel Parking",
                       variable=self.var_type, value="parallel",
                       bg=COLOR_PANEL, fg=COLOR_TEXT,
                       activebackground=COLOR_PANEL,
                       activeforeground=COLOR_TEXT,
                       selectcolor=COLOR_PANEL_2,
                       font=("Arial", 11)).pack(
            side="left", padx=4)

        tk.Label(left, text="Scenario", bg=COLOR_PANEL, fg=COLOR_TEXT,
                 font=("Arial", 11)).grid(
            row=n * 2 + 2, column=0, sticky="w", pady=(14, 2))
        scenario_frame = tk.Frame(left, bg=COLOR_PANEL)
        scenario_frame.grid(row=n * 2 + 3, column=0, sticky="w")
        tk.Radiobutton(scenario_frame, text="Clear",
                       variable=self.var_obstacle, value="none",
                       bg=COLOR_PANEL, fg=COLOR_TEXT,
                       activebackground=COLOR_PANEL,
                       activeforeground=COLOR_TEXT,
                       selectcolor=COLOR_PANEL_2,
                       font=("Arial", 11)).pack(
            side="left", padx=4)
        tk.Radiobutton(scenario_frame, text="Obstacle",
                       variable=self.var_obstacle, value="entry_blocker",
                       bg=COLOR_PANEL, fg=COLOR_TEXT,
                       activebackground=COLOR_PANEL,
                       activeforeground=COLOR_TEXT,
                       selectcolor=COLOR_PANEL_2,
                       font=("Arial", 11)).pack(
            side="left", padx=4)
        tk.Radiobutton(scenario_frame, text="Tight",
                       variable=self.var_obstacle, value="tight_lane",
                       bg=COLOR_PANEL, fg=COLOR_TEXT,
                       activebackground=COLOR_PANEL,
                       activeforeground=COLOR_TEXT,
                       selectcolor=COLOR_PANEL_2,
                       font=("Arial", 11)).pack(
            side="left", padx=4)
        tk.Radiobutton(scenario_frame, text="Parked Cars",
                       variable=self.var_obstacle, value="parked_cars",
                       bg=COLOR_PANEL, fg=COLOR_TEXT,
                       activebackground=COLOR_PANEL,
                       activeforeground=COLOR_TEXT,
                       selectcolor=COLOR_PANEL_2,
                       font=("Arial", 11)).pack(
            side="left", padx=4)

        tk.Label(left, text="Planner", bg=COLOR_PANEL, fg=COLOR_TEXT,
                 font=("Arial", 11)).grid(
            row=n * 2 + 4, column=0, sticky="w", pady=(14, 2))
        planner_frame = tk.Frame(left, bg=COLOR_PANEL)
        planner_frame.grid(row=n * 2 + 5, column=0, sticky="w")
        tk.Radiobutton(planner_frame, text="Single-step MPC",
                       variable=self.var_planner, value="single",
                       bg=COLOR_PANEL, fg=COLOR_TEXT,
                       activebackground=COLOR_PANEL,
                       activeforeground=COLOR_TEXT,
                       selectcolor=COLOR_PANEL_2,
                       font=("Arial", 11)).pack(
            side="left", padx=4)
        tk.Radiobutton(planner_frame, text="Multi-step MPC",
                       variable=self.var_planner, value="multi",
                       bg=COLOR_PANEL, fg=COLOR_TEXT,
                       activebackground=COLOR_PANEL,
                       activeforeground=COLOR_TEXT,
                       selectcolor=COLOR_PANEL_2,
                       font=("Arial", 11)).pack(
            side="left", padx=4)
        tk.Radiobutton(planner_frame, text="Hybrid A*",
                       variable=self.var_planner, value="hybrid_astar",
                       bg=COLOR_PANEL, fg=COLOR_TEXT,
                       activebackground=COLOR_PANEL,
                       activeforeground=COLOR_TEXT,
                       selectcolor=COLOR_PANEL_2,
                       font=("Arial", 11)).pack(
            side="left", padx=4)
        tk.Radiobutton(planner_frame, text="Q-learning (RL)",
                       variable=self.var_planner, value="qlearn",
                       bg=COLOR_PANEL, fg=COLOR_TEXT,
                       activebackground=COLOR_PANEL,
                       activeforeground=COLOR_TEXT,
                       selectcolor=COLOR_PANEL_2,
                       font=("Arial", 11)).pack(
            side="left", padx=4)
        tk.Radiobutton(planner_frame, text="Hierarchical RL",
                       variable=self.var_planner, value="hrl",
                       bg=COLOR_PANEL, fg=COLOR_TEXT,
                       activebackground=COLOR_PANEL,
                       activeforeground=COLOR_TEXT,
                       selectcolor=COLOR_PANEL_2,
                       font=("Arial", 11)).pack(
            side="left", padx=4)
        planner_frame2 = tk.Frame(left, bg=COLOR_PANEL)
        planner_frame2.grid(row=n * 2 + 6, column=0, sticky="w")
        tk.Radiobutton(planner_frame2, text="DQN (RL)",
                       variable=self.var_planner, value="dqn",
                       bg=COLOR_PANEL, fg=COLOR_TEXT,
                       activebackground=COLOR_PANEL,
                       activeforeground=COLOR_TEXT,
                       selectcolor=COLOR_PANEL_2,
                       font=("Arial", 11)).pack(
            side="left", padx=4)
        tk.Radiobutton(planner_frame2, text="RRT*",
                       variable=self.var_planner, value="rrt_star",
                       bg=COLOR_PANEL, fg=COLOR_TEXT,
                       activebackground=COLOR_PANEL,
                       activeforeground=COLOR_TEXT,
                       selectcolor=COLOR_PANEL_2,
                       font=("Arial", 11)).pack(
            side="left", padx=4)

        # ---- Right panel: expandable canvas ----
        right = tk.Frame(root, bg=COLOR_PANEL, padx=10, pady=8,
                         highlightthickness=1, highlightbackground=COLOR_BORDER)
        right.grid(row=1, column=1, sticky="nsew", padx=(0, 14))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)   # canvas row expands

        tk.Label(right, text="Live Preview", font=("Arial", 11, "bold"),
                 bg=COLOR_PANEL, fg=COLOR_TEXT).grid(row=0, column=0, sticky="w")

        self.canvas = tk.Canvas(right, bg="#11151b",
                                highlightthickness=1,
                                highlightbackground=COLOR_BORDER)
        self.canvas.grid(row=1, column=0, sticky="nsew",
                         padx=4, pady=(4, 2))
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.status_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.status_var, fg="#e53935",
                 bg=COLOR_PANEL, font=("Arial", 9)).grid(
            row=2, column=0, sticky="w")

        # ---- Start button: full width at the bottom ----
        tk.Button(root, text="Start Simulation", command=self._on_confirm,
                  font=("Arial", 13, "bold"), bg=COLOR_ACCENT, fg="white",
                  activebackground="#2563eb", activeforeground="white",
                  relief="flat", padx=16, pady=8).grid(
            row=2, column=0, columnspan=2, pady=(16, 14))

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _on_change(self, *_):
        self._update_preview()

    def _on_canvas_resize(self, event):
        self._update_preview()

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
            obstacle_scenario=self.var_obstacle.get(),
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
                      text="Spot", fill=spot_color, font=("Arial", 10))

        # Static obstacles / occupied cells
        for obs in obstacles_for(lot):
            ox1, oy1 = w2c(obs.x, obs.top)
            ox2, oy2 = w2c(obs.right, obs.y)
            c.create_rectangle(ox1, oy1, ox2, oy2,
                               fill="#dc2626", outline="#fecaca",
                               width=1)

        # Car body (blue rectangle)
        corners = lot.car_corners(lot.car_start_pose)
        pts = [w2c(wx, wy) for wx, wy in corners]
        flat = [coord for pt in pts for coord in pt]
        c.create_polygon(flat, fill=COLOR_CAR, outline="#dbeafe", width=1)
        # Front edge (darker) — corners[1] and corners[2]
        c.create_line(*pts[1], *pts[2], fill=COLOR_CAR_FRONT, width=3)

        # Lane width annotation (left side)
        lx, _ = w2c(r.x, 0)
        _, ly_bot = w2c(0, r.y)
        _, ly_top = w2c(0, r.top)
        c.create_line(lx - 8, ly_bot, lx - 8, ly_top,
                      fill=COLOR_MUTED, width=1, arrow=tk.BOTH,
                      arrowshape=(5, 7, 3))
        c.create_text(lx - 20, (ly_bot + ly_top) / 2,
                      text=f"{lot.pc.lane_width:.1f}m",
                      fill=COLOR_MUTED, font=("Arial", 9), angle=90)

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
        return ParkingConfig(
            lane_width=round(self.var_lane_w.get(), 1),
            spot_length=round(self.var_spot_len.get(), 1),
            spot_width=round(self.var_spot_w.get(), 1),
            parking_type=self.var_type.get(),
            obstacle_scenario=self.var_obstacle.get(),
            planner=self.var_planner.get(),
        )

    def _read_car_config(self) -> CarConfig:
        return CarConfig(
            length=round(self.var_car_len.get(), 1),
            width=round(self.var_car_w.get(), 1),
        )

    def run(self) -> Optional[Tuple[ParkingConfig, CarConfig]]:
        self.root.mainloop()
        return self.result
