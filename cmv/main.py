"""
Classical Mechanics Visualizer — Entry point
Usage:  python -m cmv.main
"""
from __future__ import annotations
import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure as MplFigure

from cmv.physics.engine import SimulationEngine
from cmv.physics import (
    # 2D motions
    PendulumParams, DoublePendulumParams, ProjectileParams,
    SHMParams, CircularParams, KeplerParams,
    PendulumODE, DoublePendulumODE, ProjectileODE,
    SHMODE, CircularODE, KeplerODE,
    # 3D motions
    SphericalPendulumParams, LorenzParams, MagneticParticleParams,
    SphericalPendulumODE, LorenzODE, MagneticParticleODE,
)
from cmv.viz.visualizer import Visualizer
from cmv.viz.ml_viewer import MLViewer
from cmv.viz.style import PAL


# ── Motion registry ────────────────────────────────────────────────────────
MOTIONS: dict[str, tuple] = {
    "단진자":           (PendulumODE,          PendulumParams),
    "이중진자":          (DoublePendulumODE,    DoublePendulumParams),
    "포물체 운동":       (ProjectileODE,        ProjectileParams),
    "SHM / 강제진동":   (SHMODE,               SHMParams),
    "원운동":           (CircularODE,          CircularParams),
    "케플러 궤도":       (KeplerODE,            KeplerParams),
    "구면진자 (3D)":    (SphericalPendulumODE, SphericalPendulumParams),
    "로렌츠 어트랙터":   (LorenzODE,            LorenzParams),
    "자기장 입자 (3D)": (MagneticParticleODE,  MagneticParticleParams),
}

# Slider definitions per motion type
SLIDERS: dict[str, list[tuple]] = {
    "pendulum": [
        ("L",      "길이 L [m]",      0.1,   5.0,   1.0),
        ("m",      "질량 m [kg]",     0.1,  10.0,   1.0),
        ("g",      "중력 g [m/s²]",   0.1,  20.0,   9.81),
        ("theta0", "초기각 θ₀ [rad]", -3.14, 3.14,  0.5),
        ("gamma",  "감쇠 γ",          0.0,   2.0,   0.0),
    ],
    "double_pendulum": [
        ("theta1_0", "θ₁₀ [rad]", -3.14, 3.14, 1.57),
        ("theta2_0", "θ₂₀ [rad]", -3.14, 3.14, 0.0),
        ("L1",       "L₁ [m]",     0.2,   3.0,  1.0),
        ("L2",       "L₂ [m]",     0.2,   3.0,  1.0),
        ("g",        "중력 g",       0.1,  20.0,  9.81),
    ],
    "projectile": [
        ("v0",        "초속 v₀ [m/s]", 1.0, 50.0, 20.0),
        ("angle_deg", "발사각 [°]",    1.0, 89.0, 45.0),
        ("k",         "공기저항 k",     0.0,  1.0,  0.0),
        ("g",         "중력 g",         0.1, 20.0,  9.81),
    ],
    "shm": [
        ("k",       "스프링 k [N/m]",   0.1, 50.0, 10.0),
        ("gamma",   "감쇠 γ",            0.0,  5.0,  0.0),
        ("F0",      "구동 진폭 F₀",      0.0,  5.0,  0.0),
        ("omega_d", "구동 ω_d [rad/s]",  0.1, 10.0,  3.0),
    ],
    "circular": [
        ("r",     "반경 r [m]",       0.1, 5.0, 1.0),
        ("omega", "각속도 ω [rad/s]",  0.1, 5.0, 1.0),
    ],
    "kepler": [
        ("ecc", "이심률 e",      0.0,  0.95, 0.0),
        ("r0",  "초기거리 r₀",   0.5,  5.0,  1.0),
        ("GM",  "GM [AU³/yr²]", 1.0, 100.0, 39.478),
    ],
    "spherical_pendulum": [
        ("L",       "길이 L [m]",        0.2, 3.0,  1.0),
        ("theta0",  "극각 θ₀ [rad]",     0.05, 3.0, 0.8),
        ("dphi0",   "방위 속도 dφ₀",    -5.0, 5.0,  1.5),
        ("dtheta0", "극 속도 dθ₀",      -3.0, 3.0,  0.0),
        ("g",       "중력 g [m/s²]",     0.1, 20.0, 9.81),
    ],
    "lorenz": [
        ("sigma", "σ (확산)",    1.0, 20.0, 10.0),
        ("rho",   "ρ (발산)",    1.0, 50.0, 28.0),
        ("beta",  "β (소산)",    0.1,  6.0,  2.667),
    ],
    "magnetic_particle": [
        ("Bz",  "자기장 Bz [T]",   0.1, 5.0,  1.0),
        ("vx0", "초기 vx₀ [m/s]", 0.1, 5.0,  1.0),
        ("vz0", "수직 vz₀ [m/s]", -3.0, 3.0, 0.5),
        ("Ez",  "전기장 Ez",      -2.0, 2.0,  0.0),
    ],
}

ML_MOTION_LABELS = [
    "단진자", "이중진자", "포물체 운동",
    "SHM / 강제진동", "원운동", "케플러 궤도",
    "구면진자 (3D)", "로렌츠 어트랙터", "자기장 입자 (3D)",
]

ML_LABEL_TO_KEY = {
    "단진자":           "pendulum",
    "이중진자":          "double_pendulum",
    "포물체 운동":       "projectile",
    "SHM / 강제진동":   "shm",
    "원운동":           "circular",
    "케플러 궤도":       "kepler",
    "구면진자 (3D)":    "spherical_pendulum",
    "로렌츠 어트랙터":   "lorenz",
    "자기장 입자 (3D)": "magnetic_particle",
}


# ── App ────────────────────────────────────────────────────────────────────
class CMVApp:
    def __init__(self):
        # ── Root window ────────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("Classical Mechanics Visualizer")
        self.root.geometry("1720x960")
        self.root.configure(bg=PAL.get("fig_bg", "#1E2128"))

        # ── Notebook style ─────────────────────────────────────────────────
        style = ttk.Style(self.root)
        style.theme_use("clam")
        bg   = PAL.get("fig_bg",   "#1E2128")
        tab  = PAL.get("panel_bg", "#2D3139")
        sel  = "#3B4252"
        fg   = PAL.get("text",     "#F8FAFC")
        sub  = PAL.get("subtext",  "#94A3B8")
        style.configure("TNotebook",
                         background=bg, borderwidth=0, tabmargins=0)
        style.configure("TNotebook.Tab",
                         background=tab, foreground=sub,
                         padding=[18, 6],
                         font=("Helvetica", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", sel)],
                  foreground=[("selected", fg)])
        style.configure("Dark.TFrame", background=bg)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        # ── Tab 1: Simulation ───────────────────────────────────────────────
        frame_sim = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(frame_sim, text="  시뮬레이션  ")

        fig_sim = MplFigure(figsize=(17, 9), facecolor=PAL["fig_bg"])
        canvas_sim = FigureCanvasTkAgg(fig_sim, master=frame_sim)
        canvas_sim.get_tk_widget().pack(fill="both", expand=True)

        ode_cls, param_cls = MOTIONS["단진자"]
        params = param_cls()
        ode    = ode_cls(params)
        engine = SimulationEngine(ode, params)

        self.viz = Visualizer(engine, fig=fig_sim)
        self.viz.set_motion_radio(
            ax_rect=[0.005, 0.05, 0.098, 0.90],
            labels=list(MOTIONS.keys()),
            callback=self._on_motion_changed,
        )
        self._current_sim_label = "단진자"
        self._load_motion("단진자")

        # ── Tab 2: ML Viewer ────────────────────────────────────────────────
        frame_ml = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(frame_ml, text="  ML 예측  ")

        fig_ml = MplFigure(figsize=(17, 9), facecolor=PAL["fig_bg"])
        canvas_ml = FigureCanvasTkAgg(fig_ml, master=frame_ml)
        canvas_ml.get_tk_widget().pack(fill="both", expand=True)

        self.ml_viz = MLViewer(fig=fig_ml)
        self.ml_viz.set_motion_radio(
            ax_rect=[0.005, 0.05, 0.098, 0.90],
            labels=ML_MOTION_LABELS,
            callback=self._on_ml_motion_changed,
        )
        # Set default motion key for the first label
        self.ml_viz.set_current_motion(
            ML_LABEL_TO_KEY[ML_MOTION_LABELS[0]]
        )

        # ── Tab switch handler ──────────────────────────────────────────────
        self._active_tab = 0
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    # ── Simulation tab ─────────────────────────────────────────────────────
    def _load_motion(self, label: str):
        ode_cls, param_cls = MOTIONS[label]
        params = param_cls()
        ode    = ode_cls(params)
        engine = SimulationEngine(ode, params)
        self.viz.engine = engine
        self.viz.setup_sliders(SLIDERS.get(ode.name, []))
        result = engine.run()
        self.viz.build_animation(result)
        self.viz.fig.canvas.draw_idle()

    def _on_motion_changed(self, label: str):
        self._current_sim_label = label
        self._load_motion(label)

    # ── ML viewer tab ──────────────────────────────────────────────────────
    def _on_ml_motion_changed(self, label: str):
        motion_key = ML_LABEL_TO_KEY.get(label, "pendulum")
        self.ml_viz.set_current_motion(motion_key)

    # ── Tab switching: pause inactive animation to save CPU ────────────────
    def _on_tab_changed(self, _event=None):
        import time as _t
        import numpy as _np

        tab = self.notebook.index("current")
        if tab == self._active_tab:
            return

        viz = self.viz
        ml  = self.ml_viz

        if tab == 1:
            # ── Entering ML tab ─────────────────────────────────────────
            # 1) Freeze simulation
            if viz.anim is not None and not viz._paused:
                try:
                    viz.anim.event_source.stop()
                except Exception:
                    pass
                viz._stop_energy_timer()

            # 2) Resume ML animation (if one exists and isn't user-paused)
            if ml.anim is not None and not ml._paused:
                # Re-anchor wall-clock start so playback continues from
                # where it left off rather than jumping forward.
                if ml._pred_times is not None and ml._anim_start_time is not None:
                    fi  = min(ml._current_fi, len(ml._pred_times) - 1)
                    t_r = (float(ml._pred_times[fi])
                           - float(ml._pred_times[0]))
                    ml._anim_start_time = _t.perf_counter() - t_r
                try:
                    ml.anim.event_source.start()
                except Exception:
                    pass
                ml._start_info_timer()

        elif tab == 0:
            # ── Entering simulation tab ─────────────────────────────────
            # 1) Freeze ML animation
            if ml.anim is not None and not ml._paused:
                try:
                    ml.anim.event_source.stop()
                except Exception:
                    pass
                ml._stop_info_timer()

            # 2) Resume simulation (if it isn't user-paused)
            if viz.anim is not None and not viz._paused:
                if viz.result is not None and viz._anim_start_time is not None:
                    fi  = min(viz._current_fi,
                              len(viz.result.timestamps) - 1)
                    t_r = (float(viz.result.timestamps[fi])
                           - float(viz.result.timestamps[0]))
                    viz._anim_start_time = _t.perf_counter() - t_r
                try:
                    viz.anim.event_source.start()
                except Exception:
                    pass
                viz._start_energy_timer()

        self._active_tab = tab

    def run(self):
        self.root.mainloop()


def main():
    app = CMVApp()
    app.run()


if __name__ == "__main__":
    main()
