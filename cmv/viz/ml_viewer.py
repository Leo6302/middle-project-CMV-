"""ML Prediction Viewer — mirrors main Visualizer layout, supports 2D and 3D."""
from __future__ import annotations
import itertools
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.lines as mlines
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp

from cmv.viz.style import apply_style, PAL


# ── Layout (identical coordinates to visualizer.py) ────────────────────────
_MOT_RECT  = [0.005, 0.05, 0.098, 0.90]
_MAIN_RECT = [0.115, 0.28, 0.490, 0.67]
_INFO_RECT = [0.620, 0.28, 0.370, 0.64]

_SLD_ZONE  = dict(x0=0.115, y_top=0.25, w=0.490)

_BTNS = [
    ("초기화",    [0.622, 0.178, 0.178, 0.082]),
    ("저장",      [0.812, 0.178, 0.178, 0.082]),
    ("모델 로드", [0.622, 0.086, 0.178, 0.082]),
    ("모델 정보", [0.812, 0.086, 0.178, 0.082]),
    ("폴더 열기", [0.622, 0.020, 0.178, 0.058]),
    ("일시정지",  [0.812, 0.020, 0.178, 0.058]),
]

TRAIL_LEN_2D = 120
TRAIL_LEN_3D = 200

_STATUS_COLOR = {"info": PAL["subtext"], "ok": "#16A34A",
                 "warn": "#D97706",      "error": "#DC2626"}
_STATUS_BG    = {"info": "#F8FAFC",      "ok": "#F0FDF4",
                 "warn": "#FFFBEB",      "error": "#FEF2F2"}

_3D_MOTIONS = {"spherical_pendulum", "lorenz", "magnetic_particle"}

_MOTION_TITLES = {
    "pendulum":           "단진자",
    "double_pendulum":    "이중진자",
    "projectile":         "포물체 운동",
    "shm":                "단순조화진동",
    "circular":           "등속 원운동",
    "kepler":             "케플러 궤도",
    "spherical_pendulum": "구면진자 (3D)",
    "lorenz":             "로렌츠 어트랙터 (3D)",
    "magnetic_particle":  "자기장 입자 (3D)",
}

# Slider defs per group — ASCII labels only (no Unicode subscript)
_SLD_DEFS_2D = [
    ("x0",    "x0 [m]",    -5.0,  5.0,  0.0),
    ("y0",    "y0 [m]",    -5.0,  5.0,  1.0),
    ("vx0",   "vx0 [m/s]", -10.0, 10.0, 0.0),
    ("vy0",   "vy0 [m/s]", -10.0, 10.0, 0.0),
    ("t_end", "t end [s]",  1.0,  30.0, 10.0),
]

_SLD_DEFS_3D = [
    ("x0",    "x0 [m]",    -5.0,  5.0,  0.0),
    ("y0",    "y0 [m]",    -5.0,  5.0,  0.0),
    ("z0",    "z0 [m]",    -5.0,  5.0,  0.0),
    ("vx0",   "vx0 [m/s]", -10.0, 10.0, 0.0),
    ("vy0",   "vy0 [m/s]", -10.0, 10.0, 0.0),
    ("vz0",   "vz0 [m/s]", -10.0, 10.0, 0.0),
    ("t_end", "t end [s]",  1.0,  50.0, 20.0),
]

_SLD_DEFS_LORENZ = [
    ("x0",    "x0",        -20.0, 20.0, 1.0),
    ("y0",    "y0",        -20.0, 20.0, 0.0),
    ("z0",    "z0",          0.0, 50.0, 0.0),
    ("t_end", "t end [s]",   1.0, 60.0, 30.0),
]

_DEFAULT_IC: dict[str, dict] = {
    "pendulum":           dict(x0=0.95, y0=-0.31, vx0=0.00, vy0=0.00, t_end=10.0),
    "double_pendulum":    dict(x0=0.00, y0=-1.00, vx0=0.80, vy0=0.00, t_end=10.0),
    "projectile":         dict(x0=0.00, y0=0.00,  vx0=14.1, vy0=14.1, t_end=5.0),
    "shm":                dict(x0=1.00, y0=0.00,  vx0=0.00, vy0=0.00, t_end=10.0),
    "circular":           dict(x0=1.00, y0=0.00,  vx0=0.00, vy0=1.00, t_end=10.0),
    "kepler":             dict(x0=1.00, y0=0.00,  vx0=0.00, vy0=6.28, t_end=10.0),
    "spherical_pendulum": dict(x0=0.70, y0=0.00, z0=-0.71, vx0=0.00, vy0=1.20, vz0=0.00, t_end=20.0),
    "magnetic_particle":  dict(x0=0.00, y0=0.00, z0=0.00,  vx0=1.00, vy0=0.00, vz0=0.50, t_end=20.0),
    "lorenz":             dict(x0=1.00, y0=0.00, z0=0.00, t_end=30.0),
}


def _get_sld_defs(motion_key: str) -> list:
    if motion_key == "lorenz":
        return _SLD_DEFS_LORENZ
    if motion_key in _3D_MOTIONS:
        return _SLD_DEFS_3D
    return _SLD_DEFS_2D


# ── Standalone model loader ─────────────────────────────────────────────────
def _load_trainer(motion_key: str, model_dir: Path):
    """
    Load a model from a named model directory (containing model.pt).
    Returns a minimal trainer-like object for ML-ODE inference.
    """
    import json
    import torch
    from cmv.ml.model import PhysicsMLPModel
    from cmv.data.recorder import _SimpleScaler

    model_dir  = Path(model_dir)
    model_path = model_dir / "model.pt"

    # Try model_meta.json first; fall back to state_dict shape inspection
    meta_path = model_dir / "model_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta_data = json.load(f)
        input_dim    = meta_data["input_dim"]
        output_dim   = meta_data["output_dim"]
        feature_cols = meta_data["feature_cols"]
        target_cols  = meta_data["target_cols"]
    else:
        sd = torch.load(model_path, map_location="cpu", weights_only=True)
        input_dim  = int(sd["_entry.0.weight"].shape[1])
        output_dim = int(sd["_head.weight"].shape[0])
        if input_dim == 3:
            feature_cols = ["x", "y", "z"]
            target_cols  = ["vx", "vy", "vz"]
        elif input_dim == 6:
            feature_cols = ["x", "y", "z", "vx", "vy", "vz"]
            target_cols  = ["ax", "ay", "az"]
        else:
            feature_cols = ["x", "y", "vx", "vy"]
            target_cols  = ["ax", "ay"]
        meta_data = {}

    model = PhysicsMLPModel(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    scaler_X = _SimpleScaler()
    scaler_y = _SimpleScaler()
    scalers_path = model_dir / "scalers.npz"
    if scalers_path.exists():
        sc = np.load(scalers_path)
        scaler_X.mean_  = sc["X_mean"];  scaler_X.scale_ = sc["X_scale"]
        scaler_y.mean_  = sc["y_mean"];  scaler_y.scale_ = sc["y_scale"]
        has_scalers = True
    else:
        scaler_X.mean_  = np.zeros(input_dim);  scaler_X.scale_  = np.ones(input_dim)
        scaler_y.mean_  = np.zeros(output_dim); scaler_y.scale_ = np.ones(output_dim)
        has_scalers = False

    from types import SimpleNamespace
    bundle = SimpleNamespace(
        scaler_X=scaler_X, scaler_y=scaler_y,
        metadata={
            "feature_cols":  feature_cols,
            "target_cols":   target_cols,
            "has_scalers":   has_scalers,
            "model_name":    meta_data.get("model_name", model_dir.name),
            "created_at":    meta_data.get("created_at", ""),
            "n_sessions":    meta_data.get("n_sessions", "?"),
            "n_points":      meta_data.get("n_points", "?"),
            "session_paths": meta_data.get("session_paths", []),
        },
    )

    class _MinTrainer:
        def __init__(self, m, b):
            self.model = m
            self.bundle = b
            self.device = "cpu"

        def predict(self, X: np.ndarray) -> np.ndarray:
            import torch
            self.model.eval()
            with torch.no_grad():
                Xt = torch.tensor(X, dtype=torch.float32)
                pred = self.model(Xt).numpy()
            return self.bundle.scaler_y.inverse_transform(pred)

    return _MinTrainer(model, bundle)


# ── MLViewer ────────────────────────────────────────────────────────────────
class MLViewer:
    def __init__(self, fig_size=(17, 9), target_fps: int = 60,
                 fig=None) -> None:
        apply_style()
        self.target_fps = target_fps

        if fig is not None:
            self.fig = fig
            self.fig.patch.set_facecolor(PAL["fig_bg"])
        else:
            self.fig = plt.figure(figsize=fig_size, facecolor=PAL["fig_bg"])
            try:
                self.fig.canvas.manager.set_window_title("ML Prediction Viewer")
            except Exception:
                pass

        self._current_motion: str = "pendulum"
        self._is_3d:          bool = False
        self._trainer         = None
        self._pred_positions  = None   # (N, 2) or (N, 3)
        self._pred_times      = None   # (N,)
        self._pred_is_3d:     bool = False
        self.anim: Optional[FuncAnimation] = None

        self._anim_start_time:  Optional[float] = None
        self._anim_pause_start: float            = 0.0
        self._current_fi:       int              = 0
        self._paused:           bool             = False

        self._ic = dict(_DEFAULT_IC["pendulum"])

        self._debounce_timer: Optional[threading.Timer] = None
        self._info_timer_id   = None
        self._window_busy:    bool = False
        self._configure_timer_id   = None
        self._last_save_path: Optional[Path] = None

        self.sliders:       dict[str, Slider] = {}
        self._slider_axes:  list              = []

        self.ax_main = self.fig.add_axes(_MAIN_RECT, facecolor=PAL["panel_bg"])
        self.ax_main.grid(True, linewidth=0.5, alpha=0.6, color=PAL["grid"])
        self.ax_info = self.fig.add_axes(_INFO_RECT, facecolor=PAL["panel_bg"])

        self._build_info_artists(is_3d=False)
        self._build_buttons()
        self._draw_dividers()
        self._build_status_bar()
        self._build_sliders("pendulum")
        self._bind_window_events()

    # ── Axes type switching ────────────────────────────────────────────────
    def _ensure_axes_type(self, is_3d: bool):
        if is_3d == self._is_3d:
            return
        self.fig.delaxes(self.ax_main)
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            ax = self.fig.add_axes(_MAIN_RECT, projection="3d",
                                   facecolor=PAL["panel_bg"])
            ax.set_box_aspect([1, 1, 1])
        else:
            ax = self.fig.add_axes(_MAIN_RECT, facecolor=PAL["panel_bg"])
            ax.grid(True, linewidth=0.5, alpha=0.6, color=PAL["grid"])
        self.ax_main = ax
        self._is_3d  = is_3d

    # ── Right panel: position time series ─────────────────────────────────
    def _build_info_artists(self, is_3d: bool = False):
        ax = self.ax_info
        ax.set_title("위치 시계열", fontsize=9, fontweight="bold",
                     color=PAL["text"], pad=6)
        ax.set_xlabel("t [s]", fontsize=8)
        ax.set_ylabel("위치 [m]", fontsize=8)
        self.x_line, = ax.plot([], [], color=PAL["accent"],
                               lw=1.5, label="x(t)")
        self.y_line, = ax.plot([], [], color=PAL["ke"],
                               lw=1.5, label="y(t)", ls="--")
        if is_3d:
            self.z_line, = ax.plot([], [], color=PAL["pe"],
                                   lw=1.5, label="z(t)", ls=":")
        else:
            self.z_line = None
        ax.legend(loc="upper right", fontsize=7.5,
                  framealpha=0.9, edgecolor=PAL["border"])
        self._info_lim_last_t: float = 0.0

    def _rebuild_info_panel(self, is_3d: bool):
        self.ax_info.cla()
        self._build_info_artists(is_3d=is_3d)

    def _update_info_panel(self, fi: int):
        if self._pred_positions is None or self._pred_times is None:
            return
        pos = self._pred_positions
        t   = self._pred_times
        fi  = min(fi, len(t) - 1)
        if fi < 1:
            return
        t_now = float(t[fi])
        win   = 10.0
        t_lo  = max(float(t[0]), t_now - win)
        start = int(np.searchsorted(t, t_lo, side="left"))
        t_w   = t[start:fi + 1]
        x_w   = pos[start:fi + 1, 0]
        y_w   = pos[start:fi + 1, 1]
        self.x_line.set_data(t_w, x_w)
        self.y_line.set_data(t_w, y_w)
        if self.z_line is not None and pos.shape[1] >= 3:
            z_w = pos[start:fi + 1, 2]
            self.z_line.set_data(t_w, z_w)

        now = time.perf_counter()
        if now - self._info_lim_last_t >= 1.0:
            self.ax_info.set_xlim(t_lo, t_lo + win)
            if len(x_w) > 1:
                cols = [x_w, y_w]
                if self.z_line is not None and pos.shape[1] >= 3:
                    cols.append(pos[start:fi + 1, 2])
                all_v = np.concatenate(cols)
                lo, hi = float(all_v.min()), float(all_v.max())
                sp = max(hi - lo, 1e-3)
                self.ax_info.set_ylim(lo - 0.1 * sp, hi + 0.1 * sp)
            self._info_lim_last_t = now

    # ── Info panel timer ───────────────────────────────────────────────────
    def _start_info_timer(self):
        self._stop_info_timer()
        def _tick():
            if self._pred_positions is not None and not self._window_busy:
                self._update_info_panel(self._current_fi)
                self.ax_info.figure.canvas.draw_idle()
            try:
                self._info_timer_id = (
                    self.fig.canvas.get_tk_widget().after(200, _tick)
                )
            except Exception:
                self._info_timer_id = None
        try:
            self._info_timer_id = (
                self.fig.canvas.get_tk_widget().after(200, _tick)
            )
        except Exception:
            pass

    def _stop_info_timer(self):
        if self._info_timer_id is not None:
            try:
                self.fig.canvas.get_tk_widget().after_cancel(self._info_timer_id)
            except Exception:
                pass
            self._info_timer_id = None

    # ── Structural helpers ─────────────────────────────────────────────────
    def _draw_dividers(self):
        T = self.fig.transFigure
        for xs, ys in [
            ([0.115, 0.608], [0.265, 0.265]),
            ([0.620, 0.990], [0.265, 0.265]),
            ([0.610, 0.610], [0.020, 0.950]),
        ]:
            ln = mlines.Line2D(xs, ys, transform=T,
                               color=PAL["border"], linewidth=0.7, zorder=0)
            self.fig.add_artist(ln)

    def _build_status_bar(self):
        self._status_text = self.fig.text(
            0.50, 0.012,
            "운동 유형을 선택하고 [모델 로드]를 눌러 시작하세요",
            ha="center", va="center", fontsize=8.5,
            color=_STATUS_COLOR["info"],
            bbox=dict(facecolor=_STATUS_BG["info"], edgecolor=PAL["border"],
                      boxstyle="round,pad=0.35", alpha=0.92),
            transform=self.fig.transFigure, zorder=10,
        )

    def set_status(self, msg: str, kind: str = "info"):
        self._status_text.set_text(msg)
        self._status_text.set_color(_STATUS_COLOR.get(kind, _STATUS_COLOR["info"]))
        patch = self._status_text.get_bbox_patch()
        if patch is not None:
            patch.set_facecolor(_STATUS_BG.get(kind, _STATUS_BG["info"]))
        self.fig.canvas.draw_idle()

    # ── Buttons ────────────────────────────────────────────────────────────
    def _build_buttons(self):
        callbacks = {
            "초기화":    self._on_reset,
            "저장":      self._on_save,
            "모델 로드": self._on_load_model,
            "모델 정보": self._on_model_info,
            "폴더 열기": self._on_open_folder,
            "일시정지":  self._on_pause,
        }
        self._buttons: dict[str, Button] = {}
        for label, rect in _BTNS:
            ax  = self.fig.add_axes(rect, facecolor=PAL["btn_bg"])
            btn = Button(ax, label, color=PAL["btn_bg"],
                         hovercolor=PAL["btn_hover"])
            btn.label.set_fontsize(8.5)
            btn.label.set_color(PAL["text"])
            btn.label.set_fontweight("bold")
            btn.on_clicked(callbacks[label])
            self._buttons[label] = btn

    # ── Sliders (initial conditions, motion-aware) ─────────────────────────
    def _build_sliders(self, motion_key: str):
        for ax in self._slider_axes:
            self.fig.delaxes(ax)
        self._slider_axes.clear()
        self.sliders.clear()

        defs = _get_sld_defs(motion_key)
        n    = len(defs)
        if n == 0:
            return

        sd    = _SLD_ZONE
        avail = sd["y_top"] - 0.01
        # Auto-size h and gap so all sliders fit: gap = h * 0.2
        h   = avail / (n + 0.2 * (n - 1))
        gap = h * 0.2
        total_h = n * h + (n - 1) * gap
        y_bot   = sd["y_top"] - total_h

        for i, (pname, label, vmin, vmax, vinit) in enumerate(defs):
            y = y_bot + (n - 1 - i) * (h + gap)
            ax_sl = self.fig.add_axes(
                [sd["x0"], y, sd["w"], h],
                facecolor=PAL["slider_bg"],
            )
            sl = Slider(ax_sl, label, vmin, vmax, valinit=vinit,
                        color=PAL["slider_fg"],
                        track_color=PAL["slider_bg"])
            sl.label.set_fontsize(8)
            sl.label.set_color(PAL["text"])
            sl.valtext.set_fontsize(7.5)
            sl.valtext.set_color(PAL["subtext"])
            sl.on_changed(lambda v, p=pname: self._on_slider(p, v))
            self.sliders[pname] = sl
            self._slider_axes.append(ax_sl)

    def _on_slider(self, name: str, val: float):
        if self._debounce_timer:
            self._debounce_timer.cancel()

        def _run():
            def _on_main():
                self._ic[name] = val
                if self._trainer is not None:
                    self._run_prediction(pause_after=True)
            try:
                self.fig.canvas.get_tk_widget().after(0, _on_main)
            except Exception:
                self._ic[name] = val
                if self._trainer is not None:
                    self._run_prediction(pause_after=True)

        self._debounce_timer = threading.Timer(0.20, _run)
        self._debounce_timer.start()

    # ── Motion selector (called from main.py) ─────────────────────────────
    def set_motion_radio(self, ax_rect: list, labels: list[str], callback):
        from matplotlib.widgets import RadioButtons
        ax = self.fig.add_axes(ax_rect, facecolor=PAL["panel_bg"])
        ax.set_title("운동\n유형", fontsize=8, fontweight="bold",
                     color=PAL["text"], pad=4)
        radio = RadioButtons(ax, labels, active=0,
                             activecolor=PAL["accent"])
        for lbl in radio.labels:
            lbl.set_fontsize(7.5)
            lbl.set_color(PAL["text"])
        radio.on_clicked(callback)
        self._model_radio = radio
        return radio

    def set_current_motion(self, motion_key: str):
        self._current_motion = motion_key
        self._trainer = None   # 다른 운동 유형의 모델은 재사용 불가
        # Rebuild sliders for this motion type
        self._build_sliders(motion_key)
        # Reset IC to motion defaults and sync slider positions
        self._ic = dict(_DEFAULT_IC.get(motion_key, _DEFAULT_IC["pendulum"]))
        for k, v in self._ic.items():
            if k in self.sliders:
                self.sliders[k].set_val(v)
        title = _MOTION_TITLES.get(motion_key, motion_key)
        self.set_status(
            f"운동 유형: {title} — [모델 로드]를 눌러 모델을 선택하세요", "info"
        )

    # ── Button callbacks ───────────────────────────────────────────────────
    def _on_load_model(self, _):
        # Matplotlib 이벤트 핸들러 밖에서 다이얼로그를 열어야 wait_window()가 정상 동작
        self.fig.canvas.get_tk_widget().after(0, self._open_load_model_dialog)

    def _open_load_model_dialog(self):
        motion = self._current_motion
        from cmv.viz.dialogs import show_model_select_dialog

        def _on_done(model_dir):
            if model_dir is None:
                return   # 취소 또는 모델 없음

            self.set_status(f"모델 로드 중... [{model_dir.name}]", "warn")
            self.fig.canvas.flush_events()
            try:
                self._trainer = _load_trainer(motion, model_dir)
            except Exception as ex:
                self.set_status(f"모델 로드 실패: {ex}", "error")
                return

            feats = self._trainer.bundle.metadata.get("feature_cols", ["x", "y", "vx", "vy"])
            is_3d = len(feats) >= 3
            self._build_sliders(motion)
            self._rebuild_info_panel(is_3d)

            ic = _DEFAULT_IC.get(motion, _DEFAULT_IC["pendulum"])
            self._ic = dict(ic)
            for k, v in self._ic.items():
                if k in self.sliders:
                    self.sliders[k].set_val(v)

            meta        = self._trainer.bundle.metadata
            has_sc      = meta.get("has_scalers", False)
            model_name  = meta.get("model_name", model_dir.name)
            title       = _MOTION_TITLES.get(motion, motion)
            scaler_note = "" if has_sc else "  [scaler 없음]"
            self.set_status(
                f"모델 로드 완료 — [{model_name}]  ({title}){scaler_note}", "ok"
            )
            self._run_prediction(pause_after=False)

        show_model_select_dialog(motion, on_done=_on_done)

    def _on_reset(self, _):
        self._trainer        = None
        self._pred_positions = None
        self._pred_times     = None
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
            try:
                self.anim._stop()
            except Exception:
                pass
            self.anim = None
        self._stop_info_timer()
        self._paused = False
        btn = self._buttons.get("일시정지")
        if btn:
            btn.label.set_text("일시정지")

        # Return to 2D axes if currently in 3D
        if self._is_3d:
            self._ensure_axes_type(False)

        self.ax_main.cla()
        self.ax_main.grid(True, linewidth=0.5, alpha=0.6, color=PAL["grid"])
        self._rebuild_info_panel(False)

        # Reset IC to motion defaults
        self._ic = dict(_DEFAULT_IC.get(self._current_motion, _DEFAULT_IC["pendulum"]))
        for k, v in self._ic.items():
            if k in self.sliders:
                self.sliders[k].set_val(v)

        self.fig.canvas.draw_idle()
        self.set_status("초기화 완료", "ok")

    def _on_save(self, _):
        if self._pred_positions is None:
            self.set_status("저장할 예측 데이터가 없습니다", "warn")
            return
        import time as _t
        ts  = _t.strftime("%Y%m%d_%H%M%S")
        out = Path("ml_predictions") / self._current_motion / ts
        out.mkdir(parents=True, exist_ok=True)
        np.save(str(out / "positions.npy"),  self._pred_positions)
        np.save(str(out / "timestamps.npy"), self._pred_times)
        try:
            self.fig.savefig(str(out / "preview.png"), dpi=120,
                             bbox_inches="tight")
        except Exception:
            pass
        self._last_save_path = out
        self.set_status(f"저장 완료 → {out.resolve()}", "ok")

    def _on_model_info(self, _):
        if self._trainer is None:
            self.set_status("로드된 모델 없음 — 먼저 [모델 로드]를 눌러주세요", "warn")
            return
        m    = self._trainer
        meta = m.bundle.metadata
        has_sc     = meta.get("has_scalers", False)
        model_name = meta.get("model_name", "?")
        created_at = meta.get("created_at", "?")
        n_sess     = meta.get("n_sessions", "?")
        n_pts      = meta.get("n_points", "?")
        motion     = _MOTION_TITLES.get(self._current_motion, self._current_motion)
        total_params = sum(p.numel() for p in m.model.parameters())
        n_pts_str = f"{int(n_pts):,}" if str(n_pts).isdigit() else str(n_pts)
        try:
            import tkinter.messagebox as _mb
            _mb.showinfo(
                "모델 정보",
                f"모델 이름:    {model_name}\n"
                f"운동 유형:    {motion}\n"
                f"생성 일시:    {created_at}\n"
                f"학습 세션:    {n_sess}개   ({n_pts_str} pts)\n"
                f"파라미터 수: {total_params:,}\n"
                f"입력 피처:   {meta.get('feature_cols', [])}\n"
                f"출력 피처:   {meta.get('target_cols', [])}\n"
                f"Scaler:      {'저장됨' if has_sc else '없음 (정확도 저하 가능)'}",
            )
        except Exception:
            self.set_status(
                f"[{model_name}]  {motion} | {total_params:,} params | "
                f"scaler={'OK' if has_sc else 'MISSING'}",
                "info",
            )

    def _on_open_folder(self, _):
        import os, sys
        target = (Path(self._last_save_path) if self._last_save_path
                  else Path("ml_predictions"))
        if not target.exists():
            target = Path("ml_predictions")
        if not target.exists():
            self.set_status("저장된 폴더 없음 — 먼저 [저장]을 눌러주세요", "warn")
            return
        try:
            resolved = str(target.resolve())
            if sys.platform == "win32":
                os.startfile(resolved)
            elif sys.platform == "darwin":
                import subprocess; subprocess.Popen(["open", resolved])
            else:
                import subprocess; subprocess.Popen(["xdg-open", resolved])
            self.set_status(f"폴더 열기 → {resolved}", "ok")
        except Exception as ex:
            self.set_status(f"폴더 열기 실패: {ex}", "error")

    def _on_pause(self, _):
        if self.anim is None:
            return
        self._paused = not self._paused
        btn = self._buttons.get("일시정지")
        if self._paused:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
            self._stop_info_timer()
            if btn:
                btn.label.set_text("▶ 재개")
            self._anim_pause_start = time.perf_counter()
            self.set_status("일시정지됨 — [▶ 재개]를 눌러 계속하세요", "warn")
        else:
            # event_source.start()는 TkAgg에서 stop 후 불안정 — 플래그 해제만으로 재개
            self._start_info_timer()
            if btn:
                btn.label.set_text("일시정지")
            if self._anim_start_time is not None:
                self._anim_start_time += (
                    time.perf_counter() - self._anim_pause_start
                )
            title = _MOTION_TITLES.get(self._current_motion, self._current_motion)
            self.set_status(f"재개됨 — {title} ML 예측", "info")
        self.fig.canvas.draw_idle()

    # ── ML Prediction ODE ─────────────────────────────────────────────────
    def _run_prediction(self, pause_after: bool = False):
        if self._trainer is None:
            return
        ic      = self._ic
        t_end   = max(ic.get("t_end", 10.0), 0.5)
        n_pts   = max(200, int(t_end * 60))

        trainer  = self._trainer
        bundle   = trainer.bundle
        scaler_X = bundle.scaler_X
        feats    = bundle.metadata.get("feature_cols", ["x", "y", "vx", "vy"])
        n_feat   = len(feats)

        if n_feat == 3:
            # Lorenz 1st-order: state = (x, y, z), target = (dx/dt, dy/dt, dz/dt)
            y_init = [ic.get("x0", 1.0), ic.get("y0", 0.0), ic.get("z0", 0.0)]

            def ml_ode(t, state):
                vals = {"x": state[0], "y": state[1], "z": state[2]}
                raw  = np.array([[vals[c] for c in feats]], dtype=np.float64)
                acc  = trainer.predict(scaler_X.transform(raw))
                return [float(acc[0, 0]), float(acc[0, 1]), float(acc[0, 2])]

        elif n_feat == 6:
            # 3D 2nd-order: state = (x,y,z,vx,vy,vz), target = (ax,ay,az)
            y_init = [
                ic.get("x0", 0.0), ic.get("y0", 0.0), ic.get("z0", 0.0),
                ic.get("vx0", 0.0), ic.get("vy0", 0.0), ic.get("vz0", 0.0),
            ]

            def ml_ode(t, state):
                xc, yc, zc, vxc, vyc, vzc = state
                vals = {"x": xc, "y": yc, "z": zc, "vx": vxc, "vy": vyc, "vz": vzc}
                raw  = np.array([[vals[c] for c in feats]], dtype=np.float64)
                acc  = trainer.predict(scaler_X.transform(raw))
                return [vxc, vyc, vzc,
                        float(acc[0, 0]), float(acc[0, 1]), float(acc[0, 2])]

        else:
            # 2D 2nd-order: state = (x,y,vx,vy), target = (ax,ay)
            y_init = [
                ic.get("x0", 0.0), ic.get("y0", 1.0),
                ic.get("vx0", 0.0), ic.get("vy0", 0.0),
            ]

            def ml_ode(t, state):
                xc, yc, vxc, vyc = state
                vals = {"x": xc, "y": yc, "vx": vxc, "vy": vyc}
                raw  = np.array([[vals[c] for c in feats]], dtype=np.float64)
                acc  = trainer.predict(scaler_X.transform(raw))
                return [vxc, vyc, float(acc[0, 0]), float(acc[0, 1])]

        try:
            sol = solve_ivp(
                ml_ode, (0.0, t_end), y_init,
                t_eval=np.linspace(0.0, t_end, n_pts),
                method="RK45", rtol=1e-5, atol=1e-7,
                max_step=t_end / 50,
            )
        except Exception as ex:
            self.set_status(f"예측 실패: {ex}", "error")
            return

        if not sol.success or not np.all(np.isfinite(sol.y)):
            self.set_status("ML ODE 발산 — 초기 조건을 조정해 보세요", "warn")
            return

        is_3d = n_feat >= 3
        if is_3d:
            self._pred_positions = np.column_stack([sol.y[0], sol.y[1], sol.y[2]])
        else:
            self._pred_positions = np.column_stack([sol.y[0], sol.y[1]])
        self._pred_times  = sol.t
        self._pred_is_3d  = is_3d

        self._build_ml_animation(is_3d=is_3d, pause_after=pause_after)

        title = _MOTION_TITLES.get(self._current_motion, self._current_motion)
        if pause_after:
            self.set_status(
                f"파라미터 변경됨 — {title}  |  [▶ 재개]를 눌러 시작하세요", "warn"
            )
        else:
            self.set_status(f"ML 예측 — {title}  |  실시간 재생 (루프)", "info")

    # ── Animation ──────────────────────────────────────────────────────────
    def _build_ml_animation(self, is_3d: bool = False, pause_after: bool = False):
        pos = self._pred_positions
        t   = self._pred_times

        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
            try:
                self.anim._stop()
            except Exception:
                pass
            self.anim = None
        self._stop_info_timer()
        self._anim_start_time = None
        self._current_fi      = 0

        self._ensure_axes_type(is_3d)
        ax = self.ax_main
        ax.cla()

        title = _MOTION_TITLES.get(self._current_motion, self._current_motion)
        M = 0.12

        if is_3d:
            ax.set_facecolor(PAL["panel_bg"])
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor(PAL["border"])
            ax.grid(True, linewidth=0.4, alpha=0.5)
            for i, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
                lo, hi = float(pos[:, i].min()), float(pos[:, i].max())
                sp = max(hi - lo, 1e-3)
                setter(lo - M * sp, hi + M * sp)
            ax.set_xlabel("x", fontsize=8, color=PAL["text"], labelpad=4)
            ax.set_ylabel("y", fontsize=8, color=PAL["text"], labelpad=4)
            ax.set_zlabel("z", fontsize=8, color=PAL["text"], labelpad=4)
            ax.set_title(f"{title} — ML 예측", fontsize=10,
                         fontweight="bold", color=PAL["text"], pad=10)
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                    color=PAL["dot"], lw=0.6, alpha=0.15)

            (self.trail_line,) = ax.plot([], [], [],
                                         color=PAL["dot"], lw=1.6, alpha=0.6)
            (self.obj_dot,)    = ax.plot([], [], [], "o",
                                         color=PAL["dot"], ms=8, zorder=5)
            self.time_text = ax.text2D(
                0.02, 0.97, "", transform=ax.transAxes,
                fontsize=9, va="top", color=PAL["subtext"],
            )
            _artists = (self.trail_line, self.obj_dot, self.time_text)
            TRAIL = TRAIL_LEN_3D

            def _init():
                self.trail_line.set_data_3d([], [], [])
                self.obj_dot.set_data_3d([], [], [])
                self.time_text.set_text("")
                return _artists

            def _update(_fi):
                if self._paused or self._window_busy:
                    return _artists
                now = time.perf_counter()
                if self._anim_start_time is None:
                    self._anim_start_time = now
                elapsed  = now - self._anim_start_time
                t_target = float(t[0]) + elapsed
                if t_target >= float(t[-1]):
                    self._anim_start_time = now
                    t_target = float(t[0])
                fi = int(np.searchsorted(t, t_target, side="right")) - 1
                fi = max(0, min(fi, len(pos) - 1))
                self._current_fi = fi
                t0 = max(0, fi - TRAIL)
                self.trail_line.set_data_3d(
                    pos[t0:fi+1, 0], pos[t0:fi+1, 1], pos[t0:fi+1, 2]
                )
                self.obj_dot.set_data_3d([pos[fi, 0]], [pos[fi, 1]], [pos[fi, 2]])
                self.time_text.set_text(f"t = {t[fi]:.2f} s")
                return _artists

            use_blit = False

        else:
            ax.set_facecolor(PAL["panel_bg"])
            ax.grid(True, linewidth=0.5, alpha=0.6, color=PAL["grid"])
            x_lo, x_hi = float(pos[:, 0].min()), float(pos[:, 0].max())
            y_lo, y_hi = float(pos[:, 1].min()), float(pos[:, 1].max())
            sp_x = max(x_hi - x_lo, 1e-3); sp_y = max(y_hi - y_lo, 1e-3)
            ax.set_xlim(x_lo - M * sp_x, x_hi + M * sp_x)
            ax.set_ylim(y_lo - M * sp_y, y_hi + M * sp_y)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x [m]", fontsize=9, color=PAL["text"])
            ax.set_ylabel("y [m]", fontsize=9, color=PAL["text"])
            ax.set_title(f"{title} — ML 예측", fontsize=10,
                         fontweight="bold", color=PAL["text"])
            ax.plot(pos[:, 0], pos[:, 1],
                    color=PAL["dot"], lw=0.7, alpha=0.18)

            (self.trail_line,) = ax.plot([], [],
                                         color=PAL["dot"], lw=1.8, alpha=0.65)
            (self.obj_dot,)    = ax.plot([], [], "o",
                                         color=PAL["dot"], ms=9, zorder=5)
            self.time_text = ax.text(
                0.02, 0.97, "", transform=ax.transAxes,
                fontsize=9, va="top", color=PAL["subtext"],
                bbox=dict(facecolor=PAL["panel_bg"], edgecolor="none",
                          alpha=0.75, pad=2),
            )
            _artists = (self.trail_line, self.obj_dot, self.time_text)
            TRAIL = TRAIL_LEN_2D

            def _init():
                self.trail_line.set_data([], [])
                self.obj_dot.set_data([], [])
                self.time_text.set_text("")
                return _artists

            def _update(_fi):
                if self._paused or self._window_busy:
                    return _artists
                now = time.perf_counter()
                if self._anim_start_time is None:
                    self._anim_start_time = now
                elapsed  = now - self._anim_start_time
                t_target = float(t[0]) + elapsed
                if t_target >= float(t[-1]):
                    self._anim_start_time = now
                    t_target = float(t[0])
                fi = int(np.searchsorted(t, t_target, side="right")) - 1
                fi = max(0, min(fi, len(pos) - 1))
                self._current_fi = fi
                t0 = max(0, fi - TRAIL)
                self.trail_line.set_data(pos[t0:fi+1, 0], pos[t0:fi+1, 1])
                self.obj_dot.set_data([pos[fi, 0]], [pos[fi, 1]])
                self.time_text.set_text(f"t = {t[fi]:.2f} s")
                return _artists

            use_blit = True

        interval = max(8, int(1000 / self.target_fps))
        self.anim = FuncAnimation(
            self.fig, _update, init_func=_init,
            frames=itertools.count(), interval=interval,
            blit=use_blit, repeat=False,
            cache_frame_data=False,
        )

        self._paused = False
        btn = self._buttons.get("일시정지")
        if btn:
            btn.label.set_text("일시정지")

        if pause_after:
            self._paused = True  # _update 클로저가 플래그를 보고 프레임 갱신 건너뜀
            if btn:
                btn.label.set_text("▶ 재개")
        else:
            self._start_info_timer()

        self.fig.canvas.draw_idle()

    # ── Window event handling (identical pattern to Visualizer) ───────────
    def _bind_window_events(self):
        try:
            top = self.fig.canvas.get_tk_widget().winfo_toplevel()
            top.bind("<Configure>", self._on_win_configure, add="+")
            top.bind("<Unmap>",     self._on_win_configure, add="+")
            top.bind("<Map>",       self._on_win_restore,   add="+")
        except Exception:
            pass

    def _on_win_configure(self, event=None):
        if not self._window_busy:
            self._window_busy = True
            if self.anim is not None and not self._paused:
                try:
                    self.anim.event_source.stop()
                except Exception:
                    pass
            self._stop_info_timer()
        if self._configure_timer_id is not None:
            try:
                self.fig.canvas.get_tk_widget().after_cancel(
                    self._configure_timer_id
                )
            except Exception:
                pass
        try:
            self._configure_timer_id = (
                self.fig.canvas.get_tk_widget().after(150, self._on_win_idle)
            )
        except Exception:
            self._window_busy = False

    def _on_win_idle(self):
        self._configure_timer_id = None
        self._window_busy = False
        if self.anim is None or self._paused:
            return
        try:
            if not self.fig.canvas.get_tk_widget().winfo_ismapped():
                return
        except Exception:
            pass
        if self._pred_times is not None and self._anim_start_time is not None:
            fi = min(self._current_fi, len(self._pred_times) - 1)
            t_resume = (float(self._pred_times[fi])
                        - float(self._pred_times[0]))
            self._anim_start_time = time.perf_counter() - t_resume
        try:
            self.anim.event_source.start()
        except Exception:
            pass
        self._start_info_timer()

    def _on_win_restore(self, event=None):
        if self._configure_timer_id is not None:
            try:
                self.fig.canvas.get_tk_widget().after_cancel(
                    self._configure_timer_id
                )
            except Exception:
                pass
            self._configure_timer_id = None
        self._on_win_idle()

    def show(self):
        plt.show()
