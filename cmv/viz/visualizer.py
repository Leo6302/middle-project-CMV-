"""Main visualizer: FuncAnimation + Matplotlib Widgets, 2D/3D unified."""
from __future__ import annotations
import itertools
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

from cmv.physics.base import SimResult
from cmv.coords.transforms import pad_to_3d
from cmv.viz.style import apply_style, PAL


# ══════════════════════════════════════════════════════════════════════════════
#  Layout  (all values are normalised figure coordinates, 0–1)
#
#  ┌─────────┬──────────────────────────┬────────────────────────┐
#  │         │                          │   [C] Energy plot      │
#  │  [A]    │    [B] Main plot         │   y: 0.28 – 0.95       │
#  │  Motion │    x: 0.115 – 0.605      │                        │
#  │  select │    y: 0.28 – 0.95        │                        │
#  │         │                          │                        │
#  │         ├──────────────────────────┴─────────┬──────────────┤
#  │         │  [E] Sliders                        │  [F] Buttons │
#  │         │  x: 0.115–0.605  y: 0.02–0.25      │  2×2 grid    │
#  └─────────┴─────────────────────────────────────┴──────────────┘
#
# ══════════════════════════════════════════════════════════════════════════════

_MOT_RECT   = [0.005, 0.05, 0.098, 0.90]   # [A] motion selector

_MAIN_RECT  = [0.115, 0.28, 0.490, 0.67]   # [B] main animation plot
_ENER_RECT  = [0.620, 0.28, 0.370, 0.64]   # [C] energy plot (에너지 창 제어용 상단 여백 확보)

# [C-ctrl] 에너지 창 크기 조절 버튼 (에너지 plot 위쪽 여백)
_EWIN_HALF = [0.622, 0.930, 0.048, 0.020]  # ½× 버튼
_EWIN_DBL  = [0.673, 0.930, 0.048, 0.020]  # 2× 버튼
_EAUTO_BTN = [0.940, 0.930, 0.048, 0.020]  # 자동/수동 토글

_SLD_ZONE = dict(                           # [E] sliders  (up to 5)
    x0=0.115, y_top=0.25,
    w=0.490,  h=0.036, gap=0.007,
)

# [F] Buttons – 2×2 grid + 하단 2버튼 행
_BTNS = [
    ("초기화",    [0.622, 0.178, 0.178, 0.082]),
    ("저장",      [0.812, 0.178, 0.178, 0.082]),
    ("ML 학습",   [0.622, 0.086, 0.178, 0.082]),
    ("결과 비교", [0.812, 0.086, 0.178, 0.082]),
    ("폴더 열기", [0.622, 0.020, 0.178, 0.058]),
    ("일시정지",  [0.812, 0.020, 0.178, 0.058]),
]

TRAIL_LEN_2D = 120
TRAIL_LEN_3D = 200

_STATUS_COLOR = {
    "info":  PAL["subtext"],
    "ok":    "#16A34A",
    "warn":  "#D97706",
    "error": "#DC2626",
}
_STATUS_BG = {
    "info":  "#F8FAFC",
    "ok":    "#F0FDF4",
    "warn":  "#FFFBEB",
    "error": "#FEF2F2",
}

_MOTION_TITLES = {
    "pendulum":           "단진자",
    "double_pendulum":    "이중진자",
    "projectile":         "포물체 운동",
    "shm":                "단순조화 / 감쇠 / 강제진동",
    "circular":           "등속 원운동",
    "kepler":             "케플러 궤도",
    "spherical_pendulum": "구면진자  (3D)",
    "lorenz":             "로렌츠 어트랙터  (3D)",
    "magnetic_particle":  "자기장 속 대전입자  (3D)",
}


class Visualizer:
    def __init__(self, engine, fig_size=(17, 9), target_fps=60,
                 fig=None) -> None:
        apply_style()

        self.engine     = engine
        self.target_fps = target_fps

        if fig is not None:
            self.fig = fig
            self.fig.patch.set_facecolor(PAL["fig_bg"])
        else:
            self.fig = plt.figure(figsize=fig_size, facecolor=PAL["fig_bg"])
            try:
                self.fig.canvas.manager.set_window_title(
                    "Classical Mechanics Visualizer"
                )
            except Exception:
                pass

        self._is_3d = False
        self.ax_main   = self._make_main_axes(is_3d=False)
        self.ax_energy = self.fig.add_axes(_ENER_RECT, facecolor=PAL["panel_bg"])

        self.result: Optional[SimResult] = None
        self.anim:   Optional[FuncAnimation] = None
        self._coords: Optional[np.ndarray] = None

        self.sliders: dict[str, Slider] = {}
        self._slider_axes: list         = []
        self._debounce_timer: Optional[threading.Timer] = None
        self._ml_trainer   = None
        self._last_save_path: Optional[Path] = None
        self._training_active: bool = False
        self._training_cancel: bool = False
        self._paused: bool = False
        self._extending: bool = False
        self._extend_cancel: bool = False
        self._energy_window: float = 10.0   # 에너지 그래프 표시 시간창 (초)
        self._energy_auto: bool = True      # True=자동스크롤, False=수동(zoom/pan)
        self._current_fi: int = 0           # 현재 애니메이션 프레임 인덱스
        self._energy_timer_id = None        # Tk after() ID (에너지 독립 타이머)
        self._frame_step: float = 1.0       # 애니메이션 1 tick당 건너뛸 데이터 수 (실시간 보정)
        self._anim_start_time: Optional[float] = None   # 벽시계 기준 애니메이션 시작 시각
        self._anim_pause_start: float = 0.0             # 일시정지 시작 시각
        self._energy_lim_last_t: float = 0.0            # 에너지 축 범위 마지막 갱신 시각

        # ML 학습 진행 창 상태 (인스턴스 변수로 관리해 재표시/재생성 지원)
        self._train_fig                   = None
        self._train_ax                    = None
        self._tr_ln                       = None
        self._vl_ln                       = None
        self._tr_h: list                  = []
        self._vl_h: list                  = []
        self._train_val_target: float     = 5e-5
        self._train_max_epochs: int       = 3000

        # 창 조작 중 렌더링 중단 플래그
        self._window_busy: bool = False
        self._configure_timer_id = None

        self._build_energy_artists()
        self._build_buttons()
        self._build_energy_controls()
        self._draw_dividers()
        self._build_status_bar()
        self._bind_window_events()

    # ── Structural helpers ─────────────────────────────────────────────────
    def _make_main_axes(self, is_3d: bool):
        rect = _MAIN_RECT
        if is_3d:
            ax = self.fig.add_axes(rect, projection="3d",
                                   facecolor=PAL["panel_bg"])
            ax.set_box_aspect([1, 1, 1])
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor(PAL["border"])
            ax.grid(True, linewidth=0.4, alpha=0.5)
        else:
            ax = self.fig.add_axes(rect, facecolor=PAL["panel_bg"])
            ax.grid(True, linewidth=0.5, alpha=0.6, color=PAL["grid"])
        return ax

    def _ensure_axes_type(self, is_3d: bool):
        if is_3d == self._is_3d:
            return
        self.fig.delaxes(self.ax_main)
        self.ax_main = self._make_main_axes(is_3d)
        self._is_3d  = is_3d

    def _draw_dividers(self):
        T = self.fig.transFigure
        specs = [
            # horizontal: separates main area from controls
            ([0.115, 0.608], [0.265, 0.265]),
            # horizontal: right col – between energy and buttons
            ([0.620, 0.990], [0.265, 0.265]),
            # vertical: separates center column from right column
            ([0.610, 0.610], [0.020, 0.950]),
        ]
        for xs, ys in specs:
            ln = mlines.Line2D(xs, ys, transform=T,
                               color=PAL["border"], linewidth=0.7, zorder=0)
            self.fig.add_artist(ln)

    # ── Status bar ────────────────────────────────────────────────────────
    def _build_status_bar(self):
        self._status_text = self.fig.text(
            0.50, 0.012, "준비",
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

    # ── Energy subplot ─────────────────────────────────────────────────────
    def _build_energy_artists(self):
        ax = self.ax_energy
        ax.set_title("에너지", fontsize=9, fontweight="bold",
                     color=PAL["text"], pad=6)
        ax.set_xlabel("t [s]", fontsize=8)
        ax.set_ylabel("E [J]", fontsize=8)
        self.ke_line,   = ax.plot([], [], color=PAL["ke"],   lw=1.5, label="KE")
        self.pe_line,   = ax.plot([], [], color=PAL["pe"],   lw=1.5, label="PE")
        self.etot_line, = ax.plot([], [], color=PAL["etot"], lw=2.0,
                                  label="E total", linestyle="--")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9,
                  edgecolor=PAL["border"])

    def _setup_energy_limits(self, result: SimResult):
        p = self.engine.params
        t0, t1 = p.t_span
        self._energy_window = float(t1 - t0)   # 초기 시뮬레이션 구간을 기본 창 크기로
        t = result.timestamps
        e = result.energies
        e_min = float(e.min())
        e_max = float(e.max())
        span  = max(e_max - e_min, 1e-3)
        self.ax_energy.set_xlim(float(t[0]), float(t[-1]))
        self.ax_energy.set_ylim(e_min - 0.05 * span, e_max + 0.05 * span)

    def _update_energy(self, fi: int):
        if self.result is None:
            return
        t_all = self.result.timestamps
        e_all = self.result.energies
        fi = min(fi, len(t_all) - 1)
        if fi < 1:
            return

        if self._energy_auto:
            # ── 슬라이딩 윈도우: searchsorted로 O(log n) 범위 탐색
            t_now = float(t_all[fi])
            t_lo  = max(float(t_all[0]), t_now - self._energy_window)
            start = int(np.searchsorted(t_all, t_lo, side="left"))
            t_win = t_all[start:fi + 1]
            e_win = e_all[start:fi + 1]
            if len(t_win) < 2:
                return
            self.ke_line.set_data(t_win, e_win[:, 0])
            self.pe_line.set_data(t_win, e_win[:, 1])
            self.etot_line.set_data(t_win, e_win[:, 2])
            # 축 범위 변경은 blit 배경을 무효화하므로 최대 1 Hz로 제한
            now = time.perf_counter()
            if now - self._energy_lim_last_t >= 1.0:
                self.ax_energy.set_xlim(t_lo, t_lo + self._energy_window)
                e_min = float(e_win.min()); e_max = float(e_win.max())
                span  = max(e_max - e_min, 1e-3)
                self.ax_energy.set_ylim(e_min - 0.08 * span, e_max + 0.08 * span)
                self._energy_lim_last_t = now
        else:
            # ── 수동 모드: 선 데이터만 갱신, 축 범위는 사용자가 zoom/pan
            t_vis = t_all[:fi + 1]
            e_vis = e_all[:fi + 1]
            self.ke_line.set_data(t_vis, e_vis[:, 0])
            self.pe_line.set_data(t_vis, e_vis[:, 1])
            self.etot_line.set_data(t_vis, e_vis[:, 2])

    # ── Energy window controls ─────────────────────────────────────────────
    def _build_energy_controls(self):
        def _make_ebtn(rect, label):
            ax = self.fig.add_axes(rect, facecolor=PAL["slider_bg"])
            btn = Button(ax, label, color=PAL["slider_bg"],
                         hovercolor=PAL["btn_hover"])
            btn.label.set_fontsize(7.5)
            btn.label.set_color(PAL["text"])
            return btn

        self._ebtn_half = _make_ebtn(_EWIN_HALF, "½×")
        self._ebtn_dbl  = _make_ebtn(_EWIN_DBL,  "2×")
        self._ebtn_auto = _make_ebtn(_EAUTO_BTN, "자동▸")

        def _on_half(_):
            self._energy_window = max(1.0, self._energy_window / 2)
            self.set_status(f"에너지 창 크기: {self._energy_window:.1f}s", "info")

        def _on_dbl(_):
            self._energy_window = min(3600.0, self._energy_window * 2)
            self.set_status(f"에너지 창 크기: {self._energy_window:.1f}s", "info")

        def _on_auto(_):
            self._energy_auto = not self._energy_auto
            if self._energy_auto:
                self._ebtn_auto.label.set_text("자동▸")
                self.set_status("에너지 그래프: 자동 스크롤", "info")
            else:
                self._ebtn_auto.label.set_text("◀수동")
                self.set_status("에너지 그래프: 수동 모드 (toolbar로 zoom/pan)", "info")
            self.fig.canvas.draw_idle()

        self._ebtn_half.on_clicked(_on_half)
        self._ebtn_dbl.on_clicked(_on_dbl)
        self._ebtn_auto.on_clicked(_on_auto)

    # ── Action buttons (2×2 grid) ──────────────────────────────────────────
    def _build_buttons(self):
        callbacks = {
            "초기화":    self._on_reset,
            "저장":      self._on_save,
            "ML 학습":   self._on_ml_train,
            "결과 비교": self._on_compare,
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

    def _set_btn_enabled(self, label: str, enabled: bool) -> None:
        btn = self._buttons.get(label)
        if btn is None:
            return
        color = PAL["btn_bg"] if enabled else PAL["slider_bg"]
        text_color = PAL["text"] if enabled else PAL["subtext"]
        btn.ax.set_facecolor(color)
        btn.label.set_color(text_color)
        self.fig.canvas.draw_idle()

    def _on_pause(self, _):
        if self.anim is None:
            return
        self._paused = not self._paused
        btn = self._buttons.get("일시정지")
        if self._paused:
            if self.anim.event_source:
                self.anim.event_source.stop()
            self._stop_energy_timer()
            if btn:
                btn.label.set_text("▶ 재개")
            self._anim_pause_start = time.perf_counter()
            self.set_status("일시정지됨 — [▶ 재개]를 눌러 계속하세요", "warn")
        else:
            # event_source.start()는 TkAgg에서 stop 후 재시작이 불안정하므로
            # _paused 플래그 해제만으로 _animate_* 함수가 자동으로 재개된다.
            self._start_energy_timer()
            if btn:
                btn.label.set_text("일시정지")
            # 일시정지 동안 흐른 시간을 _anim_start_time에 보정해 실시간 유지
            if self._anim_start_time is not None:
                self._anim_start_time += time.perf_counter() - self._anim_pause_start
            self.set_status("재개됨", "info")
        self.fig.canvas.draw_idle()

    # ── Energy timer (에너지 그래프 독립 갱신 — blit 분리) ─────────────────
    def _start_energy_timer(self) -> None:
        """Start 5 Hz energy-plot refresh via Tk after() — decoupled from blit."""
        self._stop_energy_timer()
        def _tick():
            if self.result is not None and not self._window_busy:
                self._update_energy(self._current_fi)
                self.ax_energy.figure.canvas.draw_idle()
            try:
                self._energy_timer_id = (
                    self.fig.canvas.get_tk_widget().after(200, _tick)
                )
            except Exception:
                self._energy_timer_id = None
        try:
            self._energy_timer_id = (
                self.fig.canvas.get_tk_widget().after(200, _tick)
            )
        except Exception:
            pass

    def _stop_energy_timer(self) -> None:
        if self._energy_timer_id is not None:
            try:
                self.fig.canvas.get_tk_widget().after_cancel(self._energy_timer_id)
            except Exception:
                pass
            self._energy_timer_id = None

    # ── 창 조작 중 렌더링 중단 (이동/최대화/최소화 렉 방지) ──────────────────
    def _bind_window_events(self) -> None:
        try:
            top = self.fig.canvas.get_tk_widget().winfo_toplevel()
            top.bind("<Configure>", self._on_win_configure, add="+")
            top.bind("<Unmap>",     self._on_win_configure, add="+")  # 최소화
            top.bind("<Map>",       self._on_win_restore,   add="+")  # 복원
        except Exception:
            pass

    def _on_win_configure(self, event=None) -> None:
        """창 이동/크기 변경/최소화 시작 — 애니메이션과 에너지 타이머를 중단."""
        if not self._window_busy:
            self._window_busy = True
            if self.anim is not None and not self._paused:
                try:
                    self.anim.event_source.stop()
                except Exception:
                    pass
            self._stop_energy_timer()

        # 150ms 디바운스: 마지막 configure 이후 150ms 뒤에 재개
        if self._configure_timer_id is not None:
            try:
                self.fig.canvas.get_tk_widget().after_cancel(self._configure_timer_id)
            except Exception:
                pass
        try:
            self._configure_timer_id = (
                self.fig.canvas.get_tk_widget().after(150, self._on_win_idle)
            )
        except Exception:
            self._window_busy = False

    def _on_win_idle(self) -> None:
        """창 조작 완료 — 애니메이션을 현재 위치에서 재개."""
        self._configure_timer_id = None
        self._window_busy = False
        if self.anim is None or self._paused:
            return
        # 탭 전환 시 비활성 탭의 애니메이션은 재개하지 않음
        try:
            if not self.fig.canvas.get_tk_widget().winfo_ismapped():
                return
        except Exception:
            pass
        # 조작 중 흐른 시간을 반영해 현재 프레임 위치부터 재개
        if self.result is not None and self._anim_start_time is not None:
            fi = min(self._current_fi, len(self.result.timestamps) - 1)
            t_resume = float(self.result.timestamps[fi]) - float(self.result.timestamps[0])
            self._anim_start_time = time.perf_counter() - t_resume
        try:
            self.anim.event_source.start()
        except Exception:
            pass
        self._start_energy_timer()

    def _on_win_restore(self, event=None) -> None:
        """최소화 복원 시 즉시 재개 (디바운스 타이머 취소 후 바로 idle 처리)."""
        if self._configure_timer_id is not None:
            try:
                self.fig.canvas.get_tk_widget().after_cancel(self._configure_timer_id)
            except Exception:
                pass
            self._configure_timer_id = None
        self._on_win_idle()

    def _start_extension(self) -> None:
        if self._extending or self._paused:
            return
        self._extending = True
        self._extend_cancel = False
        threading.Thread(target=self._extend_simulation, daemon=True).start()

    def _extend_simulation(self) -> None:
        try:
            if self._extend_cancel or self.result is None or self.result.raw_y is None:
                return
            p = self.engine.params
            t0_orig, t1_orig = p.t_span
            chunk_dur = t1_orig - t0_orig
            n_chunk   = p.n_points

            last_t = float(self.result.timestamps[-1])
            raw_y  = self.result.raw_y.copy()

            ext = self.engine.run(
                t_span=(last_t, last_t + chunk_dur),
                n_points=n_chunk,
                y0_override=raw_y,
            )

            if self._extend_cancel:
                return

            # 첫 점은 이전 마지막 점과 중복 → 제외
            new_pos  = ext.positions[1:]
            new_vel  = ext.velocities[1:]
            new_acc  = ext.accelerations[1:]
            new_t    = ext.timestamps[1:]
            new_e    = ext.energies[1:]
            new_c    = self._get_display_coords(ext)[1:]

            # 원자적 배열 교체 (GIL 보장)
            self.result.timestamps    = np.concatenate([self.result.timestamps, new_t])
            self.result.positions     = np.vstack([self.result.positions,  new_pos])
            self.result.velocities    = np.vstack([self.result.velocities, new_vel])
            self.result.accelerations = np.vstack([self.result.accelerations, new_acc])
            self.result.energies      = np.vstack([self.result.energies, new_e])
            self.result.raw_y         = ext.raw_y
            self._coords              = np.vstack([self._coords, new_c])

            # 에너지·메인 축 범위 갱신 (메인 스레드에서 실행)
            try:
                self.fig.canvas.get_tk_widget().after(0, self._update_axes_limits)
            except Exception:
                pass
        finally:
            self._extending = False

    def _update_axes_limits(self) -> None:
        c = self._coords
        if c is None or len(c) == 0:
            return
        if not self._is_3d:
            M = 0.10
            lo0, hi0 = float(c[:, 0].min()), float(c[:, 0].max())
            lo1, hi1 = float(c[:, 1].min()), float(c[:, 1].max())
            sp0 = max(hi0 - lo0, 1e-3)
            sp1 = max(hi1 - lo1, 1e-3)
            self.ax_main.set_xlim(lo0 - M * sp0, hi0 + M * sp0)
            self.ax_main.set_ylim(lo1 - M * sp1, hi1 + M * sp1)
        # blit 배경 캐시 비동기 갱신 — 블로킹 draw() 대신 draw_idle() 사용
        self.fig.canvas.draw_idle()

    # ── Button callbacks ───────────────────────────────────────────────────
    def _on_reset(self, _):
        if self.result is None:
            return
        if self._training_active:
            self.set_status("학습 중에는 초기화할 수 없습니다 — 학습 완료 후 시도하세요", "warn")
            return
        self.set_status("초기화 중...", "warn")
        self.fig.canvas.flush_events()
        # 일시정지 해제
        self._extend_cancel = True
        self._paused = False
        btn_p = self._buttons.get("일시정지")
        if btn_p:
            btn_p.label.set_text("일시정지")
        p_cls = type(self.engine.params)
        self.engine.params = p_cls()
        self.engine.motion.p = self.engine.params
        for sl in self.sliders.values():
            sl.reset()
        self._ml_trainer     = None
        self._last_save_path = None
        self._refresh()
        self.set_status("초기화 완료 — 파라미터 및 ML 학습 결과가 초기화됐습니다", "ok")

    def _on_save(self, _):
        if self.result is None:
            self.set_status("저장할 시뮬레이션 데이터가 없습니다", "warn")
            return
        self.set_status("저장 중...", "warn")
        self.fig.canvas.flush_events()
        from cmv.data.recorder import DataRecorder
        path = DataRecorder(Path("data")).save(self.result)
        self._last_save_path = path
        self.set_status(f"저장 완료 → {path}", "ok")

    # ── ML 학습 창 관리 ───────────────────────────────────────────────────────
    def _make_train_window(self) -> None:
        """ML 학습 진행 창을 (재)생성하고 인스턴스 변수에 저장."""
        if self._train_fig is not None:
            try:
                plt.close(self._train_fig)
            except Exception:
                pass

        lf, la = plt.subplots(figsize=(6, 4))
        lf.patch.set_facecolor(PAL["fig_bg"])
        la.set_facecolor(PAL["panel_bg"])
        la.set_title(
            f"학습 손실 곡선  (목표 val ≤ {self._train_val_target:.0e},"
            f" 상한 {self._train_max_epochs} 에포크)",
            fontsize=9, fontweight="bold",
        )
        la.set_xlabel("Epoch"); la.set_ylabel("MSE Loss (log)")
        la.set_yscale("log")
        la.axhline(self._train_val_target, color="#16A34A", lw=1.0, ls=":",
                   label=f"목표 {self._train_val_target:.0e}")
        tr_ln, = la.plot([], [], color=PAL["accent"], lw=1.6, label="Train")
        vl_ln, = la.plot([], [], color=PAL["dot"],    lw=1.6,
                         label="Val", linestyle="--")
        # 기존 히스토리 데이터 복원 (재생성 시 지금까지의 곡선 유지)
        if self._tr_h:
            xs = range(len(self._tr_h))
            tr_ln.set_data(xs, self._tr_h)
            vl_ln.set_data(xs, self._vl_h)
            la.relim(); la.autoscale_view()
        la.legend(fontsize=8)
        lf.tight_layout()
        plt.show(block=False)

        self._train_fig = lf
        self._train_ax  = la
        self._tr_ln     = tr_ln
        self._vl_ln     = vl_ln

    def _raise_train_window(self) -> None:
        """학습 창을 화면 앞으로 가져온다. 창이 닫혀 있으면 재생성한다."""
        if self._train_fig is None:
            return
        try:
            mgr = self._train_fig.canvas.manager
            mgr.window.deiconify()
            mgr.window.lift()
        except Exception:
            # 창이 파괴된 경우 — 기존 히스토리로 재생성
            self._make_train_window()

    def _on_ml_train(self, _):
        if self._last_save_path is None:
            self.set_status("먼저 [저장] 버튼으로 데이터를 저장하세요", "warn")
            return

        # 학습 진행 중: 중단 요청
        if self._training_active:
            self._training_cancel = True
            btn = self._buttons.get("ML 학습")
            if btn:
                btn.label.set_text("중단 중...")
            self._set_btn_enabled("ML 학습", False)
            self.set_status("학습 중단 요청 — 현재 에포크 완료 후 중단됩니다", "warn")
            self.fig.canvas.draw_idle()
            return

        # Matplotlib 버튼 이벤트 핸들러 안에서 wait_window()를 직접 호출하면
        # Tk 재진입(re-entrancy) 문제로 GUI가 먹통이 된다.
        # after(0, ...) 로 핸들러 밖에서 실행되도록 스케줄링한다.
        self.fig.canvas.get_tk_widget().after(0, self._open_train_dialog)

    def _open_train_dialog(self):
        try:
            self._open_train_dialog_impl()
        except Exception as e:
            self.set_status(f"다이얼로그 오류: {e}", "warn")
            import traceback, sys
            traceback.print_exc(file=sys.stderr)

    def _open_train_dialog_impl(self):
        from cmv.data.recorder import DataRecorder
        from cmv.viz.dialogs import show_train_config_dialog
        recorder = DataRecorder(Path("data"))
        all_sessions = list(reversed(recorder.list_sessions(self.result.motion_type)))

        def _on_done(model_name, selected_sessions):
            if model_name is None:
                return   # 사용자가 취소
            if not selected_sessions:
                self.set_status("학습할 세션이 선택되지 않았습니다", "warn")
                return

            # ── 새 학습 시작 ──────────────────────────────────────────────
            self._training_active = True
            self._training_cancel = False
            btn = self._buttons.get("ML 학습")
            if btn:
                btn.label.set_text("학습 중단")
            self.fig.canvas.draw_idle()
            self.set_status(f"ML 학습 시작... [{model_name}]", "warn")

            self._tr_h = []
            self._vl_h = []
            self._make_train_window()

            _gui_q: queue.SimpleQueue = queue.SimpleQueue()
            tk_widget = self.fig.canvas.get_tk_widget()

            def _schedule(fn: Callable) -> None:
                _gui_q.put(fn)

            def _poll() -> None:
                while True:
                    try:
                        fn = _gui_q.get_nowait()
                        fn()
                    except queue.Empty:
                        break
                    except Exception:
                        pass
                if self._training_active:
                    tk_widget.after(50, _poll)

            tk_widget.after(50, _poll)

            def _thread():
                from cmv.ml.model import PhysicsMLPModel
                from cmv.ml.trainer import MLTrainer
                from datetime import datetime

                _MOTION_FEAT_COLS_LOCAL = {
                    "spherical_pendulum": (["x","y","z","vx","vy","vz"], ["ax","ay","az"]),
                    "magnetic_particle":  (["x","y","z","vx","vy","vz"], ["ax","ay","az"]),
                    "lorenz":             (["x","y","z"],                 ["vx","vy","vz"]),
                }
                feat_cols, tgt_cols = _MOTION_FEAT_COLS_LOCAL.get(
                    self.result.motion_type, (["x","y","vx","vy"], ["ax","ay"])
                )

                if len(selected_sessions) > 1:
                    bundle = recorder.load_multi_as_bundle(
                        selected_sessions,
                        feature_cols=feat_cols, target_cols=tgt_cols,
                    )
                else:
                    bundle = recorder.load_as_bundle(
                        selected_sessions[0],
                        feature_cols=feat_cols, target_cols=tgt_cols,
                    )

                n_feat  = bundle.X_train.shape[1]
                n_tgt   = bundle.y_train.shape[1]
                model   = PhysicsMLPModel(input_dim=n_feat, output_dim=n_tgt)
                trainer = MLTrainer(model, bundle, lr=1e-3)

                def on_ep(ep, tl, vl):
                    self._tr_h.append(tl)
                    self._vl_h.append(vl)

                    def _update_plot(ep=ep, tr=list(self._tr_h), vl_=list(self._vl_h),
                                     tl=tl, vl=vl):
                        if self._tr_ln is not None:
                            self._tr_ln.set_data(range(len(tr)), tr)
                        if self._vl_ln is not None:
                            self._vl_ln.set_data(range(len(vl_)), vl_)
                        if self._train_ax is not None:
                            self._train_ax.relim()
                            self._train_ax.autoscale_view()
                        try:
                            if ep % 5 == 0 and self._train_fig is not None:
                                self._train_fig.canvas.draw_idle()
                        except Exception:
                            pass
                        if ep % 10 == 0:
                            self.set_status(
                                f"ML 학습 중... [{model_name}]"
                                f"  {ep}/{self._train_max_epochs} 에포크"
                                f"  |  train: {tl:.5f}  val: {vl:.5f}",
                                "warn",
                            )

                    _schedule(_update_plot)

                msg  = ""
                kind = "info"
                try:
                    hist = trainer.train(
                        on_epoch_end=on_ep,
                        val_target=self._train_val_target,
                        max_epochs=self._train_max_epochs,
                        stop_fn=lambda: self._training_cancel,
                    )

                    n_total = (bundle.X_train.shape[0]
                               + bundle.X_val.shape[0]
                               + bundle.X_test.shape[0])
                    extra_meta = {
                        "model_name":    model_name,
                        "motion_type":   self.result.motion_type,
                        "created_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "n_sessions":    len(selected_sessions),
                        "n_points":      n_total,
                        "session_paths": [str(s) for s in selected_sessions],
                    }
                    pt = (Path("models") / self.result.motion_type
                          / model_name / "model.pt")
                    trainer.save(pt, extra_meta=extra_meta)
                    self._ml_trainer = trainer

                    n_ep      = len(hist["val"])
                    final_val = hist["val"][-1] if hist["val"] else float("nan")

                    if hist.get("stopped"):
                        msg  = (f"학습 중단 [{model_name}]"
                                f"  ({n_ep} 에포크 | val: {final_val:.5f})"
                                f" — 부분 학습 모델로 결과 비교 가능")
                        kind = "warn"
                    elif hist.get("converged"):
                        msg  = (f"학습 완료 [{model_name}] — 목표 달성!"
                                f"  ({n_ep} 에포크 | val: {final_val:.5f})")
                        kind = "ok"
                    else:
                        msg  = (f"학습 완료 [{model_name}]  ({n_ep} 에포크"
                                f" | val: {final_val:.5f},"
                                f" 목표 {self._train_val_target:.0e} 미달)")
                        kind = "warn"
                except Exception as ex:
                    msg  = f"ML 학습 오류: {ex}"
                    kind = "error"
                finally:
                    self._training_active = False
                    self._training_cancel = False

                def _done(msg=msg, kind=kind):
                    try:
                        if self._train_fig is not None:
                            self._train_fig.canvas.draw_idle()
                    except Exception:
                        pass
                    self.set_status(msg, kind)
                    btn = self._buttons.get("ML 학습")
                    if btn:
                        btn.label.set_text("ML 학습")
                    self._set_btn_enabled("ML 학습", True)

                _schedule(_done)

            threading.Thread(target=_thread, daemon=True).start()

        show_train_config_dialog(
            motion_type=self.result.motion_type,
            sessions=all_sessions,
            on_done=_on_done,
        )

    def _on_compare(self, _):
        if self._ml_trainer is None:
            self.set_status("먼저 ML 학습을 완료하세요", "warn")
            return
        if getattr(self.engine.motion, "is_3d", False):
            self.set_status("결과 비교는 2D 운동에만 지원됩니다", "warn")
            return
        self.set_status("ML 예측 궤적 계산 중...", "warn")
        self.fig.canvas.flush_events()
        from cmv.viz.comparator import ResultComparator
        comp   = ResultComparator(self._ml_trainer, self.result)
        n_pts  = len(self.result.timestamps)
        t1     = float(self.result.timestamps[-1])
        pred   = comp.predict_trajectory(self.result.positions[0], (0.0, t1), n_pts)
        metrics = comp.compute_metrics(pred)
        comp.show_window(pred, metrics)
        self.set_status(
            f"비교 완료 | 위치 RMSE: {metrics.position_rmse:.4f} m"
            f"  속도 RMSE: {metrics.velocity_rmse:.4f} m/s",
            "ok",
        )

    def _on_open_folder(self, _):
        import os, sys
        target = Path(self._last_save_path) if self._last_save_path else Path("data")
        if not target.exists():
            target = Path("data")
        if not target.exists():
            self.set_status("저장된 데이터 폴더가 없습니다 — 먼저 [저장]을 눌러주세요", "warn")
            return
        try:
            resolved = str(target.resolve())
            if sys.platform == "win32":
                os.startfile(resolved)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", resolved])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", resolved])
            self.set_status(f"폴더 열기 → {resolved}", "ok")
        except Exception as ex:
            self.set_status(f"폴더 열기 실패: {ex}", "error")

    # ── Sliders ────────────────────────────────────────────────────────────
    def setup_sliders(self, slider_defs: list[tuple]) -> None:
        for ax in self._slider_axes:
            self.fig.delaxes(ax)
        self._slider_axes.clear()
        self.sliders.clear()

        n = len(slider_defs)
        if n == 0:
            return

        sd      = _SLD_ZONE
        total_h = n * sd["h"] + (n - 1) * sd["gap"]
        y_bot   = sd["y_top"] - total_h

        for i, (pname, label, vmin, vmax, vinit) in enumerate(slider_defs):
            y = y_bot + (n - 1 - i) * (sd["h"] + sd["gap"])
            ax_sl = self.fig.add_axes(
                [sd["x0"], y, sd["w"], sd["h"]],
                facecolor=PAL["slider_bg"],
            )
            sl = Slider(ax_sl, label, vmin, vmax, valinit=vinit,
                        color=PAL["slider_fg"], track_color=PAL["slider_bg"])
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
            # _refresh()는 Tk 오퍼레이션(after_cancel, FuncAnimation 등)을 포함하므로
            # 반드시 메인 스레드에서 실행해야 한다. after(0, ...) 로 위임한다.
            def _on_main():
                setattr(self.engine.params, name, val)
                self._refresh(pause_after=True)
            try:
                self.fig.canvas.get_tk_widget().after(0, _on_main)
            except Exception:
                # Tk 위젯 접근 불가 시 폴백 (스레드 안전 보장 안 됨)
                setattr(self.engine.params, name, val)
                self._refresh(pause_after=True)
        self._debounce_timer = threading.Timer(0.20, _run)
        self._debounce_timer.start()

    def _refresh(self, pause_after: bool = False):
        self._extend_cancel = True   # 진행 중인 연장 취소
        self._stop_energy_timer()    # 기존 에너지 타이머 정지
        self.build_animation(self.engine.run())
        motion = _MOTION_TITLES.get(self.engine.motion.name, self.engine.motion.name)

        # 파라미터 변경 후 항상 자동 재개 (pause_after 인수는 하위호환 보존)
        self._paused = False
        btn = self._buttons.get("일시정지")
        if btn:
            btn.label.set_text("일시정지")
        self.set_status(f"파라미터 변경됨 — {motion}  |  실시간 재생", "info")

        self.fig.canvas.draw_idle()

    # ── Motion selector (called from main.py) ──────────────────────────────
    def set_motion_radio(self, ax_rect: list, labels: list[str],
                         callback: Callable[[str], None]):
        from matplotlib.widgets import RadioButtons
        ax = self.fig.add_axes(ax_rect, facecolor=PAL["panel_bg"])
        ax.set_title("운동\n유형", fontsize=8, fontweight="bold",
                     color=PAL["text"], pad=4)
        radio = RadioButtons(ax, labels, active=0, activecolor=PAL["accent"])
        for lbl in radio.labels:
            lbl.set_fontsize(7.5)
            lbl.set_color(PAL["text"])
        radio.on_clicked(callback)
        self._motion_radio = radio
        return radio

    # ── Coordinate helper ──────────────────────────────────────────────────
    def _get_display_coords(self, result: SimResult) -> np.ndarray:
        pos = result.positions
        if self._is_3d:
            return pos if pos.shape[1] == 3 else pad_to_3d(pos)
        return pad_to_3d(pos) if pos.shape[1] == 2 else pos

    # ── 2D animation ───────────────────────────────────────────────────────
    def _init_2d_artists(self):
        ax = self.ax_main
        (self.trail_line,) = ax.plot([], [], color=PAL["trail_2d"],
                                     lw=1.8, alpha=0.55)
        (self.obj_dot,)    = ax.plot([], [], "o", color=PAL["dot"],
                                     ms=9, zorder=5)
        (self.arm_line,)   = ax.plot([], [], color=PAL["arm"], lw=2.2, alpha=0.75)
        (self.pivot_dot,)  = ax.plot([0], [0], "s", color=PAL["arm"],
                                     ms=6, zorder=4)
        self.time_text = ax.text(
            0.02, 0.97, "", transform=ax.transAxes,
            fontsize=9, va="top", color=PAL["subtext"],
            bbox=dict(facecolor=PAL["panel_bg"], edgecolor="none",
                      alpha=0.75, pad=2),
        )

    def _anim_init_2d(self):
        self.trail_line.set_data([], [])
        self.obj_dot.set_data([], [])
        self.arm_line.set_data([], [])
        self.time_text.set_text("")
        # 에너지 선은 독립 타이머가 갱신 — blit 아티스트에서 제외
        self.ke_line.set_data([], [])
        self.pe_line.set_data([], [])
        self.etot_line.set_data([], [])
        return self.trail_line, self.obj_dot, self.arm_line, self.time_text

    def _animate_2d(self, _fi: int):
        if self._coords is None or self.result is None:
            return self.trail_line, self.obj_dot, self.arm_line, self.time_text
        if self._paused:
            return self.trail_line, self.obj_dot, self.arm_line, self.time_text
        c = self._coords
        t_all = self.result.timestamps

        # 벽시계 기준 실시간 재생 — 렌더 속도에 무관하게 실제 시간과 동기화
        now = time.perf_counter()
        if self._anim_start_time is None:
            self._anim_start_time = now
        elapsed = now - self._anim_start_time
        t_target = float(t_all[0]) + elapsed

        # 잔여 데이터가 3초 미만이면 선제적으로 연장 트리거
        if float(t_all[-1]) - t_target < 3.0:
            self._start_extension()

        # O(log n) 이진 탐색으로 현재 프레임 인덱스 결정
        fi = int(np.searchsorted(t_all, t_target, side="right")) - 1
        fi = max(0, min(fi, len(c) - 1))
        self._current_fi = fi   # 에너지 타이머가 읽는 현재 프레임

        t0 = max(0, fi - TRAIL_LEN_2D)
        self.trail_line.set_data(c[t0:fi+1, 0], c[t0:fi+1, 1])
        self.obj_dot.set_data([c[fi, 0]], [c[fi, 1]])
        self.time_text.set_text(f"t = {t_all[fi]:.2f} s")
        if self.result.motion_type == "double_pendulum":
            self.arm_line.set_data([0, c[fi, 0]], [0, c[fi, 1]])
        else:
            self.arm_line.set_data([], [])
        return self.trail_line, self.obj_dot, self.arm_line, self.time_text

    # ── 3D animation ───────────────────────────────────────────────────────
    def _init_3d_artists(self):
        ax = self.ax_main
        (self.trail_line,) = ax.plot([], [], [], color=PAL["trail_3d"],
                                     lw=1.4, alpha=0.5)
        (self.obj_dot,)    = ax.plot([], [], [], "o", color=PAL["dot"],
                                     ms=8, zorder=5)
        self.time_text = ax.text2D(
            0.02, 0.97, "", transform=ax.transAxes,
            fontsize=9, va="top", color=PAL["subtext"],
        )

    def _anim_init_3d(self):
        self.trail_line.set_data_3d([], [], [])
        self.obj_dot.set_data_3d([], [], [])
        self.time_text.set_text("")
        return self.trail_line, self.obj_dot, self.time_text

    def _animate_3d(self, _fi: int):
        if self._coords is None or self.result is None:
            return self.trail_line, self.obj_dot, self.time_text
        if self._paused:
            return self.trail_line, self.obj_dot, self.time_text
        c = self._coords
        t_all = self.result.timestamps

        # 벽시계 기준 실시간 재생
        now = time.perf_counter()
        if self._anim_start_time is None:
            self._anim_start_time = now
        elapsed = now - self._anim_start_time
        t_target = float(t_all[0]) + elapsed

        if float(t_all[-1]) - t_target < 3.0:
            self._start_extension()

        fi = int(np.searchsorted(t_all, t_target, side="right")) - 1
        fi = max(0, min(fi, len(c) - 1))
        self._current_fi = fi   # 에너지 타이머가 읽는 현재 프레임

        t0 = max(0, fi - TRAIL_LEN_3D)
        self.trail_line.set_data_3d(c[t0:fi+1, 0], c[t0:fi+1, 1], c[t0:fi+1, 2])
        self.obj_dot.set_data_3d([c[fi, 0]], [c[fi, 1]], [c[fi, 2]])
        self.time_text.set_text(f"t = {t_all[fi]:.2f} s")
        return self.trail_line, self.obj_dot, self.time_text

    # ── Build / rebuild animation ──────────────────────────────────────────
    def build_animation(self, result: SimResult) -> FuncAnimation:
        self.result = result
        is_3d = getattr(self.engine.motion, "is_3d", False)

        self._ensure_axes_type(is_3d)
        self._coords = self._get_display_coords(result)

        self.ax_main.cla()
        self._style_main_axes(result, is_3d)
        self._setup_energy_limits(result)

        if is_3d:
            self._init_3d_artists()
        else:
            self._init_2d_artists()

        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
            try:
                self.anim._stop()   # disconnects resize/close canvas callbacks
            except Exception:
                pass
            self.anim = None

        # 연장 상태 초기화
        self._extend_cancel = False
        self._extending = False
        self._anim_start_time = None       # 첫 프레임에서 벽시계 기준 설정
        self._energy_lim_last_t = 0.0     # 에너지 축 범위 스로틀 초기화

        interval = max(8, int(1000 / self.target_fps))
        # 실시간 재생: 1 tick(=interval ms)에 몇 데이터 포인트를 표시할지 계산
        dt_real = interval / 1000.0
        self._frame_step = max(1.0, dt_real / max(result.dt, 1e-9))

        fn_anim  = self._animate_3d  if is_3d else self._animate_2d
        fn_init  = self._anim_init_3d if is_3d else self._anim_init_2d
        self.anim = FuncAnimation(
            self.fig, fn_anim,
            init_func=fn_init,
            frames=itertools.count(),
            interval=interval,
            blit=True,
            repeat=False,
            cache_frame_data=False,
        )
        self._start_energy_timer()   # 에너지 그래프 독립 갱신 시작
        return self.anim

    def _style_main_axes(self, result: SimResult, is_3d: bool):
        ax    = self.ax_main
        c     = self._coords
        title = _MOTION_TITLES.get(result.motion_type, result.motion_type)
        M     = 0.10

        if is_3d:
            ax.set_facecolor(PAL["panel_bg"])
            for i, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
                lo, hi = c[:, i].min(), c[:, i].max()
                sp = max(hi - lo, 1e-3)
                setter(lo - M * sp, hi + M * sp)
            ax.set_xlabel("x", fontsize=8, color=PAL["text"], labelpad=4)
            ax.set_ylabel("y", fontsize=8, color=PAL["text"], labelpad=4)
            ax.set_zlabel("z", fontsize=8, color=PAL["text"], labelpad=4)
            ax.set_title(title, fontsize=10, fontweight="bold",
                         color=PAL["text"], pad=10)
        else:
            ax.set_facecolor(PAL["panel_bg"])
            ax.grid(True, linewidth=0.5, alpha=0.6, color=PAL["grid"])
            lo0, hi0 = c[:, 0].min(), c[:, 0].max()
            lo1, hi1 = c[:, 1].min(), c[:, 1].max()
            sp0 = max(hi0 - lo0, 1e-3)
            sp1 = max(hi1 - lo1, 1e-3)
            ax.set_xlim(lo0 - M * sp0, hi0 + M * sp0)
            ax.set_ylim(lo1 - M * sp1, hi1 + M * sp1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x [m]", fontsize=9, color=PAL["text"])
            ax.set_ylabel("y [m]", fontsize=9, color=PAL["text"])
            ax.set_title(title, fontsize=10, fontweight="bold", color=PAL["text"])

    def show(self):
        plt.show()
