from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button as MplButton
from scipy.integrate import solve_ivp

from cmv.physics.base import SimResult
from cmv.ml.trainer import MLTrainer
from cmv.viz.style import PAL


_MOTION_TITLES = {
    "pendulum":           "단진자",
    "double_pendulum":    "이중진자",
    "projectile":         "포물체 운동",
    "shm":                "단순조화 / 감쇠 / 강제진동",
    "circular":           "등속 원운동",
    "kepler":             "케플러 궤도",
    "spherical_pendulum": "구면진자",
    "lorenz":             "로렌츠 어트랙터",
    "magnetic_particle":  "자기장 속 대전입자",
}

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


@dataclass
class ComparisonMetrics:
    position_rmse: float
    velocity_rmse: float
    energy_error: float
    param_errors: dict[str, float] = field(default_factory=dict)


class ResultComparator:
    def __init__(self, trainer: MLTrainer, true_result: SimResult) -> None:
        self.trainer = trainer
        self.true_result = true_result

    # ── Prediction ─────────────────────────────────────────────────────────
    def predict_trajectory(
        self,
        initial_state: np.ndarray,
        t_span: tuple[float, float],
        n_points: int = 500,
    ) -> SimResult:
        bundle = self.trainer.bundle
        scaler_X = bundle.scaler_X
        feat_cols = bundle.metadata.get("feature_cols", ["x", "y", "vx", "vy"])

        initial_state = np.asarray(initial_state).ravel()
        if len(initial_state) == 2:
            idx0 = 0
            pos0 = initial_state
            vel0 = self.true_result.velocities[idx0]
            y0 = [float(pos0[0]), float(pos0[1]),
                  float(vel0[0]), float(vel0[1])]
        else:
            y0 = [float(initial_state[0]), float(initial_state[1]),
                  float(initial_state[2]), float(initial_state[3])]

        def ml_ode(t, state):
            x, y, vx, vy = state
            vals = {"x": x, "y": y, "vx": vx, "vy": vy, "t": t}
            raw = np.array([[vals[c] for c in feat_cols]])
            raw_scaled = scaler_X.transform(raw)
            accel = self.trainer.predict(raw_scaled)
            ax, ay = accel[0, 0], accel[0, 1]
            return [vx, vy, ax, ay]

        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(ml_ode, t_span, y0, t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-8)

        pos = np.column_stack([sol.y[0], sol.y[1]])
        vel = np.column_stack([sol.y[2], sol.y[3]])
        t_out = sol.t

        return SimResult(
            motion_type=self.true_result.motion_type + "_predicted",
            timestamps=t_out,
            positions=pos,
            velocities=vel,
            accelerations=np.zeros_like(pos),
            energies=np.zeros((len(t_out), 3)),
            params=self.true_result.params.copy(),
            dt=float(np.mean(np.diff(t_out))) if len(t_out) > 1 else 0.0,
        )

    # ── Metrics ────────────────────────────────────────────────────────────
    def compute_metrics(self, predicted: SimResult) -> ComparisonMetrics:
        n = min(len(predicted.timestamps), len(self.true_result.timestamps))
        pos_rmse = float(np.sqrt(
            ((predicted.positions[:n] - self.true_result.positions[:n])**2).mean()
        ))
        vel_rmse = float(np.sqrt(
            ((predicted.velocities[:n] - self.true_result.velocities[:n])**2).mean()
        ))
        # 진폭 대비 정규화 RMSE (0~1 사이 값; ML 예측은 에너지 계산 불가)
        amp = float(self.true_result.positions[:n].std())
        norm_rmse = pos_rmse / max(amp, 1e-9)

        return ComparisonMetrics(
            position_rmse=pos_rmse,
            velocity_rmse=vel_rmse,
            energy_error=norm_rmse,   # 필드 재사용: 진폭 대비 RMSE 비율
        )

    # ── Comparison window ──────────────────────────────────────────────────
    def show_window(
        self,
        predicted: SimResult,
        metrics: ComparisonMetrics,
        save_dir: Path = Path(__file__).resolve().parent.parent.parent / "comparisons",
    ) -> plt.Figure:
        from matplotlib.animation import FuncAnimation

        tr   = self.true_result
        pred = predicted
        n    = min(len(pred.timestamps), len(tr.timestamps))

        pos_pred = pred.positions[:n].copy()
        pos_pred[~np.isfinite(pos_pred)] = 0.0
        err = np.sqrt(((pos_pred - tr.positions[:n]) ** 2).sum(axis=1))

        t_tr   = tr.timestamps[:n]
        t_pred = pred.timestamps[:n]
        title  = _MOTION_TITLES.get(tr.motion_type, tr.motion_type)

        # ── Figure  (17×9, 2 rows × 3 cols — left col = animation) ──────
        fig = plt.figure(figsize=(17, 9), facecolor=PAL["fig_bg"])
        try:
            fig.canvas.manager.set_window_title(f"ML 결과 비교 — {title}")
        except Exception:
            pass
        fig.suptitle(
            f"ML 결과 비교 — {title}",
            fontsize=13, fontweight="bold", color=PAL["text"], y=0.97,
        )

        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[1.35, 1, 1],
            hspace=0.46, wspace=0.38,
            top=0.90, bottom=0.20, left=0.05, right=0.97,
        )
        ax_anim = fig.add_subplot(gs[0:2, 0])   # full-height left column
        ax_traj = fig.add_subplot(gs[0, 1])
        ax_time = fig.add_subplot(gs[0, 2])
        ax_err  = fig.add_subplot(gs[1, 1])
        ax_info = fig.add_subplot(gs[1, 2])

        # ── Panel A: real-time animation ─────────────────────────────────
        all_pos = np.vstack([tr.positions[:n], pos_pred])
        M  = 0.12
        xl = all_pos[:, 0].min(), all_pos[:, 0].max()
        yl = all_pos[:, 1].min(), all_pos[:, 1].max()
        xsp = max(xl[1] - xl[0], 1e-3)
        ysp = max(yl[1] - yl[0], 1e-3)
        ax_anim.set_xlim(xl[0] - M * xsp, xl[1] + M * xsp)
        ax_anim.set_ylim(yl[0] - M * ysp, yl[1] + M * ysp)
        ax_anim.set_aspect("equal", adjustable="box")
        ax_anim.set_title("실시간 비교", fontsize=10, fontweight="bold")
        ax_anim.set_xlabel("x [m]", fontsize=9)
        ax_anim.set_ylabel("y [m]", fontsize=9)
        ax_anim.grid(True, alpha=0.4)

        # static full-trajectory ghost lines for context
        ax_anim.plot(tr.positions[:n, 0], tr.positions[:n, 1],
                     color=PAL["accent"], lw=0.8, alpha=0.20)
        ax_anim.plot(pos_pred[:, 0], pos_pred[:, 1],
                     color=PAL["dot"], lw=0.8, alpha=0.20, ls="--")

        TRAIL = 80
        trail_tr, = ax_anim.plot([], [], color=PAL["accent"],
                                 lw=2.0, alpha=0.70, label="실제")
        trail_pr, = ax_anim.plot([], [], color=PAL["dot"],
                                 lw=1.8, alpha=0.70, ls="--", label="ML 예측")
        dot_tr,   = ax_anim.plot([], [], "o", color=PAL["accent"],
                                 ms=10, zorder=6)
        dot_pr,   = ax_anim.plot([], [], "o", color=PAL["dot"],
                                 ms=10, zorder=6)
        info_txt  = ax_anim.text(
            0.04, 0.97, "",
            transform=ax_anim.transAxes,
            fontsize=8.5, va="top", color=PAL["subtext"],
            bbox=dict(facecolor=PAL["panel_bg"], edgecolor="none",
                      alpha=0.80, pad=3),
        )
        ax_anim.legend(fontsize=8.5, loc="lower right")

        _anim_artists = (trail_tr, trail_pr, dot_tr, dot_pr, info_txt)

        def _anim_init():
            trail_tr.set_data([], [])
            trail_pr.set_data([], [])
            dot_tr.set_data([], [])
            dot_pr.set_data([], [])
            info_txt.set_text("")
            return _anim_artists

        def _anim_update(fi):
            fi  = min(fi, n - 1)
            t0  = max(0, fi - TRAIL)
            trail_tr.set_data(tr.positions[t0:fi+1, 0],
                              tr.positions[t0:fi+1, 1])
            trail_pr.set_data(pos_pred[t0:fi+1, 0],
                              pos_pred[t0:fi+1, 1])
            dot_tr.set_data([tr.positions[fi, 0]], [tr.positions[fi, 1]])
            dot_pr.set_data([pos_pred[fi, 0]],     [pos_pred[fi, 1]])
            info_txt.set_text(
                f"t = {t_pred[fi]:.2f} s\n|Δr| = {err[fi]:.4f} m"
            )
            return _anim_artists

        anim = FuncAnimation(
            fig, _anim_update, init_func=_anim_init,
            frames=n, interval=33, blit=True, repeat=True,
        )
        fig._anim = anim   # keep reference — prevents GC

        # ── Panel 1: trajectory overlay (static) ─────────────────────────
        ax_traj.plot(tr.positions[:, 0], tr.positions[:, 1],
                     color=PAL["accent"], lw=2.0, label="실제", alpha=0.85)
        ax_traj.plot(pos_pred[:, 0], pos_pred[:, 1],
                     color=PAL["dot"], lw=1.8, ls="--", label="ML 예측", alpha=0.85)
        ax_traj.plot(*tr.positions[0], "o", color="#22C55E",
                     ms=8, zorder=5, label="시작점")
        ax_traj.set_xlabel("x [m]", fontsize=9)
        ax_traj.set_ylabel("y [m]", fontsize=9)
        ax_traj.set_title("궤적 비교", fontsize=10, fontweight="bold")
        ax_traj.legend(fontsize=8)
        ax_traj.set_aspect("equal", adjustable="box")
        ax_traj.grid(True, alpha=0.4)

        # ── Panel 2: x(t) and y(t) time series ───────────────────────────
        ax_time.plot(t_tr,   tr.positions[:n, 0],
                     color=PAL["accent"], lw=1.6, label="x (실제)")
        ax_time.plot(t_pred, pos_pred[:, 0],
                     color=PAL["accent"], lw=1.4, ls="--", alpha=0.75, label="x (ML)")
        ax_time.plot(t_tr,   tr.positions[:n, 1],
                     color=PAL["ke"],    lw=1.6, label="y (실제)")
        ax_time.plot(t_pred, pos_pred[:, 1],
                     color=PAL["ke"],    lw=1.4, ls="--", alpha=0.75, label="y (ML)")
        ax_time.set_xlabel("t [s]", fontsize=9)
        ax_time.set_ylabel("위치 [m]", fontsize=9)
        ax_time.set_title("위치 시계열 비교", fontsize=10, fontweight="bold")
        ax_time.legend(fontsize=7.5, ncol=2)
        ax_time.grid(True, alpha=0.4)

        # ── Panel 3: position error |Δr(t)| ──────────────────────────────
        ax_err.plot(t_pred, err, color=PAL["dot"], lw=1.8)
        ax_err.fill_between(t_pred, err, alpha=0.20, color=PAL["dot"])
        ax_err.axhline(float(err.mean()), color=PAL["subtext"],
                       lw=1.0, ls=":", label=f"평균 {err.mean():.4f} m")
        ax_err.set_xlabel("t [s]", fontsize=9)
        ax_err.set_ylabel("|Δr| [m]", fontsize=9)
        ax_err.set_title("위치 오차 (시간)", fontsize=10, fontweight="bold")
        ax_err.legend(fontsize=8)
        ax_err.grid(True, alpha=0.4)

        # ── Panel 4: metrics summary ──────────────────────────────────────
        ax_info.axis("off")
        ratio = metrics.energy_error   # 진폭 대비 정규화 RMSE
        if ratio < 0.10:
            verdict, vcol = "● 우수 (Good)", "#16A34A"
        elif ratio < 0.35:
            verdict, vcol = "● 보통 (Fair)", "#D97706"
        else:
            verdict, vcol = "● 불량 (Poor)", "#DC2626"

        rows = [
            ("위치 RMSE",        f"{metrics.position_rmse:.4f} m"),
            ("속도 RMSE",        f"{metrics.velocity_rmse:.4f} m/s"),
            ("정규화 RMSE",      f"{ratio * 100:.2f} %"),
            ("최대 오차",        f"{float(err.max()):.4f} m"),
            ("평균 오차",        f"{float(err.mean()):.4f} m"),
        ]
        ax_info.add_patch(mpatches.FancyBboxPatch(
            (0.04, 0.04), 0.92, 0.92,
            transform=ax_info.transAxes,
            boxstyle="round,pad=0.02",
            linewidth=1.2, edgecolor=PAL["border"],
            facecolor=PAL["panel_bg"], zorder=0,
        ))
        ax_info.text(0.50, 0.93, "성능 평가",
                     transform=ax_info.transAxes,
                     fontsize=11, fontweight="bold", ha="center", va="top",
                     color=PAL["text"])
        y0 = 0.76
        for label, val in rows:
            ax_info.text(0.13, y0, label + ":",
                         transform=ax_info.transAxes,
                         fontsize=9, ha="left", va="top", color=PAL["subtext"])
            ax_info.text(0.87, y0, val,
                         transform=ax_info.transAxes,
                         fontsize=9, ha="right", va="top", fontweight="bold",
                         color=PAL["text"])
            y0 -= 0.135
        ax_info.text(0.50, 0.10, verdict,
                     transform=ax_info.transAxes,
                     fontsize=11, fontweight="bold", ha="center", va="top",
                     color=vcol)

        # ── Save button ───────────────────────────────────────────────────
        btn_ax = fig.add_axes(
            [0.38, 0.07, 0.24, 0.07],
            facecolor=PAL["btn_bg"],
        )
        btn_save = MplButton(btn_ax, "결과 저장",
                             color=PAL["btn_bg"], hovercolor=PAL["btn_hover"])
        btn_save.label.set_fontsize(10)
        btn_save.label.set_fontweight("bold")
        btn_save.label.set_color(PAL["text"])

        status_txt = fig.text(
            0.50, 0.025, "저장 버튼을 눌러 PNG · JSON · npy를 저장합니다",
            ha="center", va="center", fontsize=10,
            fontweight="bold",
            color=PAL["subtext"],
            bbox=dict(facecolor=_STATUS_BG["info"], edgecolor=PAL["border"],
                      boxstyle="round,pad=0.45", linewidth=1.5, alpha=0.95),
            transform=fig.transFigure, zorder=10,
        )

        def _set_win_status(msg: str, kind: str = "info"):
            status_txt.set_text(msg)
            status_txt.set_color(_STATUS_COLOR[kind])
            patch = status_txt.get_bbox_patch()
            if patch is not None:
                patch.set_facecolor(_STATUS_BG[kind])
            # draw() + flush_events(): blit 재개 전에 화면에 실제로 반영
            fig.canvas.draw()
            try:
                fig.canvas.flush_events()
            except Exception:
                pass

        def _on_save(_):
            # 애니메이션 중단 (blit 배경 덮어쓰기 방지)
            anim_src = getattr(fig._anim, "event_source", None) if hasattr(fig, "_anim") else None
            if anim_src is not None:
                anim_src.stop()

            _set_win_status("저장 중...", "warn")

            save_ok   = False
            out_path  = None
            error_msg = ""
            try:
                out_path = self._save_results(fig, pred, metrics, save_dir)
                save_ok  = True
            except Exception as ex:
                error_msg = str(ex)

            if save_ok:
                resolved = str(Path(out_path).resolve())
                # 경로가 길면 status 텍스트가 잘릴 수 있으므로 말줄임 처리
                short = resolved if len(resolved) <= 60 else "..." + resolved[-57:]
                _set_win_status(f"저장 완료 → {short}", "ok")
                # blit이 상태 텍스트를 덮어쓰는 경우를 대비한 명시적 팝업
                try:
                    import tkinter.messagebox as _msgbox
                    _tk_top = fig.canvas.get_tk_widget().winfo_toplevel()
                    _msgbox.showinfo(
                        "저장 완료",
                        f"비교 결과가 저장됐습니다.\n\n저장 위치:\n{resolved}",
                        parent=_tk_top,
                    )
                except Exception:
                    pass
            else:
                _set_win_status(f"저장 실패: {error_msg}", "error")
                try:
                    import tkinter.messagebox as _msgbox
                    _tk_top = fig.canvas.get_tk_widget().winfo_toplevel()
                    _msgbox.showerror(
                        "저장 실패",
                        f"저장 중 오류가 발생했습니다:\n\n{error_msg}",
                        parent=_tk_top,
                    )
                except Exception:
                    pass

            if anim_src is not None:
                anim_src.start()

        btn_save.on_clicked(_on_save)

        plt.show(block=False)
        return fig

    # ── Save helper ────────────────────────────────────────────────────────
    def _save_results(
        self,
        fig: plt.Figure,
        predicted: SimResult,
        metrics: ComparisonMetrics,
        save_dir: Path,
    ) -> Path:
        ts  = time.strftime("%Y%m%d_%H%M%S")
        out = Path(save_dir) / self.true_result.motion_type / ts
        out.mkdir(parents=True, exist_ok=True)

        fig.savefig(out / "comparison.png", dpi=150, bbox_inches="tight")

        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "motion_type":         self.true_result.motion_type,
                "position_rmse_m":     metrics.position_rmse,
                "velocity_rmse_ms":    metrics.velocity_rmse,
                "energy_error_frac":   metrics.energy_error,
            }, f, indent=2, ensure_ascii=False)

        np.save(str(out / "true_positions.npy"),      self.true_result.positions)
        np.save(str(out / "predicted_positions.npy"), predicted.positions)
        np.save(str(out / "timestamps.npy"),          predicted.timestamps)

        return out

    # ── Legacy shim ────────────────────────────────────────────────────────
    def plot_overlay(
        self,
        predicted: SimResult,
        metrics: ComparisonMetrics,
        ax=None,
    ) -> plt.Figure:
        return self.show_window(predicted, metrics)
