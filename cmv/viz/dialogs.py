"""Tkinter modal dialogs for CMV: training config and model selection.

wait_window() / wait_variable() 은 Matplotlib TkAgg 애니메이션과 함께
사용하면 중첩 이벤트 루프 충돌이 발생한다.
따라서 모든 다이얼로그는 콜백(on_done) 방식으로 구현한다 —
창이 열린 뒤 즉시 반환하며, 사용자가 확인/취소할 때 on_done 이 호출된다.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

# ── Color palette (matches PAL in style.py) ─────────────────────────────────
_BG    = "#1E2128"
_PANEL = "#2D3139"
_FG    = "#F8FAFC"
_SUB   = "#94A3B8"
_ENTRY = "#3B4252"
_HOVER = "#4B5262"
_ACC   = "#6366F1"
_ACC_H = "#4F52C1"


# ── Helpers ──────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _sanitize_name(name: str) -> str:
    """Strip filesystem-illegal chars; keep Korean, alphanumeric, spaces, - _"""
    sanitized = re.sub(r'[\\/:*?"<>|\x00]', "_", name.strip())
    sanitized = sanitized.strip(". ")
    return sanitized or "모델"


def _center(win: tk.Toplevel) -> None:
    # withdraw 상태에서 레이아웃을 계산한 뒤 정확한 크기로 deiconify한다.
    win.update_idletasks()
    w  = win.winfo_reqwidth()
    h  = win.winfo_reqheight()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    win.geometry(f"{w}x{h}+{max(0,(sw-w)//2)}+{max(0,(sh-h)//2)}")
    win.deiconify()
    win.update_idletasks()  # grab_set() 전에 창이 실제로 맵핑되도록 보장
    win.lift()
    win.focus_force()


def _btn(parent, text, command, primary=False, **kw):
    bg = _ACC if primary else _ENTRY
    hv = _ACC_H if primary else _HOVER
    b = tk.Button(parent, text=text, command=command,
                  bg=bg, fg=_FG, relief="flat",
                  font=("Helvetica", 9, "bold" if primary else "normal"),
                  padx=14, pady=4,
                  activebackground=hv, activeforeground=_FG, **kw)
    return b


# ── Model registry ────────────────────────────────────────────────────────────
def list_saved_models(motion_type: str) -> list[Path]:
    """Return list of model dirs (newest first) that contain model.pt."""
    base = Path("models") / motion_type
    if not base.exists():
        return []
    dirs = [d for d in base.iterdir()
            if d.is_dir() and (d / "model.pt").exists()]

    def _key(d: Path) -> str:
        return _load_json(d / "model_meta.json").get("created_at", d.name)

    return sorted(dirs, key=_key, reverse=True)


# ── Training config dialog ────────────────────────────────────────────────────
def show_train_config_dialog(
    motion_type: str,
    sessions: list[Path],
    on_done,           # callable(model_name: str | None, paths: list | None)
) -> None:
    """
    Non-blocking modal: enter model name + choose data sessions.
    Returns immediately; calls on_done(model_name, paths) on confirm
    or on_done(None, None) on cancel / close.
    """
    win = tk.Toplevel()
    win.withdraw()  # 레이아웃 계산 전까지 숨김 — _center에서 deiconify
    win.title("ML 학습 설정")
    win.configure(bg=_BG)
    win.resizable(False, False)

    # ── title ─────────────────────────────────────────────────────────────
    tk.Label(win, text=f"ML 학습 설정  —  {motion_type}",
             bg=_BG, fg=_FG, font=("Helvetica", 11, "bold")
             ).pack(fill="x", padx=14, pady=(12, 8))

    # ── model name ────────────────────────────────────────────────────────
    nf = tk.Frame(win, bg=_PANEL)
    nf.pack(fill="x", padx=14, pady=(0, 10))

    tk.Label(nf, text="모델 이름 :", bg=_PANEL, fg=_FG,
             font=("Helvetica", 9)
             ).grid(row=0, column=0, padx=(10, 6), pady=8, sticky="w")

    name_var = tk.StringVar(value="기본모델")
    name_ent = tk.Entry(nf, textvariable=name_var,
                        bg=_ENTRY, fg=_FG, insertbackground=_FG,
                        font=("Helvetica", 9), width=28,
                        relief="flat", bd=4)
    name_ent.grid(row=0, column=1, padx=(0, 10), pady=8, sticky="ew")
    nf.columnconfigure(1, weight=1)

    tk.Label(nf, text="※ 같은 이름으로 학습하면 기존 모델을 덮어씁니다.",
             bg=_PANEL, fg=_SUB, font=("Helvetica", 8)
             ).grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 6), sticky="w")

    # ── session list header ───────────────────────────────────────────────
    hf = tk.Frame(win, bg=_BG)
    hf.pack(fill="x", padx=14, pady=(0, 2))

    tk.Label(hf, text="학습 데이터 선택",
             bg=_BG, fg=_FG, font=("Helvetica", 9, "bold")
             ).pack(side="left")

    lb_ref: list = []

    def _select_all():
        lb_ref[0].selection_set(0, "end")

    def _deselect_all():
        lb_ref[0].selection_clear(0, "end")

    _btn(hf, "전체 해제", _deselect_all).pack(side="right", padx=(4, 0))
    _btn(hf, "전체 선택", _select_all).pack(side="right", padx=4)

    # ── listbox ───────────────────────────────────────────────────────────
    lf = tk.Frame(win, bg=_PANEL)
    lf.pack(fill="both", expand=True, padx=14, pady=(0, 4))

    sb = tk.Scrollbar(lf, bg=_PANEL, troughcolor=_BG)
    sb.pack(side="right", fill="y")

    n_rows = min(10, max(4, len(sessions)))
    lb = tk.Listbox(lf, selectmode="multiple", height=n_rows,
                    bg=_PANEL, fg=_FG,
                    selectbackground=_ACC, selectforeground=_FG,
                    font=("Helvetica", 9), activestyle="none",
                    yscrollcommand=sb.set, relief="flat", bd=0)
    lb.pack(side="left", fill="both", expand=True, padx=(4, 0))
    sb.config(command=lb.yview)
    lb_ref.append(lb)

    for s in sessions:
        meta    = _load_json(s / "metadata.json")
        created = meta.get("created_at", s.name)
        n_pts   = meta.get("n_points", "?")
        lb.insert("end", f"  {created}   ({n_pts} pts)")
        lb.selection_set(lb.size() - 1)   # all selected by default

    if not sessions:
        lb.insert("end", "  (저장된 세션 없음)")

    tk.Label(win, text="Ctrl+클릭으로 다중 선택 / 위 버튼으로 전체 선택·해제",
             bg=_BG, fg=_SUB, font=("Helvetica", 8)
             ).pack(anchor="w", padx=14, pady=(0, 6))

    # ── buttons ───────────────────────────────────────────────────────────
    bf = tk.Frame(win, bg=_BG)
    bf.pack(fill="x", padx=14, pady=(0, 12))

    def _cancel():
        win.destroy()
        on_done(None, None)

    def _ok():
        name = _sanitize_name(name_var.get())
        sel_idx = lb.curselection()
        if not sel_idx and sessions:
            messagebox.showwarning("선택 오류",
                                   "학습할 세션을 하나 이상 선택하세요.",
                                   parent=win)
            return
        paths = [sessions[i] for i in sel_idx]
        win.destroy()
        on_done(name, paths)

    _btn(bf, "취소", _cancel).pack(side="right", padx=(4, 0))
    _btn(bf, "학습 시작", _ok, primary=True).pack(side="right", padx=4)

    win.protocol("WM_DELETE_WINDOW", _cancel)
    win.bind("<Return>", lambda _e: _ok())
    win.bind("<Escape>", lambda _e: _cancel())

    _center(win)
    win.grab_set()  # 창이 완전히 표시된 뒤에 grab 설정
    name_ent.focus_set()
    name_ent.select_range(0, "end")
    # 반환 — 이벤트 루프 블로킹 없음. on_done 이 결과를 전달한다.


# ── Model selection dialog ────────────────────────────────────────────────────
def show_model_select_dialog(
    motion_type: str,
    on_done,           # callable(model_dir: Path | None)
) -> None:
    """
    Non-blocking modal: choose one of the saved named models.
    Returns immediately; calls on_done(model_dir) on confirm
    or on_done(None) on cancel / close.
    """
    model_dirs = list_saved_models(motion_type)

    win = tk.Toplevel()
    win.withdraw()  # 레이아웃 계산 전까지 숨김 — _center에서 deiconify
    win.title("모델 선택")
    win.configure(bg=_BG)
    win.resizable(False, False)

    tk.Label(win, text=f"모델 선택  —  {motion_type}",
             bg=_BG, fg=_FG, font=("Helvetica", 11, "bold")
             ).pack(fill="x", padx=14, pady=(12, 8))

    if not model_dirs:
        tk.Label(win,
                 text="저장된 모델이 없습니다.\n"
                      "[시뮬레이션] 탭에서 ML 학습을 먼저 진행하세요.",
                 bg=_BG, fg=_SUB,
                 font=("Helvetica", 9), justify="center"
                 ).pack(padx=24, pady=12)

        def _close_empty():
            win.destroy()
            on_done(None)

        _btn(win, "닫기", _close_empty).pack(pady=(0, 14))
        win.protocol("WM_DELETE_WINDOW", _close_empty)
        _center(win)
        win.grab_set()
        return

    tk.Label(win, text="불러올 모델을 선택하세요 (더블클릭으로 로드):",
             bg=_BG, fg=_FG, font=("Helvetica", 9, "bold")
             ).pack(anchor="w", padx=14, pady=(0, 2))

    lf = tk.Frame(win, bg=_PANEL)
    lf.pack(fill="both", expand=True, padx=14, pady=(0, 6))

    sb = tk.Scrollbar(lf, bg=_PANEL, troughcolor=_BG)
    sb.pack(side="right", fill="y")

    n_rows = min(10, max(4, len(model_dirs)))
    lb = tk.Listbox(lf, selectmode="single", height=n_rows,
                    bg=_PANEL, fg=_FG,
                    selectbackground=_ACC, selectforeground=_FG,
                    font=("Helvetica", 9), activestyle="none",
                    yscrollcommand=sb.set, relief="flat", bd=0)
    lb.pack(side="left", fill="both", expand=True, padx=(4, 0))
    sb.config(command=lb.yview)

    for d in model_dirs:
        meta    = _load_json(d / "model_meta.json")
        created = meta.get("created_at", "")
        n_sess  = meta.get("n_sessions", "?")
        n_pts   = meta.get("n_points", "?")
        feat    = meta.get("feature_cols", [])
        dim_tag = f"  [{len(feat)}D]" if feat else ""
        parts = []
        if created:
            parts.append(created)
        if n_sess != "?":
            parts.append(f"세션 {n_sess}개")
        if n_pts != "?":
            n_str = f"{int(n_pts):,}" if str(n_pts).isdigit() else str(n_pts)
            parts.append(f"{n_str} pts")
        info = "   |   ".join(parts)
        lb.insert("end", f"  {d.name}{dim_tag}   —   {info}")

    lb.selection_set(0)

    def _cancel():
        win.destroy()
        on_done(None)

    def _ok():
        sel = lb.curselection()
        if not sel:
            return
        result_dir = model_dirs[sel[0]]
        win.destroy()
        on_done(result_dir)

    lb.bind("<Double-Button-1>", lambda _e: _ok())
    win.bind("<Return>", lambda _e: _ok())
    win.bind("<Escape>", lambda _e: _cancel())
    win.protocol("WM_DELETE_WINDOW", _cancel)

    bf = tk.Frame(win, bg=_BG)
    bf.pack(fill="x", padx=14, pady=(4, 12))
    _btn(bf, "취소", _cancel).pack(side="right", padx=(4, 0))
    _btn(bf, "로드", _ok, primary=True).pack(side="right", padx=4)

    _center(win)
    win.grab_set()
    lb.focus_set()
    # 반환 — 이벤트 루프 블로킹 없음. on_done 이 결과를 전달한다.
