"""Global style configuration for CMV."""
from __future__ import annotations
import matplotlib
import matplotlib.font_manager as fm


# ── Color palette ─────────────────────────────────────────────────────────
PAL = {
    "fig_bg":      "#F0F4F8",   # figure background
    "panel_bg":    "#FFFFFF",   # axes / panel background
    "border":      "#CBD5E0",   # border / separator
    "accent":      "#3B82F6",   # primary blue
    "accent2":     "#6366F1",   # purple accent (3D)
    "trail_2d":    "#3B82F6",   # 2D trail colour
    "trail_3d":    "#8B5CF6",   # 3D trail colour
    "dot":         "#EF4444",   # moving object dot
    "ke":          "#F97316",   # kinetic energy
    "pe":          "#3B82F6",   # potential energy
    "etot":        "#22C55E",   # total energy
    "text":        "#1E293B",   # primary text
    "subtext":     "#64748B",   # secondary text
    "slider_bg":   "#E2E8F0",   # slider rail
    "slider_fg":   "#3B82F6",   # slider handle
    "btn_bg":      "#EFF6FF",   # button background
    "btn_hover":   "#DBEAFE",   # button hover
    "grid":        "#E2E8F0",   # gridlines
    "arm":         "#475569",   # pendulum rod colour
}


def apply_style() -> str:
    """Apply global matplotlib style and return detected Korean font name."""
    # ── Korean font ────────────────────────────────────────────────────────
    korean_candidates = [
        "Malgun Gothic", "NanumGothic", "Apple SD Gothic Neo",
        "AppleGothic", "NanumBarunGothic", "Gulim", "Batang",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    font_found = None
    for font in korean_candidates:
        if font in available:
            font_found = font
            break

    matplotlib.rcParams["font.family"] = font_found if font_found else "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False

    # ── Global rcParams ────────────────────────────────────────────────────
    matplotlib.rcParams.update({
        "figure.facecolor":  PAL["fig_bg"],
        "axes.facecolor":    PAL["panel_bg"],
        "axes.edgecolor":    PAL["border"],
        "axes.labelcolor":   PAL["text"],
        "axes.titlecolor":   PAL["text"],
        "axes.grid":         True,
        "grid.color":        PAL["grid"],
        "grid.linewidth":    0.6,
        "grid.alpha":        0.8,
        "xtick.color":       PAL["subtext"],
        "ytick.color":       PAL["subtext"],
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "axes.labelsize":    9,
        "axes.titlesize":    10,
        "axes.titlepad":     8,
        "legend.fontsize":   8,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  PAL["border"],
        "lines.linewidth":   1.8,
        "text.color":        PAL["text"],
        "savefig.facecolor": PAL["fig_bg"],
    })

    return font_found or "DejaVu Sans"
