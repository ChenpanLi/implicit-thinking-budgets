"""The figure style shared by every analysis script: fonts, colours, sizes."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

FONTS = Path(__file__).resolve().parent / "fonts"

TEXT = "#2c3e5c"
TICK = "#6b7a90"
BG_AX = "#f5f6f8"
GRID = "#e3e6eb"

# One continuous CIELCh ramp, sampled at equal perceptual arclength; Opus is the anchor.
MODEL_COLORS = {
    "r1-qwen-distill-7b": "#7A1F2B",
    "r1-llama-distill-8b": "#C8413F",
    "r1-qwen3-8b": "#E6B33D",
    "qwen3-4b-thinking": "#8FA9C9",
    "qwen3-8b": "#6FB3D9",
    "qwen3-14b": "#2F6FB0",
    "qwen3-8-27b": "#1B2F5E",
    "claude-opus-5": "#D97757",
}

# Isolated trace-length bins: Low (<= 20%), Medium (20-80%), High (> 80%).
BIN_COLORS = ["#2F6FB0", "#E6B33D", "#C8413F"]
BIN_LABELS = ["Low", "Med.", "High"]


def pt_to_in(pt: float) -> float:
    """TeX points to inches (1 in = 72.27 pt)."""
    return pt / 72.27


def use_paper_style() -> None:
    """Light Roboto on a pale grey plot area, ticks pointing out, no top/right spines."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Roboto", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.weight": "light",
        "axes.titleweight": "light",
        "axes.labelweight": "light",
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": TEXT,
        "axes.facecolor": BG_AX,
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelcolor": TICK,
        "ytick.labelcolor": TICK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
    })
    for name in ("Roboto-Light.ttf", "Roboto-SemiBold.ttf"):
        font_manager.fontManager.addfont(str(FONTS / name))
    plt.rcParams.update({
        "font.family": "Roboto",
        "font.weight": "300",
        "axes.titleweight": "300",
        "axes.labelweight": "300",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Roboto:light",
        "mathtext.bf": "Roboto:bold",
    })
