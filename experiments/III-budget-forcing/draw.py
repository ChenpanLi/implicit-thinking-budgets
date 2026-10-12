"""Drawing helpers shared by plot_main.py and plot_grids.py (not a script).

Both figures put the composed points (circles, joined in k order, one trajectory per
isolated trace-length bin) beside the budget-forced points (pale diamonds) on the same
axes: token reduction `1 - r_i(k)` against the accuracy change in percentage points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.transforms import Bbox

from llm import paths
from llm.plotting import BIN_COLORS, BIN_LABELS

OUT = paths.results("III-budget-forcing")
KS_SHOWN = [2, 3, 4, 6, 8, 10, 12, 15, 18, 20]
GROUPS = ["low", "medium", "high"]
WEAK_GREY = "#adb5bd"
K_LABEL_COLOR = "#42505c"
# The y axis turns symmetric-log when the forced points reach more than this many times
# past the composed range (see symlog_y).
SYMLOG_RATIO = 3
K_LABEL_TRIES = [(0, 8), (0, -9), (9, 8), (-9, 8), (9, -9), (-9, -9), (0, 16), (0, -17)]   # points
LEGEND_STYLE = dict(frameon=True, facecolor="#ffffff", edgecolor="#ced4da", framealpha=1.0,
                    borderpad=0.6, handlelength=1.0, handletextpad=0.5,
                    shadow={"color": "#303559", "alpha": 0.7})


def load_points():
    """points[model][dataset] (composed) and forced[model][dataset], on the same axes."""
    composed = pd.read_csv(OUT / "composed-bins.csv")
    forced = pd.read_csv(OUT / "forced-bins.csv")
    points = {m: {d: f.reset_index(drop=True) for d, f in g.groupby("dataset")}
              for m, g in composed.groupby("model")}
    forced = {m: {d: f.assign(saved_tokens=1 - f.token_ratio, acc_change_pp=-f.accuracy_drop_pp,
                              group=f.length_bin).reset_index(drop=True)
                  for d, f in g.groupby("dataset")}
              for m, g in forced.groupby("model")}
    return points, forced


def signed_formatter(decimals):
    def fmt(v, _):
        if abs(v) < 1e-9:
            return "0"
        s = f"{int(round(v)):+d}" if decimals == 0 else f"{v:+.{decimals}f}"
        return s.replace("-", "−")
    return FuncFormatter(fmt)


def marker(color, shape="o", alpha=1.0, size=6):
    """A legend handle: a filled circle, or a diamond drawn slightly smaller."""
    return Line2D([], [], linestyle="none", marker=shape, markersize=size if shape == "o" else size - 0.5,
                  markerfacecolor=color, markeredgewidth=0, alpha=alpha)


def legend_entries(high_first=False, size=6):
    """Handles and labels: the three length bins, then the two conditions in a weak grey."""
    bins = list(zip(BIN_COLORS, BIN_LABELS))
    if high_first:
        bins = bins[::-1]
    handles = [marker(c, size=size) for c, _ in bins] + \
        [marker(WEAK_GREY, size=size), marker(WEAK_GREY, "D", 0.5, size=size)]
    return handles, [label for _, label in bins] + ["comp.", "forced"]


def draw_panel(ax, points, forced, circle_size=36, diamond_size=30, line_width=1.2):
    """Composed circles joined in k order, forced diamonds at 50%, the zero line.

    Returns the k labels still to place (k=2, k=6 and the largest k shown, per bin),
    the composed circles' data coordinates, the composed y values, and every y value drawn.
    """
    ax.set_facecolor("#f8f9fa")
    ax.axhline(0, color="#42505c", linewidth=0.8, zorder=1)
    k_labels, circles, comp_ys, ys = [], [], [], []
    for group, color in zip(GROUPS, BIN_COLORS):
        q = points.loc[(points.group == group) & points.k.isin(KS_SHOWN)].sort_values("k")
        f = forced.loc[(forced.group == group) & forced.k.isin(KS_SHOWN)]
        ax.plot(q.saved_tokens.values, q.acc_change_pp.values, color=color, linewidth=line_width, zorder=3)
        ax.scatter(q.saved_tokens.values, q.acc_change_pp.values, color=color, s=circle_size, zorder=4)
        ax.scatter(f.saved_tokens.values, f.acc_change_pp.values, color=color, marker="D",
                   s=diamond_size, alpha=0.5, linewidths=0, zorder=2)
        comp_ys.append(q.acc_change_pp.values)
        ys += [q.acc_change_pp.values, f.acc_change_pp.values]
        for k in sorted({2, 6, int(q.k.max())} & set(q.k)):
            row = q.loc[q.k == k].iloc[0]
            k_labels.append((row.saved_tokens, row.acc_change_pp, f"$k$={k}"))
        circles += list(zip(q.saved_tokens.values, q.acc_change_pp.values))
    return k_labels, circles, np.concatenate(comp_ys), np.concatenate(ys)


def thin_xticklabels(ax, pad_px=4):
    """Blank the x tick labels that would touch their right neighbour, keeping the rightmost.

    The x axis stretches large token reductions, so even steps crowd at its left end.
    Call after fig.canvas.draw()."""
    renderer = ax.figure.canvas.get_renderer()
    ticks = sorted(((t.get_window_extent(renderer), v)
                    for t, v in zip(ax.get_xticklabels(), ax.get_xticks()) if t.get_text()),
                   key=lambda bv: bv[0].x0, reverse=True)
    keep, left_edge = [], None
    for box, v in ticks:
        if left_edge is None or box.x1 + pad_px <= left_edge:
            keep.append(v)
            left_edge = box.x0
    fmt = ax.xaxis.get_major_formatter()
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, pos: fmt(v, pos) if any(abs(v - k) < 1e-9 for k in keep) else ""))


def nice_ceil(v):
    """The smallest of 1, 2, 5 x 10^n that is >= v."""
    e = 10 ** np.floor(np.log10(v))
    return next(m * e for m in (1, 2, 5, 10) if v <= m * e * (1 + 1e-9))


def symlog_y(ax, comp_ys, ys):
    """Keep the composed lines readable when the forced points reach much further.

    When the forced points reach more than SYMLOG_RATIO times past the composed range,
    the y axis turns symmetric-log: linear between -L and +L, where L is the composed
    range rounded up to 1, 2 or 5 x 10^n, logarithmic beyond. The visible linear zone
    gets the same height as L..the furthest point, and dashed lines at +-L mark where
    the scale changes. Returns False, touching nothing, when a linear axis is fine.
    """
    comp_reach = max(np.abs(comp_ys).max(), 0.5)
    lo, hi = min(ys.min(), 0), max(ys.max(), 0)
    reach = max(-lo, hi)
    if reach <= SYMLOG_RATIO * comp_reach:
        return False
    lim = nice_ceil(comp_reach)
    # matplotlib draws each L of the linear zone linscale / (1 - 1/10) decades tall, and
    # L..reach log10(reach / L) decades; the visible linear zone spans `span` L (between
    # 1 and 2: below 0 and, when the composed lines rise above it, above), so dividing
    # by it gives the visible linear zone the same height as the log zone
    span = (min(np.abs(comp_ys[comp_ys > 0]).max(initial=0), lim)
            + min(np.abs(comp_ys[comp_ys < 0]).max(initial=0), lim)) / lim
    span = min(max(span, 1.0), 2.0)
    ax.set_yscale("symlog", linthresh=lim, linscale=0.9 * np.log10(reach / lim) / span)

    step = lim / 2
    linear = [v for v in np.arange(-lim, lim + step / 2, step) if lo - step <= v <= hi + step]
    mantissas = (1, 2, 5) if reach / lim <= 30 else (1, 5)
    far = [sign * m * 10 ** e
           for e in range(int(np.floor(np.log10(lim))), int(np.ceil(np.log10(reach))) + 1)
           for m in mantissas for sign in (-1, 1)
           if lim < m * 10 ** e <= reach * 1.05 and (lo if sign < 0 else hi) * sign >= m * 10 ** e * 0.5]
    ax.yaxis.set_major_locator(FixedLocator(sorted(set(linear) | set(far))))
    # whole numbers print without decimals ("−10", not "−10.0"); halves keep one
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "0" if abs(v) < 1e-9 else
        (f"{int(round(v)):+d}" if abs(v - round(v)) < 1e-9 else f"{v:+.1f}").replace("-", "−")))
    ax.yaxis.set_minor_locator(NullLocator())
    for edge in (-lim, lim):
        if lo <= edge <= hi:
            ax.axhline(edge, color="#9aa5b1", linewidth=0.6, linestyle=(0, (3, 3)), zorder=1)
    return True


def circle_boxes(ax, circles, circle_size):
    """Display-space boxes of the composed circles, for the labels to avoid."""
    r = np.sqrt(circle_size) / 2 * ax.figure.dpi / 72          # radius in pixels
    return [Bbox.from_extents(px - r, py - r, px + r, py + r)
            for px, py in ax.transData.transform(np.array(circles))]


def place_k_labels(ax, labels, blocked, fontsize=7, same_spot_px=24, keep_inside=True):
    """Small k labels next to their points, avoiding each other and `blocked` boxes.

    Call after the figure is drawn, so positions are final. Each label takes the first
    offset in K_LABEL_TRIES that clears every box placed so far (and, with keep_inside,
    stays inside the axes, so it never runs into a title or a tick label); a label whose
    text repeats one already within `same_spot_px` pixels is dropped, since it marks the
    same spot.
    """
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    placed, anchors = list(blocked), []

    def annotate(text, x, y, dx, dy):
        return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                           ha="center", va="center", fontsize=fontsize, color=K_LABEL_COLOR, zorder=6)

    for x, y, text in labels:
        at = ax.transData.transform((x, y))
        if any(t == text and np.hypot(*(at - a)) < same_spot_px for t, a in anchors):
            continue
        fallback = None
        for dx, dy in K_LABEL_TRIES:
            ann = annotate(text, x, y, dx, dy)
            box = ann.get_window_extent(renderer)
            inside = not keep_inside or (frame.x0 <= box.x0 and box.x1 <= frame.x1
                                         and frame.y0 <= box.y0 and box.y1 <= frame.y1)
            if inside and not any(box.overlaps(b) for b in placed):
                break
            if inside and fallback is None:
                fallback = (dx, dy)
            ann.remove()
        else:   # nowhere free: the first offset inside the axes, overlap or not
            ann = annotate(text, x, y, *(fallback or K_LABEL_TRIES[0]))
            box = ann.get_window_extent(renderer)
        placed.append(box)
        anchors.append((text, at))
