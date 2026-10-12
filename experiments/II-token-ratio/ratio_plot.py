"""Drawing helpers shared by the two token-ratio figures."""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from llm.plotting import BIN_COLORS, BIN_LABELS
from llm.specs import TRACE_KS

# The figure name of each dataset, bold.
TITLES = {
    "integrals": r"$\mathbf{Integrals}$",
    "mmlu": r"$\mathbf{MMLU\text{-}Redux}$",
    "onewordqa": r"$\mathbf{OneWordQA}$",
    "gpqa-diamond": r"$\mathbf{GPQA\text{-}Diamond}$",
    "gsm8k": r"$\mathbf{GSM8K}$",
    "math500": r"$\mathbf{MATH\text{-}500}$",
}


def by_bin(points):
    """{bin: Series ratio indexed by k}, restricted to the ks every model ran."""
    points = points[points.k.isin(TRACE_KS)]
    return {b: points[points.bin == b].set_index("k").ratio for b in BIN_LABELS + ["all"]}


def origin_slope(ratio) -> float:
    """Log-log least squares with no intercept: the line passes through (1, 1)."""
    lx, ly = np.log2(ratio.index.values.astype(float)), np.log2(ratio.values)
    return float(np.dot(lx, ly) / np.dot(lx, lx))


def plot_bins(ax, d, markersize=6, linewidth=1.5, x_start=2.0):
    """One bin per colour: its mean ratios and its power law through (1, 1).

    Returns one (colour, slope, line x, line y, point x, point y) per bin, for
    slope_labels."""
    ax.set_facecolor("#f8f9fa")
    ax.set_xlim(1.5, d["all"].index.max() + 1.5)
    fits = []
    for color, label in zip(BIN_COLORS, BIN_LABELS):
        q = d[label]
        m = origin_slope(q)
        ax.loglog(q.index, q.values, "o", color=color, markersize=markersize,
                  markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        x_fit = np.logspace(np.log10(x_start), np.log10(q.index.max()), 200)
        ax.loglog(x_fit, x_fit ** m, "-", color=color, linewidth=linewidth, label=label)
        fits.append((color, m, x_fit, x_fit ** m, q.index.values.astype(float), q.values))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    return fits


# Where slope_labels may put a label: fractions along the line (in log k), and the gap
# to the line in points, on either side.
SLOPE_SPOTS = np.linspace(0.05, 0.95, 19)
SLOPE_GAPS = [2, 4, 7, 11, 16, 22]
# What a spot costs: each point it covers, each other line between it and its own line,
# being on the less natural side of the line, each point of gap, and distance from the
# middle of the line.
COST_DOT, COST_BETWEEN, COST_SIDE, COST_GAP, COST_OFF_CENTRE = 3.0, 5.0, 1.0, 0.05, 0.5


def slope_labels(ax, fits, fontsize, markersize, avoid=()):
    """Write each line's power law, "∝ k^m", horizontally, just above or below the line.

    Call once the layout is final (after fig.canvas.draw()). Each label goes inside the
    axes, in its line's colour, where no other line, label or box in `avoid` (the legend)
    crosses it. Among those spots it takes the cheapest (see the COST_ constants): one
    that covers no point, has no other line between it and its own line, sits on the
    natural side (above the top line, below the bottom one), close to the line and near
    its middle. A label with no spot at all is left out.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    px_per_pt = fig.dpi / 72
    to_px = ax.transData.transform
    lines = [to_px(np.column_stack([x, y])) for _, _, x, y, _, _ in fits]
    dots = np.vstack([to_px(np.column_stack([x, y])) for *_, x, y in fits])
    dot_r = markersize / 2 * px_per_pt
    heights = [line[:, 1].mean() for line in lines]
    natural = [-1 if h == min(heights) else 1 for h in heights]   # +1: above the line
    placed = []                                          # (centre, u, n, half w, half h)

    def corners(c, u, n, hw, hh):
        return [c + a * hw * u + b * hh * n for a in (-1, 1) for b in (-1, 1)]

    def inside(p, rect, pad=0.0):
        c, u, n, hw, hh = rect
        v = p - c
        return abs(v @ u) <= hw + pad and abs(v @ n) <= hh + pad

    def overlaps(a, b):
        # two rectangles overlap unless one of their four axes separates them
        for axis in (a[1], a[2], b[1], b[2]):
            pa = [p @ axis for p in corners(*a)]
            pb = [p @ axis for p in corners(*b)]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
        return True

    def box_rect(box):
        return (np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2]), np.array([1.0, 0.0]),
                np.array([0.0, 1.0]), box.width / 2, box.height / 2)

    def y_at(line, x):
        return np.interp(x, line[:, 0], line[:, 1]) if line[0, 0] <= x <= line[-1, 0] else None

    def lines_between(i, anchor, c):
        """The other lines that pass between line i and the label centre c, measured at
        the label's anchor on line i and, where line i reaches, below the centre."""
        count = 0
        for x, own in ((anchor[0], anchor[1]), (c[0], y_at(lines[i], c[0]))):
            if own is None:
                continue
            ys = [y_at(other, x) for k, other in enumerate(lines) if k != i]
            count = max(count, sum(y is not None and min(own, c[1]) < y < max(own, c[1])
                                   for y in ys))
        return count

    for i, (color, m, x, y, _, _) in enumerate(fits):
        text = ax.text(0, 0, rf"$\propto\,k^{{{m:.2f}}}$", fontsize=fontsize, color=color,
                       ha="center", va="center", zorder=6)
        size = text.get_window_extent(renderer)
        hw, hh = size.width / 2 + 1, size.height / 2
        line = lines[i]
        others = [p for k, other in enumerate(lines) if k != i for p in other]
        best = None
        for frac in SLOPE_SPOTS:
            j = int(round(frac * (len(line) - 1)))
            a, b = line[max(j - 5, 0)], line[min(j + 5, len(line) - 1)]
            steep = abs((b - a)[1] / (b - a)[0])         # the line's slope on screen
            u, n = np.array([1.0, 0.0]), np.array([0.0, 1.0])
            for side in (1, -1):                         # +1: above the line
                for gap in SLOPE_GAPS:
                    # far enough up or down that the label's corner on the downhill
                    # side still clears the line
                    c = line[j] + side * n * (hh + hw * steep + gap * px_per_pt)
                    rect = (c, u, n, hw, hh)
                    if not all(frame.x0 + 2 <= p[0] <= frame.x1 - 2 and
                               frame.y0 + 2 <= p[1] <= frame.y1 - 2 for p in corners(*rect)):
                        continue
                    if (any(overlaps(rect, r) for r in placed)
                            or any(overlaps(rect, box_rect(box)) for box in avoid)
                            or any(inside(p, rect) for p in others)):
                        continue
                    cost = (COST_DOT * sum(inside(p, rect, dot_r) for p in dots)
                            + COST_BETWEEN * lines_between(i, line[j], c)
                            + COST_SIDE * (side != natural[i])
                            + COST_GAP * gap
                            + COST_OFF_CENTRE * abs(frac - 0.5))
                    if best is None or cost < best[0]:
                        best = (cost, rect)
        if best is None:                                 # nowhere free: leave it out
            text.remove()
            continue
        c = best[1][0]
        text.set_position(ax.transData.inverted().transform(c))
        placed.append(best[1])


def sparse_ticks(ax, d, n, round_y=False):
    """n task counts on x; n log-spaced ratios on y, to two significant digits."""
    xs = np.sort(d["all"].index.unique())
    idx = np.unique(np.linspace(0, len(xs) - 1, n).round().astype(int))
    ax.xaxis.set_major_locator(FixedLocator(xs[idx]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
    ys = np.concatenate([d[b].values for b in BIN_LABELS])
    yt = np.unique([float(f"{v:.2g}") for v in np.geomspace(ys.min(), ys.max(), n)])
    if round_y:
        yt = [round(v * 10) / 10 for v in yt]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2g}"))


def bin_legend(ax, fontsize, borderaxespad=0.5):
    """The length-bin legend, longest bin on top, in the lower-left corner."""
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=9, markerfacecolor=c,
                      markeredgecolor="white", markeredgewidth=1.0) for c in BIN_COLORS]
    leg = ax.legend(handles[::-1], BIN_LABELS[::-1], title="Isolated \ntrace len.",
                    fontsize=fontsize, title_fontsize=fontsize, loc="lower left",
                    frameon=True, facecolor="#ffffff", edgecolor="#ced4da", framealpha=1.0,
                    borderpad=0.6, labelspacing=0.4, handlelength=1.0, handletextpad=0.5,
                    borderaxespad=borderaxespad, shadow={"color": "#303559", "alpha": 0.7})
    leg.set_zorder(10)
    return leg
