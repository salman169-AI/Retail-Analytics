"""Stats panel drawn beside the video: stat tiles, footfall by minute, a second chart.

Rendered with matplotlib into a fixed-size image (default 640x1080, to sit next
to a 1920x1080 frame). Dark surface so it reads as part of the video, not a
pasted slide. Colours follow the project's chart rules: one hue per series in a
fixed order (blue, then orange), text in ink tones rather than series colours,
thin bars with the value at the tip, hairline grid, no second y-axis.

The panel is a pure function of a `PanelState`, so the renderer can rebuild it
from whatever the replay has computed up to the current frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
SERIES = ["#3987e5", "#d95926"]  # validated pair (dark surface): blue, orange
FONT = ["DejaVu Sans"]  # same face as the video overlay (free licence, ships with matplotlib)


@dataclass
class PanelState:
    title: str
    clock_s: float
    tiles: list[tuple[str, str]]  # (label, value), first one is the hero
    minutes: int  # x extent of the footfall chart
    footfall: dict[str, list[int]]  # series name -> count per minute (1 or 2 series)
    footfall_title: str
    # Second chart: either ranked bars (label -> value) or a line over time.
    bars: dict[str, float] = field(default_factory=dict)
    bars_title: str = ""
    bars_unit: str = "s"
    # Lines over time on one axis, one unit: series name -> (t_s, value). Up to two.
    lines: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)
    line_title: str = ""
    line_max: float | None = None
    footer: list[str] = field(default_factory=list)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def render_panel(state: PanelState, size: tuple[int, int] = (640, 1080)) -> np.ndarray:
    """Panel image (BGR uint8, h x w x 3)."""
    w, h = size
    plt.rcParams["font.family"] = FONT
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=SURFACE)

    # Header
    fig.text(0.06, 0.965, state.title, color=INK, fontsize=17, fontweight="bold", va="top")
    m, s = divmod(int(state.clock_s), 60)
    fig.text(0.94, 0.965, f"{m:02d}:{s:02d}", color=INK_2, fontsize=15, va="top", ha="right")

    # Stat tiles: first is the hero figure, the rest smaller.
    y0 = 0.915
    label, value = state.tiles[0]
    fig.text(0.06, y0, label, color=INK_2, fontsize=12, va="top")
    fig.text(0.06, y0 - 0.025, value, color=INK, fontsize=44, fontweight="bold", va="top")
    for i, (label, value) in enumerate(state.tiles[1:]):
        x = 0.06 + 0.45 * i
        fig.text(x, y0 - 0.105, label, color=INK_2, fontsize=11, va="top")
        fig.text(x, y0 - 0.125, value, color=INK, fontsize=24, fontweight="bold", va="top")

    # Footfall by minute (columns; 1 or 2 series side by side)
    ax = fig.add_axes((0.10, 0.44, 0.84, 0.22))
    _style(ax)
    names = list(state.footfall)
    x = np.arange(state.minutes)
    n = len(names)
    bw = 0.36 if n == 2 else 0.5
    top = max([1] + [max(v) for v in state.footfall.values() if v])
    for k, name in enumerate(names):
        vals = state.footfall[name]
        offs = (k - (n - 1) / 2) * (bw + 0.04)
        for xi, v in zip(x, vals, strict=False):
            if v > 0:
                ax.bar(xi + offs, v, width=bw, color=SERIES[k])
                ax.text(xi + offs, v + top * 0.03, str(v), color=INK_2, fontsize=10,
                        ha="center", va="bottom")
    ax.set_xticks(x, [f"{i}–{i + 1}" for i in x])
    ax.set_xlim(-0.6, state.minutes - 0.4)
    ax.set_ylim(0, top * 1.25)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
    ax.set_xlabel("minute of clip", color=MUTED, fontsize=10)
    fig.text(0.06, 0.695, state.footfall_title, color=INK, fontsize=13, fontweight="bold")
    if n > 1:  # legend for two series: swatch + ink text
        for k, name in enumerate(names):
            lx = 0.06 + 0.22 * k
            fig.patches.append(matplotlib.patches.Rectangle(
                (lx, 0.669), 0.022, 0.011, color=SERIES[k], transform=fig.transFigure))
            fig.text(lx + 0.03, 0.674, name, color=INK_2, fontsize=11, va="center")

    # Second chart
    ax2 = fig.add_axes((0.06, 0.10, 0.86, 0.24))
    _style(ax2)
    if state.bars:
        fig.text(0.06, 0.375, state.bars_title, color=INK, fontsize=13, fontweight="bold")
        items = sorted(state.bars.items(), key=lambda kv: kv[1])
        ys = np.arange(len(items))
        vmax = max([1.0] + [v for _, v in items])
        ax2.barh(ys, [v for _, v in items], height=0.32, color=SERIES[0])
        for y, (name, v) in zip(ys, items, strict=True):
            # Name above its bar, value at the tip: long names never clip.
            ax2.text(0, y + 0.24, name, color=INK_2, fontsize=11, va="bottom")
            ax2.text(v + vmax * 0.02, y, f"{v:.0f} {state.bars_unit}", color=INK,
                     fontsize=11, va="center")
        ax2.set_yticks([])
        ax2.set_ylim(-0.5, len(items) - 0.2)
        ax2.set_xlim(0, vmax * 1.3)
        ax2.grid(axis="y", visible=False)
        ax2.grid(axis="x", color=GRID, linewidth=1)
        ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    elif state.lines:
        fig.text(0.06, 0.375, state.line_title, color=INK, fontsize=13, fontweight="bold")
        top = 1.0
        for k, (name, (t, v)) in enumerate(state.lines.items()):
            t = np.array(t) / 60
            ax2.step(t, v, where="post", color=SERIES[k], linewidth=2)
            if len(t):
                ax2.plot([t[-1]], [v[-1]], "o", color=SERIES[k], markersize=8,
                         markeredgecolor=SURFACE, markeredgewidth=2, clip_on=False)
                # Direct label at the line end, in ink (the dot carries the colour).
                ax2.annotate(f"{name} {v[-1]}", (t[-1], v[-1]), xytext=(8, 0),
                             textcoords="offset points", color=INK_2, fontsize=11,
                             va="center")
            top = max(top, max(v) if len(v) else 0)
        # Room on the right for the end labels, so they're never clipped.
        ax2.set_xlim(0, state.minutes * 1.22)
        ax2.set_xticks(range(state.minutes + 1))
        ax2.spines["bottom"].set_bounds(0, state.minutes)
        ax2.set_ylim(0, (state.line_max or top) * 1.15)
        ax2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
        ax2.set_xlabel("minute of clip", color=MUTED, fontsize=10)

    for i, line in enumerate(state.footer):
        fig.text(0.06, 0.045 - 0.022 * i, line, color=MUTED, fontsize=9.5, va="top")

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1].copy()
    plt.close(fig)
    return img
