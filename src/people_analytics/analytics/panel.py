"""Responsive stats panel, drawn only from the supplied replay state."""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import ImageFont  # noqa: E402

from people_analytics.analytics.theme import (  # noqa: E402
    ACCENT,
    BASELINE,
    CARD,
    DANGER,
    FONT_BOLD,
    FONT_REGULAR,
    GRID,
    INK,
    INK_2,
    MUTED,
    SERIES,
    SUCCESS,
    SURFACE,
)


@dataclass
class PanelState:
    title: str
    clock_s: float
    tiles: list[tuple[str, str]]  # (label, value), in display order
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


def render_panel(state: PanelState, size: tuple[int, int] = (640, 1080)) -> np.ndarray:
    """Render a BGR panel with pixel-aligned cards and independently sized charts."""
    w, h = size
    scale = h / 1080
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=SURFACE)
    width = w / scale
    margin = 26
    inner = width - margin * 2

    def prop(px: float, bold: bool = False) -> FontProperties:
        return FontProperties(fname=FONT_BOLD if bold else FONT_REGULAR, size=px * scale * .72)

    def wrap(text: str, px: int, available: float, bold: bool = False) -> str:
        face = ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, px)
        rows, line = [], ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if face.getlength(trial) > available and line:
                rows.append(line)
                line = ""
            for char in word:
                trial = line + char
                if face.getlength(trial) > available and line:
                    rows.append(line)
                    line = ""
                line += char
            line += " "
        rows.append(line.rstrip())
        return "\n".join(row.rstrip() for row in rows)

    def text(x: float, y: float, value: str, px: int = 18, color: str = INK_2,
             bold: bool = False, ha: str = "left", available: float | None = None) -> None:
        if available is not None:
            value = wrap(value, px, available, bold)
        fig.text(x / width, 1 - y / 1080, value, color=color, fontproperties=prop(px, bold),
                 ha=ha, va="top", linespacing=1.25)

    def rect(x: float, y: float, rw: float, rh: float, color: str,
             rounded: bool = False) -> None:
        bounds = (x / width, 1 - (y + rh) / 1080, rw / width, rh / 1080)
        patch = (FancyBboxPatch(bounds[:2], *bounds[2:], boxstyle="round,pad=0,rounding_size=.009",
                               facecolor=color, edgecolor=GRID, linewidth=.7)
                 if rounded else Rectangle(bounds[:2], *bounds[2:], color=color, linewidth=0))
        patch.set_transform(fig.transFigure)
        fig.patches.append(patch)

    def axes(y: float, height: float) -> matplotlib.axes.Axes:
        ax = fig.add_axes(((margin + 26) / width, 1 - (y + height) / 1080,
                           (inner - 38) / width, height / 1080))
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=MUTED, length=0, pad=8)
        ax.grid(axis="y", color=GRID, linewidth=.7)
        ax.set_axisbelow(True)
        return ax

    def ticks(ax: matplotlib.axes.Axes) -> None:
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontproperties(prop(17))

    def heading(y: float, title: str) -> None:
        main, sep, qualifier = title.partition(" (")
        text(margin, y, main, 23, INK, True, available=inner)
        if sep:
            text(margin, y + 32, qualifier.rstrip(")").capitalize(), 17, MUTED)

    rect(0, 0, 2, 1080, BASELINE)
    text(margin, 28, "CAMERA ANALYTICS", 15, ACCENT, True)
    text(margin, 59, state.title, 32, INK, True, available=inner - 115)
    m, s = divmod(int(state.clock_s), 60)
    text(width - margin, 61, f"{m:02d}:{s:02d}", 28, INK, ha="right")
    text(width - margin, 96, "CLIP TIME", 12, MUTED, ha="right")

    gap = 14
    card_w = (inner - gap) / 2
    for i, (label, value) in enumerate(state.tiles):
        x, y, cw, ch = (margin + i * (card_w + gap), 136, card_w, 142) if i < 2 else (
            margin, 292 + (i - 2) * 100, inner, 94)
        rect(x, y, cw, ch, CARD, True)
        text(x + 18, y + 16, label, 18, INK_2, available=cw - 36)
        status = SUCCESS if value == "Staffed" else DANGER if value.startswith("Empty ") else INK
        value_px = 52 if i < 2 else 29
        face = ImageFont.truetype(FONT_BOLD, value_px)
        while face.getlength(value) > cw - 58 and value_px > 18:
            value_px -= 1
            face = ImageFont.truetype(FONT_BOLD, value_px)
        text(x + 18, y + (57 if i < 2 else 45), value, value_px, status, True)
        if label == "Counter":
            fig.patches.append(Ellipse(((x + cw - 25) / width, 1 - (y + 60) / 1080),
                                       10 / width, 10 / 1080, transform=fig.transFigure,
                                      facecolor=status, edgecolor="none"))

    heading(426, state.footfall_title)
    ax = axes(506, 170)
    names = list(state.footfall)
    x = np.arange(state.minutes)
    n = len(names)
    bw = .36 if n == 2 else .5
    top = max([1] + [max(v) for v in state.footfall.values() if v])
    for k, name in enumerate(names):
        color = SERIES[k % len(SERIES)]
        offset = (k - (n - 1) / 2) * (bw + .04)
        for xi, value in zip(x, state.footfall[name], strict=False):
            if value > 0:
                ax.bar(xi + offset, value, width=bw, color=color, zorder=3)
                ax.text(xi + offset, value + top * .04, str(value), color=INK,
                        fontproperties=prop(19, True), ha="center", va="bottom")
        if n > 1:
            lx = margin + k * inner / n
            rect(lx, 479, 10, 10, color)
            text(lx + 17, 474, name, 16)
    stride = max(1, int(np.ceil(state.minutes / 8)))
    ax.set_xticks(x[::stride], [f"{i}\u2013{i + 1}" for i in x[::stride]])
    ax.set_xlim(-.6, max(.6, state.minutes - .4))
    ax.set_ylim(0, top * 1.28)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, integer=True))
    ax.set_xlabel("Minute of clip", color=MUTED, fontproperties=prop(16), labelpad=12)
    ticks(ax)

    rect(margin, 743, inner, 1, GRID)
    heading(766, state.bars_title or state.line_title)
    if state.bars:
        items = sorted(state.bars.items(), key=lambda kv: kv[1], reverse=True)
        vmax = max([1.0] + [v for _, v in items])
        label_px = 18
        while True:
            labels = [wrap(name, label_px, inner - 85) for name, _ in items]
            heights = [(label.count("\n") + 1) * label_px * 1.25 + 18 for label in labels]
            if sum(heights) <= 186 or label_px <= 10:
                break
            label_px -= 1
        extra = max(0, min(14, (186 - sum(heights)) / len(items)))
        y = 823
        for label, row_h, (_, value) in zip(labels, heights, items, strict=True):
            row_h += extra
            text(margin, y, label, label_px)
            text(width - margin, y, f"{value:.0f} {state.bars_unit}", 19, INK, True, ha="right")
            rect(margin, y + row_h - 11, inner, 5, GRID)
            rect(margin, y + row_h - 11, inner * value / vmax, 5, ACCENT)
            y += row_h
    elif state.lines:
        ax2 = axes(838, 136)
        top = 1.0
        for k, (name, (times, values)) in enumerate(state.lines.items()):
            color = SERIES[k % len(SERIES)]
            t = np.asarray(times) / 60
            ax2.step(t, values, where="post", color=color, linewidth=2)
            if len(t):
                ax2.plot(t[-1], values[-1], "o", color=color, markersize=5)
                text(margin + k * inner / len(state.lines), 806,
                     f"{name}  {values[-1]}", 18, color, True)
            top = max(top, max(values) if len(values) else 0)
        ax2.set_xlim(0, max(1, state.minutes))
        ax2.set_ylim(0, (state.line_max or top) * 1.15)
        ax2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, integer=True))
        ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, integer=True))
        ticks(ax2)
    else:
        text(margin, 843, "No completed visits yet", 19, MUTED)

    rect(margin, 1019, inner, 1, GRID)
    y = 1030
    for line in state.footer:
        wrapped = wrap(line, 14, inner)
        text(margin, y, wrapped, 14, MUTED)
        y += (wrapped.count("\n") + 1) * 18

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1].copy()
    plt.close(fig)
    return img
