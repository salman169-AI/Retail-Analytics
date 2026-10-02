"""Overlay drawing for the annotated video.

`sv.PolygonZoneAnnotator` draws a hairline outline and a bare number, which
disappears against busy street footage. These helpers fill each zone with a
translucent wash, outline it, and label it with its name *and* live occupancy,
so a viewer can tell at a glance what is being measured where.

One look across the video and the stats panel: a single typeface (DejaVu Sans,
bundled with matplotlib, free licence) in rounded "pill" labels, and one palette,
the same categorical hues as the panel's charts (bright steps, for video).
"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import matplotlib
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont


def _hex(code: str) -> sv.Color:
    return sv.Color.from_hex(code)


# Categorical order shared with the panel (blue, orange, aqua, yellow, magenta).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    _hex("#2a78d6"), _hex("#eb6834"), _hex("#1baf7a"), _hex("#eda100"), _hex("#e87ba4"),
)
# Zones take blue, magenta, yellow; orange is the queue's, aqua the door's.
ZONE_PALETTE = [BLUE, MAGENTA, YELLOW]

_FILL_ALPHA = 0.22
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
FONT_REGULAR = os.path.join(_FONT_DIR, "DejaVuSans.ttf")
FONT_BOLD = os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")


@lru_cache(maxsize=16)
def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


@lru_cache(maxsize=512)
def _pill_rgba(text: str, size: int, bg: tuple[int, int, int], fg: tuple[int, int, int],
               bold: bool) -> np.ndarray:
    """A rounded label as an RGBA patch (cached: the same counts recur every frame)."""
    f = font(size, bold)
    x0, y0, x1, y1 = f.getbbox(text)
    pad_x, pad_y = int(size * 0.6), int(size * 0.35)
    w, h = x1 - x0 + 2 * pad_x, y1 - y0 + 2 * pad_y
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=(*bg, 235))
    d.text((pad_x - x0, pad_y - y0), text, font=f, fill=(*fg, 255))
    return np.asarray(img)


def blit(frame: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend an RGBA patch onto a BGR frame at (x, y), clipped to the frame."""
    h, w = rgba.shape[:2]
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    patch = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = patch[..., 3:4].astype(np.float32) / 255.0
    region = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (region * (1 - a) + patch[..., 2::-1][..., :3] * a).astype(np.uint8)


def pill(frame: np.ndarray, text: str, anchor: tuple[int, int], color: sv.Color,
         size: int = 26, align: str = "center", bold: bool = True) -> tuple[int, int, int, int]:
    """Draw a rounded label; returns its box. `align`: center | left (anchor = left-middle)."""
    ink = _ink_for(color)
    rgba = _pill_rgba(text, size, (color.r, color.g, color.b), ink[::-1], bold)
    h, w = rgba.shape[:2]
    H, W = frame.shape[:2]
    x = anchor[0] - (w // 2 if align == "center" else 0)
    x = int(np.clip(x, 4, W - w - 4))
    y = int(np.clip(anchor[1] - h // 2, 4, H - h - 4))
    blit(frame, rgba, x, y)
    return x, y, w, h


def zone_color(index: int) -> sv.Color:
    return ZONE_PALETTE[index % len(ZONE_PALETTE)]


def _ink_for(color: sv.Color) -> tuple[int, int, int]:
    """Black or white text, whichever stays readable on this badge colour."""
    luminance = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b
    return (20, 20, 20) if luminance > 140 else (255, 255, 255)


def _badge(frame: np.ndarray, text: str, anchor: tuple[int, int], color: sv.Color) -> None:
    # Kept inside the frame even when the zone hugs an edge (see `pill`).
    pill(frame, text, anchor, color)


def draw_zones(
    frame: np.ndarray,
    zones: list[tuple[str, np.ndarray, sv.Color]],
    counts: dict[str, int],
) -> np.ndarray:
    """Fill, outline and label each zone. `counts` maps zone name -> people inside."""
    if not zones:
        return frame

    wash = frame.copy()
    for _name, polygon, color in zones:
        cv2.fillPoly(wash, [polygon], color.as_bgr())
    frame = cv2.addWeighted(wash, _FILL_ALPHA, frame, 1 - _FILL_ALPHA, 0)

    for name, polygon, color in zones:
        cv2.polylines(frame, [polygon], True, color.as_bgr(), 3, cv2.LINE_AA)
        centre = polygon.mean(axis=0).astype(int)
        _badge(frame, f"{name}  {counts.get(name, 0)}", tuple(centre), color)
    return frame


DOOR_COLOR = AQUA
QUEUE_COLOR = ORANGE


def draw_door(
    frame: np.ndarray, outer: np.ndarray, inner: np.ndarray, text: str
) -> np.ndarray:
    """Threshold line solid, inner line thinner, band lightly filled, counts badge.

    The badge sits just past the far end of the threshold line, not on the band:
    the band is exactly where people walk, and a badge there hides them.
    """
    outer = np.asarray(outer, dtype=np.int32)
    inner = np.asarray(inner, dtype=np.int32)
    band = np.array([outer[0], outer[1], inner[1], inner[0]], dtype=np.int32)
    wash = frame.copy()
    cv2.fillPoly(wash, [band], DOOR_COLOR.as_bgr())
    frame = cv2.addWeighted(wash, _FILL_ALPHA * 0.6, frame, 1 - _FILL_ALPHA * 0.6, 0)
    cv2.line(frame, tuple(outer[0]), tuple(outer[1]), DOOR_COLOR.as_bgr(), 4, cv2.LINE_AA)
    cv2.line(frame, tuple(inner[0]), tuple(inner[1]), DOOR_COLOR.as_bgr(), 2, cv2.LINE_AA)
    beyond = outer[1] + (outer[1] - outer[0]) * 0.3
    _badge(frame, text, tuple(beyond.astype(int)), DOOR_COLOR)
    return frame


def draw_queue(frame: np.ndarray, polygon: np.ndarray, text: str) -> np.ndarray:
    """Queue area filled and outlined, with its length/wait badge."""
    polygon = np.asarray(polygon, dtype=np.int32)
    wash = frame.copy()
    cv2.fillPoly(wash, [polygon], QUEUE_COLOR.as_bgr())
    frame = cv2.addWeighted(wash, _FILL_ALPHA, frame, 1 - _FILL_ALPHA, 0)
    cv2.polylines(frame, [polygon], True, QUEUE_COLOR.as_bgr(), 3, cv2.LINE_AA)
    top = polygon[np.argmin(polygon[:, 1])]
    _badge(frame, text, (int(polygon[:, 0].mean()), int(top[1]) + 28), QUEUE_COLOR)
    return frame
