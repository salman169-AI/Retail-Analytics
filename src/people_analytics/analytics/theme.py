"""Shared display tokens for charts, video overlays and presentation cards."""

from pathlib import Path

SURFACE = "#0B1017"
SIDEBAR = "#0D141E"
ACCENT_WASH = "#17263D"
CARD = "#131B26"
ELEVATED = "#1C2838"
INK = "#F4F7FB"
INK_2 = "#C2CEDB"
MUTED = "#94A6BA"
GRID = "#283344"
BASELINE = "#344256"
ACCENT = "#5BC9B1"
SUCCESS = "#39D2B0"
DANGER = "#FF6474"
QUEUE = "#F0A44B"
VIOLET = "#B5A1EE"
CYAN = "#83CFC4"
ZONE_NEUTRAL = "#C0CFCC"
ZONE_SECONDARY = "#91AFB5"
SERIES = (ACCENT, CYAN)
ZONE_COLORS = (CYAN, ZONE_NEUTRAL, ZONE_SECONDARY)
UI_BG = "#F0F4F3"
UI_CARD = "#FFFFFF"
UI_TEXT = "#18312D"
UI_SECONDARY = "#465E59"
UI_MUTED = "#687C77"
UI_LINE = "#DFE7E4"
UI_ACCENT = "#087F6C"
UI_TINT = "#E8F5EF"
UI_BAR = "#A2CFC2"
UI_RAIL = "#143D35"
UI_RAIL_ACTIVE = "#28584E"
UI_ON_DARK = "#D1E6DF"
UI_SHADOW = "#E7EDEA"
UI_DANGER = "#B53547"
UI_DANGER_TINT = "#FFF0F1"
CREDIT = "Video: MEVA dataset (mevadata.org), CC BY 4.0"

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
FONT_REGULAR = str(FONT_DIR / "IBMPlexSans-Regular.ttf")
FONT_BOLD = str(FONT_DIR / "IBMPlexSans-SemiBold.ttf")


def rgb(code: str) -> tuple[int, int, int]:
    """Convert a shared hex token for PIL."""
    return tuple(int(code[i:i + 2], 16) for i in (1, 3, 5))


def bgr(code: str) -> tuple[int, int, int]:
    """Convert a shared hex token for OpenCV."""
    return rgb(code)[::-1]
