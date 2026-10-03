"""Application-style presentation of the existing camera replay state."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from people_analytics.analytics import theme as t
from people_analytics.analytics.panel import PanelState
from people_analytics.analytics.render import font

WIDTH, HEIGHT = 1920, 1080
CAMERA = (128, 358, 1200, 675)


class Canvas:
    """Draw UI components in one PIL pass, then return an OpenCV frame."""

    def __init__(self, background: str = t.UI_BG) -> None:
        self.image = Image.new("RGB", (WIDTH * 2, HEIGHT * 2), background)
        self.draw = ImageDraw.Draw(self.image)

    def text(self, x: int, y: int, value: str, size: int = 18,
             color: str = t.UI_TEXT, bold: bool = False, width: int | None = None,
             align: str = "left") -> int:
        face = font(size * 2, bold)
        lines, line = [], ""
        for word in value.split():
            trial = (line + " " + word).strip()
            if width and face.getlength(trial) > width * 2 and line:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        for line in lines:
            dx = face.getlength(line) / (2 if align == "center" else 1) if align != "left" else 0
            self.draw.text((x * 2 - dx, y * 2), line, font=face, fill=color)
            y += int(size * 1.35)
        return y

    def box(self, x: int, y: int, w: int, h: int, fill: str | None = t.UI_CARD,
            border: str | None = t.UI_LINE, radius: int = 10) -> None:
        self.draw.rounded_rectangle((x * 2, y * 2, (x + w) * 2, (y + h) * 2),
                                    radius=radius * 2, fill=fill, outline=border, width=2)

    def card(self, x: int, y: int, w: int, h: int, fill: str = t.UI_CARD) -> None:
        self.box(x, y + 3, w, h, t.UI_SHADOW, None, 14)
        self.box(x, y, w, h, fill, t.UI_LINE, 14)

    def line(self, points: list[tuple[int, int]], color: str = t.UI_LINE, width: int = 1) -> None:
        self.draw.line([(x * 2, y * 2) for x, y in points], fill=color, width=width * 2)

    def dot(self, x: int, y: int, color: str, radius: int = 4) -> None:
        self.draw.ellipse(((x - radius) * 2, (y - radius) * 2,
                           (x + radius) * 2, (y + radius) * 2), fill=color)

    def icon(self, x: int, y: int, kind: str, color: str = t.UI_MUTED) -> None:
        class IconDraw:
            def rounded_rectangle(_self, box: tuple[int, int, int, int], radius: int,
                                  outline: str, width: int) -> None:
                self.draw.rounded_rectangle(tuple(v * 2 for v in box), radius=radius * 2,
                                            outline=outline, width=width * 2)

            def ellipse(_self, box: tuple[int, int, int, int], outline: str,
                        width: int) -> None:
                self.draw.ellipse(tuple(v * 2 for v in box), outline=outline, width=width * 2)

        d = IconDraw()
        if kind == "camera":
            d.rounded_rectangle((x, y + 3, x + 17, y + 19), radius=3, outline=color, width=2)
            self.line([(x + 18, y + 8), (x + 24, y + 4), (x + 24, y + 19),
                       (x + 18, y + 15)], color, 2)
        elif kind == "grid":
            for dx, dy in ((0, 0), (13, 0), (0, 13), (13, 13)):
                d.rounded_rectangle((x + dx, y + dy, x + dx + 8, y + dy + 8),
                                    radius=2, outline=color, width=2)
        elif kind == "chart":
            for dx, ht in ((1, 9), (9, 18), (17, 13)):
                self.line([(x + dx, y + 22), (x + dx, y + 22 - ht)], color, 3)
        elif kind == "bell":
            self.line([(x + 2, y + 18), (x + 5, y + 14), (x + 5, y + 7),
                       (x + 9, y + 3), (x + 15, y + 3), (x + 19, y + 7),
                       (x + 19, y + 14), (x + 22, y + 18), (x + 2, y + 18)], color, 2)
            self.line([(x + 9, y + 22), (x + 15, y + 22)], color, 2)
        elif kind == "clock":
            d.ellipse((x, y, x + 22, y + 22), outline=color, width=2)
            self.line([(x + 11, y + 4), (x + 11, y + 11), (x + 16, y + 14)], color, 2)
        else:
            d.rounded_rectangle((x + 2, y + 2, x + 22, y + 22), radius=5,
                                outline=color, width=2)
            self.dot(x + 12, y + 12, color, 3)

    def array(self) -> np.ndarray:
        image = self.image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        return np.asarray(image)[..., ::-1].copy()


def clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def shell(active: str, title: str, subtitle: str) -> Canvas:
    c = Canvas()
    c.box(0, 0, 88, HEIGHT, t.UI_RAIL, None, 0)
    c.box(20, 20, 48, 48, t.UI_CARD, None, 14)
    c.icon(32, 32, "camera", t.UI_ACCENT)
    for i, (label, kind) in enumerate((("Overview", "grid"), ("Cameras", "camera"),
                                      ("Analytics", "chart"), ("Notifications", "bell"))):
        y = 126 + i * 76
        if label == active:
            c.box(17, y - 12, 54, 49, t.UI_RAIL_ACTIVE, None, 12)
            c.box(0, y - 2, 3, 28, t.SUCCESS, None, 1)
        c.icon(32, y, kind, t.INK if label == active else t.UI_ON_DARK)
    c.line([(24, 970), (64, 970)], t.UI_RAIL_ACTIVE)
    c.icon(32, 1000, "settings", t.UI_ON_DARK)

    c.box(88, 0, 1832, 78, t.UI_CARD, None, 0)
    c.line([(88, 78), (WIDTH, 78)])
    c.text(128, 23, "Retail vision", 24, bold=True)
    current = {"Overview": "Overview", "Analytics": "Reports",
               "Notifications": "Activity"}.get(active, active)
    for x, label in ((492, "Overview"), (654, "Activity"), (798, "Reports")):
        selected = current == label
        c.text(x, 27, label, 18, t.UI_ACCENT if selected else t.UI_MUTED, selected)
        if selected:
            c.box(x, 74, int(font(18).getlength(label)), 3, t.UI_ACCENT, None, 1)
    c.box(1454, 20, 182, 39, t.UI_CARD, t.UI_LINE, 8)
    c.icon(1469, 28, "camera", t.UI_MUTED)
    c.text(1506, 26, "Cafe", 17)
    c.line([(1610, 36), (1615, 41), (1620, 36)], t.UI_MUTED, 2)
    c.box(1654, 23, 162, 33, t.UI_BG, None, 16)
    c.dot(1671, 40, t.UI_MUTED, 3)
    c.text(1684, 28, "Session replay", 14, t.UI_SECONDARY)
    c.icon(1851, 28, "bell", t.UI_MUTED)
    c.text(128, 99, title, 35, bold=True)
    c.text(128, 149, subtitle, 17, t.UI_SECONDARY)
    return c


def kpis(c: Canvas, state: PanelState, y: int = 184) -> None:
    gap, width = 20, (1752 - 40) // 3
    for i, (label, value) in enumerate(state.tiles[:3]):
        x = 128 + i * (width + gap)
        hero = i == 0
        c.card(x, y, width, 102, t.UI_RAIL if hero else t.UI_CARD)
        c.text(x + 24, y + 12, label, 17, t.UI_ON_DARK if hero else t.UI_SECONDARY)
        c.icon(x + width - 49, y + 17, ("grid", "clock", "camera")[i],
               t.SUCCESS if hero else t.UI_MUTED)
        if label == "Counter":
            color = t.UI_ACCENT if value == "Staffed" else t.UI_DANGER
            fill = t.UI_TINT if value == "Staffed" else t.UI_DANGER_TINT
            pill_w = min(width - 48, int(font(29).getlength(value)) + 48)
            c.box(x + 24, y + 46, pill_w, 42, fill, None, 8)
            c.dot(x + 40, y + 67, color, 4)
            c.text(x + 54, y + 46, value, 29, color, True)
        else:
            c.text(x + 24, y + 39, value, 43, t.INK if hero else t.UI_TEXT, True)


def visits(c: Canvas, state: PanelState, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    c.card(x, y, w, h)
    c.text(x + 24, y + 17, "Visit activity", 23, bold=True)
    c.text(x + 24, y + 49, "Visits started per minute / after clip start", 14, t.UI_MUTED)
    left, right, top, bottom = x + 55, x + w - 25, y + 100, y + h - 36
    vmax = max([1] + [v for vals in state.footfall.values() for v in vals])
    limit = max(2, math.ceil(vmax / 10) * 10)
    for value in (0, limit // 2, limit):
        yy = bottom - int(value / limit * (bottom - top))
        c.line([(left, yy), (right, yy)])
        c.text(x + 39, yy - 10, str(value), 14, t.UI_MUTED, align="right")
    step = (right - left) / max(1, state.minutes)
    n = max(1, len(state.footfall))
    for k, values in enumerate(state.footfall.values()):
        for i, value in enumerate(values):
            bx = int(left + (i + .5) * step + (k - (n - 1) / 2) * step * .3)
            bw = int(min(52, step * .55 / n))
            bh = int(value / limit * (bottom - top))
            if value:
                color = t.UI_ACCENT if value == vmax else t.UI_BAR
                c.box(bx - bw // 2, bottom - bh, bw, bh, color, None, 5)
                c.text(bx, bottom - bh - 27, str(value), 17, t.UI_TEXT, True, align="center")
    for i in range(state.minutes):
        bx = int(left + (i + .5) * step)
        c.text(bx, bottom + 9, f"{i}\u2013{i + 1}", 14, t.UI_MUTED, align="center")


def dwell(c: Canvas, state: PanelState, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    c.card(x, y, w, h)
    c.text(x + 24, y + 17, "Time by area", 23, bold=True)
    c.text(x + 24, y + 49, "Average duration of completed visits", 14, t.UI_MUTED)
    items = sorted(state.bars.items(), key=lambda item: -item[1])
    maximum = max([1.] + [value for _, value in items])
    if not items:
        c.text(x + 24, y + 104, "No completed visits yet", 17, t.UI_MUTED)
    row_h = min(67, (h - 94) / max(1, len(items)))
    for i, (label, value) in enumerate(items):
        yy = int(y + 85 + i * row_h)
        size = 18
        while font(size, False).getlength(label) > w - 130 and size > 11:
            size -= 1
        c.text(x + 24, yy, label, size, t.UI_SECONDARY)
        c.text(x + w - 24, yy - 1, f"{value:.0f} {state.bars_unit}", 19,
               t.UI_TEXT, True, align="right")
        c.box(x + 24, yy + 30, w - 48, 6, t.UI_BG, None, 3)
        c.box(x + 24, yy + 30, int((w - 48) * value / maximum), 6,
              t.UI_ACCENT if i == 0 else t.UI_BAR, None, 3)


def monitor(state: PanelState, activity: list[dict], total_s: float,
            caption: str, sub: str) -> np.ndarray:
    c = shell("Overview", "Cafe overview", sub if "threshold" in sub else caption)
    kpis(c, state)
    c.card(128, 310, 1200, 757)
    c.icon(152, 324, "camera", t.UI_MUTED)
    c.text(190, 318, "Cafe camera", 22, bold=True)
    c.box(1127, 320, 176, 28, t.UI_TINT, None, 6)
    c.dot(1143, 334, t.UI_ACCENT, 3)
    c.text(1155, 323, "Tracking overlay", 14, t.UI_ACCENT)
    c.line([(152, 1043), (152, 1057)], t.UI_SECONDARY, 2)
    c.line([(159, 1043), (159, 1057)], t.UI_SECONDARY, 2)
    c.text(180, 1038, clock(state.clock_s) + " / " + clock(total_s), 14, t.UI_SECONDARY)
    c.box(323, 1049, 827, 3, t.UI_LINE, None, 1)
    position = int(827 * min(1, state.clock_s / max(total_s, 1)))
    c.box(323, 1049, position, 3, t.UI_ACCENT, None, 1)
    c.dot(323 + position, 1050, t.UI_ACCENT, 4)
    c.text(1206, 1038, "REPLAY", 12, t.UI_MUTED, True)
    visits(c, state, (1352, 310, 528, 240))
    dwell(c, state, (1352, 568, 528, 285))
    c.card(1352, 871, 528, 196)
    c.text(1376, 887, "Recent activity", 23, bold=True)
    names = {"zone_enter": "Visit started", "zone_exit": "Visit completed",
             "queue_join": "Joined queue", "queue_leave": "Left queue"}
    recent = [r for r in activity if float(r["time_s"]) <= state.clock_s
              and r["event"] in names][-3:]
    for i, row in enumerate(reversed(recent)):
        yy = 929 + i * 43
        c.dot(1380, yy + 12, t.UI_ACCENT, 4)
        if i < len(recent) - 1:
            c.line([(1380, yy + 21), (1380, yy + 46)])
        c.text(1398, yy - 1, names[row["event"]], 16, t.UI_TEXT, True)
        c.text(1398, yy + 20, row["place"], 13, t.UI_MUTED)
        c.text(1856, yy, clock(float(row["time_s"])), 14, t.UI_MUTED, align="right")
    if not recent:
        c.text(1376, 949, "Waiting for activity", 16, t.UI_MUTED)
    return c.array()


def report(states: list[PanelState], notes: list[str], caption: str, sub: str) -> np.ndarray:
    c = shell("Analytics", "Session summary", sub)
    state = states[0]
    kpis(c, state)
    visits(c, state, (128, 310, 852, 414))
    dwell(c, state, (1000, 310, 880, 414))
    c.text(128, 748, "Session insights", 25, bold=True)
    c.text(1880, 754, "Full clip / " + clock(state.clock_s), 16,
           t.UI_MUTED, align="right")
    width = (1752 - 20 * (len(notes) - 1)) // max(1, len(notes))
    for i, note in enumerate(notes):
        x = 128 + i * (width + 20)
        c.card(x, 795, width, 251)
        c.box(x + 24, 819, 40, 40, t.UI_TINT, None, 11)
        c.icon(x + 32, 827, "chart" if i == 0 else "grid", t.UI_ACCENT)
        c.text(x + 80, 823, "Area dwell" if i == 0 else "Queue demand", 21, bold=True)
        c.text(x + 24, 889, note, 28, t.UI_SECONDARY, width=width - 48)
    return c.array()


def email(subject: str, body: str) -> np.ndarray:
    c = shell("Notifications", "Notifications", "Counter monitoring / Email notifications")
    c.card(128, 204, 350, 843)
    c.text(152, 229, "Inbox", 25, bold=True)
    c.box(152, 282, 302, 42, t.UI_BG, None, 8)
    c.icon(168, 291, "bell", t.UI_MUTED)
    c.text(204, 290, "Counter alerts", 17, t.UI_SECONDARY)
    c.line([(128, 351), (478, 351)])
    c.box(142, 366, 322, 142, t.UI_TINT, None, 10)
    c.dot(165, 390, t.UI_DANGER)
    c.text(181, 376, "Counter unattended", 20, bold=True)
    c.text(165, 416, "Cafe camera alerts", 16, t.UI_SECONDARY)

    c.card(502, 204, 1378, 843)
    c.box(534, 230, 48, 48, t.UI_TINT, None, 13)
    c.icon(546, 242, "bell", t.UI_ACCENT)
    c.text(601, 230, "Cafe camera alerts", 23, bold=True)
    c.text(601, 263, "To: Cafe manager", 16, t.UI_MUTED)
    c.line([(502, 310), (1880, 310)])
    y = c.text(534, 345, subject, 31, bold=True, width=1290)
    lines = [line for line in body.splitlines() if line.strip()]
    y += 35
    c.box(534, y, 1314, 70, t.UI_DANGER_TINT, None, 10)
    c.dot(559, y + 35, t.UI_DANGER)
    c.text(579, y + 20, lines[0], 23, t.UI_DANGER, width=1233)
    y += 112
    for i, line in enumerate(lines[1:4]):
        label, _, value = line.partition(":")
        x = 534 + i * 445
        c.text(x, y, label.upper(), 13, t.UI_MUTED, True)
        c.text(x, y + 29, value.strip(), 24, t.UI_TEXT, width=414)
    return c.array()


def measured(rows: list[tuple[str, str, str]], comparison: tuple[int, int] | None) -> np.ndarray:
    c = shell("Analytics", "Tracking performance",
              "Validation against MEVA ground-truth annotations")
    if len(rows) == 1 and comparison:
        label, big, rest = rows[0]
        c.box(236, 191, 1648, 200)
        c.text(264, 213, label, 20, t.UI_MUTED)
        c.text(264, 252, big, 66, t.UI_ACCENT, True)
        c.text(785, 258, rest, 25, t.UI_SECONDARY, width=1045)
        c.box(236, 415, 988, 597)
        c.text(266, 439, "Identity switches", 25, bold=True)
        c.text(266, 480, "Lower is better / Full 5-minute clip", 17, t.UI_MUTED)
        maximum = max(*comparison, 1)
        for i, (label, value) in enumerate(zip(("This system", "ByteTrack"), comparison,
                                              strict=True)):
            y = 557 + i * 150
            c.text(266, y, label, 23, t.UI_SECONDARY)
            c.text(1104, y, str(value), 25, t.UI_TEXT, True)
            c.box(266, y + 49, 918, 37, t.UI_LINE, None, 5)
            c.box(266, y + 49, int(918 * value / maximum), 37,
                  t.UI_ACCENT if i == 0 else t.UI_MUTED, None, 5)
        c.text(266, 923, "Same detections for both trackers", 19, t.UI_MUTED)
        c.box(1248, 415, 636, 597)
        c.text(1278, 439, "Evaluation context", 25, bold=True)
        y = 506
        for label, value in (("DATASET", "MEVA annotations"),
                             ("EVALUATION", "Footage the system was not tuned on"),
                             ("REFERENCE", "ByteTrack (motion only)"),
                             ("METHOD & LIMITS", "metrics/meva_eval.md")):
            c.text(1278, y, label, 13, t.UI_MUTED, True)
            c.text(1278, y + 27, value, 22, t.UI_SECONDARY, width=568)
            y += 113
    else:
        for i, (label, big, rest) in enumerate(rows):
            y = 191 + i * 255
            c.box(236, y, 1648, 235)
            c.text(262, y + 20, label, 22, t.UI_MUTED)
            c.text(262, y + 69, big, 60, t.UI_ACCENT, True)
            c.text(780, y + 88, rest, 26, t.UI_SECONDARY, width=1060)
    return c.array()


def end(title: list[str], author: str) -> np.ndarray:
    c = shell("Overview", "Retail vision analytics", "Computer vision for retail operations")
    c.box(236, 191, 1648, 821)
    c.box(290, 245, 216, 34, t.UI_TINT, None, 5)
    c.text(305, 251, "MEVA / PROJECT DEMO", 14, t.UI_ACCENT, True)
    y = 345
    for line in title:
        y = c.text(290, y, line, 51, bold=True, width=1530) + 12
    c.text(290, y + 50, author, 27, t.UI_SECONDARY, True)
    c.text(290, y + 102, "Personal demo on the MEVA dataset (CC BY 4.0)", 23, t.UI_MUTED)
    c.line([(290, 781), (1826, 781)])
    for i, (icon, label) in enumerate((("camera", "Queue monitoring"),
                                      ("clock", "Time per area"), ("bell", "Staff alerts"))):
        x = 290 + i * 518
        c.icon(x, 833, icon, t.UI_ACCENT)
        c.text(x + 40, 830, label, 23, t.UI_SECONDARY)
    return c.array()


def thumbnail(frame: np.ndarray, boxes: list) -> np.ndarray:
    """Portfolio cover with a large title and a focused crop of actual detections."""
    c = Canvas(t.UI_RAIL)
    c.box(92, 221, 64, 7, t.SUCCESS, None, 3)
    c.text(88, 284, "RETAIL", 142, t.INK, True)
    c.text(88, 456, "ANALYTICS", 126, t.INK, True)
    c.text(94, 672, "Queues. Dwell. Alerts.", 45, t.UI_ON_DARK)

    source = Image.fromarray(frame[..., ::-1])
    sw, sh = source.size
    ph, pw = 932, 994
    crop_h = int(sh * .93)
    crop_w = int(crop_h * pw / ph)
    left = max(0, min(sw - crop_w, int(sw * .48) - crop_w // 2))
    top = max(0, sh - crop_h)
    camera = source.crop((left, top, left + crop_w, top + crop_h)).resize(
        (pw * 2, ph * 2), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(camera)
    visible = [row for row in boxes if left < (row[1][0] + row[1][2]) / 2 < left + crop_w]
    visible.sort(key=lambda row: (row[1][2] - row[1][0]) * (row[1][3] - row[1][1]), reverse=True)
    # A few true detections communicate tracking without a wall of tiny labels.
    for _, (x1, y1, x2, y2), _ in visible[:3]:
        bounds = ((x1 - left) / crop_w * pw * 2, (y1 - top) / crop_h * ph * 2,
                  (x2 - left) / crop_w * pw * 2, (y2 - top) / crop_h * ph * 2)
        d.rectangle(bounds, outline=t.SUCCESS, width=7)
    mask = Image.new("L", camera.size)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, pw * 2 - 1, ph * 2 - 1), radius=36, fill=255)
    c.box(845, 64, 1014, 952, t.UI_RAIL_ACTIVE, None, 24)
    c.image.paste(camera, (855 * 2, 74 * 2), mask)
    return c.array()


def _rgb(code: str) -> tuple[int, int, int]:
    code = code.lstrip("#")
    return tuple(int(code[i:i + 2], 16) for i in (0, 2, 4))


def _fit_font(text: str, max_w: int, start: int, bold: bool = True):
    size = start
    while size > 12 and font(size, bold).getlength(text) > max_w:
        size -= 2
    return font(size, bold)


def cover(
    frame: np.ndarray,
    boxes: list,
    queue_ids: set[int],
    queue_poly: list,
    staff_poly: list,
    waiting: int,
    wait_s: float | None,
    staffed: bool,
    features: tuple[tuple[str, str], ...] = (
        ("People counting", t.SUCCESS), ("Dwell time", t.CYAN),
        ("Queue & wait", t.QUEUE), ("Staff alerts", t.DANGER),
    ),
) -> np.ndarray:
    """1600x1200 camera cover with a few overlays from the replayed state."""
    width, height, scale = 1600, 1200, 2
    safe_x = 130  # Keep overlay text inside the portfolio tile's side crops.
    safe_w = width - 2 * safe_x
    source = Image.fromarray(frame[..., ::-1])
    crop_h = min(source.height, int(source.width * height / width))
    crop_h = int(crop_h * .98)
    crop_w = int(crop_h * width / height)
    top = source.height - crop_h
    out = source.crop((0, top, crop_w, top + crop_h)).resize(
        (width * scale, height * scale), Image.Resampling.LANCZOS).convert("RGBA")
    overlay = Image.new("RGBA", out.size)
    d = ImageDraw.Draw(overlay)

    def point(x: float, y: float) -> tuple[float, float]:
        return x / crop_w * width * scale, (y - top) / crop_h * height * scale

    d.polygon([point(*p) for p in queue_poly], fill=(*t.rgb(t.QUEUE), 20),
              outline=(*t.rgb(t.QUEUE), 255), width=4 * scale)
    staff_color = t.SUCCESS if staffed else t.DANGER
    d.polygon([point(*p) for p in staff_poly], fill=(*t.rgb(staff_color), 38),
              outline=(*t.rgb(staff_color), 255), width=4 * scale)
    selected = [row for row in boxes if row[0] in queue_ids]
    if not selected:
        selected = sorted(boxes, key=lambda row: (row[1][2] - row[1][0]) *
                          (row[1][3] - row[1][1]), reverse=True)[:3]
    for gid, (x1, y1, x2, y2), _ in selected:
        color = t.QUEUE if gid in queue_ids else t.SUCCESS
        d.rectangle((*point(x1, y1), *point(x2, y2)), outline=(*t.rgb(color), 255),
                    width=4 * scale)
    out = Image.alpha_composite(out, overlay)

    # Gentle edge shading supports the overlays without darkening people's faces.
    shade = Image.new("RGBA", (1, height * scale))
    for y in range(height * scale):
        logical_y = y / scale
        strength = max(0, (260 - logical_y) / 260, (logical_y - 900) / 300)
        shade.putpixel((0, y), (*t.rgb(t.UI_RAIL), int(100 * strength)))
    out = Image.alpha_composite(out, shade.resize(out.size))
    overlay = Image.new("RGBA", out.size)
    d = ImageDraw.Draw(overlay)

    def rect(x: int, y: int, w: int, h: int, color: str, alpha: int = 255,
             radius: int = 14) -> None:
        d.rounded_rectangle((x * scale, y * scale, (x + w) * scale, (y + h) * scale),
                            radius=radius * scale, fill=(*t.rgb(color), alpha))

    def text(x: int, y: int, value: str, size: int, color: str = t.INK,
             bold: bool = False, max_w: int | None = None) -> None:
        face = _fit_font(value, max_w * scale, size * scale, bold) if max_w else font(
            size * scale, bold)
        d.text((x * scale, y * scale), value, font=face, fill=color, anchor="lt")

    # Keep each class label attached to its box and above the person's head.
    label_w = int(font(34 * scale, True).getlength("Person") / scale) + 28
    for gid, (x1, y1, _, _), _ in selected:
        x, y = point(x1, y1)
        x = max(safe_x, min(width - safe_x - label_w, int(x / scale)))
        y = max(64, int(y / scale) - 48)
        color = t.QUEUE if gid in queue_ids else t.SUCCESS
        rect(x, y, label_w, 48, color, radius=6)
        text(x + 14, y + 9, "Person", 34, t.UI_TEXT, True)

    rect(safe_x, 64, 314, 62, t.UI_RAIL, 238)
    d.ellipse(((safe_x + 22) * scale, 85 * scale, (safe_x + 40) * scale, 103 * scale),
              fill=t.SUCCESS)
    text(safe_x + 58, 80, "People tracking", 32, bold=True)

    alert = not staffed and waiting > 0
    status_color = t.UI_DANGER if alert else t.UI_ACCENT
    alert_x = width - safe_x - 606
    rect(alert_x, 70, 606, 178, t.UI_RAIL, 40)
    rect(alert_x, 64, 606, 178, t.UI_CARD, 249)
    rect(alert_x, 64, 7, 178, status_color, radius=3)
    text(alert_x + 32, 86, "STAFF ALERT" if alert else "COUNTER STATUS", 27, status_color, True)
    status = "Counter unattended" if alert else "Counter staffed" if staffed else "Counter empty"
    text(alert_x + 32, 128, status, 43, t.UI_TEXT, True, max_w=542)
    text(alert_x + 32, 190, "Email notification" if alert else "Service area occupied" if staffed
         else "No customers waiting", 30, t.UI_SECONDARY, max_w=542)

    # Label the counter immediately above its polygon, clear of the customers.
    counter_y = max(190, min(830, int(min(point(*p)[1] for p in staff_poly) / scale) - 66))
    label = "Counter staffed" if staffed else "Counter empty"
    face = font(35 * scale, True)
    label_w = int(face.getlength(label) / scale) + 40
    rect(safe_x, counter_y, label_w, 56, t.UI_RAIL, 240)
    rect(safe_x, counter_y, 5, 56, staff_color, radius=2)
    text(safe_x + 20, counter_y + 11, label, 35, bold=True)

    # Keep the queue readout on the lower floor so faces remain unobstructed.
    queue_x = width - safe_x - 576
    rect(queue_x, 922, 576, 142, t.UI_RAIL, 240)
    rect(queue_x, 922, 6, 142, t.QUEUE, radius=3)
    text(queue_x + 32, 944, f"{waiting} in queue", 58, bold=True, max_w=512)
    wait = f"~{int(round(wait_s))} s wait" if wait_s is not None else "Wait estimate pending"
    text(queue_x + 32, 1013, wait, 40, t.UI_ON_DARK, max_w=512)

    # A single translucent row communicates the remaining capabilities.
    rect(safe_x, 1090, safe_w, 70, t.UI_RAIL, 235)
    feature_w = safe_w / max(1, len(features))
    for i, (label, color) in enumerate(features):
        x = int(safe_x + 22 + i * feature_w)
        d.ellipse((x * scale, 1118 * scale, (x + 14) * scale, 1132 * scale), fill=color)
        text(x + 28, 1107, label, 35, bold=True, max_w=int(feature_w - 60))

    out = Image.alpha_composite(out, overlay).convert("RGB").resize(
        (width, height), Image.Resampling.LANCZOS)
    return np.asarray(out)[..., ::-1].copy()
