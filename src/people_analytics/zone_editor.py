"""Draw zones, the queue and the staff area on a frame, and save them to a config.

    people-analytics zone-editor --config configs/meva_cafe.yaml

The background is the clip's person-free median frame (or `--frame-s` seconds in).

Keys (also shown in the window, H toggles the help):

    Z  new area zone        Q  new queue area        S  staff area (behind the counter)
    click   add a point     right-click / Backspace   remove the last point
    Enter   finish the shape, then type its name and press Enter again
    drag a corner of any shape to move it
    D  then click inside a shape to delete it
    W  save to the config   Esc  quit (twice if there are unsaved changes)

Saving rewrites only the `zones:`, `queues:` and `staff:` sections of the YAML;
everything else (tracker, notify, comments elsewhere) is left as it was. Settings
of an existing queue or staff area (speeds, alert timing) are kept when it is
redrawn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml

from people_analytics.analytics import theme
from people_analytics.analytics.render import AQUA, BLUE, MAGENTA, ORANGE, YELLOW, pill

KIND_KEYS = {"zone": "zones", "queue": "queues", "staff": "staff"}
DEFAULT_NAMES = {"queue": "counter queue", "staff": "behind the counter"}
QUEUE_DEFAULTS = {"anchor": "bottom_center", "max_speed": 0.5, "min_visit_s": 3.0, "recent": 10}
STAFF_DEFAULTS = {"anchor": "center", "alert_after_s": 120, "min_waiting": 1,
                  "back_after_s": 1.0, "repeat_after_s": 300}
ZONE_COLOURS = [BLUE, MAGENTA, YELLOW]
_DARK = sv.Color.from_hex(theme.SURFACE)


@dataclass
class Shape:
    kind: str  # zone | queue | staff
    name: str
    points: list[tuple[int, int]]
    extra: dict = field(default_factory=dict)  # other settings, kept on save

    def contains(self, x: float, y: float) -> bool:
        poly = np.array(self.points, np.float32)
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


# --- load / save ---------------------------------------------------------------


def load_shapes(config_path: str | Path) -> list[Shape]:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    shapes: list[Shape] = []
    for z in raw.get("zones") or []:
        shapes.append(Shape("zone", z["name"], [tuple(p) for p in z["polygon"]],
                            {k: v for k, v in z.items() if k not in ("name", "polygon")}))
    for q in raw.get("queues") or []:
        shapes.append(Shape("queue", q["name"], [tuple(p) for p in q["polygon"]],
                            {k: v for k, v in q.items() if k not in ("name", "polygon")}))
    s = raw.get("staff")
    if s:
        shapes.append(Shape("staff", s.get("name", DEFAULT_NAMES["staff"]),
                            [tuple(p) for p in s["polygon"]],
                            {k: v for k, v in s.items() if k not in ("name", "polygon")}))
    return shapes


def _item(shape: Shape, indent: str, first: str) -> list[str]:
    pts = json.dumps([list(map(int, p)) for p in shape.points])
    lines = [f"{indent}{first}name: {json.dumps(shape.name)}",
             f"{indent}{' ' * len(first)}polygon: {pts}"]
    for k, v in shape.extra.items():
        lines.append(f"{indent}{' ' * len(first)}{k}: {json.dumps(v)}")
    return lines


def _block(key: str, shapes: list[Shape]) -> str | None:
    if not shapes:
        return None
    if key == "staff":
        return "\n".join(["staff:", *_item(shapes[0], "  ", "")]) + "\n"
    lines = [f"{key}:"]
    for sh in shapes:
        lines += _item(sh, "  ", "- ")
    return "\n".join(lines) + "\n"


_TOP_KEY = re.compile(r"^[A-Za-z_][\w-]*:")


def replace_block(text: str, key: str, block: str | None, before: str = "analytics") -> str:
    """Replace (or insert, or remove) the top-level `key:` section of a YAML text.

    The section runs from its key line to the next top-level key; comment and
    blank lines just above that next key stay with it.
    """
    lines = text.splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if _TOP_KEY.match(ln)]
    here = next((i for i in starts if lines[i].startswith(f"{key}:")), None)
    if here is not None:
        nxt = next((i for i in starts if i > here), len(lines))
        end = nxt
        while end - 1 > here and (lines[end - 1].startswith("#") or not lines[end - 1].strip()):
            end -= 1
        new = [] if block is None else [block]
        if block is not None and end < len(lines):
            new.append("\n")
        return "".join(lines[:here] + new + lines[end:])
    if block is None:
        return text
    anchor = next((i for i in starts if lines[i].startswith(f"{before}:")), None)
    if anchor is None:
        sep = "" if text.endswith("\n") or not text else "\n"
        return text + sep + "\n" + block
    return "".join(lines[:anchor] + [block, "\n"] + lines[anchor:])


def save_shapes(config_path: str | Path, shapes: list[Shape]) -> None:
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    queues = [s for s in shapes if s.kind == "queue"]
    staff = [s for s in shapes if s.kind == "staff"][-1:]
    for sh in queues:
        for k, v in QUEUE_DEFAULTS.items():
            sh.extra.setdefault(k, v)
    for sh in staff:
        for k, v in STAFF_DEFAULTS.items():
            sh.extra.setdefault(k, v)
        if queues:
            sh.extra.setdefault("queue", queues[0].name)
    text = replace_block(text, "zones", _block("zones", [s for s in shapes if s.kind == "zone"]))
    text = replace_block(text, "queues", _block("queues", queues))
    text = replace_block(text, "staff", _block("staff", staff))
    yaml.safe_load(text)  # never write a config that no longer parses
    path.write_text(text, encoding="utf-8")


# --- the window ----------------------------------------------------------------

HELP = [
    "Z new zone   Q new queue   S staff area (behind the counter)",
    "click: add point   right-click/Backspace: undo point   Enter: finish + name",
    "drag a corner to move it   D + click: delete a shape",
    "W save   Esc quit   H hide help",
]


class Editor:
    def __init__(self, image: np.ndarray, shapes: list[Shape], config_path: Path,
                 max_size: tuple[int, int] = (1600, 900)):
        self.image = image
        self.shapes = shapes
        self.path = config_path
        self.scale = min(max_size[0] / image.shape[1], max_size[1] / image.shape[0], 1.0)
        self.mode = "idle"  # idle | draw | name | delete
        self.kind = "zone"
        self.points: list[tuple[int, int]] = []
        self.name = ""
        self.mouse = (0, 0)
        self.drag: tuple[int, int] | None = None  # (shape index, vertex index)
        self.dirty = False
        self.help = True
        self.status = "Ready"
        self.quit_armed = False

    def to_img(self, x: int, y: int) -> tuple[int, int]:
        return int(round(x / self.scale)), int(round(y / self.scale))

    def _near_vertex(self, x: int, y: int) -> tuple[int, int] | None:
        r = 10 / self.scale
        for i, sh in enumerate(self.shapes):
            for j, (px, py) in enumerate(sh.points):
                if (px - x) ** 2 + (py - y) ** 2 <= r * r:
                    return i, j
        return None

    def on_mouse(self, event, x, y, flags, _param) -> None:
        ix, iy = self.to_img(x, y)
        self.mouse = (ix, iy)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "draw":
                self.points.append((ix, iy))
            elif self.mode == "delete":
                hit = next((i for i in range(len(self.shapes) - 1, -1, -1)
                            if self.shapes[i].contains(ix, iy)), None)
                if hit is not None:
                    gone = self.shapes.pop(hit)
                    self.dirty = True
                    self.status = f"Deleted {gone.kind} '{gone.name}'"
                self.mode = "idle"
            elif self.mode == "idle":
                self.drag = self._near_vertex(ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            i, j = self.drag
            self.shapes[i].points[j] = (ix, iy)
            self.dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None
        elif event == cv2.EVENT_RBUTTONDOWN and self.mode == "draw" and self.points:
            self.points.pop()

    def on_key(self, key: int) -> bool:
        """Handle a key; returns False to quit."""
        if key < 0:
            return True
        if self.mode == "name":
            if key in (13, 10):
                self._finish_shape()
            elif key == 27:
                self.mode, self.points = "idle", []
                self.status = "Cancelled"
            elif key == 8:
                self.name = self.name[:-1]
            elif 32 <= key < 127:
                self.name += chr(key)
            return True
        ch = chr(key).lower() if 0 <= key < 256 else ""
        if key == 27:
            if self.mode in ("draw", "delete"):
                self.mode, self.points = "idle", []
                self.status = "Cancelled"
                return True
            if self.dirty and not self.quit_armed:
                self.quit_armed = True
                self.status = "Unsaved changes: W to save, Esc again to quit without saving"
                return True
            return False
        self.quit_armed = False
        if self.mode == "draw":
            if key in (13, 10) and len(self.points) >= 3:
                self.mode = "name"
                n = sum(1 for s in self.shapes if s.kind == "zone") + 1
                self.name = DEFAULT_NAMES.get(self.kind, f"zone {n}")
            elif key == 8 and self.points:
                self.points.pop()
            return True
        if ch in ("z", "q", "s"):
            self.kind = {"z": "zone", "q": "queue", "s": "staff"}[ch]
            self.mode, self.points = "draw", []
            self.status = f"Drawing a {self.kind}: click the corners, Enter when done"
        elif ch == "d":
            self.mode = "delete"
            self.status = "Click inside the shape to delete"
        elif ch == "w":
            save_shapes(self.path, self.shapes)
            self.dirty = False
            self.status = f"Saved to {self.path}"
        elif ch == "h":
            self.help = not self.help
        return True

    def _finish_shape(self) -> None:
        name = self.name.strip() or DEFAULT_NAMES.get(self.kind, "zone")
        if self.kind == "staff":  # one staff area: a new one replaces the old
            old = next((s for s in self.shapes if s.kind == "staff"), None)
            extra = old.extra if old else {}
            self.shapes = [s for s in self.shapes if s.kind != "staff"]
        else:
            same = next((s for s in self.shapes if s.kind == self.kind and s.name == name), None)
            extra = same.extra if same else {}
            if same:
                self.shapes.remove(same)
        self.shapes.append(Shape(self.kind, name, list(self.points), dict(extra)))
        self.mode, self.points, self.dirty = "idle", [], True
        self.status = f"Added {self.kind} '{name}' (W to save)"

    def _colour(self, sh: Shape):
        if sh.kind == "queue":
            return ORANGE
        if sh.kind == "staff":
            return AQUA
        zones = [s for s in self.shapes if s.kind == "zone"]
        return ZONE_COLOURS[zones.index(sh) % len(ZONE_COLOURS)]

    def render(self) -> np.ndarray:
        img = self.image.copy()
        for sh in self.shapes:
            col = self._colour(sh)
            pts = np.array(sh.points, np.int32)
            cv2.polylines(img, [pts], True, col.as_bgr(), 2, cv2.LINE_AA)
            for p in sh.points:
                cv2.circle(img, tuple(map(int, p)), 6, col.as_bgr(), -1, cv2.LINE_AA)
            c = pts.mean(axis=0).astype(int)
            pill(img, f"{sh.kind}: {sh.name}", (int(c[0]), int(c[1])), col, size=20)
        if self.mode in ("draw", "name") and self.points:
            pts = np.array(self.points + ([self.mouse] if self.mode == "draw" else []), np.int32)
            cv2.polylines(img, [pts], self.mode == "name", theme.bgr(theme.CYAN), 2, cv2.LINE_AA)
            for p in self.points:
                cv2.circle(img, p, 6, theme.bgr(theme.CYAN), -1, cv2.LINE_AA)
        view = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        lines = list(HELP) if self.help else []
        if self.mode == "name":
            lines.append(f"Name: {self.name}_   (Enter to accept, Esc to cancel)")
        lines.append(self.status + ("   *unsaved*" if self.dirty else ""))
        y = 10
        for ln in lines:
            _, _, _, h = pill(view, ln, (10, y + 14), _DARK, size=16, align="left", bold=False)
            y += h + 4
        return view

    def run(self, title: str = "zone editor") -> None:
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self.on_mouse)
        while True:
            cv2.imshow(title, self.render())
            if not self.on_key(cv2.waitKey(30)):
                break
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()




def background(config_path: str | Path, frame_s: float | None) -> np.ndarray:
    """The scene without people (or one frame, at `frame_s` seconds)."""
    from people_analytics.analytics.sitemap import load_background
    from people_analytics.config import load_config

    cfg = load_config(config_path)
    if frame_s is None:
        # Person-masked median when the clip has been analysed (cleaner in a busy
        # scene), plain median otherwise. Cached either way.
        events = Path(cfg.io.output_dir) / "analytics" / f"{Path(cfg.io.source).stem}_events.csv"
        cache = Path(cfg.io.output_dir) / "editor"
        return load_background(cfg.io.source, cache, str(events) if events.exists() else None)
    cap = cv2.VideoCapture(cfg.io.source)
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read a frame at {frame_s} s")
    return frame
