"""Site-map heatmap and walking paths from several cameras on one floor plan.

Each camera's per-detection export (`*_events.csv`) is projected to its room's
floor in metres (`analytics.floor`), the room is placed on a shared plan
(rotation + offset, in metres), and everything is drawn in plan pixels:

- **floor**: each camera's person-free background warped top-down, so the plan
  shows the real floor (court lines, tiles, mats) rather than a drawing;
- **heat**: person-seconds per square metre, blurred by ~0.5 m, coloured on a
  scale per room (top = that room's busiest spot), so a quiet lobby still reads
  next to a packed gym; each room's top value is reported alongside;
- **paths**: each identity's feet trail, smoothed, for tracks long enough to be
  a walk rather than a fragment.

Where two cameras see the same floor, each floor point is taken from one camera
only (its `owns` polygon), so the overlap isn't counted twice.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from people_analytics.analytics.floor import (
    FloorProjector,
    krtd_projector,
    mats_homography,
    median_background,
    tile_homography,
)

INK = (235, 235, 235)
PAPER = (25, 26, 26)  # BGR of #1a1a19, the stats panel's surface
WALL = (90, 96, 104)


@dataclass
class Room:
    name: str
    label: str
    outline: np.ndarray  # room-local metres
    rotate_deg: float = 0.0
    offset: tuple[float, float] = (0.0, 0.0)

    def to_plan(self, pts: np.ndarray) -> np.ndarray:
        """Room-local metres -> plan metres (x right, y up)."""
        a = np.radians(self.rotate_deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return np.asarray(pts, float) @ R.T + np.asarray(self.offset, float)

    def matrix(self) -> np.ndarray:
        a = np.radians(self.rotate_deg)
        return np.array([[np.cos(a), -np.sin(a), self.offset[0]],
                         [np.sin(a), np.cos(a), self.offset[1]], [0, 0, 1]])


@dataclass
class Camera:
    name: str
    video: str
    events: str
    room: Room
    projector: FloorProjector
    owns: np.ndarray | None = None  # room-local polygon this camera is trusted for
    fps: float = 30.0
    tracks: dict[int, list[tuple[int, float, float]]] = field(default_factory=dict)


class SitePlan:
    """Plan canvas: plan metres (x right, y up) -> pixels (y down).

    Sized to the rooms it shows, plus room for labels around them, so the
    layout in the YAML can move rooms without also re-tuning a canvas size.
    """

    # Margins in pixels: left, top (title + labels), right (side labels), bottom (notes).
    MARGIN_PX = (40, 90, 40, 170)

    def __init__(self, rooms: list[Room], px_per_m: float, extra_right_px: int = 0):
        pts = np.vstack([r.to_plan(r.outline) for r in rooms])
        self.lo, self.hi = pts.min(axis=0), pts.max(axis=0)
        self.ppm = px_per_m
        left, top, right, bottom = self.MARGIN_PX
        self.pad = (left, top)
        self.wh = (int((self.hi[0] - self.lo[0]) * px_per_m) + left + right + extra_right_px,
                   int((self.hi[1] - self.lo[1]) * px_per_m) + top + bottom)

    def matrix(self) -> np.ndarray:
        """Plan metres -> pixels."""
        s = self.ppm
        left, top = self.pad
        return np.array([[s, 0, left - self.lo[0] * s], [0, -s, top + self.hi[1] * s], [0, 0, 1]])

    def px(self, pts: np.ndarray) -> np.ndarray:
        p = np.column_stack([pts, np.ones(len(pts))]) @ self.matrix().T
        return p[:, :2]


def _inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, np.float32)
    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in pts])


def build_projector(floor: dict, video: str, cache_dir: Path) -> tuple[FloorProjector, dict]:
    # The tile fit always uses the plain median background, so the calibration
    # doesn't change depending on whether an analytics run exists yet.
    method = floor["method"]
    if method == "krtd":
        return krtd_projector(floor["krtd"]), {"method": "krtd"}
    if method == "tiles":
        bg = load_background(video, cache_dir)
        projector, diag = tile_homography(
            bg, tuple(floor["roi"]), tuple(map(tuple, floor["line_angles"])), floor["tile_m"]
        )
        return projector, {"method": "tiles", **diag}
    if method == "mats":
        projector, diag = mats_homography(
            np.array(floor["corners"]), floor["mat_width_m"],
            tuple(floor.get("principal_point", (959.5, 535.5))),
        )
        return projector, {"method": "mats", **diag}
    raise ValueError(f"Unknown floor method '{method}'")


def load_background(video: str, cache_dir: Path, events: str | None = None) -> np.ndarray:
    """Person-free view of the scene, cached. Uses the detections when available."""
    clean = events is not None and Path(events).exists() and Path(events).stat().st_size > 200
    # Keyed on the export's size + mtime: a rerun (or a run still being written)
    # must not leave a stale cached background behind.
    tag = ""
    if clean:
        st = Path(events).stat()
        tag = f"clean{st.st_size}_{int(st.st_mtime)}_"
    path = cache_dir / f"bg_{tag}{Path(video).stem}.png"
    if path.exists():
        return cv2.imread(str(path))
    cache_dir.mkdir(parents=True, exist_ok=True)
    bg = clean_background(video, events) if clean else median_background(video)
    cv2.imwrite(str(path), bg)
    return bg


def clean_background(video: str, events: str, n: int = 60, pad: int = 12) -> np.ndarray:
    """Median of `n` frames using only pixels no detected person covers in that frame.

    A plain median keeps anyone who stands in one place for most of the clip (the
    gym crowd round the chairs smears across the floor). Masking each frame's
    person boxes leaves only real floor samples; where a pixel is covered in every
    sampled frame the plain median fills it in.
    """
    boxes: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    with open(events, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            boxes[int(r["frame"]) - 1].append(  # events frames are 1-based
                (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
            )
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, masks = [], []
    for f in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            continue
        m = np.zeros(frame.shape[:2], bool)
        for x1, y1, x2, y2 in boxes.get(int(f), []):
            m[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = True
        frames.append(frame)
        masks.append(m)
    cap.release()
    stack, covered = np.stack(frames), np.stack(masks)
    out = np.median(stack, axis=0).astype(np.uint8)
    # Row strips keep the float copy small (60 x 1080 x 1920 x 3 would be ~1.5 GB).
    for y0 in range(0, stack.shape[1], 120):
        chunk = stack[:, y0:y0 + 120].astype(np.float32)
        chunk[covered[:, y0:y0 + 120]] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(chunk, axis=0)
        ok = ~np.isnan(med)
        out[y0:y0 + 120][ok] = med[ok].astype(np.uint8)
    return out


def load_tracks(camera: Camera, min_track_s: float) -> None:
    """Feet of every detection, projected to room metres, grouped by identity.

    Boxes touching the bottom of the image are skipped: their bottom edge is the
    frame border, not the person's feet, and it lands on a false floor spot (a
    row of fake hot spots right under the camera).
    """
    cap = cv2.VideoCapture(camera.video)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    rows: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    with open(camera.events, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            x1, x2, y2 = float(r["x1"]), float(r["x2"]), float(r["y2"])
            if y2 >= frame_h - 3:
                continue
            rows[int(r["global_id"])].append((int(r["frame"]), (x1 + x2) / 2, y2))
    min_frames = min_track_s * camera.fps
    for gid, pts in rows.items():
        if len(pts) < min_frames:
            continue
        arr = np.array(pts, float)
        floor = camera.projector(arr[:, 1:3])
        keep = _inside(camera.room.outline, floor)
        if camera.owns is not None:
            keep &= _inside(camera.owns, floor)
        if keep.sum() >= 2:
            camera.tracks[gid] = [(int(f), x, y) for f, (x, y) in zip(arr[keep, 0], floor[keep],
                                                                      strict=True)]


def floor_layer(plan: SitePlan, cameras: list[Camera], cache_dir: Path) -> np.ndarray:
    """Each room's floor, seen from its camera(s), warped onto the plan."""
    canvas = np.full((plan.wh[1], plan.wh[0], 3), PAPER, np.uint8)
    for cam in cameras:
        bg = load_background(cam.video, cache_dir, cam.events)
        p = cam.projector
        if p.K is not None and p.D is not None:
            bg = cv2.undistort(bg, p.K, p.D)
        M = plan.matrix() @ cam.room.matrix() @ p.H
        warped = cv2.warpPerspective(bg, M, plan.wh)
        mask = np.zeros(canvas.shape[:2], np.uint8)
        region = cam.owns if cam.owns is not None else cam.room.outline
        cv2.fillPoly(mask, [plan.px(cam.room.to_plan(region)).astype(np.int32)], 255)
        # Only where this camera actually sees the floor (in front of it).
        visible = cv2.warpPerspective(np.full(bg.shape[:2], 255, np.uint8), M, plan.wh)
        mask &= visible
        dim = (warped.astype(np.float32) * 0.55).astype(np.uint8)  # keep heat readable on top
        canvas[mask > 0] = dim[mask > 0]
    return canvas


def heat_layer(plan: SitePlan, cameras: list[Camera], sigma_m: float = 0.5) -> np.ndarray:
    """Person-seconds per square metre on the plan grid (float32, plan pixels)."""
    heat = np.zeros((plan.wh[1], plan.wh[0]), np.float32)
    for cam in cameras:
        for pts in cam.tracks.values():
            xy = cam.room.to_plan(np.array([(x, y) for _, x, y in pts]))
            px = plan.px(xy).astype(int)
            w, h = plan.wh
            ok = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
            np.add.at(heat, (px[ok, 1], px[ok, 0]), 1.0 / cam.fps)
    heat *= plan.ppm ** 2  # per pixel -> per square metre
    return cv2.GaussianBlur(heat, (0, 0), sigma_m * plan.ppm)


def colourise(heat: np.ndarray, vmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Inferno-coloured heat plus an alpha mask (transparent where there's none)."""
    norm = np.clip(heat / max(vmax, 1e-9), 0, 1)
    colour = cv2.applyColorMap((np.sqrt(norm) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = np.clip(np.sqrt(norm) * 1.6, 0, 0.9)
    alpha[norm < 0.02] = 0
    return colour, alpha


def smooth_path(pts: np.ndarray, window: int = 31) -> np.ndarray:
    """Moving average over ~1 s: far from a camera, feet jitter mostly in depth."""
    if len(pts) < window:
        return pts
    k = np.ones(window) / window
    return np.column_stack([np.convolve(pts[:, i], k, mode="valid") for i in range(2)])


def path_pieces(arr: np.ndarray, max_gap_frames: int = 15, max_step_m: float = 1.0):
    """Split a track [frame, x, y] where it skips time or jumps in space.

    Points outside the room (or another camera's half of a shared room) are
    dropped before this, so a plain polyline would join what's left with long
    straight lines that nobody walked.
    """
    if len(arr) < 2:
        return []
    gap = np.diff(arr[:, 0]) > max_gap_frames
    jump = np.hypot(*np.diff(arr[:, 1:3], axis=0).T) > max_step_m
    cuts = np.flatnonzero(gap | jump) + 1
    return [p for p in np.split(arr, cuts) if len(p) >= 2]


def draw_paths(canvas: np.ndarray, plan: SitePlan, cameras: list[Camera],
               upto_frame: dict[str, int] | None = None, alpha: float = 0.35) -> np.ndarray:
    overlay = canvas.copy()
    for cam in cameras:
        for pts in cam.tracks.values():
            arr = np.array(pts, float)
            if upto_frame is not None:
                arr = arr[arr[:, 0] <= upto_frame.get(cam.name, 0)]
            for piece in path_pieces(arr):
                xy = smooth_path(cam.room.to_plan(piece[:, 1:3]))
                cv2.polylines(overlay, [plan.px(xy).astype(np.int32)], False, (255, 230, 160),
                              1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)


def _put_text(canvas: np.ndarray, items: list[tuple[str, tuple[int, int], int, tuple]]) -> None:
    """Draw (text, (x, y) top-left, px size, BGR) items in the overlay typeface."""
    from PIL import Image, ImageDraw

    from people_analytics.analytics.render import font

    pil = Image.fromarray(canvas[..., ::-1])
    d = ImageDraw.Draw(pil)
    for text, (x, y), size, colour in items:
        d.text((x, y), text, font=font(size, bold=False), fill=tuple(colour[::-1]))
    canvas[:] = np.asarray(pil)[..., ::-1]


def draw_frame(canvas: np.ndarray, plan: SitePlan, rooms: list[Room], title: str,
               notes: list[str]) -> np.ndarray:
    """Room outlines and labels, title, scale bar, credit (in the overlay typeface)."""
    from people_analytics.analytics.render import font

    size = max(14, int(plan.ppm * 0.8))  # label size follows the plan scale
    items: list[tuple[str, tuple[int, int], int, tuple]] = []
    for room in rooms:
        poly = plan.px(room.to_plan(room.outline)).astype(np.int32)
        cv2.polylines(canvas, [poly], True, WALL, 3, cv2.LINE_AA)
        x = int(poly[:, 0].min()) + 4
        y = int(poly[:, 1].min()) - int(size * 1.5)
        items.append((room.label, (x, max(y, 4)), size, INK))
    if title:
        items.append((title, (16, 14), int(size * 1.4), INK))
    h, w = canvas.shape[:2]
    lines: list[str] = []
    note_px = max(12, int(size * 0.85))
    f = font(note_px, bold=False)
    for note in notes:  # wrap each note to the canvas width
        words, line = note.split(), ""
        for word in words:
            trial = f"{line} {word}".strip()
            if f.getlength(trial) > w - 32 and line:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    step = int(note_px * 1.35)
    for i, text in enumerate(reversed(lines)):
        items.append((text, (16, h - 10 - step * (i + 1)), note_px, (180, 180, 180)))
    # 5 m scale bar, just under the rooms' bottom-right corner
    corners = np.vstack([plan.px(r.to_plan(r.outline)) for r in rooms])
    x1 = int(corners[:, 0].max())
    y = min(int(corners[:, 1].max()) + int(size * 2.2), h - 24 - step * len(lines))
    x0 = int(x1 - 5 * plan.ppm)
    cv2.line(canvas, (x0, y), (x1, y), INK, 2)
    for x in (x0, x1):
        cv2.line(canvas, (x, y - 6), (x, y + 6), INK, 2)
    items.append(("5 m", (x0, y - int(size * 1.4)), size, INK))
    _put_text(canvas, items)
    return canvas


def load_sitemap(
    path: str | Path, px_per_m: float | None = None
) -> tuple[SitePlan, list[Room], list[Camera], dict]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rooms = {
        r["name"]: Room(r["name"], r["label"], np.array(r["outline"], float),
                        r.get("rotate_deg", 0.0), tuple(r.get("offset", (0, 0))))
        for r in cfg["rooms"]
    }
    plan = SitePlan(list(rooms.values()), (px_per_m or cfg["plan"]["px_per_m"]))
    # Labels are drawn above each room's top-left corner: widen the canvas if a
    # label would run past the right edge.
    overflow = 0
    for room in rooms.values():
        x0 = plan.px(room.to_plan(room.outline))[:, 0].min() + 8
        from people_analytics.analytics.render import font

        size = max(14, int((px_per_m or cfg["plan"]["px_per_m"]) * 0.8))
        width = int(font(size, bold=False).getlength(room.label))
        overflow = max(overflow, int(x0 + width + 20 - plan.wh[0]))
    if overflow > 0:
        plan = SitePlan(list(rooms.values()), (px_per_m or cfg["plan"]["px_per_m"]), overflow)
    cache = Path(cfg.get("cache_dir", "outputs/meva/sitemap"))
    cameras, fits = [], {}
    for c in cfg["cameras"]:
        projector, diag = build_projector(c["floor"], c["video"], cache)
        fits[c["name"]] = diag
        cameras.append(Camera(c["name"], c["video"], c["events"], rooms[c["room"]], projector,
                              np.array(c["owns"], float) if c.get("owns") else None))
    return plan, list(rooms.values()), cameras, {"config": cfg, "fits": fits}


def render_sitemap(
    path: str | Path,
    out_dir: str | Path = "outputs/meva/sitemap",
    min_track_s: float = 3.0,
) -> dict:
    """Floor + heat + paths for every camera whose events export exists; returns stats."""
    import json

    from loguru import logger

    plan, rooms, cameras, meta = load_sitemap(path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(meta["config"].get("cache_dir", out))

    live = []
    for cam in cameras:
        if not Path(cam.events).exists():
            logger.warning(f"{cam.name}: no events export yet ({cam.events}); skipped")
            continue
        load_tracks(cam, min_track_s)
        live.append(cam)

    canvas = floor_layer(plan, cameras, cache)
    with_heat = canvas.astype(np.float32)
    room_top: dict[str, float] = {}
    for room in rooms:
        cams = [c for c in live if c.room is room]
        if not cams:
            continue
        heat = heat_layer(plan, cams)
        mask = np.zeros(heat.shape, np.uint8)
        cv2.fillPoly(mask, [plan.px(room.to_plan(room.outline)).astype(np.int32)], 1)
        heat *= mask  # the blur must not bleed one room's scale into another
        if not (heat > 0).any():
            continue
        vmax = float(np.percentile(heat[heat > 0], 99.5))
        room_top[room.name] = round(vmax, 2)
        colour, alpha = colourise(heat, vmax)
        with_heat = with_heat * (1 - alpha[..., None]) + colour * alpha[..., None]
    final = draw_paths(with_heat.astype(np.uint8), plan, live)
    notes = [
        "Schematic plan: floors measured per room (camera models, floor tiles, entrance mats);"
        " room placement after the MEVA site map, approximate.",
        "Heat = person-seconds per m2, scaled per room (brightest = that room's busiest spot);"
        f" paths = tracks of {min_track_s:.0f} s or longer. Each room is its own 5-minute clip.",
        "Video: MEVA dataset (mevadata.org), CC BY 4.0",
    ]
    final = draw_frame(final, plan, rooms, "Where people spent time", notes)
    cv2.imwrite(str(out / "site_heatmap.png"), final)
    cv2.imwrite(str(out / "site_floor.png"), draw_frame(canvas.copy(), plan, rooms,
                                                         "Site plan (floors from video)", notes))

    stats = {
        "fits": meta["fits"],
        "cameras": {
            c.name: {
                "tracks_drawn": len(c.tracks),
                "person_seconds_on_plan": round(sum(len(t) for t in c.tracks.values()) / c.fps, 1),
            }
            for c in live
        },
        # Top of each room's colour scale (99.5th percentile of its heat).
        "heat_scale_top_person_s_per_m2": room_top,
        "min_track_s": min_track_s,
    }
    (out / "site_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"Site map -> {out}")
    return stats
