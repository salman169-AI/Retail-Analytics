"""The ~60 s portfolio demo video, assembled from a shot list (configs/meva_demo.yaml).

Shot kinds:

- **sitemap**: the site plan filling with heat and walking paths, all cameras at
  once, time-lapsed (each room's clip played back in a few seconds).
- **scene**: one camera with its overlay, the live stats panel on the right and a
  caption under the video (`speed` < 1 for slow motion).
- **summary**: two end-of-clip panels side by side, with the busiest minute.
- **measured**: the evaluation results, read from metrics/*.json at render time.
- **end**: credits.

Layout is 1920x1080: video 1440 wide, panel 480 wide, captions in the band under
the video. Shots dip through the background colour for a few frames at each cut.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger
from PIL import Image, ImageDraw

from people_analytics.analytics.composite import FFmpegWriter, SceneStats, _load, iter_scene
from people_analytics.analytics.panel import render_panel
from people_analytics.analytics.render import font
from people_analytics.config import load_config

W, H = 1920, 1080
VIDEO_W = 1440
PANEL_W = W - VIDEO_W
BG = (25, 26, 26)  # BGR of the panel surface #1a1a19
INK = (255, 255, 255)
INK_2 = (183, 194, 195)  # BGR of #c3c2b7
MUTED = (129, 135, 137)
ACCENT = (52, 104, 235)  # BGR of the overlay orange #eb6834
CREDIT = "Video: MEVA dataset (mevadata.org), CC BY 4.0"
FADE = 6  # frames dipped to the background at each cut


def _text(img: np.ndarray, lines: list[tuple[str, int, tuple[int, int, int], bool]],
          x: int, y: int, gap: int = 14, max_w: int | None = None) -> int:
    """Draw lines (text, px size, BGR colour, bold) top-down from (x, y); returns end y."""
    pil = Image.fromarray(img[..., ::-1])
    d = ImageDraw.Draw(pil)
    for text, size, color, bold in lines:
        f = font(size, bold)
        words, line, rows = text.split(), "", []
        for word in words:  # wrap to max_w
            trial = f"{line} {word}".strip()
            if max_w and d.textlength(trial, font=f) > max_w and line:
                rows.append(line)
                line = word
            else:
                line = trial
        rows.append(line)
        for row in rows:
            d.text((x, y), row, font=f, fill=color[::-1])
            y += int(size * 1.2)
        y += gap
    img[:] = np.asarray(pil)[..., ::-1]
    return y


def _blank() -> np.ndarray:
    return np.full((H, W, 3), BG, np.uint8)


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale to fit inside w x h, centred on the background."""
    s = min(w / img.shape[1], h / img.shape[0])
    small = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                       interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), BG, np.uint8)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def _caption_band(canvas: np.ndarray, x: int, y: int, w: int, caption: str, sub: str) -> None:
    _text(canvas, [(caption, 46, INK, True), (sub, 30, INK_2, False)], x + 40, y + 34,
          gap=10, max_w=w - 80)


# --- shots ---------------------------------------------------------------------


def _config(shot: dict):
    """Load the shot's config and apply its `overrides` ({"staff.alert_after_s": 1.5})."""
    cfg = load_config(shot["config"])
    for dotted, value in (shot.get("overrides") or {}).items():
        section, key = dotted.split(".")
        setattr(getattr(cfg, section), key, value)
    return cfg


def shot_scene(shot: dict, fps: float):
    cfg = _config(shot)
    caption_y = int(VIDEO_W * 9 / 16)  # 810
    for img, panel, _ in iter_scene(cfg, shot["title"], shot["start_s"], shot["seconds"],
                                    speed=shot.get("speed", 1.0), panel_size=(PANEL_W, H),
                                    highlight=set(shot.get("highlight", []))):
        canvas = _blank()
        canvas[:caption_y, :VIDEO_W] = _fit(img, VIDEO_W, caption_y)
        canvas[:, VIDEO_W:] = panel
        _caption_band(canvas, 0, caption_y, VIDEO_W, shot["caption"], shot["sub"])
        yield canvas


def shot_sitemap(shot: dict, fps: float):
    from people_analytics.analytics.sitemap import (
        colourise,
        draw_frame,
        draw_paths,
        floor_layer,
        heat_layer,
        load_sitemap,
        load_tracks,
    )

    plan, rooms, cameras, meta = load_sitemap(shot["config"], shot.get("px_per_m"))
    cache = Path(meta["config"].get("cache_dir", "outputs/meva/sitemap"))
    for cam in cameras:
        load_tracks(cam, 3.0)
    last = {c.name: max((p[-1][0] for p in c.tracks.values()), default=1) for c in cameras}
    floor = floor_layer(plan, cameras, cache)
    masks, tops = {}, {}
    for room in rooms:
        m = np.zeros(floor.shape[:2], np.float32)
        cv2.fillPoly(m, [plan.px(room.to_plan(room.outline)).astype(np.int32)], 1.0)
        masks[room.name] = m
        full = heat_layer(plan, [c for c in cameras if c.room is room]) * m
        tops[room.name] = float(np.percentile(full[full > 0], 99.5)) if (full > 0).any() else 1

    n = int(shot["seconds"] * fps)
    band = 190
    # Crop the plan to its rooms (+ labels) so the map fills the frame.
    full = draw_frame(floor.copy(), plan, rooms, "", [])
    ink = np.any(np.abs(full.astype(int) - np.array(BG)) > 6, axis=2)
    ys, xs = np.nonzero(ink)
    pad = 30
    crop = (max(0, ys.min() - pad), min(full.shape[0], ys.max() + pad),
            max(0, xs.min() - pad), min(full.shape[1], xs.max() + pad))
    for i in range(n):
        p = min(1.0, (i + 1) / (n * 0.85))  # fill in the first 85%, then hold
        upto = {c.name: int(last[c.name] * p) for c in cameras}
        img = floor.astype(np.float32)
        for room in rooms:
            cams = []
            for c in (c for c in cameras if c.room is room):
                trimmed = {g: [q for q in pts if q[0] <= upto[c.name]]
                           for g, pts in c.tracks.items()}
                c2 = type(c)(c.name, c.video, c.events, c.room, c.projector, c.owns, c.fps,
                             {g: v for g, v in trimmed.items() if v})
                cams.append(c2)
            heat = heat_layer(plan, cams) * masks[room.name]
            colour, alpha = colourise(heat, tops[room.name])
            img = img * (1 - alpha[..., None]) + colour * alpha[..., None]
        img = draw_paths(img.astype(np.uint8), plan, cameras, upto_frame=upto, alpha=0.3)
        img = draw_frame(img, plan, rooms, "", [])
        y0, y1, x0, x1 = crop
        canvas = _blank()
        canvas[:H - band] = _fit(img[y0:y1, x0:x1], W, H - band)
        _caption_band(canvas, 0, H - band, W, shot["caption"], shot["sub"])
        _text(canvas, [(CREDIT, 20, MUTED, False)], W - 520, H - 44)
        yield canvas


def busiest_minute(activity_csv: Path, fps: float, minutes: int) -> tuple[int, int]:
    counts = [0] * minutes
    with open(activity_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["event"] in ("door_in", "door_out"):
                counts[min(int(int(r["frame"]) / fps // 60), minutes - 1)] += 1
    m = int(np.argmax(counts))
    return m, counts[m]


def shot_summary(shot: dict, fps: float):
    canvas = _blank()
    notes = []
    for k, spec in enumerate(shot["panels"]):
        cfg = load_config(spec["config"])
        _, activity, cfps, total = _load(cfg)
        stats = SceneStats(cfg, activity, cfps, total, spec["title"])
        canvas[:, k * PANEL_W:(k + 1) * PANEL_W] = render_panel(stats.state(total, None),
                                                               (PANEL_W, H))
        if cfg.doors:
            stem = Path(cfg.io.source).stem
            m, n = busiest_minute(Path(cfg.io.output_dir) / "analytics" / f"{stem}_activity.csv",
                                  cfps, stats.minutes)
            notes.append(f"Busiest minute at the doors: minute {m}–{m + 1}, "
                         f"{n} people through.")
        if cfg.zones:
            ranked = sorted(stats.state(total, None).bars.items(), key=lambda kv: -kv[1])
            if len(ranked) >= 2:
                (a, va), (b, vb) = ranked[:2]
                notes.append(f"Longest average stay: {a} ({va:.0f} s), then {b} ({vb:.0f} s).")
        if cfg.queues:
            stem = Path(cfg.io.source).stem
            summary = json.loads((Path(cfg.io.output_dir) / "analytics" / f"{stem}_summary.json")
                                 .read_text(encoding="utf-8"))
            q = summary["queues"].get(cfg.queues[0].name, {})
            if q:
                notes.append(f"Most people queueing at once: {q['peak_length']}; "
                             f"{q['visits']} queue visits in 5 minutes.")
    x = len(shot["panels"]) * PANEL_W + 70
    _text(canvas, [(shot["caption"], 52, INK, True),
                   (shot.get("sub", "Totals for the full 5-minute clip."), 30, INK_2, False)]
          + [(n, 30, INK, False) for n in notes], x, 140, gap=26, max_w=W - x - 70)
    for _ in range(int(shot["seconds"] * fps)):
        yield canvas


def _metrics() -> dict:
    doors = json.loads(Path("metrics/meva_doors_eval.json").read_text(encoding="utf-8"))[0]
    idsw = {}
    for scene in ("cafe", "entrance"):
        rows = json.loads(Path(f"metrics/meva_{scene}_idsw.json").read_text(encoding="utf-8"))
        by = {r["variant"]: r for r in rows}
        idsw[scene] = (by["base"]["gt_idsw"], by["bytetrack"]["gt_idsw"])
    return {"doors": doors, "idsw": idsw}


def shot_measured(shot: dict, fps: float):
    """Results card; `items` picks rows: cafe_idsw, entrance_idsw, doors."""
    m = _metrics()
    d = m["doors"]
    rows = {
        "doors": ("Door counting",
                  f"{d['in']['matched'] + d['out']['matched']} of "
                  f"{d['in']['labels'] + d['out']['labels']}",
                  f"labelled entries and exits found, "
                  f"{d['in']['false_counts'] + d['out']['false_counts']} false counts"),
    }
    for scene, place in (("cafe", "this cafe clip"), ("entrance", "the entrance clip")):
        ours, bt = m["idsw"][scene]
        rows[f"{scene}_idsw"] = ("Tracking people through a crowd", f"{1 - ours / bt:.0%} fewer",
                                 f"identity switches than ByteTrack in {place} "
                                 f"({ours} vs {bt}, full 5 minutes)")
    canvas = _blank()
    y = _text(canvas, [("Measured on footage the system was not tuned on", 52, INK, True),
                       ("Ground truth: MEVA's own annotations", 30, INK_2, False)], 140, 150,
              gap=20)
    y += 50
    for key in shot.get("items", ["doors", "cafe_idsw", "entrance_idsw"]):
        label, big, rest = rows[key]
        _text(canvas, [(label, 30, INK_2, False)], 140, y)
        _text(canvas, [(big, 64, INK, True)], 140, y + 44)
        _text(canvas, [(rest, 32, INK_2, False)], 620, y + 62, max_w=W - 760)
        y += 210
    cv2.line(canvas, (140, y - 30), (W - 140, y - 30), (44, 44, 44), 1)
    _text(canvas, [("Full results, method and limits: metrics/meva_eval.md", 26, MUTED, False),
                   (CREDIT, 26, MUTED, False)], 140, y, gap=6)
    for _ in range(int(shot["seconds"] * fps)):
        yield canvas


def shot_email(shot: dict, fps: float):
    """The alert email the run generates, shown as a message card."""
    from people_analytics import notify
    from people_analytics.analytics.activity import build_activity
    from people_analytics.analytics.staff import StaffAlert

    cfg = _config(shot)
    stem = Path(cfg.io.source).stem
    _, _, cfps, total = _load(cfg)
    events = Path(cfg.io.output_dir) / "analytics" / f"{stem}_events.csv"
    rows = [r for r in build_activity(cfg, events, cfps, 1, total)
            if r["event"] == "alert_no_staff"]
    if not rows:
        raise ValueError("No staff alert in this clip with these settings")
    r = rows[0]
    alert = StaffAlert(r["frame"], r["time_s"], r["id"], r["value_s"], cfg.staff.name)
    subject, body = notify.alert_text(cfg.notify.site_name, alert.message, alert.time_s,
                                      "cafe camera")
    canvas = _blank()
    _text(canvas, [(shot["caption"], 52, INK, True), (shot["sub"], 30, INK_2, False)],
          140, 110, gap=14, max_w=W - 280)
    x0, y0, x1, y1 = 140, 330, W - 140, 900
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (44, 44, 44), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y0 + 8), ACCENT, -1)
    y = _text(canvas, [("From: Cafe camera alerts", 26, INK_2, False),
                       ("To: Cafe manager", 26, INK_2, False),
                       (f"Subject: {subject}", 30, INK, True)], x0 + 40, y0 + 40, gap=8)
    cv2.line(canvas, (x0 + 40, y + 10), (x1 - 40, y + 10), (70, 70, 70), 1)
    _text(canvas, [(ln, 28, INK, False) for ln in body.splitlines() if ln.strip()],
          x0 + 40, y + 40, gap=10, max_w=x1 - x0 - 80)
    for _ in range(int(shot["seconds"] * fps)):
        yield canvas


def shot_end(shot: dict, fps: float, author: str):
    canvas = _blank()
    title = shot.get("title", ["People counting, dwell time & heatmaps",
                               "with occlusion-robust tracking"])
    _text(canvas, [(line, 60, INK, True) for line in title], 140, 300, gap=4)
    _text(canvas, [("Personal demo on the MEVA dataset (CC BY 4.0)", 36, INK_2, False),
                   (author, 36, INK, True),
                   (CREDIT, 26, MUTED, False)], 140, 560, gap=22)
    cv2.rectangle(canvas, (140, 270), (230, 278), ACCENT, -1)
    for _ in range(int(shot["seconds"] * fps)):
        yield canvas


# --- assembly ------------------------------------------------------------------


def _with_fades(frames, fade: int = FADE):
    """Dip the first and last `fade` frames of a shot towards the background."""
    buf: list[np.ndarray] = []
    bg = _blank().astype(np.float32)
    for i, fr in enumerate(frames):
        if i < fade:
            a = (i + 1) / (fade + 1)
            fr = (fr.astype(np.float32) * a + bg * (1 - a)).astype(np.uint8)
        buf.append(fr)
        if len(buf) > fade:
            yield buf.pop(0)
    for j, fr in enumerate(buf):
        a = 1 - (j + 1) / (len(buf) + 1)
        yield (fr.astype(np.float32) * a + bg * (1 - a)).astype(np.uint8)


def build_demo(config: str | Path) -> dict:
    spec = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    fps = float(spec.get("fps", 30))
    out = FFmpegWriter(spec["out"], (W, H), fps, crf=22)
    thumb = None
    # Thumbnail: [shot index, seconds into that shot] (raw frame, before any fade).
    t_shot, t_sec = spec.get("thumbnail_at", [0, 0.0])
    durations = []
    for idx, shot in enumerate(spec["shots"]):
        kind = shot["kind"]
        gen = {
            "scene": lambda s=shot: shot_scene(s, fps),
            "sitemap": lambda s=shot: shot_sitemap(s, fps),
            "summary": lambda s=shot: shot_summary(s, fps),
            "measured": lambda s=shot: shot_measured(s, fps),
            "email": lambda s=shot: shot_email(s, fps),
            "end": lambda s=shot: shot_end(s, fps, spec.get("author", "")),
        }[kind]()
        grab: list[np.ndarray] = []

        def capture(frames, want=int(t_sec * fps) if idx == t_shot else -1, holder=grab):
            for i, fr in enumerate(frames):
                if i == want:
                    holder.append(fr.copy())
                yield fr

        n = 0
        for frame in _with_fades(capture(gen)):
            out.write(frame)
            n += 1
        if grab:
            thumb = grab[0]
        durations.append((kind, round(n / fps, 2)))
        logger.info(f"{kind}: {n / fps:.1f} s")
    path = out.close()
    if thumb is not None and spec.get("thumbnail"):
        Path(spec["thumbnail"]).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(spec["thumbnail"], thumb)
    return {"video": str(path), "seconds": round(sum(d for _, d in durations), 2),
            "shots": durations, "mb": round(path.stat().st_size / 1e6, 1)}
