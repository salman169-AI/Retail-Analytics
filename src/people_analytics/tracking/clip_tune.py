"""Tune detector/tracker settings on a MEVA clip against its partial annotations.

MEVA boxes only the people taking part in an annotated activity (see
`people_analytics.data.meva`), so the usual MOT scores don't apply: every
unannotated person we track would count as a false positive. Two numbers *are*
sound on a partial ground truth, and they are the two this module reports:

- **recall** — share of annotated person boxes our tracks cover (IoU >= 0.3;
  MEVA boxes are loose hand-drawn boxes, 0.5 rejects many correct matches);
- **IDSW** — identity switches on the annotated people (motmetrics' count only
  looks at ground-truth objects, so unannotated people can't inflate it).

Detections are computed once per clip segment at a low confidence and cached to
`.npz`, so tracker variants (and higher detector thresholds, applied by filtering
the cache) are compared on identical boxes.

Tune on a *different* clip from the one used for evaluation in phase 4, so the
reported numbers stay out-of-sample.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from loguru import logger

from people_analytics.config import Config
from people_analytics.detection import PersonDetector
from people_analytics.tracking.benchmark import _xyxy_to_xywh
from people_analytics.tracking.interpolate import interpolate_tracks
from people_analytics.tracking.tracker import MultiObjectTracker

MATCH_IOU = 0.3
CACHE_CONF = 0.1  # low enough for the trackers' second-stage association


def _frames(src: str | Path, start: int, n: int):
    """Yield (frame_index, frame) with 0-based indices matching MEVA's ts0."""
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    try:
        for idx in range(start, start + n):
            ok, frame = cap.read()
            if not ok:
                return
            yield idx, frame
    finally:
        cap.release()


def cache_detections(
    cfg: Config, src: str | Path, start: int, n: int, cache_path: str | Path
) -> dict[int, np.ndarray]:
    """frame -> [x1, y1, x2, y2, conf] at CACHE_CONF; loaded from `cache_path` if present."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path)
        return {int(k): data[k] for k in data.files}
    det_cfg = cfg.detector.model_copy(update={"confidence": CACHE_CONF})
    detector = PersonDetector(det_cfg)
    out: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    for idx, frame in _frames(src, start, n):
        d = detector.detect(frame)
        conf = d.confidence if d.confidence is not None else np.ones(len(d))
        out[idx] = np.column_stack([d.xyxy, conf]).astype(np.float32) if len(d) else (
            np.empty((0, 5), np.float32)
        )
    logger.info(f"Detected {len(out)} frames at {len(out) / (time.perf_counter() - t0):.1f} FPS")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **{str(k): v for k, v in out.items()})
    return out


def _to_detections(rows: np.ndarray, min_conf: float) -> sv.Detections:
    rows = rows[rows[:, 4] >= min_conf]
    if len(rows) == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=rows[:, :4].astype(float),
        confidence=rows[:, 4].astype(float),
        class_id=np.zeros(len(rows), dtype=int),
    )


def run_variant(
    cfg: Config,
    src: str | Path,
    detections: dict[int, np.ndarray],
    start: int,
    n: int,
    fps: float,
) -> tuple[dict[int, np.ndarray], float]:
    """Track the cached segment with `cfg`; returns (frame -> [id, x, y, w, h, conf], FPS)."""
    gallery = None
    if cfg.reid_gallery.enabled:
        from people_analytics.analytics.pipeline import _build_gallery

        gallery = _build_gallery(cfg)
    mot = MultiObjectTracker(
        cfg.tracker.name,
        reid_name=cfg.tracker.reid_model,
        device=cfg.detector.device,
        frame_rate=int(fps),
        cfg=cfg.tracker,
        gallery=gallery,
    )
    tracks: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    for idx, frame in _frames(src, start, n):
        tracked = mot.update(_to_detections(detections[idx], cfg.detector.confidence), frame)
        if len(tracked):
            tracks[idx] = np.column_stack(
                [tracked.tracker_id, _xyxy_to_xywh(tracked.xyxy), tracked.confidence]
            )
    speed = len(detections) / (time.perf_counter() - t0)
    if cfg.analytics.interpolate:
        tracks = interpolate_tracks(tracks, max_gap=cfg.analytics.interpolate_max_gap)
    return tracks, speed


def score(
    tracks: dict[int, np.ndarray], gt: dict[int, np.ndarray], start: int, n: int, fps: float
) -> dict:
    """Recall + IDSW on the annotated people, plus track-length stats on everyone."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    for f in range(start, start + n):
        g = gt.get(f, np.empty((0, 5)))
        r = tracks.get(f, np.empty((0, 6)))
        g_xywh = _xyxy_to_xywh(g[:, 1:5]) if len(g) else np.empty((0, 4))
        dists = mm.distances.iou_matrix(g_xywh, r[:, 1:5], max_iou=1 - MATCH_IOU)
        acc.update(g[:, 0].astype(int).tolist(), r[:, 0].astype(int).tolist(), dists)
    m = mm.metrics.create().compute(
        acc, metrics=["recall", "num_switches", "num_unique_objects"], name="x"
    ).iloc[0]

    lengths: dict[int, int] = {}
    for rows in tracks.values():
        for tid in rows[:, 0].astype(int):
            lengths[tid] = lengths.get(tid, 0) + 1
    secs = np.array(sorted(lengths.values())) / fps if lengths else np.zeros(1)
    return {
        "gt_recall": round(float(m["recall"]), 3),
        "gt_idsw": int(m["num_switches"]),
        "gt_tracks": int(m["num_unique_objects"]),
        "ids": len(lengths),
        "short_ids_lt1s": int((secs < 1.0).sum()),
        "median_track_s": round(float(np.median(secs)), 1),
        "boxes_per_frame": round(sum(len(r) for r in tracks.values()) / max(n, 1), 2),
    }


# Each variant changes one thing from the scene config, so a row's difference
# from its neighbour is attributable. "base" is the scene config as written.
VARIANTS: dict[str, dict] = {
    "base": {},
    "bytetrack": {"tracker.name": "bytetrack", "tracker.reid_model": None,
                  "reid_gallery.enabled": False},
    "no_gallery": {"reid_gallery.enabled": False},
    "osnet_ain": {"tracker.reid_model": "osnet_ain"},
    "buffer150": {"tracker.track_buffer": 150},
    "conf0.1": {"detector.confidence": 0.1},
    "conf0.4": {"detector.confidence": 0.4},
    "no_interp": {"analytics.interpolate": False},
    "cmc_on": {"tracker.use_cmc": True},
    "gates0.4": {"tracker.track_high_thresh": 0.4, "tracker.new_track_thresh": 0.4},
    "gates0.3": {"tracker.track_high_thresh": 0.3, "tracker.new_track_thresh": 0.3},
    "gallery0.65": {"reid_gallery.cosine_thresh": 0.65},
}


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    trial = cfg.model_copy(deep=True)
    for dotted, value in overrides.items():
        section, key = dotted.split(".")
        setattr(getattr(trial, section), key, value)
    return trial


def busiest_window(gt: dict[int, np.ndarray], n: int, total: int) -> int:
    """Start frame of the `n`-frame window holding the most annotated boxes."""
    counts = np.zeros(total)
    for f, rows in gt.items():
        if f < total:
            counts[f] = len(rows)
    window = np.convolve(counts, np.ones(n), mode="valid")
    return int(np.argmax(window)) if len(window) else 0


def run_on_clip(
    cfg: Config,
    clip: str,
    out_dir: str | Path,
    variants: list[str] | None = None,
    seconds: float | None = None,
) -> list[dict]:
    """Run variants on `clip` (its busiest `seconds`, or all of it) and score each."""
    from people_analytics.data.meva import load_person_boxes
    from people_analytics.io import get_video_info

    src = Path(cfg.meva.clips_dir) / f"{clip}.r13.avi"
    info = get_video_info(src)
    gt = load_person_boxes(cfg.meva.annotations, clip)
    if seconds is None:
        start, n = 0, info.total_frames
    else:
        n = int(seconds * info.fps)
        start = busiest_window(gt, n, info.total_frames)
    logger.info(f"Running on {clip} frames {start}-{start + n}")

    out_dir = Path(out_dir)
    tag = "tiled" if cfg.detector.slicer.enabled else "full"
    dets = cache_detections(cfg, src, start, n, out_dir / f"dets_{tag}_{start}_{n}.npz")

    rows = []
    for name in variants or list(VARIANTS):
        # "a+b" combines variants, e.g. "gates0.4+gallery0.65".
        overrides = {k: v for part in name.split("+") for k, v in VARIANTS[part].items()}
        trial = apply_overrides(cfg, overrides)
        tracks, speed = run_variant(trial, src, dets, start, n, info.fps)
        row = {"variant": name, "track_fps": round(speed, 1),
               **score(tracks, gt, start, n, info.fps)}
        logger.info(row)
        rows.append(row)
    return [{"clip": clip, "start": start, "frames": n, "detector": tag, **r} for r in rows]


def tune_on_clip(
    cfg: Config, seconds: float, out_dir: str | Path, variants: list[str] | None = None
) -> list[dict]:
    """Run every variant on the busiest `seconds` of the scene's tuning clip."""
    if cfg.meva is None or cfg.meva.tune_clip is None:
        raise ValueError("Config has no meva.tune_clip")
    return run_on_clip(cfg, cfg.meva.tune_clip, out_dir, variants, seconds)
