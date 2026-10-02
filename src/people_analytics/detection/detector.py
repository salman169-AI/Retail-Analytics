"""YOLO26 person detector wrapped to emit `supervision` detections.

Design notes:
- Weights are cached under `weights/` (auto-downloaded from the Ultralytics
  release on first use) rather than polluting the repo root.
- `detect()` returns an `sv.Detections` already filtered to the configured
  classes (person by default), so downstream code never sees raw YOLO output.
- The optional `InferenceSlicer` path tiles each frame for dense/overhead scenes
  (Phase 2.2). Supervision expects overlap in *pixels*, so we convert the
  config's overlap *ratio* against the slice size.
- Tiling alone reintroduces a classic artefact on eye-level footage: a person
  taller than a tile is proposed once as a full body (or large fragment) and
  again as an upper/lower half from the neighbouring tile. Those two boxes have
  *low* IoU and only moderate IoS, so default NMS keeps both. We therefore run
  a full-frame pass alongside the slicer and merge with an IoS + vertical-stack
  rule that prefers complete bodies (see `merge_full_and_tiled`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv
import torch
from loguru import logger
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from people_analytics.config import DetectorConfig


def resolve_device(requested: str) -> str:
    """Honor the requested device, but fall back to CPU if CUDA is unavailable."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(f"Device '{requested}' requested but CUDA is unavailable; using cpu.")
        return "cpu"
    return requested


def overlap_wh_from_ratio(slice_wh: tuple[int, int], ratio: float) -> tuple[int, int]:
    """Convert a fractional overlap into the pixel overlap InferenceSlicer wants."""
    return int(slice_wh[0] * ratio), int(slice_wh[1] * ratio)


_OVERLAP_METRICS = {"iou": sv.OverlapMetric.IOU, "ios": sv.OverlapMetric.IOS}


def _overlap_metric(name: str) -> sv.OverlapMetric:
    """Resolve the tile-merge metric name (see SlicerConfig for why it matters)."""
    if name.lower() not in _OVERLAP_METRICS:
        raise ValueError(
            f"Unknown slicer merge_metric '{name}'. Options: {sorted(_OVERLAP_METRICS)}"
        )
    return _OVERLAP_METRICS[name.lower()]


def ensure_weights(model: str, weights_dir: str | Path) -> str:
    """Return a local path to the weights, downloading into `weights_dir` if needed."""
    target = Path(weights_dir) / Path(model).name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Fetching detector weights -> {target}")
        attempt_download_asset(str(target))
    return str(target)


def clean_detections(detections: sv.Detections) -> sv.Detections:
    """Drop backend-specific metadata so `sv.Detections.merge` is safe.

    Ultralytics attaches `data['class_name']`; our post-merge boxes do not.
    Merging mixed sources then raises. Keep only the fields the rest of the
    pipeline uses (boxes, confidence, class_id).
    """
    if len(detections) == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.asarray(detections.xyxy, dtype=np.float32),
        confidence=(
            None
            if detections.confidence is None
            else np.asarray(detections.confidence, dtype=np.float32)
        ),
        class_id=(
            None
            if detections.class_id is None
            else np.asarray(detections.class_id, dtype=int)
        ),
    )


def box_ios(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-smaller for two xyxy boxes."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def nms_ios(detections: sv.Detections, threshold: float) -> sv.Detections:
    """Greedy NMS scored by intersection-over-smaller; keep higher confidence."""
    if len(detections) == 0:
        return detections
    conf = (
        detections.confidence
        if detections.confidence is not None
        else np.ones(len(detections), dtype=np.float32)
    )
    order = np.argsort(-conf)
    keep: list[int] = []
    suppressed = np.zeros(len(detections), dtype=bool)
    xyxy = detections.xyxy
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        for j in order:
            if suppressed[j] or j == idx:
                continue
            if box_ios(xyxy[idx], xyxy[j]) >= threshold:
                suppressed[j] = True
    return detections[np.array(keep, dtype=int)]


def is_vertical_partial(
    a: np.ndarray,
    b: np.ndarray,
    *,
    ios_thr: float = 0.45,
    center_x_tol_ratio: float = 0.55,
    min_x_overlap: float = 0.5,
    min_partial_ios: float = 0.2,
) -> bool:
    """True when `a` and `b` look like fragments of the same standing person.

    Covers:
    - nested halves (high IoS) — pure containment;
    - upper/lower tile splits: centres aligned in x, strong horizontal overlap,
      moderate IoS, one box typically much shorter than the other.
    """
    s = box_ios(a, b)
    if s >= ios_thr:
        return True

    wa = float(a[2] - a[0])
    ha = float(a[3] - a[1])
    wb = float(b[2] - b[0])
    hb = float(b[3] - b[1])
    if min(wa, wb, ha, hb) <= 0:
        return False

    cxa = 0.5 * (float(a[0]) + float(a[2]))
    cxb = 0.5 * (float(b[0]) + float(b[2]))
    if abs(cxa - cxb) > center_x_tol_ratio * min(wa, wb):
        return False

    x_inter = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    if x_inter / min(wa, wb) < min_x_overlap:
        return False

    y_inter = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    if y_inter <= 0 and s < min_partial_ios:
        return False

    # Prefer the "one is a fragment" signal so two full-height people standing
    # one behind the other (similar height, moderate IoS) are less likely to
    # collapse into a single box.
    shorter_ratio = min(ha, hb) / max(ha, hb)
    return s >= min_partial_ios and shorter_ratio < 0.85


def merge_full_and_tiled(
    full: sv.Detections,
    tiled: sv.Detections,
    *,
    ios_threshold: float = 0.45,
) -> sv.Detections:
    """Combine a whole-frame pass with tiled detections without duplicate bodies.

    Strategy:
    1. Keep every full-frame box (these are complete for near/large people).
    2. Keep a tiled box only if it is *not* a vertical partial / contained
       duplicate of any full-frame box — this is how far/arch people survive.
    3. Among the surviving tiled boxes, suppress mutual vertical partials by
       keeping the taller (more complete) box.
    4. Final IoS NMS at the same threshold as a safety net.
    """
    full = clean_detections(full)
    tiled = clean_detections(tiled)

    if len(tiled) == 0:
        return full
    if len(full) == 0:
        return nms_ios(_suppress_vertical_partials(tiled, ios_threshold), ios_threshold)

    keep_tiled = np.ones(len(tiled), dtype=bool)
    for i in range(len(tiled)):
        ti = tiled.xyxy[i]
        for j in range(len(full)):
            if is_vertical_partial(ti, full.xyxy[j], ios_thr=ios_threshold):
                keep_tiled[i] = False
                break

    tiled_kept = tiled[keep_tiled] if keep_tiled.any() else sv.Detections.empty()
    if len(tiled_kept) == 0:
        return full

    tiled_kept = _suppress_vertical_partials(tiled_kept, ios_threshold)
    if len(tiled_kept) == 0:
        return full

    merged = sv.Detections.merge([full, tiled_kept])
    return nms_ios(merged, ios_threshold)


def _suppress_vertical_partials(
    detections: sv.Detections,
    ios_threshold: float,
) -> sv.Detections:
    """Drop the shorter box when two detections are vertical partials of one person."""
    if len(detections) < 2:
        return detections

    conf = (
        detections.confidence
        if detections.confidence is not None
        else np.ones(len(detections), dtype=np.float32)
    )
    # Prefer taller boxes; break ties on confidence.
    heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    order = np.lexsort((-conf, -heights))
    keep = np.ones(len(detections), dtype=bool)
    xyxy = detections.xyxy

    for pos, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[pos + 1 :]:
            if not keep[j]:
                continue
            if is_vertical_partial(xyxy[i], xyxy[j], ios_thr=ios_threshold):
                # `i` is taller (or equal+higher conf) because of sort order.
                keep[j] = False
    return detections[keep]


class PersonDetector:
    """Config-driven YOLO26 detector that outputs `sv.Detections` (person-only)."""

    def __init__(self, cfg: DetectorConfig, weights_dir: str | None = None):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        weights = ensure_weights(cfg.model, weights_dir or cfg.weights_dir)
        self.model = YOLO(weights)
        self.model.to(self.device)

        self._merge_threshold = (
            cfg.iou if cfg.slicer.merge_threshold is None else cfg.slicer.merge_threshold
        )
        # When tiling is on we always fuse a full-frame pass (see detect()).
        # Hybrid can be disabled for pure-slicer ablations via config.
        self._hybrid = cfg.slicer.enabled and cfg.slicer.hybrid

        self._slicer: sv.InferenceSlicer | None = None
        if cfg.slicer.enabled:
            slice_wh = tuple(cfg.slicer.slice_wh)
            self._slicer = sv.InferenceSlicer(
                callback=self._predict,
                slice_wh=slice_wh,
                overlap_wh=overlap_wh_from_ratio(slice_wh, cfg.slicer.overlap_ratio),
                overlap_metric=_overlap_metric(cfg.slicer.merge_metric),
                iou_threshold=self._merge_threshold,
            )
        logger.info(
            f"PersonDetector ready: {Path(weights).name} on {self.device} "
            f"(classes={cfg.classes}, conf={cfg.confidence}, slicer={cfg.slicer.enabled}, "
            f"hybrid={self._hybrid}, merge={cfg.slicer.merge_metric}@{self._merge_threshold})"
        )

    def _predict(self, image: np.ndarray) -> sv.Detections:
        result = self.model(
            image,
            conf=self.cfg.confidence,
            iou=self.cfg.iou,
            classes=self.cfg.classes,
            device=self.device,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Detect persons in a BGR frame, tiling first if the slicer is enabled."""
        if self._slicer is None:
            return clean_detections(self._predict(frame))

        tiled = self._slicer(frame)
        if not self._hybrid:
            # Pure slicer path (historical). Still apply vertical-partial cleanup
            # so half-body duplicates from neighbouring tiles do not pass through.
            return nms_ios(
                _suppress_vertical_partials(clean_detections(tiled), self._merge_threshold),
                self._merge_threshold,
            )

        full = self._predict(frame)
        return merge_full_and_tiled(
            full, tiled, ios_threshold=self._merge_threshold
        )
