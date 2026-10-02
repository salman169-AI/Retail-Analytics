"""BoxMOT tracker factory + a Supervision-friendly wrapper.

We drive the tracker classes directly (`boxmot.trackers.bbox.*`) because the
high-level `BoxMOT` facade in 19.0.0 has a broken import (`boxmot.data`).

Tracker registry:
    botsort     : BoT-SORT with ReID appearance
    deepocsort  : Deep OC-SORT
    boosttrack  : BoostTrack++ (rich-similarity / soft-boost / varying-thresh on)
    bytetrack   : motion-only baseline (no appearance) — the before/after anchor
"""

from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv
from loguru import logger

from people_analytics.config import TrackerConfig
from people_analytics.tracking.reid import build_reid

# Names that need a ReID appearance model.
APPEARANCE_TRACKERS = {"botsort", "deepocsort", "boosttrack"}
MOTION_ONLY = {"bytetrack"}


def _normalize(name: str) -> str:
    key = name.lower().replace("-", "").replace("_", "").replace(" ", "").replace("+", "")
    aliases = {
        "botsort": "botsort",
        "botsortreid": "botsort",
        "deepocsort": "deepocsort",
        "deepoc": "deepocsort",
        "boosttrack": "boosttrack",
        "boosttrackpp": "boosttrack",
        "bytetrack": "bytetrack",
        "byte": "bytetrack",
    }
    if key not in aliases:
        raise KeyError(f"Unknown tracker '{name}'. Options: {sorted(set(aliases.values()))}")
    return aliases[key]


def build_tracker(
    name: str,
    reid_model: Any | None = None,
    frame_rate: int = 30,
    cfg: TrackerConfig | None = None,
) -> Any:
    """Instantiate a BoxMOT tracker, wiring config knobs each class supports."""
    from boxmot.trackers.bbox.boosttrack import BoostTrack
    from boxmot.trackers.bbox.botsort import BotSort
    from boxmot.trackers.bbox.bytetrack import ByteTrack
    from boxmot.trackers.bbox.deepocsort import DeepOcSort

    canonical = _normalize(name)
    track_buffer = cfg.track_buffer if cfg else 30
    match_thresh = cfg.match_thresh if cfg else 0.8
    appearance_thresh = cfg.appearance_thresh if cfg else 0.25
    use_cmc = cfg.use_cmc if cfg else True
    track_high_thresh = cfg.track_high_thresh if cfg else 0.5
    new_track_thresh = cfg.new_track_thresh if cfg else 0.6
    with_reid = reid_model is not None

    if canonical == "botsort":
        return BotSort(
            reid_model=reid_model,
            with_reid=with_reid,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            appearance_thresh=appearance_thresh,
            use_cmc=use_cmc,
            track_high_thresh=track_high_thresh,
            new_track_thresh=new_track_thresh,
            frame_rate=frame_rate,
            half=False,
        )
    if canonical == "deepocsort":
        return DeepOcSort(reid_model=reid_model, embedding_off=not with_reid, half=False)
    if canonical == "boosttrack":
        # The BoostTrack++ enhancements over vanilla BoostTrack.
        return BoostTrack(
            reid_model=reid_model,
            with_reid=with_reid,
            use_rich_s=True,
            use_sb=True,
            use_vt=True,
            s_sim_corr=True,
            half=False,
        )
    # bytetrack — motion only
    return ByteTrack(track_buffer=track_buffer, match_thresh=match_thresh, frame_rate=frame_rate)


def detections_to_boxmot(det: sv.Detections) -> np.ndarray:
    """sv.Detections -> Nx6 float array [x1, y1, x2, y2, conf, cls] BoxMOT expects."""
    if len(det) == 0:
        return np.empty((0, 6), dtype=np.float32)
    conf = det.confidence if det.confidence is not None else np.ones(len(det))
    cls = det.class_id if det.class_id is not None else np.zeros(len(det))
    return np.hstack([det.xyxy, conf[:, None], cls[:, None]]).astype(np.float32)


def boxmot_to_detections(out: np.ndarray) -> sv.Detections:
    """BoxMOT Nx8 output [x1,y1,x2,y2,id,conf,cls,idx] -> tracked sv.Detections."""
    out = np.asarray(out)
    if out.size == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=out[:, :4].astype(float),
        confidence=out[:, 5].astype(float),
        class_id=out[:, 6].astype(int),
        tracker_id=out[:, 4].astype(int),
    )


class MultiObjectTracker:
    """Wrap a BoxMOT tracker so it consumes/returns `sv.Detections`."""

    def __init__(
        self,
        tracker_name: str,
        reid_name: str | None = None,
        device: str = "cuda:0",
        frame_rate: int = 30,
        cfg: TrackerConfig | None = None,
        gallery: object | None = None,
    ):
        canonical = _normalize(tracker_name)
        reid_model = None
        if canonical in APPEARANCE_TRACKERS and reid_name:
            reid_model = build_reid(reid_name, device=device)
        elif canonical in APPEARANCE_TRACKERS and not reid_name:
            logger.warning(f"{canonical} supports ReID but no reid_name given; motion-only.")
        self.name = canonical
        self.reid_name = reid_name if reid_model is not None else None
        self.reid_model = reid_model
        self.frame_rate = max(1, frame_rate)
        self.gallery = gallery
        self._frame_idx = 0
        if gallery is not None and reid_model is None:
            logger.warning("Re-entry gallery enabled but tracker has no ReID model; disabling.")
            self.gallery = None
        self.tracker = build_tracker(canonical, reid_model, frame_rate, cfg)

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """Advance the tracker one frame; return detections carrying a stable id.

        With a re-entry gallery attached, ``tracker_id`` is remapped to a stable
        ``global_id`` that survives track loss / re-entry.
        """
        out = self.tracker.update(detections_to_boxmot(detections), frame)
        det = boxmot_to_detections(out)
        self._frame_idx += 1
        if self.gallery is not None and self.reid_model is not None and len(det):
            embeddings = self.reid_model.get_features(det.xyxy.astype(float), frame)
            gids = self.gallery.update(
                det.tracker_id.tolist(),
                embeddings,
                timestamp=self._frame_idx / self.frame_rate,
                frame_idx=self._frame_idx,
            )
            det.tracker_id = np.asarray(gids, dtype=int)
        return det
