"""Detection layer (YOLO26 -> sv.Detections)."""

from people_analytics.detection.detector import (
    PersonDetector,
    box_ios,
    clean_detections,
    ensure_weights,
    is_vertical_partial,
    merge_full_and_tiled,
    nms_ios,
    overlap_wh_from_ratio,
    resolve_device,
)

__all__ = [
    "PersonDetector",
    "box_ios",
    "clean_detections",
    "ensure_weights",
    "is_vertical_partial",
    "merge_full_and_tiled",
    "nms_ios",
    "overlap_wh_from_ratio",
    "resolve_device",
]
