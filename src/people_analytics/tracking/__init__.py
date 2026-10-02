"""Tracking layer: BoxMOT integration + benchmark harness."""

from people_analytics.tracking.reid import REID_WEIGHTS, build_reid
from people_analytics.tracking.tracker import (
    MultiObjectTracker,
    boxmot_to_detections,
    build_tracker,
    detections_to_boxmot,
)

__all__ = [
    "REID_WEIGHTS",
    "MultiObjectTracker",
    "boxmot_to_detections",
    "build_reid",
    "build_tracker",
    "detections_to_boxmot",
]
