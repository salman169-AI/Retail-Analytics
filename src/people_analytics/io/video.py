"""Minimal OpenCV-backed video reader (Phase 1 stub reader).

Kept dependency-light on purpose: this is the interface every later phase reads
frames through, so it must import cleanly before the ML stack is installed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Basic properties of a video source."""

    width: int
    height: int
    fps: float
    total_frames: int

    @property
    def resolution_wh(self) -> tuple[int, int]:
        return self.width, self.height


def get_video_info(source: str | Path) -> VideoInfo:
    """Probe a video file for width/height/fps/frame-count."""
    source = str(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")
    try:
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def frame_generator(
    source: str | Path,
    max_frames: int | None = None,
    start: int = 0,
) -> Iterator[np.ndarray]:
    """Yield BGR frames from a video source, from frame `start`, capped at `max_frames`."""
    source = str(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    try:
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        cap.release()
