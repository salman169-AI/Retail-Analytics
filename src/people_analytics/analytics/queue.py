"""Queue length and estimated wait at a counter.

The queue area is a floor polygon in front of the counter. Two numbers come out
of it each frame:

- **Queue length**: people whose feet are in the polygon *and* who are moving
  slowly. Someone cutting across the area to the tables is in the polygon for a
  moment but is not queuing; speed separates the two.
- **Estimated wait**: the median time recent visitors spent in the polygon,
  over the last `recent` completed visits of at least `min_visit_s`. A median of
  recent visits is what a board above a real counter would show, and it ignores
  the odd person who stood chatting for ten minutes.

Speed is measured in *body heights per second* (feet displacement over a short
window, divided by the person's box height). Pixel speeds mean different things
at the front and back of a perspective view; body heights don't.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv


@dataclass
class _Visit:
    first_frame: int
    last_frame: int
    history: deque = field(default_factory=lambda: deque(maxlen=64))


class QueueMonitor:
    """Counts slow-moving people in a polygon and estimates wait from recent visits.

    Args:
        polygon: floor area in front of the counter (image coordinates).
        fps: source frame rate, to turn frames into seconds.
        max_speed: people moving faster than this many body heights per second
            are passing through, not queuing.
        speed_window_s: window over which speed is measured.
        min_visit_s: visits shorter than this don't feed the wait estimate.
        leave_grace_s: a visit only ends after the person has been out of the
            polygon (or untracked) this long, so a missed detection or a step
            outside the edge doesn't split one wait into two.
        recent: how many completed visits the median is taken over.
    """

    def __init__(
        self,
        polygon: np.ndarray,
        fps: float,
        anchor: sv.Position = sv.Position.BOTTOM_CENTER,
        max_speed: float = 0.5,
        speed_window_s: float = 1.0,
        min_visit_s: float = 3.0,
        leave_grace_s: float = 1.0,
        recent: int = 10,
    ):
        self.polygon = np.asarray(polygon, dtype=np.float32)
        self.fps = float(fps)
        self.anchor = anchor
        self.max_speed = float(max_speed)
        self.window = max(2, int(round(speed_window_s * fps)))
        self.min_visit_frames = int(round(min_visit_s * fps))
        self.grace = int(round(leave_grace_s * fps))
        self.completed: deque[float] = deque(maxlen=recent)
        self.all_visits_s: list[float] = []
        # (tracker_id, first_frame, last_frame) of every visit that counted.
        self.visit_log: list[tuple[int, int, int]] = []

        self.length = 0
        self.peak_length = 0
        self.queuing_ids: list[int] = []
        self._open: dict[int, _Visit] = {}
        self._frame = 0

    def _inside(self, point: np.ndarray) -> bool:
        return cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False) >= 0

    def _speed(self, visit: _Visit) -> float | None:
        """Body heights per second over the speed window, or None if too new to tell."""
        h = [p for p in visit.history if self._frame - p[0] <= self.window]
        if len(h) < 2 or h[-1][0] - h[0][0] < self.window // 2:
            return None
        (f0, x0, y0, _), (f1, x1, y1, height) = h[0], h[-1]
        dist = float(np.hypot(x1 - x0, y1 - y0))
        return dist / max(height, 1.0) / ((f1 - f0) / self.fps)

    def update(self, detections: sv.Detections) -> int:
        """Feed one frame of tracked detections; returns the current queue length."""
        self._frame += 1
        queuing: list[int] = []
        if len(detections) and detections.tracker_id is not None:
            feet = detections.get_anchors_coordinates(self.anchor)
            heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
            for tid, point, height in zip(
                detections.tracker_id.astype(int), feet, heights, strict=True
            ):
                if not self._inside(point):
                    continue
                visit = self._open.setdefault(int(tid), _Visit(self._frame, self._frame))
                visit.last_frame = self._frame
                visit.history.append((self._frame, point[0], point[1], float(height)))
                speed = self._speed(visit)
                # Too new to measure: count them once they've been here a moment.
                slow = speed is None and self._frame - visit.first_frame >= self.window // 2
                if slow or (speed is not None and speed <= self.max_speed):
                    queuing.append(int(tid))

        for tid in [t for t, v in self._open.items() if self._frame - v.last_frame > self.grace]:
            self._close(tid)

        self.queuing_ids = queuing
        self.length = len(queuing)
        self.peak_length = max(self.peak_length, self.length)
        return self.length

    def _close(self, tid: int) -> None:
        visit = self._open.pop(tid)
        frames = visit.last_frame - visit.first_frame + 1
        if frames >= self.min_visit_frames:
            self.completed.append(frames / self.fps)
            self.all_visits_s.append(frames / self.fps)
            self.visit_log.append((tid, visit.first_frame, visit.last_frame))

    def finish(self) -> None:
        """Close every open visit (end of clip) so they count toward the totals."""
        for tid in list(self._open):
            self._close(tid)

    @property
    def wait_estimate_s(self) -> float | None:
        return float(np.median(self.completed)) if self.completed else None

    def time_in_queue_s(self, tid: int) -> float | None:
        visit = self._open.get(int(tid))
        return None if visit is None else (self._frame - visit.first_frame + 1) / self.fps
