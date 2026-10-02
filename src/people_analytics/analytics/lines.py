"""Counting lines with a crossing-angle gate.

`sv.LineZone` counts *any* crossing of the segment. For a doorway that sits
alongside a busy pavement that over-counts badly: people walking **past** the
door clip the line at a shallow angle and register as an entry.

`DirectionalLineZone` wraps `sv.LineZone` and only accepts a crossing when the
person was actually moving *through* the doorway — i.e. their travel direction
is within `max_angle_deg` of the line's normal. Walking parallel to the line no
longer counts.

The wrapper exposes `vector` / `in_count` / `out_count`, so it drops straight
into `sv.LineZoneAnnotator` in place of the real thing.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import supervision as sv

# Judge direction over a short window rather than a single frame. One frame of
# motion is mostly detector jitter, and jitter that happens to sit astride the
# line is both tiny *and* perfectly perpendicular — the angle test alone can't
# reject it. Over several frames a real walker accumulates displacement and a
# wobbling stationary person doesn't.
_LOOKBACK_FRAMES = 5

# Drop remembered positions for identities not seen for this many frames, so a
# long clip can't grow the history without bound. Generous on purpose: a track
# that vanishes behind an obstruction and reappears on the far side should still
# be judged against where it went in, not treated as brand new.
_FORGET_AFTER_FRAMES = 900


class DirectionalLineZone:
    """A counting line that ignores crossings made at a glancing angle.

    Args:
        start, end: line endpoints.
        triggering_anchors: which point on the box is tested (feet, centre, ...).
        max_angle_deg: how far off perpendicular a crossing may be and still
            count. 90 accepts everything (plain `sv.LineZone` behaviour); 45
            means "must be travelling more through the line than along it".
        min_travel_px: ignore crossings where the anchor barely moved across the
            lookback window — a few pixels of wobble is noise, not intent.
        cooldown_frames: after counting someone, ignore that same identity for
            this many frames. Nobody walks through a door twice in half a second;
            a box hovering on the line does.
    """

    def __init__(
        self,
        start: sv.Point,
        end: sv.Point,
        triggering_anchors: list[sv.Position],
        max_angle_deg: float = 90.0,
        min_travel_px: float = 4.0,
        cooldown_frames: int = 0,
    ):
        self._zone = sv.LineZone(
            start=start, end=end, triggering_anchors=triggering_anchors
        )
        self._anchor = triggering_anchors[0]
        self.max_angle_deg = float(max_angle_deg)
        self.min_travel_px = float(min_travel_px)
        self.cooldown_frames = int(cooldown_frames)

        self.in_count = 0
        self.out_count = 0
        self.rejected_oblique = 0   # crossings dropped for being too shallow
        self.rejected_repeat = 0    # same identity re-counted inside the cooldown

        # Unit normal of the line: the direction a perpendicular crossing travels.
        along = np.array([end.x - start.x, end.y - start.y], dtype=float)
        normal = np.array([-along[1], along[0]], dtype=float)
        self._normal = normal / (np.linalg.norm(normal) or 1.0)

        # tracker_id -> recent anchor positions, plus the frame it was last seen.
        self._history: dict[int, deque[np.ndarray]] = {}
        self._last_frame: dict[int, int] = {}
        self._counted_at: dict[int, int] = {}
        self._frame = 0

    @property
    def vector(self) -> sv.Vector:
        """Delegate so `sv.LineZoneAnnotator` can draw us."""
        return self._zone.vector

    def _accepts(self, previous: np.ndarray, current: np.ndarray) -> bool:
        travel = current - previous
        distance = float(np.linalg.norm(travel))
        if distance < self.min_travel_px:
            return False
        if self.max_angle_deg >= 90.0:
            return True
        alignment = abs(float(np.dot(travel / distance, self._normal)))
        return alignment >= math.cos(math.radians(self.max_angle_deg))

    def trigger(self, detections: sv.Detections) -> tuple[np.ndarray, np.ndarray]:
        """Update counts; returns supervision's (crossed_in, crossed_out) masks
        with the rejected crossings cleared."""
        self._frame += 1
        crossed_in, crossed_out = self._zone.trigger(detections)
        if len(detections) == 0 or detections.tracker_id is None:
            return crossed_in, crossed_out

        anchors = detections.get_anchors_coordinates(self._anchor)
        for i, tracker_id in enumerate(detections.tracker_id):
            tracker_id = int(tracker_id)
            current = anchors[i]
            history = self._history.setdefault(
                tracker_id, deque(maxlen=_LOOKBACK_FRAMES)
            )
            if crossed_in[i] or crossed_out[i]:
                counted_at = self._counted_at.get(tracker_id)
                repeat = (
                    counted_at is not None
                    and self._frame - counted_at < self.cooldown_frames
                )
                # Oldest position still in the window: the further back we can
                # look, the less the direction is dominated by per-frame noise.
                if repeat:
                    crossed_in[i] = False
                    crossed_out[i] = False
                    self.rejected_repeat += 1
                elif history and self._accepts(history[0], current):
                    if crossed_in[i]:
                        self.in_count += 1
                    else:
                        self.out_count += 1
                    self._counted_at[tracker_id] = self._frame
                else:
                    crossed_in[i] = False
                    crossed_out[i] = False
                    self.rejected_oblique += 1
            history.append(current)
            self._last_frame[tracker_id] = self._frame

        if self._frame % 300 == 0:
            stale = [
                k
                for k, seen in self._last_frame.items()
                if self._frame - seen > _FORGET_AFTER_FRAMES
            ]
            for k in stale:
                del self._history[k]
                del self._last_frame[k]
                self._counted_at.pop(k, None)
        return crossed_in, crossed_out
