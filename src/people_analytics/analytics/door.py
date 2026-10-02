"""Two-line door counter with running occupancy.

A single counting line double-counts whenever a person's feet jitter on it, and
a doorway is exactly where people stop, hold the door and shuffle. So a door is
drawn as **two roughly parallel lines**: an *outer* one at the threshold and an
*inner* one a step or two into the room. Between them is a dead band.

Each identity is remembered only by the last side it was settled on — outside
the outer line or inside the inner line. Standing in the band changes nothing.
A count happens only when an identity that was settled on one side is later
settled on the other, i.e. its feet crossed *both* lines in order:

    outside -> (band) -> inside   = IN
    inside  -> (band) -> outside  = OUT

Jitter inside the band can never count, and a person who steps into the band
and back out never counts either.

Three guards handle crowds, measured on MEVA G420 (a busy doorway, high camera):

- `settle_frames`: a side only becomes "settled" after the feet have stayed on
  it this many consecutive frames. When someone's legs are hidden behind the
  person in front, their box bottom jumps up the image for a few frames and can
  land beyond both lines; it must not count as walking back out.
- `cooldown_frames`: after counting an identity, a reversal by the same identity
  inside this window is ignored (its side stays where it was counted). Nobody
  walks in and back out through a door within a second or two; a box that jumps
  between two people in a doorway crowd does.
- `forget_after_frames`: an identity unseen for longer than this loses its
  settled side. Re-identification (the re-entry gallery) can hand a newcomer at
  the door the id of someone who went in half a minute ago; judged against that
  person's old "inside" side, the newcomer would count OUT and then IN. A
  continuously tracked person never hits this (short blinks are interpolated).

Occupancy is running IN minus OUT, clamped at zero (people already inside when
the clip starts are not known to the counter).

Points beyond the ends of the door (along the line direction) are ignored, so
someone walking past the doorway further down the hall is not "outside".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

OUTSIDE, BAND, INSIDE = -1, 0, 1

# Default for `forget_after_frames`: generous, so someone who stops behind a door
# frame for a while is still judged against where they were settled before.
_FORGET_AFTER_FRAMES = 900


@dataclass
class DoorEvent:
    frame: int
    tracker_id: int
    direction: str  # "in" | "out"


class DoorCounter:
    """IN/OUT counts across a doorway drawn as an outer + inner line.

    Args:
        outer: (start, end) of the threshold line.
        inner: (start, end) of the line a step into the room. Its side of the
            outer line defines "inside".
        anchor: point of the box tested against the lines (feet for a floor door).
        span_margin: how far past the door ends, as a fraction of the door
            width, a point may lie and still be judged.
    """

    def __init__(
        self,
        outer: tuple[tuple[float, float], tuple[float, float]],
        inner: tuple[tuple[float, float], tuple[float, float]],
        anchor: sv.Position = sv.Position.BOTTOM_CENTER,
        span_margin: float = 0.1,
        settle_frames: int = 1,
        cooldown_frames: int = 0,
        forget_after_frames: int = _FORGET_AFTER_FRAMES,
    ):
        self.outer = np.asarray(outer, dtype=float)
        self.inner = np.asarray(inner, dtype=float)
        self.anchor = anchor
        self.span_margin = float(span_margin)
        self.settle_frames = max(1, int(settle_frames))
        self.cooldown_frames = int(cooldown_frames)
        self.forget_after_frames = int(forget_after_frames)

        self._origin = self.outer[0]
        along = self.outer[1] - self.outer[0]
        self._along = along / (np.dot(along, along) or 1.0)
        normal = np.array([-along[1], along[0]])
        normal /= np.linalg.norm(normal) or 1.0
        # Orient the normal so the inner line sits on the positive side.
        if np.dot(self.inner.mean(axis=0) - self._origin, normal) < 0:
            normal = -normal
        self._normal = normal

        self.in_count = 0
        self.out_count = 0
        self.occupancy = 0
        self.peak_occupancy = 0
        self.events: list[DoorEvent] = []
        self.rejected_reversals = 0  # reversals dropped by the cooldown

        self._side: dict[int, int] = {}
        # Candidate side not yet settled, and for how many frames it has held.
        self._pending: dict[int, tuple[int, int]] = {}
        self._counted_at: dict[int, int] = {}
        self._last_seen: dict[int, int] = {}
        self._frame = 0

    def region(self, point: np.ndarray) -> int | None:
        """OUTSIDE / BAND / INSIDE for a point, or None if it is beside the door."""
        t = float(np.dot(point - self._origin, self._along))
        if not -self.span_margin <= t <= 1 + self.span_margin:
            return None
        if np.dot(point - self._origin, self._normal) < 0:
            return OUTSIDE
        if np.dot(point - self.inner[0], self._normal) > 0:
            return INSIDE
        return BAND

    def trigger(self, detections: sv.Detections) -> list[DoorEvent]:
        """Update with one frame's tracked detections; returns this frame's crossings."""
        self._frame += 1
        new_events: list[DoorEvent] = []
        if len(detections) and detections.tracker_id is not None:
            points = detections.get_anchors_coordinates(self.anchor)
            for tid, point in zip(detections.tracker_id.astype(int), points, strict=True):
                tid = int(tid)
                last = self._last_seen.get(tid)
                if last is not None and self._frame - last > self.forget_after_frames:
                    self._forget(tid)
                self._last_seen[tid] = self._frame
                side = self.region(point)
                if side is None or side == BAND:
                    self._pending.pop(tid, None)
                    continue
                pending_side, held = self._pending.get(tid, (side, 0))
                held = held + 1 if pending_side == side else 1
                self._pending[tid] = (side, held)
                if held < self.settle_frames:
                    continue
                previous = self._side.get(tid)
                if previous is None or previous == side:
                    self._side[tid] = side
                    continue
                counted_at = self._counted_at.get(tid)
                if counted_at is not None and self._frame - counted_at < self.cooldown_frames:
                    if held == self.settle_frames:  # once per reversal, not per frame
                        self.rejected_reversals += 1
                    continue
                self._side[tid] = side
                self._counted_at[tid] = self._frame
                if side == INSIDE:
                    self.in_count += 1
                    self.occupancy += 1
                    direction = "in"
                else:
                    self.out_count += 1
                    self.occupancy = max(0, self.occupancy - 1)
                    direction = "out"
                self.peak_occupancy = max(self.peak_occupancy, self.occupancy)
                new_events.append(DoorEvent(self._frame, tid, direction))

        if self._frame % 300 == 0:
            for tid in [t for t, f in self._last_seen.items()
                        if self._frame - f > self.forget_after_frames]:
                self._forget(tid)
                del self._last_seen[tid]
        self.events.extend(new_events)
        return new_events

    def _forget(self, tid: int) -> None:
        self._side.pop(tid, None)
        self._pending.pop(tid, None)
        self._counted_at.pop(tid, None)
