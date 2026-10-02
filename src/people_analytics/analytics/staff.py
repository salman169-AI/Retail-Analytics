"""Is anyone behind the counter? Alert when customers wait at an unstaffed counter.

Each frame, the counter counts as **staffed** if a tracked person's anchor (box
centre by default: the counter hides staff legs) is inside the staff area. A
single missed detection must not look like the server walking off, and a single
stray detection must not look like them coming back, so the state only flips
after it has held for a moment:

- staffed -> empty: the moment nobody has been seen for `back_after_s` (the
  absence is then timed from the last frame staff were seen);
- empty -> staffed: someone has been seen for `back_after_s` in a row.

An **alert** fires when the counter has been empty for `alert_after_s` and at
least `min_waiting` people are in the queue. It fires once per absence, then
again every `repeat_after_s` if nothing changes (0 disables repeats).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv


@dataclass
class StaffAlert:
    frame: int
    time_s: float
    waiting: int
    unstaffed_s: float
    counter: str

    @property
    def message(self) -> str:
        m, s = divmod(int(round(self.unstaffed_s)), 60)
        people = "person is" if self.waiting == 1 else "people are"
        # Fixed wording (the area's name is free text and doesn't always fit a sentence).
        return (f"{self.waiting} {people} waiting and no one has been at the counter "
                f"for {m}:{s:02d}.")


class StaffMonitor:
    def __init__(
        self,
        polygon: np.ndarray,
        fps: float,
        anchor: sv.Position = sv.Position.CENTER,
        counter: str = "the counter",
        alert_after_s: float = 120.0,
        min_waiting: int = 1,
        back_after_s: float = 1.0,
        repeat_after_s: float = 300.0,
    ):
        self.polygon = np.asarray(polygon, dtype=np.float32)
        self.fps = float(fps)
        self.anchor = anchor
        self.counter = counter
        self.alert_after = alert_after_s * fps
        self.min_waiting = int(min_waiting)
        self.settle = max(1, int(round(back_after_s * fps)))
        self.repeat = repeat_after_s * fps

        self.staffed = True  # assume staffed until shown otherwise
        self.staff_now = 0
        self.last_seen = 0  # frame staff were last seen
        self.unstaffed_since: int | None = None
        self._seen_run = 0
        self._last_alert: int | None = None
        self._frame = 0
        self.alerts: list[StaffAlert] = []
        self.changes: list[tuple[int, bool]] = []  # (frame, staffed)

    def _count(self, detections: sv.Detections) -> int:
        if len(detections) == 0:
            return 0
        pts = detections.get_anchors_coordinates(self.anchor)
        return sum(cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0
                   for x, y in pts)

    def update(self, detections: sv.Detections, waiting: int) -> StaffAlert | None:
        """Feed one frame; `waiting` = people currently in the queue."""
        self._frame += 1
        f = self._frame
        self.staff_now = self._count(detections)
        if self.staff_now:
            self.last_seen = f
            self._seen_run += 1
        else:
            self._seen_run = 0

        if self.staffed and f - self.last_seen >= self.settle:
            self.staffed = False
            self.unstaffed_since = self.last_seen + 1
            self._last_alert = None
            self.changes.append((self.unstaffed_since, False))
        elif not self.staffed and self._seen_run >= self.settle:
            self.staffed = True
            self.unstaffed_since = None
            self.changes.append((f - self.settle + 1, True))

        if self.staffed or self.unstaffed_since is None:
            return None
        empty_for = f - self.unstaffed_since + 1
        due = (self._last_alert is None and empty_for >= self.alert_after) or (
            self._last_alert is not None and self.repeat > 0
            and f - self._last_alert >= self.repeat
        )
        if due and waiting >= self.min_waiting:
            self._last_alert = f
            alert = StaffAlert(f, f / self.fps, waiting, empty_for / self.fps, self.counter)
            self.alerts.append(alert)
            return alert
        return None

    @property
    def unstaffed_s(self) -> float:
        """How long the counter has been empty (0 while staffed)."""
        if self.staffed or self.unstaffed_since is None:
            return 0.0
        return (self._frame - self.unstaffed_since + 1) / self.fps


_ANCHORS = {
    "center": sv.Position.CENTER,
    "bottom_center": sv.Position.BOTTOM_CENTER,
    "feet": sv.Position.BOTTOM_CENTER,
    "top_center": sv.Position.TOP_CENTER,
}


def build_staff_monitor(cfg, fps: float) -> StaffMonitor | None:
    """StaffMonitor from `cfg.staff` (a `people_analytics.config.Config`), or None."""
    s = cfg.staff
    if s is None:
        return None
    return StaffMonitor(
        np.array(s.polygon), fps, anchor=_ANCHORS[s.anchor or "center"], counter=s.name,
        alert_after_s=s.alert_after_s, min_waiting=s.min_waiting,
        back_after_s=s.back_after_s, repeat_after_s=s.repeat_after_s,
    )


def waiting_count(cfg, queues: list) -> int:
    """People in the queue the staff alert watches (`staff.queue`, else the first)."""
    if not queues:
        return 0
    name = cfg.staff.queue if cfg.staff is not None else None
    for q, qm in queues:
        if name is None or q.name == name:
            return qm.length
    return 0
