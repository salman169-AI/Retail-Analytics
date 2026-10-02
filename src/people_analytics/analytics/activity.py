"""Event log: one row per door crossing, zone visit and queue visit.

Built by replaying a run's per-detection export (`*_events.csv`: every tracked
box with its zones, frame by frame) through the same `DoorCounter` and
`QueueMonitor` the pipeline uses online, so the log can be rebuilt for any past
run without tracking again, and always agrees with the run's summary.

Rows: `time_s, frame, event, id, place, value_s`, where `event` is one of
door_in, door_out, zone_enter, zone_exit, queue_join, queue_leave, and `value_s`
is the visit length on zone_exit / queue_leave (dwell, time in queue).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import supervision as sv

from people_analytics.analytics.door import DoorCounter
from people_analytics.analytics.queue import QueueMonitor
from people_analytics.config import Config

FIELDS = ["time_s", "frame", "event", "id", "place", "value_s"]

_ANCHORS = {
    "center": sv.Position.CENTER,
    "bottom_center": sv.Position.BOTTOM_CENTER,
    "feet": sv.Position.BOTTOM_CENTER,
    "top_center": sv.Position.TOP_CENTER,
}


def _read_detections(path: str | Path) -> tuple[dict[int, list], dict[tuple[str, int], list[int]]]:
    """frame -> [[id, x1, y1, x2, y2]], and (zone, id) -> frames present."""
    per_frame: dict[int, list] = defaultdict(list)
    presence: dict[tuple[str, int], list[int]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            f, gid = int(r["frame"]), int(r["global_id"])
            per_frame[f].append([gid, float(r["x1"]), float(r["y1"]),
                                 float(r["x2"]), float(r["y2"])])
            for zone in filter(None, r["zones"].split("|")):
                presence[(zone, gid)].append(f)
    return per_frame, presence


def _detections(rows: list) -> sv.Detections:
    if not rows:
        return sv.Detections.empty()
    a = np.asarray(rows, dtype=float)
    return sv.Detections(xyxy=a[:, 1:5], tracker_id=a[:, 0].astype(int),
                         confidence=np.ones(len(a)), class_id=np.zeros(len(a), dtype=int))


def zone_visits(frames: list[int], gap_frames: int) -> list[tuple[int, int]]:
    """Split sorted presence frames into visits; gaps up to `gap_frames` don't split."""
    if not frames:
        return []
    frames = sorted(frames)
    visits, start, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f - prev > gap_frames + 1:
            visits.append((start, prev))
            start = f
        prev = f
    visits.append((start, prev))
    return visits


def build_activity(
    cfg: Config, detections_csv: str | Path, fps: float, first_frame: int, last_frame: int,
    gap_s: float = 1.0,
) -> list[dict]:
    """All events of one run, sorted by time. Frames are the export's (1-based, absolute)."""
    per_frame, presence = _read_detections(detections_csv)
    default = cfg.analytics.anchor
    doors = [(d, DoorCounter(d.outer, d.inner, anchor=_ANCHORS[d.anchor or default],
                             settle_frames=d.settle_frames, cooldown_frames=d.cooldown_frames,
                             forget_after_frames=d.forget_after_frames)) for d in cfg.doors]
    queues = [(q, QueueMonitor(np.array(q.polygon), fps, anchor=_ANCHORS[q.anchor or default],
                               max_speed=q.max_speed, min_visit_s=q.min_visit_s,
                               recent=q.recent)) for q in cfg.queues]

    def row(frame: int, event: str, gid: int, place: str, value: float | None = None) -> dict:
        return {"time_s": round(frame / fps, 2), "frame": frame, "event": event, "id": gid,
                "place": place, "value_s": "" if value is None else round(value, 2)}

    rows: list[dict] = []
    for f in range(first_frame, last_frame + 1):
        det = _detections(per_frame.get(f, []))
        for d, counter in doors:
            for ev in counter.trigger(det):
                rows.append(row(f, f"door_{ev.direction}", ev.tracker_id, d.name))
        for _, qm in queues:
            qm.update(det)
    for q, qm in queues:
        qm.finish()
        for gid, first, last in qm.visit_log:  # monitor frames count from 1
            a, b = first_frame + first - 1, first_frame + last - 1
            rows.append(row(a, "queue_join", gid, q.name))
            rows.append(row(b, "queue_leave", gid, q.name, (b - a + 1) / fps))

    min_frames = cfg.analytics.min_dwell_s * fps
    for (zone, gid), frames in presence.items():
        for a, b in zone_visits(frames, int(round(gap_s * fps))):
            if b - a + 1 < min_frames:
                continue
            rows.append(row(a, "zone_enter", gid, zone))
            rows.append(row(b, "zone_exit", gid, zone, (b - a + 1) / fps))
    return sorted(rows, key=lambda r: (r["frame"], r["event"], r["id"]))


def write_activity(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path
