"""Video + stats panel, side by side, replayed from a run's exports.

Nothing is detected or tracked here: boxes come from `*_events.csv`, counts and
visits from `*_activity.csv`, and the queue length is recomputed frame by frame
with the same `QueueMonitor`. So a composite can be re-cut for any time window
in seconds, and every number on screen is the run's own.

The panel is re-drawn once per `panel_every_s` (matplotlib is ~0.1 s a frame);
counts change slower than that. Output is encoded by ffmpeg (x264) through a
pipe, which is several times smaller than OpenCV's mp4v at the same quality.
"""

from __future__ import annotations

import csv
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from people_analytics.analytics.activity import _ANCHORS
from people_analytics.analytics.panel import PanelState, render_panel
from people_analytics.analytics.queue import QueueMonitor
from people_analytics.analytics.render import (
    ORANGE,
    YELLOW,
    draw_alert_banner,
    draw_door,
    draw_queue,
    draw_staff,
    draw_zones,
    pill,
    zone_color,
)
from people_analytics.analytics.staff import StaffMonitor, build_staff_monitor, waiting_count
from people_analytics.config import Config

CREDIT = "Video: MEVA dataset (mevadata.org), CC BY 4.0"
BOX = (235, 235, 235)
WHITE = sv.Color(r=235, g=235, b=235)


def _load(cfg: Config) -> tuple[dict, list[dict], float, int]:
    stem = Path(cfg.io.source).stem
    out = Path(cfg.io.output_dir) / "analytics"
    boxes: dict[int, list] = defaultdict(list)
    with open(out / f"{stem}_events.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            boxes[int(r["frame"])].append(
                (int(r["global_id"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")],
                 [z for z in r["zones"].split("|") if z])
            )
    with open(out / f"{stem}_activity.csv", newline="", encoding="utf-8") as fh:
        activity = list(csv.DictReader(fh))
    cap = cv2.VideoCapture(cfg.io.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return boxes, activity, fps, total


class SceneStats:
    """Everything the panel shows, computed up to a given frame from the event log."""

    def __init__(self, cfg: Config, activity: list[dict], fps: float, total: int, title: str):
        self.cfg, self.fps, self.title = cfg, fps, title
        # A "5-minute" clip is 300.03 s: don't give it an empty sixth minute.
        self.minutes = max(1, int(np.ceil(total / fps / 60 - 1 / 60)))
        self.events = [(int(r["frame"]), r["event"], r["place"],
                        float(r["value_s"]) if r["value_s"] else None) for r in activity]
        self.occupancy: list[tuple[float, int]] = [(0.0, 0)]

    def _upto(self, frame: int):
        return [e for e in self.events if e[0] <= frame]

    def state(self, frame: int, queue: QueueMonitor | None,
              staff: StaffMonitor | None = None) -> PanelState:
        t = frame / self.fps
        ev = self._upto(frame)
        footer = [CREDIT]
        if self.cfg.doors:
            ins = [f for f, e, _, _ in ev if e == "door_in"]
            outs = [f for f, e, _, _ in ev if e == "door_out"]
            last = self.minutes - 1
            per_min = {
                "In": [sum(1 for f in ins if min(int(f / self.fps // 60), last) == m)
                       for m in range(self.minutes)],
                "Out": [sum(1 for f in outs if min(int(f / self.fps // 60), last) == m)
                        for m in range(self.minutes)],
            }

            def cumulative(frames: list[int]) -> tuple[list[float], list[int]]:
                ts = [0.0] + [f / self.fps for f in frames] + [t]
                return ts, list(range(len(frames) + 1)) + [len(frames)]

            # Headline is traffic through the door, not "inside now": a clip can
            # start with the building full, and the counter can't know that.
            return PanelState(
                title=self.title, clock_s=t,
                tiles=[("Through the door", str(len(ins) + len(outs))),
                       ("Entered", str(len(ins))), ("Left", str(len(outs)))],
                minutes=self.minutes, footfall=per_min,
                footfall_title="People through the door per minute",
                lines={"In": cumulative(ins), "Out": cumulative(outs)},
                line_title="Running total since clip start",
                footer=footer,
            )

        starts = [(f, p) for f, e, p, _ in ev if e in ("zone_enter", "queue_join")]
        arrivals = [(f, p) for f, p in starts if f > self.fps]
        done: dict[str, list[float]] = defaultdict(list)
        for _, e, p, v in ev:
            if e in ("zone_exit", "queue_leave") and v is not None:
                done[p].append(v)
        per_min = [sum(1 for f, _ in arrivals if min(int(f / self.fps // 60),
                                                      self.minutes - 1) == m)
                   for m in range(self.minutes)]
        bars = {p: float(np.mean(v)) for p, v in done.items() if v}
        if queue is not None:
            wait = queue.wait_estimate_s
            third = ("Visits so far", str(len(starts)))
            if staff is not None:
                m, sec = divmod(int(staff.unstaffed_s), 60)
                third = ("Counter", "Staffed" if staff.staffed else f"Empty {m}:{sec:02d}")
            tiles = [("In queue now", str(queue.length)),
                     ("Est. wait", "–" if wait is None else f"{wait:.0f} s"), third]
        else:
            tiles = [("Visits so far", str(len(starts))),
                     ("Areas", str(len(self.cfg.zones))),
                     ("Longest visit", f"{max([0.0] + [max(v) for v in done.values()]):.0f} s")]
        return PanelState(
            title=self.title, clock_s=t, tiles=tiles, minutes=self.minutes,
            footfall={"Visits": per_min},
            footfall_title="Visits started per minute (after clip start)",
            bars=bars, bars_title="Average visit by area (finished)",
            footer=[f"A visit = {self.cfg.analytics.min_dwell_s:.0f} s or more in an area.",
                    *footer],
        )


def iter_scene(
    cfg: Config,
    title: str,
    start_s: float = 0.0,
    seconds: float | None = None,
    speed: float = 1.0,
    panel_size: tuple[int, int] = (640, 1080),
    panel_every_s: float = 0.5,
    highlight: set[int] | None = None,
):
    """Yield (annotated frame, panel image, frame number) for a cut of the scene.

    The queue is replayed from the clip start so its state is right at the cut.
    `speed` < 1 repeats frames (0.5 = half speed). `highlight` ids get an accent box.
    """
    boxes, activity, fps, total = _load(cfg)
    stats = SceneStats(cfg, activity, fps, total, title)
    default = cfg.analytics.anchor
    queues = [(q, QueueMonitor(np.array(q.polygon), fps, anchor=_ANCHORS[q.anchor or default],
                               max_speed=q.max_speed, min_visit_s=q.min_visit_s,
                               recent=q.recent)) for q in cfg.queues]
    zone_shapes = [(z.name, np.array(z.polygon, np.int32), zone_color(i))
                   for i, z in enumerate(cfg.zones)]
    staff = build_staff_monitor(cfg, fps)
    # An alert banner stays up for 8 s; once staff are back it turns "resolved".
    alert_text, alert_until, resolved = "", -1, False
    start_f = int(start_s * fps) + 1
    end_f = total if seconds is None else min(total, start_f + int(seconds * fps) - 1)
    repeat = max(1, round(1 / speed))

    cap = cv2.VideoCapture(cfg.io.source)
    panel, panel_at = None, -1e9
    door_cfg = cfg.doors[0] if cfg.doors else None
    for f in range(1, end_f + 1):
        ok, frame = cap.read()
        if not ok:
            break
        rows = boxes.get(f, [])
        det = (sv.Detections(xyxy=np.array([b for _, b, _ in rows]),
                             tracker_id=np.array([g for g, _, _ in rows]),
                             confidence=np.ones(len(rows)), class_id=np.zeros(len(rows), int))
               if rows else sv.Detections.empty())
        for _, qm in queues:
            qm.update(det)
        if staff is not None:
            alert = staff.update(det, waiting_count(cfg, queues))
            if alert is not None:
                alert_text, alert_until, resolved = alert.message, f + int(8 * fps), False
            if staff.staffed and f <= alert_until:
                resolved = True
        if f < start_f:  # replay from the clip start, but draw nothing yet
            continue

        queuing = {g for _, qm in queues for g in qm.queuing_ids}
        for gid, (x1, y1, x2, y2), _ in rows:
            if highlight and gid in highlight:
                col = YELLOW
            elif gid in queuing:
                col = ORANGE
            else:
                col = None
            bgr = col.as_bgr() if col is not None else BOX
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), bgr,
                          2 if col is not None else 1, cv2.LINE_AA)
            if cfg.analytics.show_labels:
                pill(frame, f"#{gid}", (int(x1), int(y1) - 2), col or WHITE, size=17,
                     align="left")

        # Areas and their labels go on top of the people, so no box crosses a label.
        counts = {n: sum(1 for _, _, zs in rows if n in zs) for n, _, _ in zone_shapes}
        img = draw_zones(frame, zone_shapes, counts)
        for q, qm in queues:
            wait = qm.wait_estimate_s
            img = draw_queue(img, np.array(q.polygon),
                             f"{q.name}  {qm.length} waiting"
                             + (f"  ~{wait:.0f} s wait" if wait is not None else ""))
        if door_cfg is not None:
            ev = stats._upto(f)
            n_in = sum(1 for e in ev if e[1] == "door_in")
            n_out = sum(1 for e in ev if e[1] == "door_out")
            img = draw_door(img, np.array(door_cfg.outer), np.array(door_cfg.inner),
                            f"{door_cfg.in_label} {n_in}   {door_cfg.out_label} {n_out}")
        if staff is not None:
            img = draw_staff(img, np.array(cfg.staff.polygon), cfg.staff.name,
                             staff.staffed, staff.unstaffed_s)
            if f <= alert_until:
                img = draw_alert_banner(img, alert_text, resolved)
        if f / fps - panel_at >= panel_every_s:
            q = queues[0][1] if queues else None
            panel = render_panel(stats.state(f, q, staff), panel_size)
            panel_at = f / fps
        for _ in range(repeat):
            yield img, panel, f
    cap.release()


def _pad_to(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] >= h:
        return img[:h]
    return cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                              value=(25, 26, 26))


class FFmpegWriter:
    """Raw BGR frames -> x264 mp4 through an ffmpeg pipe."""

    def __init__(self, path: str | Path, size: tuple[int, int], fps: float, crf: int = 23):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{size[0]}x{size[1]}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
             "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(self.path)],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> Path:
        self.proc.stdin.close()
        self.proc.wait()
        return self.path


def render_composite(
    cfg: Config,
    title: str,
    out_path: str | Path,
    start_s: float = 0.0,
    seconds: float | None = None,
    panel_every_s: float = 0.5,
    crf: int = 26,
    scale: float = 1.0,
) -> Path:
    """Video (1920 wide) + panel (640 wide) side by side for a cut of the scene."""
    fps = cv2.VideoCapture(cfg.io.source).get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(2560 * scale) // 2 * 2, int(1080 * scale) // 2 * 2)
    out = FFmpegWriter(out_path, size, fps, crf)
    for img, panel, _ in iter_scene(cfg, title, start_s, seconds, panel_every_s=panel_every_s):
        out.write(np.hstack([_pad_to(img, 1080)[:, :1920], panel]))
    return out.close()
