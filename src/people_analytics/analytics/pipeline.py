"""End-to-end analytics: detect -> track (+gallery) -> zones / line / heatmap.

Produces, for a full clip:
  - an annotated video (boxes + stable IDs + zone polygons + counting line),
  - a per-zone **dwell-time** breakdown per identity,
  - **entry/exit** counts + running **occupancy** from a LineZone,
  - an accumulated occupancy **heatmap** image,
  - CSV (per-detection events) and JSON (summary) exports.
"""

from __future__ import annotations

import contextlib
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from loguru import logger

from people_analytics.analytics.door import DoorCounter
from people_analytics.analytics.dwell import DwellTracker
from people_analytics.analytics.lines import DirectionalLineZone
from people_analytics.analytics.queue import QueueMonitor
from people_analytics.analytics.render import draw_door, draw_queue, draw_zones, zone_color
from people_analytics.config import Config
from people_analytics.detection import PersonDetector
from people_analytics.io import frame_generator, get_video_info
from people_analytics.tracking import MultiObjectTracker
from people_analytics.tracking.interpolate import count_filled, interpolate_tracks

# Where on each detection box a zone/line "presence" is tested.
_ANCHORS = {
    "center": sv.Position.CENTER,
    "bottom_center": sv.Position.BOTTOM_CENTER,
    "feet": sv.Position.BOTTOM_CENTER,
    "top_center": sv.Position.TOP_CENTER,
}


def _anchor(name: str | None, default: str = "center") -> sv.Position:
    """Resolve an anchor name, falling back to the scene-wide default."""
    name = name or default
    if name not in _ANCHORS:
        raise ValueError(f"Unknown anchor '{name}'. Options: {sorted(_ANCHORS)}")
    return _ANCHORS[name]


def _build_gallery(cfg: Config):
    if not cfg.reid_gallery.enabled:
        return None
    from people_analytics.reid import ReEntryGallery

    g = cfg.reid_gallery
    return ReEntryGallery(
        cosine_thresh=g.cosine_thresh,
        time_window_s=g.time_window_s,
        max_gallery_size=g.max_gallery_size,
        ema_alpha=g.ema_alpha,
    )


def drop_resurrected(smoothed: sv.Detections, live_ids: set[int]) -> sv.Detections:
    """Remove smoothed boxes for identities the tracker did not report this frame.

    `sv.DetectionsSmoother` keeps a vanished track alive for the whole length of
    its window: it records a `None` for the missing frame but still emits the
    average of whatever boxes remain. The result is a ghost box that drifts along
    at a stale position for several frames, overlapping whoever is really there —
    which reads as one person wearing several boxes, and as boxes sliding around
    on people. Smoothing is still worth having for jitter, so keep it and just
    drop anything the tracker isn't currently reporting.
    """
    if len(smoothed) == 0:
        return smoothed
    keep = np.array([int(i) in live_ids for i in smoothed.tracker_id], dtype=bool)
    return smoothed[keep]


def _detections_to_rows(det: sv.Detections) -> np.ndarray:
    """sv.Detections -> MOT rows [id, x, y, w, h, conf] (top-left xywh)."""
    xywh = det.xyxy.copy().astype(float)
    xywh[:, 2] -= xywh[:, 0]
    xywh[:, 3] -= xywh[:, 1]
    conf = det.confidence if det.confidence is not None else np.ones(len(det))
    return np.column_stack([det.tracker_id.astype(float), xywh, conf])


def _rows_to_detections(rows: np.ndarray | None) -> sv.Detections:
    """Inverse of `_detections_to_rows`, for rendering interpolated tracks."""
    if rows is None or len(rows) == 0:
        return sv.Detections.empty()
    xyxy = rows[:, 1:5].astype(float).copy()
    xyxy[:, 2] += xyxy[:, 0]
    xyxy[:, 3] += xyxy[:, 1]
    return sv.Detections(
        xyxy=xyxy,
        tracker_id=rows[:, 0].astype(int),
        confidence=rows[:, 5].astype(float),
        class_id=np.zeros(len(rows), dtype=int),
    )


def _downsample_timeline(series: list[int], fps: float, max_points: int = 400) -> list[dict]:
    """Return [{t_seconds, count}] for the occupancy series, downsampled."""
    if not series:
        return []
    step = max(1, len(series) // max_points)
    return [
        {"t": round(i / fps, 2), "count": series[i]}
        for i in range(0, len(series), step)
    ]


def run_analytics(
    cfg: Config,
    source: str | None = None,
    max_frames: int | None = None,
    output_dir: str | None = None,
    heatmap_overlay: bool = False,
) -> dict:
    """Run the full analytics pipeline over a clip; write video + exports. Returns summary."""
    src = source or cfg.io.source
    if not Path(src).exists():
        raise FileNotFoundError(f"Video source not found: {src}")

    vinfo = get_video_info(src)
    fps = vinfo.fps or 30.0
    n_frames = max_frames if max_frames is not None else cfg.io.max_frames
    start = cfg.io.start_frame

    out_dir = Path(output_dir or cfg.io.output_dir) / "analytics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem

    # --- pipeline components -------------------------------------------------
    detector = PersonDetector(cfg.detector)
    mot = MultiObjectTracker(
        cfg.tracker.name,
        reid_name=cfg.tracker.reid_model,
        device=cfg.detector.device,
        frame_rate=int(fps),
        cfg=cfg.tracker,
        gallery=_build_gallery(cfg),
    )
    smoother = sv.DetectionsSmoother(length=cfg.analytics.smoothing_window)

    default_anchor = cfg.analytics.anchor
    zones, zone_shapes = [], []
    for i, z in enumerate(cfg.zones):
        polygon = np.array(z.polygon, dtype=np.int32)
        pz = sv.PolygonZone(
            polygon=polygon.astype(np.int64),
            triggering_anchors=[_anchor(z.anchor, default_anchor)],
        )
        zones.append((z.name, pz))
        zone_shapes.append((z.name, polygon, zone_color(i)))
    dwell = DwellTracker([z.name for z in cfg.zones], fps, cfg.analytics.min_dwell_s)

    lines, line_annotators = [], []
    for line_cfg in cfg.lines:
        lz = DirectionalLineZone(
            start=sv.Point(*line_cfg.start),
            end=sv.Point(*line_cfg.end),
            triggering_anchors=[_anchor(line_cfg.anchor, default_anchor)],
            max_angle_deg=line_cfg.max_angle_deg,
            cooldown_frames=line_cfg.cooldown_frames,
        )
        lines.append((line_cfg.name, lz))
        # Green + thick so the counting line reads distinctly from the dwell zones.
        line_annotators.append(
            sv.LineZoneAnnotator(
                thickness=4, text_scale=1.0, color=sv.Color(r=0, g=220, b=0),
                custom_in_text=line_cfg.in_label, custom_out_text=line_cfg.out_label,
            )
        )

    doors = [
        (d, DoorCounter(
            d.outer, d.inner, anchor=_anchor(d.anchor, default_anchor),
            settle_frames=d.settle_frames, cooldown_frames=d.cooldown_frames,
            forget_after_frames=d.forget_after_frames,
        ))
        for d in cfg.doors
    ]
    door_events: list[dict] = []
    queues = [
        (q, QueueMonitor(
            np.array(q.polygon), fps, anchor=_anchor(q.anchor, default_anchor),
            max_speed=q.max_speed, min_visit_s=q.min_visit_s, recent=q.recent,
        ))
        for q in cfg.queues
    ]

    box = sv.BoxAnnotator(thickness=2)
    label = sv.LabelAnnotator(
        text_scale=max(0.5, vinfo.width / 2400), text_position=sv.Position.TOP_CENTER
    )
    heat = sv.HeatMapAnnotator(opacity=0.5, radius=max(20, vinfo.width // 60))

    # Privacy: obscure each person's pixels before any overlay is drawn.
    anonymizer = None
    if cfg.privacy.enabled:
        anonymizer = (
            sv.PixelateAnnotator()
            if cfg.privacy.method == "pixelate"
            else sv.BlurAnnotator()
        )
        logger.info(f"Privacy anonymization ON ({cfg.privacy.method}).")

    # --- run -----------------------------------------------------------------
    write_video = cfg.analytics.write_video
    write_heatmap = cfg.analytics.heatmap
    if not write_video:
        logger.info("Annotated video disabled (analytics.write_video=false) - exports only.")
    video_path = out_dir / f"{stem}_analytics.mp4"
    events_path = out_dir / f"{stem}_events.csv"
    sink_info = sv.VideoInfo(width=vinfo.width, height=vinfo.height, fps=fps)

    peak_occupancy = 0
    occupancy_sum = 0
    frames_done = 0
    background = None
    seen_ids: set[int] = set()
    occupancy_series: list[int] = []
    events_file = events_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(events_file)
    writer.writerow(["frame", "time_s", "global_id", "x1", "y1", "x2", "y2", "zones"])

    def process(frame: np.ndarray, detections: sv.Detections, sink: sv.VideoSink) -> None:
        """Analytics + render for one frame's final detections (shared by both modes)."""
        nonlocal peak_occupancy, occupancy_sum, frames_done, background
        if background is None:
            background = frame.copy()
        frames_done += 1
        # Absolute frame (1-based, as before) and clip time, so a cropped run's
        # events still line up with the source video and its annotations.
        frame_no = start + frames_done
        t = frame_no / fps

        # zones -> dwell + per-detection membership
        membership: dict[int, list[str]] = {}
        zone_masks = []
        for name, pz in zones:
            mask = pz.trigger(detections)
            zone_masks.append((name, mask))
            ids = [int(i) for i in detections.tracker_id[mask]] if len(detections) else []
            dwell.update(name, ids, frame_no)
            for gid in ids:
                membership.setdefault(gid, []).append(name)

        for _, lz in lines:
            lz.trigger(detections)
        for d, counter in doors:
            for ev in counter.trigger(detections):
                door_events.append({
                    "door": d.name, "frame": frame_no, "time_s": round(t, 2),
                    "global_id": ev.tracker_id, "direction": ev.direction,
                })
        for _, qm in queues:
            qm.update(detections)

        # `sv.Detections.empty()` carries `tracker_id=None` rather than an empty
        # array, so every plain iteration over it raises on a frame the tracker
        # reported nothing for. Continuously-busy clips never reach that state; a
        # street corner with a quiet moment does, and it took the whole run down.
        # Normalising once here keeps the rest of this function honest about the
        # empty case instead of guarding it site by site.
        track_ids = (
            detections.tracker_id if detections.tracker_id is not None
            else np.empty(0, dtype=int)
        )

        occupancy = len(detections)
        peak_occupancy = max(peak_occupancy, occupancy)
        occupancy_sum += occupancy
        occupancy_series.append(occupancy)
        seen_ids.update(int(i) for i in track_ids)

        for i in range(len(detections)):
            gid = int(track_ids[i])
            x1, y1, x2, y2 = detections.xyxy[i]
            writer.writerow(
                [frame_no, f"{t:.2f}", gid, int(x1), int(y1), int(x2), int(y2),
                 "|".join(membership.get(gid, []))]
            )

        # Heatmap accumulation is independent of the video: it still needs every
        # frame's detections even when nothing is being rendered. When the overlay
        # is on, the overlay call below does the accumulating, so doing it here as
        # well would count every detection twice.
        overlay_now = heatmap_overlay and sink is not None
        if write_heatmap and not overlay_now:
            heat.annotate(frame.copy(), detections)

        # Everything below only exists to produce the annotated MP4. On a headless
        # analytics run it is pure cost — a full-frame copy plus five annotator
        # passes plus an H.264 encode, per frame — so it is skipped entirely
        # rather than rendered and thrown away.
        if sink is None:
            return

        canvas = frame.copy()
        if anonymizer is not None:
            canvas = anonymizer.annotate(canvas, detections)
        if overlay_now:
            canvas = heat.annotate(canvas, detections)

        annotated = draw_zones(
            canvas, zone_shapes, {name: int(mask.sum()) for name, mask in zone_masks}
        )
        for d, counter in doors:
            annotated = draw_door(
                annotated, counter.outer, counter.inner,
                f"{d.in_label} {counter.in_count}  {d.out_label} {counter.out_count}"
                f"  INSIDE {counter.occupancy}",
            )
        for q, qm in queues:
            wait = qm.wait_estimate_s
            annotated = draw_queue(
                annotated, np.array(q.polygon),
                f"{q.name}  {qm.length} waiting"
                + (f"  ~{wait:.0f}s wait" if wait is not None else ""),
            )
        annotated = box.annotate(annotated, detections)
        if cfg.analytics.show_labels:
            labels = []
            for gid in (int(i) for i in track_ids):
                # Live timer: the longest dwell so far among the zones they're in.
                secs = max(
                    (dwell.seconds_so_far(z, gid) for z in membership.get(gid, [])),
                    default=None,
                )
                labels.append(f"#{gid}" if secs is None else f"#{gid} {secs:.0f}s")
            annotated = label.annotate(annotated, detections, labels)
        for (_name, lz), la in zip(lines, line_annotators, strict=True):
            annotated = la.annotate(annotated, lz)
        sink.write_frame(annotated)

    interp_added = 0
    # A null context when the annotated video is switched off, so the two passes
    # below stay a single code path and `process` simply receives sink=None.
    sink_ctx = (
        sv.VideoSink(str(video_path), sink_info) if write_video
        else contextlib.nullcontext(None)
    )
    with sink_ctx as sink:
        if not cfg.analytics.interpolate:
            for frame in frame_generator(src, max_frames=n_frames, start=start):
                tracked = mot.update(detector.detect(frame), frame)
                live_ids = {int(i) for i in tracked.tracker_id} if len(tracked) else set()
                detections = drop_resurrected(
                    smoother.update_with_detections(tracked), live_ids
                )
                process(frame, detections, sink)
        else:
            # Pass 1: track every frame, collecting raw tracks (no smoothing).
            raw: dict[int, np.ndarray] = {}
            frames = frame_generator(src, max_frames=n_frames, start=start)
            for idx, frame in enumerate(frames, start=1):
                tracked = mot.update(detector.detect(frame), frame)
                if len(tracked):
                    raw[idx] = _detections_to_rows(tracked)
            filled = interpolate_tracks(raw, max_gap=cfg.analytics.interpolate_max_gap)
            interp_added = count_filled(raw, filled)
            logger.info(
                f"Interpolation bridged {interp_added} boxes across "
                f"{len(filled) - len(raw)} newly-populated frames."
            )
            # Pass 2: re-read frames, render from the gap-filled tracks.
            frames = frame_generator(src, max_frames=n_frames, start=start)
            for idx, frame in enumerate(frames, start=1):
                process(frame, _rows_to_detections(filled.get(idx)), sink)

    events_file.close()
    for _, qm in queues:
        qm.finish()

    # Event log (door crossings, zone and queue visits), replayed from the export
    # just written so it is identical to what `people-analytics activity` rebuilds.
    from people_analytics.analytics.activity import build_activity, write_activity

    activity_path = write_activity(
        build_activity(cfg, events_path, fps, start + 1, start + frames_done),
        out_dir / f"{stem}_activity.csv",
    )

    # --- heatmap still -------------------------------------------------------
    heatmap_path = out_dir / f"{stem}_heatmap.png"
    if write_heatmap and background is not None:
        still = heat.annotate(background.copy(), sv.Detections.empty())
        cv2.imwrite(str(heatmap_path), still)

    # --- summary -------------------------------------------------------------
    summary = {
        "source": src,
        "start_frame": start,
        "frames": frames_done,
        "duration_s": round(frames_done / fps, 2),
        "interpolation": {
            "enabled": cfg.analytics.interpolate,
            "max_gap": cfg.analytics.interpolate_max_gap,
            "boxes_added": interp_added,
        },
        "unique_visitors": len(seen_ids),
        "zone_visitors": len({r.global_id for r in dwell.records()}),
        "occupancy": {
            "peak": peak_occupancy,
            "mean": round(occupancy_sum / frames_done, 2) if frames_done else 0.0,
            # per-frame timeline (downsampled to <=400 pts) for the dashboard chart
            "timeline": _downsample_timeline(occupancy_series, fps),
        },
        "lines": {
            lc.name: {
                "in": lz.in_count,
                "out": lz.out_count,
                "net_inside": lz.in_count - lz.out_count,
                "in_label": lc.in_label,
                "out_label": lc.out_label,
                "max_angle_deg": lc.max_angle_deg,
                # Crossings seen but rejected as too shallow to be a real
                # through-the-line movement (people walking *past* a doorway).
                "rejected_oblique": lz.rejected_oblique,
                # Re-triggers by an identity already counted (box hovering on
                # the line) suppressed by the cooldown.
                "rejected_repeat": lz.rejected_repeat,
            }
            for lc, (_name, lz) in zip(cfg.lines, lines, strict=True)
        },
        "doors": {
            d.name: {
                "in": c.in_count,
                "out": c.out_count,
                "occupancy_end": c.occupancy,
                "peak_occupancy": c.peak_occupancy,
                "rejected_reversals": c.rejected_reversals,
                "settle_frames": d.settle_frames,
                "cooldown_frames": d.cooldown_frames,
                "in_label": d.in_label,
                "out_label": d.out_label,
            }
            for d, c in doors
        },
        "door_events": door_events,
        "queues": {
            q.name: {
                "peak_length": qm.peak_length,
                "visits": len(qm.all_visits_s),
                "median_wait_s": (
                    round(float(np.median(qm.all_visits_s)), 2) if qm.all_visits_s else None
                ),
                "max_speed_heights_per_s": q.max_speed,
                "min_visit_s": q.min_visit_s,
            }
            for q, qm in queues
        },
        "min_dwell_s": cfg.analytics.min_dwell_s,
        "privacy": {
            "anonymization": cfg.privacy.method if cfg.privacy.enabled else "off",
            # We persist only boxes, IDs, timestamps and aggregate metrics. Appearance
            # embeddings live in memory (re-entry gallery) and are never written; no
            # raw person crops are stored at any point.
            "persisted_data": "bboxes + ids + timestamps + metrics only",
            "embeddings_persisted": False,
            "raw_crops_persisted": False,
        },
        "zones": dwell.zone_summary(),
        "dwell_per_identity": [
            {"zone": r.zone, "global_id": r.global_id, "dwell_s": round(r.seconds(fps), 2)}
            for r in sorted(dwell.records(), key=lambda r: r.seconds(fps), reverse=True)
        ],
        "exports": {
            "video": str(video_path) if write_video else None,
            "events_csv": str(events_path),
            "activity_csv": str(activity_path),
            "heatmap": str(heatmap_path) if write_heatmap else None,
        },
    }
    summary_path = out_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Analytics written to {out_dir}")
    logger.info(
        f"visitors={summary['unique_visitors']} peak_occupancy={peak_occupancy} "
        f"lines={summary['lines']}"
    )
    return summary
