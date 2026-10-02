"""Dwell-time error on hand-timed visits.

MEVA has no dwell ground truth, so visits are timed by hand:

1. `sample` (done once, seeded) picks visits from a run's event log, spread over
   the scene's zones and queue: `metrics/meva_dwell_sample.csv`.
2. `review_sheets` draws, for each visit, frames around the predicted start and
   end (zone outlined, the tracked person in red, everyone else grey, clip time on
   each frame) so a person can read off when the visitor really arrived and left.
3. The true times go in `metrics/meva_dwell_truth.csv`; `dwell_error` compares.

The truth is the visitor's actual arrival/departure even when the tracker broke
the visit up, so identity switches show up as dwell error rather than vanishing.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from people_analytics.config import Config


def _boxes(events_csv: str | Path) -> dict[int, list[tuple[int, list[float]]]]:
    by_frame: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    with open(events_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_frame[int(r["frame"])].append(
                (int(r["global_id"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")])
            )
    return by_frame


def _place_polygon(cfg: Config, place: str) -> np.ndarray:
    for z in list(cfg.zones) + list(cfg.queues):
        if z.name == place:
            return np.array(z.polygon, np.int32)
    raise KeyError(place)


def _sheet(cap, frames: list[int], fps: float, poly: np.ndarray, gid: int,
           boxes: dict, title: str) -> np.ndarray:
    tiles = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f - 1))  # export frames are 1-based
        ok, im = cap.read()
        if not ok:
            im = np.zeros((1080, 1920, 3), np.uint8)
        cv2.polylines(im, [poly], True, (0, 200, 255), 3, cv2.LINE_AA)
        for tid, (x1, y1, x2, y2) in boxes.get(f, []):
            hit = tid == gid
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 0, 255) if hit else (160, 160, 160), 5 if hit else 1)
            if hit:
                cv2.circle(im, (int((x1 + x2) / 2), int(y2)), 9, (0, 0, 255), -1)
        cv2.rectangle(im, (0, 0), (420, 70), (0, 0, 0), -1)
        cv2.putText(im, f"t={f / fps:7.1f}s  f={f}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 255, 255), 3, cv2.LINE_AA)
        tiles.append(cv2.resize(im, (640, 360)))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    bar = np.zeros((44, grid.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA)
    return np.vstack([bar, grid])


def review_sheets(cfg: Config, sample_csv: str | Path, events_csv: str | Path, fps: float,
                  out_dir: str | Path, around_s: float = 4.0, step_s: float = 1.0) -> None:
    """Entry and exit sheets for every sampled visit."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    boxes = _boxes(events_csv)
    cap = cv2.VideoCapture(cfg.io.source)
    offsets = np.arange(-around_s, around_s + 1e-6, step_s)
    with open(sample_csv, newline="", encoding="utf-8") as fh:
        for v in csv.DictReader(fh):
            gid, poly = int(v["id"]), _place_polygon(cfg, v["place"])
            for edge in ("start", "end"):
                centre = int(v[edge])
                frames = [int(centre + o * fps) for o in offsets]
                title = (f"visit {v['visit']}  {v['place']}  id {gid}  predicted {edge} "
                         f"t={centre / fps:.1f}s")
                cv2.imwrite(str(out / f"visit{int(v['visit']):02d}_{edge}.jpg"),
                            _sheet(cap, frames, fps, poly, gid, boxes, title),
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
    cap.release()


def dwell_error(truth_csv: str | Path, fps: float) -> dict:
    """Per-visit and mean absolute dwell error (seconds)."""
    with open(truth_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    errs, out = [], []
    for r in rows:
        pred = (int(r["end"]) - int(r["start"]) + 1) / fps
        true = float(r["true_end_s"]) - float(r["true_start_s"])
        errs.append(pred - true)
        out.append({"visit": int(r["visit"]), "place": r["place"], "predicted_s": round(pred, 2),
                    "true_s": round(true, 2), "error_s": round(pred - true, 2)})
    e = np.array(errs)
    return {
        "visits": len(rows),
        "mean_abs_error_s": round(float(np.mean(np.abs(e))), 2),
        "median_abs_error_s": round(float(np.median(np.abs(e))), 2),
        "mean_signed_error_s": round(float(np.mean(e)), 2),
        "per_visit": out,
    }
