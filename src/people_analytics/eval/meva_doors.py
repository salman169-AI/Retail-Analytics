"""Door counts against MEVA's enter/exit labels.

MEVA labels `person_enters_scene_through_structure` and
`person_exits_scene_through_structure`; on the entrance camera (G420) the only
structure is the pair of double doors, and every label's path runs through them
(checked: outputs/meva/phase4_label_paths.jpg). So each enter label is one true
IN and each exit label one true OUT.

Two views of the result:

- **counts**: our IN/OUT totals against the label totals (what a footfall report
  shows);
- **matched events**: each of our crossings is paired with at most one label of
  the same direction whose time span (+/- `tolerance_s`) contains it. Unpaired
  labels are misses, unpaired crossings are false counts. This catches errors that
  cancel out in the totals.
"""

from __future__ import annotations

import csv
from pathlib import Path

from people_analytics.data.meva import ENTER, EXIT, load_activities


def evaluate_doors(
    activity_csv: str | Path,
    annotations: str | Path,
    clip: str,
    door: str,
    fps: float,
    tolerance_s: float = 2.0,
) -> dict:
    labels = load_activities(annotations, clip, {ENTER, EXIT})
    ours: dict[str, list[int]] = {"in": [], "out": []}
    with open(activity_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["place"] == door and r["event"] in ("door_in", "door_out"):
                ours[r["event"][5:]].append(int(r["frame"]) - 1)  # export frames are 1-based

    tol = int(round(tolerance_s * fps))
    result: dict = {"clip": clip, "tolerance_s": tolerance_s}
    for direction, name in (("in", ENTER), ("out", EXIT)):
        spans = [(a["start"] - tol, a["end"] + tol) for a in labels if a["name"] == name]
        events = sorted(ours[direction])
        used: set[int] = set()
        tp = 0
        # Labels in time order, each takes the closest free crossing inside its span.
        for lo, hi in sorted(spans):
            free = [i for i, f in enumerate(events) if i not in used and lo <= f <= hi]
            if free:
                mid = (lo + hi) / 2
                used.add(min(free, key=lambda i: abs(events[i] - mid)))
                tp += 1
        n_true, n_ours = len(spans), len(events)
        result[direction] = {
            "labels": n_true,
            "counted": n_ours,
            "count_error": n_ours - n_true,
            "matched": tp,
            "missed": n_true - tp,
            "false_counts": n_ours - tp,
            "recall": round(tp / n_true, 3) if n_true else None,
            "precision": round(tp / n_ours, 3) if n_ours else None,
        }
    return result
