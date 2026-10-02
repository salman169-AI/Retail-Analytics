"""Linear gap-filling for finished tracks (offline post-processing).

A track fragments whenever the detector misses a person for a few frames — an
occluder passes, or a small far detection drops out. The tracker ends the track
and, when the person reappears, may open a new id. That single event is both a
**fragmentation** and, to the evaluator, an **identity switch**: the ground-truth
person that was matched to tracker id 5 is now matched to id 37.

`interpolate_tracks` runs over the completed results and, for each id, linearly
fills short gaps *within that id* — so a track that blinks out for a handful of
frames stays continuous. This is the standard MOTChallenge / StrongSORT
post-process. It raises IDF1 and recall and lowers fragmentation, and it lowers
identity switches because the evaluator keeps matching the same tracker id
across the gap instead of re-assigning.

Two deliberate limits:

- It only bridges gaps up to `max_gap` frames. A longer gap is a genuine
  disappearance/re-entry, not a blink; linearly interpolating a straight line
  across it would invent a wrong path (the person may have turned). Those are
  the re-entry gallery's job, not interpolation's.
- It only fills *within* an existing id. It never links id 5 to id 37 — that is
  re-identification, a separate step.

This is an **offline** operation: it needs the whole track, so it belongs to
batch evaluation / recorded-video analysis, not the online frame-by-frame path.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def interpolate_tracks(
    results: dict[int, np.ndarray], max_gap: int = 20
) -> dict[int, np.ndarray]:
    """Fill short within-id gaps by linear interpolation.

    Args:
        results: frame index -> rows of ``[id, x, y, w, h, conf]`` (MOT xywh,
            top-left origin), as produced by ``run_combo``.
        max_gap: longest gap (in frames) to bridge. Gaps longer than this are
            left as-is — they are treated as real disappearances.

    Returns:
        A new results dict with the original rows plus interpolated rows. The
        input is not mutated.
    """
    if max_gap < 2:
        return {f: rows.copy() for f, rows in results.items()}

    # id -> {frame: [x, y, w, h, conf]}
    timeline: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for frame, rows in results.items():
        for row in rows:
            timeline[int(row[0])][frame] = np.asarray(row[1:6], dtype=float)

    filled: dict[int, list[np.ndarray]] = defaultdict(list)
    for frame, rows in results.items():
        for row in rows:
            filled[frame].append(np.asarray(row, dtype=float))

    for tid, frames in timeline.items():
        ordered = sorted(frames)
        for f0, f1 in zip(ordered, ordered[1:], strict=False):
            gap = f1 - f0
            if not 2 <= gap <= max_gap:
                continue
            start, end = frames[f0], frames[f1]
            for k in range(1, gap):
                box = start + (end - start) * (k / gap)  # x, y, w, h, conf
                filled[f0 + k].append(np.concatenate([[float(tid)], box]))

    return {f: np.array(rows, dtype=float) for f, rows in sorted(filled.items())}


def count_filled(
    original: dict[int, np.ndarray], interpolated: dict[int, np.ndarray]
) -> int:
    """How many box rows interpolation added (for reporting)."""
    before = sum(len(r) for r in original.values())
    after = sum(len(r) for r in interpolated.values())
    return after - before
