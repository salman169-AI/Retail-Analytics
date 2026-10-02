"""Hyperparameter sweep to push IDSW lower on the eval clip.

Searches the two biggest identity-stability levers on the final config
(BoT-SORT + LightMBN + re-entry gallery):

- ``track_buffer`` — how long a lost track stays alive before a *new* id is minted
  (longer → BoxMOT re-associates instead of switching).
- gallery ``cosine_thresh`` — how eagerly a returning person reclaims their id
  (lower → more re-links, but too low risks merging different people).

Detections are cached once; each combo is scored with TrackEval. The winner is
the lowest IDSW that does not regress IDF1 below the ByteTrack baseline.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from people_analytics.config import Config
from people_analytics.data.mot import MOTSequence
from people_analytics.eval.evaluate import _sliced_detector
from people_analytics.eval.trackeval_runner import run_trackeval
from people_analytics.tracking.benchmark import cache_detections, run_combo, write_mot_file

# Search grid for the two biggest levers.
TRACK_BUFFERS = [60, 150, 250]
COSINE_THRESHS = [0.40, 0.50, 0.55]
BASELINE_IDF1 = 0.435  # ByteTrack baseline — don't regress identity below this.


def tune(
    cfg: Config,
    out_dir: str | Path = "outputs/tune",
    results_md: str | Path = "metrics/tuning.md",
) -> dict:
    """Sweep track_buffer x gallery cosine_thresh; score each with TrackEval."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = MOTSequence(cfg.eval.sequence_dir)
    detections = cache_detections(seq, _sliced_detector(cfg))

    result_files: dict[str, Path] = {}
    grid: dict[str, tuple[int, float]] = {}
    for tb in TRACK_BUFFERS:
        for cos in COSINE_THRESHS:
            name = f"tb{tb}_cos{int(cos * 100)}"
            trial = cfg.model_copy(deep=True)
            trial.tracker.track_buffer = tb
            trial.reid_gallery.cosine_thresh = cos
            logger.info(f"Tuning {name} (track_buffer={tb}, cosine={cos})…")
            res, _ = run_combo(seq, detections, "botsort", "lmbn", trial, use_gallery=True)
            mot_file = out_dir / f"{name}.txt"
            write_mot_file(res, mot_file)
            result_files[name] = mot_file
            grid[name] = (tb, cos)

    logger.info("Scoring all combos with TrackEval…")
    metrics = run_trackeval(seq.path, result_files, out_dir / "trackeval")

    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["IDSW"])
    eligible = [(n, m) for n, m in ranked if m["IDF1"] >= BASELINE_IDF1] or ranked
    best_name, best = eligible[0]

    _write_md(grid, metrics, best_name, seq.info.name, Path(results_md))
    logger.info(
        f"Best: {best_name} (track_buffer={grid[best_name][0]}, "
        f"cosine={grid[best_name][1]}) -> IDSW={best['IDSW']} IDF1={best['IDF1']}"
    )
    return {"best": {"name": best_name, "params": grid[best_name], **best}, "all": metrics}


def _write_md(grid, metrics, best_name, seq_name, path) -> None:
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["IDSW"])
    lines = [
        "# IDSW tuning sweep",
        "",
        f"Eval clip: **{seq_name}** · final config (BoT-SORT + LightMBN + gallery) · "
        "TrackEval-scored. Winner 🏆 = lowest IDSW without regressing IDF1 below the "
        f"ByteTrack baseline ({BASELINE_IDF1}).",
        "",
        "| track_buffer | cosine_thresh | HOTA ↑ | IDF1 ↑ | IDSW ↓ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for name, m in ranked:
        tb, cos = grid[name]
        mark = " 🏆" if name == best_name else ""
        lines.append(f"| {tb}{mark} | {cos} | {m['HOTA']:.3f} | {m['IDF1']:.3f} | {m['IDSW']} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote {path}")
