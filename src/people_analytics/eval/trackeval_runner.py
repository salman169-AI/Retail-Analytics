"""Run TrackEval (HOTA/CLEAR/Identity) on MOT-format results.

The official motchallenge.net host is unreachable here, but TrackEval computes
HOTA fully offline from local files, so we assemble the MOTChallenge folder
layout it expects in a work dir and evaluate our tracker outputs against the GT.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


def _shim_numpy() -> None:
    """TrackEval uses np.int/np.float/np.bool, removed in NumPy 2.x. Restore them."""
    for name, t in {"float": float, "int": int, "bool": bool, "object": object}.items():
        if not hasattr(np, name):
            setattr(np, name, t)


def run_trackeval(
    gt_seq_dir: str | Path,
    tracker_results: dict[str, str | Path],
    work_dir: str | Path,
    benchmark: str = "MOT20",
    split: str = "train",
) -> dict[str, dict[str, float]]:
    """Evaluate each named MOT result file against the sequence GT.

    Args:
        gt_seq_dir: sequence dir with ``gt/gt.txt`` and ``seqinfo.ini``.
        tracker_results: name -> MOT-format results file.
        work_dir: scratch dir for the TrackEval folder layout.
    Returns:
        name -> {"HOTA", "MOTA", "IDF1", "IDSW"}.
    """
    _shim_numpy()
    import trackeval
    from trackeval.metrics import CLEAR, HOTA, Identity

    gt_seq_dir = Path(gt_seq_dir)
    seq = gt_seq_dir.name
    work = Path(work_dir)
    if work.exists():
        shutil.rmtree(work)
    split_fol = f"{benchmark}-{split}"

    gt_dst = work / "gt" / split_fol / seq
    (gt_dst / "gt").mkdir(parents=True)
    shutil.copy(gt_seq_dir / "gt" / "gt.txt", gt_dst / "gt" / "gt.txt")
    shutil.copy(gt_seq_dir / "seqinfo.ini", gt_dst / "seqinfo.ini")
    seqmaps = work / "gt" / "seqmaps"
    seqmaps.mkdir(parents=True)
    seqmap_file = seqmaps / f"{split_fol}.txt"
    seqmap_file.write_text(f"name\n{seq}\n", encoding="utf-8")

    for name, results_file in tracker_results.items():
        dst = work / "trackers" / split_fol / name / "data"
        dst.mkdir(parents=True)
        shutil.copy(results_file, dst / f"{seq}.txt")

    eval_config = {
        **trackeval.Evaluator.get_default_eval_config(),
        "USE_PARALLEL": False,
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
        "TIME_PROGRESS": False,
    }
    ds_config = {
        **trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
        "GT_FOLDER": str(work / "gt"),
        "TRACKERS_FOLDER": str(work / "trackers"),
        "BENCHMARK": benchmark,
        "SPLIT_TO_EVAL": split,
        "TRACKERS_TO_EVAL": list(tracker_results.keys()),
        "SEQMAP_FILE": str(seqmap_file),
        "DO_PREPROC": True,
        "PRINT_CONFIG": False,
    }

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(ds_config)
    res, _ = evaluator.evaluate([dataset], [HOTA(), CLEAR(), Identity()])

    key = next(iter(res))
    out: dict[str, dict[str, float]] = {}
    for name in tracker_results:
        ped = res[key][name]["COMBINED_SEQ"]["pedestrian"]
        out[name] = {
            "HOTA": round(float(np.mean(ped["HOTA"]["HOTA"])), 4),
            "MOTA": round(float(ped["CLEAR"]["MOTA"]), 4),
            "IDF1": round(float(ped["Identity"]["IDF1"]), 4),
            "IDSW": int(ped["CLEAR"]["IDSW"]),
        }
    return out
