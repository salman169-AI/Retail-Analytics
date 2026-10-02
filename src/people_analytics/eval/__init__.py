"""Evaluation layer (TrackEval HOTA/IDF1/IDSW, dwell accuracy, before/after assets)."""

from people_analytics.eval.evaluate import dwell_accuracy, run_evaluation
from people_analytics.eval.trackeval_runner import run_trackeval

__all__ = ["dwell_accuracy", "run_evaluation", "run_trackeval"]
