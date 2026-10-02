"""Dataset acquisition and loading (sample clips + MOT-format eval sequences)."""

from people_analytics.data.mot import MOTSequence, download_mot_sequence, load_mot_sequence
from people_analytics.data.samples import RETAIL_CLIPS, download_sample_clips

__all__ = [
    "RETAIL_CLIPS",
    "MOTSequence",
    "download_mot_sequence",
    "download_sample_clips",
    "load_mot_sequence",
]
