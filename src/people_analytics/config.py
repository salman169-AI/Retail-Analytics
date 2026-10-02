"""Typed, validated configuration models loaded from YAML.

The whole pipeline is config-driven. `load_config` reads a YAML file into a
validated `Config` object so every module receives already-checked values.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SlicerConfig(BaseModel):
    enabled: bool = False
    slice_wh: tuple[int, int] = (640, 640)
    overlap_ratio: float = 0.2
    # How detections from neighbouring tiles are de-duplicated.
    #
    # "iou" is supervision's default and it cannot remove the characteristic
    # tiling artefact: a person taller than a tile is detected once in full and
    # again as a half-body from the adjacent tile. Those two boxes have *low*
    # IoU, so IoU-NMS keeps both and one person ends up with several ids.
    #
    # "ios" (intersection over smaller) scores that pair on how much of the
    # smaller box sits inside the larger one, which is exactly the containment
    # case, and removes it. Pure IoS still misses *offset* upper/lower splits
    # (IoS ~0.4–0.55); those are handled by hybrid full-frame fusion below.
    merge_metric: str = "iou"
    # Threshold for the metric above. None reuses `detector.iou`, which is the
    # historical behaviour the MOT20 benchmark numbers were produced with.
    merge_threshold: float | None = None
    # When True (default with slicing), also run a whole-frame pass and fuse it
    # with the tiled boxes. Full-frame boxes are complete for near/large people;
    # tiled boxes only contribute detections that are not vertical partials of a
    # full-frame box (far/arch people). Disable only for pure-slicer ablations.
    hybrid: bool = True


class IOConfig(BaseModel):
    source: str
    output_dir: str = "outputs"
    max_frames: int | None = None
    # First frame to process (0-based). Event frames stay absolute, so they line
    # up with the source's own annotations however the run is cropped.
    start_frame: int = 0


class EvalConfig(BaseModel):
    sequence_dir: str = "data/eval/MOT20-01"


class DetectorConfig(BaseModel):
    model: str = "yolo26m.pt"
    weights_dir: str = "weights"
    device: str = "cuda:0"
    confidence: float = 0.3
    iou: float = 0.7
    classes: list[int] = Field(default_factory=lambda: [0])
    slicer: SlicerConfig = Field(default_factory=SlicerConfig)


class TrackerConfig(BaseModel):
    name: str = "botsort_reid"
    reid_model: str = "clip"
    track_buffer: int = 60
    appearance_thresh: float = 0.25
    match_thresh: float = 0.8
    # BoT-SORT camera-motion compensation (ECC). Needed for handheld/PTZ footage;
    # pure cost on a fixed CCTV camera, where it cut throughput to ~7 FPS on MEVA.
    use_cmc: bool = True
    # BoT-SORT confidence gates (BoxMOT defaults). Detections below
    # `track_high_thresh` only join the second, motion-only association, and a
    # new track needs `new_track_thresh`. In a crowd most partly hidden people
    # score below 0.6, so they never get a track at all.
    track_high_thresh: float = 0.5
    new_track_thresh: float = 0.6


class ReidGalleryConfig(BaseModel):
    enabled: bool = True
    cosine_thresh: float = 0.55
    time_window_s: float = 120.0
    max_gallery_size: int = 500
    ema_alpha: float = 0.9  # running-embedding smoothing (higher = slower to change)


class ZoneConfig(BaseModel):
    name: str
    polygon: list[tuple[int, int]]
    # Optional per-zone override of `analytics.anchor` (see AnalyticsConfig).
    anchor: str | None = None


class LineConfig(BaseModel):
    name: str
    start: tuple[int, int]
    end: tuple[int, int]
    # Optional per-line override of `analytics.anchor`. A doorway line needs a
    # ground-plane anchor (feet) even when the zones are testing body centres.
    anchor: str | None = None
    in_label: str = "IN"
    out_label: str = "OUT"
    # How far off perpendicular a crossing may be and still count, in degrees.
    # 90 = count any crossing. Lower it for a doorway beside a busy walkway so
    # people passing *along* the line aren't counted as going through it.
    max_angle_deg: float = 90.0
    # Ignore the same identity for this many frames after counting it, so a box
    # hovering on the line can't register over and over. 0 disables it.
    cooldown_frames: int = 0


class DoorConfig(BaseModel):
    """A doorway drawn as two lines; see `analytics.door` for the counting rule."""

    name: str
    # Threshold line, and a line a step or two into the room. The inner line's
    # side of the outer one is "inside", so IN = outside -> inside.
    outer: tuple[tuple[int, int], tuple[int, int]]
    inner: tuple[tuple[int, int], tuple[int, int]]
    anchor: str | None = None
    in_label: str = "IN"
    out_label: str = "OUT"
    # Frames the feet must stay on a side before it counts as settled (1 = off).
    settle_frames: int = 1
    # Ignore a reversal by an identity this soon after counting it (0 = off).
    cooldown_frames: int = 0
    # Forget an identity's side after it has been unseen this long, so a
    # re-identified newcomer isn't judged against someone else's old side.
    forget_after_frames: int = 900


class QueueConfig(BaseModel):
    """Floor area in front of a counter; see `analytics.queue`."""

    name: str
    polygon: list[tuple[int, int]]
    anchor: str | None = None
    # Faster than this (body heights per second) = walking past, not queuing.
    max_speed: float = 0.5
    # Visits shorter than this don't feed the wait estimate.
    min_visit_s: float = 3.0
    # The wait estimate is the median over this many recent visits.
    recent: int = 10


class AnalyticsConfig(BaseModel):
    smoothing_window: int = 5
    heatmap: bool = True
    # Write the annotated MP4. Defaults to True so existing behaviour is unchanged.
    # Turn it off for headless runs that only need the CSV/JSON exports.
    #
    # Measured on a 500-frame 1080p segment: 208 s without the video against 199 s
    # with it — i.e. **no speed benefit**. Drawing and H.264 encoding are CPU work
    # and are not the bottleneck; tiled detection and Re-ID on the GPU are. The
    # real saving is disk (~41 MB per 500 frames, ~300 MB across a corpus) and not
    # having to write files nothing reads. Do not switch this off expecting the
    # run to finish sooner.
    write_video: bool = True
    # Anchor used to test a person against zones/line: center | bottom_center | top_center.
    anchor: str = "center"
    # Offline gap-filling: run a first tracking pass, linearly bridge short gaps
    # within each id, then render from the filled tracks. Needs two passes over
    # the clip, so it's opt-in. Cannot remove a continuously-tracked person; only
    # adds boxes during a track's own gaps.
    interpolate: bool = False
    interpolate_max_gap: int = 20
    # Ignore identities whose total time in a zone is under this, so people
    # cutting across a zone don't count as visitors. 0 keeps everyone.
    min_dwell_s: float = 0.0
    # Per-person "#id 12s" labels on the video. A wide crowd view with many
    # short tracks turns them into an unreadable pile; boxes alone still show.
    show_labels: bool = True


class PrivacyConfig(BaseModel):
    enabled: bool = False
    method: str = "blur"


class MevaConfig(BaseModel):
    # Where the KPF annotations live (cloned meva-data-repo).
    annotations: str = "data/meva/meva-data-repo"
    clips_dir: str = "data/meva/clips"
    # A different clip from the same camera, used for tuning so the scene's own
    # clip stays out-of-sample for evaluation.
    tune_clip: str | None = None


class Config(BaseModel):
    io: IOConfig
    meva: MevaConfig | None = None
    eval: EvalConfig = Field(default_factory=EvalConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    reid_gallery: ReidGalleryConfig = Field(default_factory=ReidGalleryConfig)
    zones: list[ZoneConfig] = Field(default_factory=list)
    lines: list[LineConfig] = Field(default_factory=list)
    doors: list[DoorConfig] = Field(default_factory=list)
    queues: list[QueueConfig] = Field(default_factory=list)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a `Config`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
