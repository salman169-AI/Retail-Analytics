"""MOT-format eval sequences: download, parse, and iterate.

We use MOT20 as the labeled eval clip (ground-truth person boxes + track IDs),
which lets us skip hand-labeling. The official host (motchallenge.net) is often
unreachable, so we pull a single sequence from a public HuggingFace mirror into
the standard MOTChallenge layout that TrackEval expects:

    data/eval/<SEQ>/
      ├─ img1/000001.jpg ...      # frames
      ├─ gt/gt.txt                # frame,id,x,y,w,h,conf,class,visibility
      ├─ det/det.txt              # public detections (unused by us)
      └─ seqinfo.ini              # name, frameRate, seqLength, imWidth/Height, imExt

gt.txt line format (MOTChallenge):
    frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
For MOT20 ground truth: `class == 1` is pedestrian and `conf` is a 0/1 "consider"
flag. `only_pedestrian` filtering below applies both.
"""

from __future__ import annotations

import configparser
import urllib.parse
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

HF_REPO = "Lekim89/MOT20"
HF_SPLIT = "ablation"  # sequences with public ground truth in this mirror
DEFAULT_SEQUENCE = "MOT20-01"  # 214 frames @ 25fps (~8.5s), 1920x1080, crowded


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _hf_tree(repo: str) -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https host
        import json

        return json.load(resp)


def _hf_resolve_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"


def download_mot_sequence(
    sequence: str = DEFAULT_SEQUENCE,
    dest_root: str | Path = "data/eval",
    repo: str = HF_REPO,
    split: str = HF_SPLIT,
    workers: int = 8,
) -> Path:
    """Download one MOT sequence into `dest_root/<sequence>/` (MOTChallenge layout).

    Skips files already present with the expected size. Returns the sequence dir.
    """
    prefix = f"{split}/{sequence}/"
    tree = _hf_tree(repo)
    files = [(d["path"], d.get("size", 0)) for d in tree if d["type"] == "file"]
    wanted = [(p, s) for p, s in files if p.startswith(prefix)]
    if not wanted:
        raise FileNotFoundError(
            f"Sequence '{sequence}' not found under '{split}/' in {repo}. "
            f"Available: {sorted({p.split('/')[1] for p, _ in files if p.startswith(split + '/')})}"
        )

    seq_dir = Path(dest_root) / sequence
    logger.info(f"Downloading {sequence}: {len(wanted)} files -> {seq_dir}")

    def fetch(item: tuple[str, int]) -> None:
        remote_path, size = item
        rel = remote_path[len(prefix) :]  # strip "<split>/<seq>/"
        out = seq_dir / rel
        if out.exists() and (size == 0 or out.stat().st_size == size):
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        url = _hf_resolve_url(repo, remote_path)
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed https host
            out.write_bytes(resp.read())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, wanted))

    logger.info(f"{sequence}: download complete")
    return seq_dir


# --------------------------------------------------------------------------- #
# Load / parse
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MOTFrameInfo:
    name: str
    frame_rate: float
    seq_length: int
    im_width: int
    im_height: int
    im_ext: str
    im_dir: str


class MOTSequence:
    """A loaded MOT-format sequence: seqinfo + ground truth + frame access."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.info = self._read_seqinfo(self.path / "seqinfo.ini")
        gt_path = self.path / "gt" / "gt.txt"
        # frame index (1-based) -> Nx6 array of [track_id, x, y, w, h, visibility]
        self.gt: dict[int, np.ndarray] = self._read_gt(gt_path)

    @staticmethod
    def _read_seqinfo(path: Path) -> MOTFrameInfo:
        if not path.exists():
            raise FileNotFoundError(f"Missing seqinfo.ini: {path}")
        cp = configparser.ConfigParser()
        cp.read(path)
        s = cp["Sequence"]
        return MOTFrameInfo(
            name=s.get("name", path.parent.name),
            frame_rate=s.getfloat("frameRate", 25.0),
            seq_length=s.getint("seqLength", 0),
            im_width=s.getint("imWidth", 0),
            im_height=s.getint("imHeight", 0),
            im_ext=s.get("imExt", ".jpg"),
            im_dir=s.get("imDir", "img1"),
        )

    @staticmethod
    def _read_gt(path: Path, only_pedestrian: bool = True) -> dict[int, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"Missing ground truth: {path}")
        raw = np.loadtxt(path, delimiter=",", ndmin=2)
        if raw.size == 0:
            return {}
        # columns: frame,id,x,y,w,h,conf,class,visibility (+ ignore extras)
        if only_pedestrian and raw.shape[1] >= 8:
            keep = (raw[:, 6] == 1) & (raw[:, 7] == 1)  # consider-flag & pedestrian class
            raw = raw[keep]
        by_frame: dict[int, np.ndarray] = {}
        vis_col = 8 if raw.shape[1] >= 9 else None
        for frame in np.unique(raw[:, 0]).astype(int):
            rows = raw[raw[:, 0] == frame]
            vis = rows[:, vis_col] if vis_col is not None else np.ones(len(rows))
            by_frame[frame] = np.column_stack([rows[:, 1:6], vis])  # id,x,y,w,h,vis
        return by_frame

    # -- convenience --------------------------------------------------------- #
    @property
    def num_frames(self) -> int:
        return self.info.seq_length

    @property
    def track_ids(self) -> set[int]:
        ids: set[int] = set()
        for arr in self.gt.values():
            ids.update(arr[:, 0].astype(int).tolist())
        return ids

    def frame_path(self, frame_idx: int) -> Path:
        # MOT frames are 1-based, zero-padded to 6 digits.
        return self.path / self.info.im_dir / f"{frame_idx:06d}{self.info.im_ext}"

    def frame_generator(self) -> Iterator[np.ndarray]:
        """Yield BGR frames in order from the sequence's image directory."""
        for i in range(1, self.num_frames + 1):
            fp = self.frame_path(i)
            img = cv2.imread(str(fp))
            if img is None:
                raise FileNotFoundError(f"Could not read frame: {fp}")
            yield img


def load_mot_sequence(path: str | Path) -> MOTSequence:
    """Load a MOT-format sequence directory."""
    return MOTSequence(path)
