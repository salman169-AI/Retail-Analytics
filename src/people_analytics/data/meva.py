"""MEVA clip discovery: rank clips from the KPF annotations, fetch, contact sheets.

MEVA (mevadata.org, CC BY 4.0) is ~330 h of 5-minute clips from 29 cameras at one
training site. Picking three good clips by eye is hopeless, so we rank them from
the Kitware annotations in `meva-data-repo` (cloned under `data/meva/`):

  - ``*.types.yml``      object id -> type; we count ids typed ``person``.
  - ``*.activities.yml`` one line per activity instance (enter/exit scene, ...).
  - ``*.geom.yml``       one line per box per frame; gives people visible at once.

Read the person numbers carefully. MEVA annotates people *taking part in an
activity*, for the span of that activity, and the same walker can get a new id
per activity. So "annotated person tracks" over-counts distinct people (median
track ~2-3 s), and "peak at once" under-counts how many people are on screen.
Both are still good for ranking clips against each other, which is all we use
them for here.

Video comes from the public bucket ``s3://mevadata-public-01`` over plain HTTPS
(anonymous, same as ``aws s3 cp --no-sign-request``), so no AWS tooling is needed.
"""

from __future__ import annotations

import csv
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

S3_HOST = "https://mevadata-public-01.s3.amazonaws.com/"
S3_PREFIX = "drops-123-r13/"
ANNOTATION_SETS = ("kitware", "kitware-meva-training")
FRAMES_PER_CLIP = 9000  # 5 min @ 30 fps; used to turn summed boxes into a mean

ENTER = "person_enters_scene_through_structure"
EXIT = "person_exits_scene_through_structure"

# KPF is YAML, but full YAML parsing of the multi-GB geom files is far too slow.
# Every record is a single line with a fixed key order, so regexes are safe here.
_PERSON_RE = re.compile(r"'person': 1\.0\}, 'id1': (\d+)")
_ACT_RE = re.compile(r"'act2': \{'(\w+)'")
_GEOM_RE = re.compile(r"'id1': (\d+), .*?'ts0': (\d+)")


@dataclass
class ClipStats:
    clip: str
    camera: str
    location: str
    interior: str
    sensor: str  # EO (colour) or IR (thermal)
    resolution: str
    person_tracks: int
    enters: int
    exits: int
    door_opens: int
    purchases: int
    sits: int
    s3_key: str | None = None
    size_mb: int = 0
    peak_at_once: int | None = None
    mean_at_once: float | None = None


def _annotation_root(repo: Path) -> Path:
    return Path(repo) / "annotation" / "DIVA-phase-2" / "MEVA"


def load_cameras(cameras_csv: str | Path) -> dict[str, dict[str, str]]:
    """meva-kf1-cameras.csv (from the KF1 metadata zip) keyed by 'G###'."""
    with open(cameras_csv, newline="", encoding="utf-8") as fh:
        return {f"G{row['Camera ID']}": row for row in csv.DictReader(fh)}


def list_s3_clips(prefix: str = S3_PREFIX) -> dict[str, tuple[str, int]]:
    """Clip name -> (S3 key, bytes) for every ground-camera .avi in the bucket."""
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    out: dict[str, tuple[str, int]] = {}
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(S3_HOST + "?" + urllib.parse.urlencode(query)) as resp:
            root = ET.fromstring(resp.read())
        for item in root.findall("s:Contents", ns):
            key = item.find("s:Key", ns).text
            name = Path(key).name.removesuffix(".r13.avi")
            out[name] = (key, int(item.find("s:Size", ns).text))
        if root.find("s:IsTruncated", ns).text != "true":
            return out
        token = root.find("s:NextContinuationToken", ns).text


def scan_annotations(
    repo: str | Path,
    cameras: dict[str, dict[str, str]],
    s3: dict[str, tuple[str, int]] | None = None,
) -> list[ClipStats]:
    """Per-clip person/activity counts for every annotated clip (types + activities)."""
    stats: dict[str, ClipStats] = {}
    for sub in ANNOTATION_SETS:
        for types in (_annotation_root(Path(repo)) / sub).rglob("*.types.yml"):
            clip = types.name.removesuffix(".types.yml")
            acts_path = types.with_name(f"{clip}.activities.yml")
            acts = (
                Counter(_ACT_RE.findall(acts_path.read_text(encoding="utf-8")))
                if acts_path.exists() else Counter()
            )
            location, camera = clip.split(".")[-2:]
            cam = cameras.get(camera, {})
            key, size = (s3 or {}).get(clip, (None, 0))
            stats[clip] = ClipStats(
                clip=clip,
                camera=camera,
                location=location,
                interior=cam.get("interior / exterior", "?"),
                sensor=cam.get("EO/IR", "?"),
                resolution=cam.get("resolution", "?"),
                person_tracks=len(set(_PERSON_RE.findall(types.read_text(encoding="utf-8")))),
                enters=acts[ENTER],
                exits=acts[EXIT],
                door_opens=acts["person_opens_facility_door"],
                purchases=acts["person_purchases"],
                sits=acts["person_sits_down"],
                s3_key=key,
                size_mb=round(size / 1e6),
            )
    return sorted(stats.values(), key=lambda s: s.clip)


def add_occupancy(repo: str | Path, stat: ClipStats) -> ClipStats:
    """Fill peak/mean annotated people per frame from the clip's geom file (slow-ish)."""
    geom = next(_annotation_root(Path(repo)).rglob(f"{stat.clip}.geom.yml"))
    types = geom.with_name(f"{stat.clip}.types.yml").read_text(encoding="utf-8")
    persons = set(_PERSON_RE.findall(types))
    per_frame: Counter[int] = Counter()
    with geom.open(encoding="utf-8") as fh:
        for line in fh:
            m = _GEOM_RE.search(line)
            if m and m.group(1) in persons:
                per_frame[int(m.group(2))] += 1
    stat.peak_at_once = max(per_frame.values(), default=0)
    stat.mean_at_once = round(sum(per_frame.values()) / FRAMES_PER_CLIP, 1)
    return stat


_GEOM_BOX_RE = re.compile(r"'g0': '(\d+) (\d+) (\d+) (\d+)', .*?'id1': (\d+), .*?'ts0': (\d+)")


def load_person_boxes(repo: str | Path, clip: str) -> dict[int, np.ndarray]:
    """MEVA annotated person boxes: frame (0-based, = ts0) -> rows [id, x1, y1, x2, y2].

    Only people inside an annotated activity are boxed (see module docstring), so
    this is a *partial* ground truth: fine for recall and for identity switches on
    the people it covers, meaningless for precision.
    """
    geom = next(_annotation_root(Path(repo)).rglob(f"{clip}.geom.yml"))
    types = geom.with_name(f"{clip}.types.yml").read_text(encoding="utf-8")
    persons = {int(i) for i in _PERSON_RE.findall(types)}
    rows: dict[int, list[list[int]]] = {}
    with geom.open(encoding="utf-8") as fh:
        for line in fh:
            m = _GEOM_BOX_RE.search(line)
            if m and int(m.group(5)) in persons:
                x1, y1, x2, y2, tid, frame = (int(g) for g in m.groups())
                rows.setdefault(frame, []).append([tid, x1, y1, x2, y2])
    return {f: np.asarray(r, dtype=float) for f, r in rows.items()}


def load_activities(repo: str | Path, clip: str, names: set[str]) -> list[dict]:
    """Activity instances of the given types: [{name, start, end, actors}] (0-based frames)."""
    acts = next(_annotation_root(Path(repo)).rglob(f"{clip}.activities.yml"))
    span_re = re.compile(r"'tsr0': \[(\d+), (\d+)\]")
    actor_re = re.compile(r"'id1': (\d+)")
    out = []
    for line in acts.read_text(encoding="utf-8").splitlines():
        m = _ACT_RE.search(line)
        if not m or m.group(1) not in names:
            continue
        spans = [(int(a), int(b)) for a, b in span_re.findall(line)]
        out.append({
            "name": m.group(1),
            "start": min(a for a, _ in spans),
            "end": max(b for _, b in spans),
            "actors": sorted({int(a) for a in actor_re.findall(line)}),
        })
    return sorted(out, key=lambda a: a["start"])


def write_stats_csv(stats: list[ClipStats], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(stats[0])))
        writer.writeheader()
        writer.writerows(asdict(s) for s in stats)
    return path


def download_clip(s3_key: str, dest_dir: str | Path) -> Path:
    """Fetch one clip from the public bucket; skips files already complete on disk."""
    dest = Path(dest_dir) / Path(s3_key).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = S3_HOST + urllib.parse.quote(s3_key)
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
        expected = int(resp.headers["Content-Length"])
    if dest.exists() and dest.stat().st_size == expected:
        logger.info(f"{dest.name}: already present")
        return dest
    logger.info(f"Downloading {dest.name} ({expected / 1e6:.0f} MB)")
    urllib.request.urlretrieve(url, dest)
    if dest.stat().st_size != expected:
        raise RuntimeError(f"{dest.name}: size mismatch")
    return dest


def sample_frames(video: str | Path, n: int = 4) -> list[np.ndarray]:
    """`n` frames spread evenly across the clip (skipping the very first/last)."""
    cap = cv2.VideoCapture(str(video))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        return frames
    finally:
        cap.release()


def contact_sheet(
    rows: list[tuple[str, Path]], out_path: str | Path, n: int = 4, thumb_w: int = 480
) -> Path:
    """One row per clip: a caption strip plus `n` frames across it, saved as a JPEG."""
    thumb_h = thumb_w * 9 // 16
    strips = []
    for caption, video in rows:
        frames = sample_frames(video, n)
        tiles = [cv2.resize(f, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA) for f in frames]
        tiles += [np.zeros((thumb_h, thumb_w, 3), np.uint8)] * (n - len(tiles))
        for t, tile in enumerate(tiles):
            cv2.putText(tile, f"{(t + 0.5) / n:.0%}", (8, thumb_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        bar = np.full((34, thumb_w * n, 3), 30, np.uint8)
        cv2.putText(bar, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1, cv2.LINE_AA)
        strips.append(np.vstack([bar, np.hstack(tiles)]))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out
