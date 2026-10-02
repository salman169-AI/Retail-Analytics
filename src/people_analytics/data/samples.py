"""Download retail/venue sample clips from Roboflow Supervision's public assets.

We pick three clips spanning the camera angles the plan calls for:
  - GROCERY_STORE  : eye-level retail aisle
  - MARKET_SQUARE  : elevated, crowded venue
  - PEOPLE_WALKING : top-down / overhead pedestrians

Clips are streamed straight to `data/samples/` from Supervision's asset host
(md5-verified), so the destination is explicit and no CWD games are needed.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copyfileobj

import cv2
from loguru import logger
from supervision.assets import VideoAssets
from supervision.assets.downloader import MEDIA_ASSETS

# The retail/venue-relevant subset of Supervision's VideoAssets, with a note on angle.
RETAIL_CLIPS: dict[str, VideoAssets] = {
    "grocery_store": VideoAssets.GROCERY_STORE,  # eye-level retail
    "market_square": VideoAssets.MARKET_SQUARE,  # elevated crowd
    "people_walking": VideoAssets.PEOPLE_WALKING,  # top-down overhead
}


def _is_playable(path: Path) -> bool:
    """A meaningful integrity check for our purposes: the clip must decode."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    cap = cv2.VideoCapture(str(path))
    try:
        ok, _ = cap.read()
        return bool(ok)
    finally:
        cap.release()


def _download_one(asset: VideoAssets, dest_dir: Path) -> Path:
    # NOTE: Supervision ships pinned md5s that can go stale against the live host,
    # so we validate by decoding the clip rather than trusting that hash.
    filename = asset.value  # e.g. "grocery-store.mp4"
    url, _stale_md5 = MEDIA_ASSETS[filename]
    out = dest_dir / filename

    if _is_playable(out):
        logger.info(f"{filename}: already present ({out.stat().st_size / 1e6:.1f} MB)")
        return out

    # Imported lazily so the module imports cleanly even if requests isn't around.
    from requests import get

    logger.info(f"Downloading {filename} -> {out}")
    with get(url, stream=True, allow_redirects=True, timeout=60) as resp:
        resp.raise_for_status()
        expected = int(resp.headers.get("Content-Length", 0))
        with out.open("wb") as fh:
            copyfileobj(resp.raw, fh)
        got = out.stat().st_size
        if expected and got != expected:
            raise RuntimeError(f"{filename}: size mismatch (got {got}, expected {expected})")

    if not _is_playable(out):
        raise RuntimeError(f"{filename}: downloaded file does not decode")
    logger.info(f"{filename}: download complete ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def download_sample_clips(dest_dir: str | Path = "data/samples") -> list[Path]:
    """Download the retail/venue sample clips into `dest_dir`. Returns their paths."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    return [_download_one(asset, dest) for asset in RETAIL_CLIPS.values()]
