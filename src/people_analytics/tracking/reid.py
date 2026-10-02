"""Appearance (Re-ID) model builder for BoxMOT trackers.

The plan calls for CLIP-ReID as the accuracy model, but the installed BoxMOT
(19.0.0) ships no pretrained CLIP-ReID checkpoint (only OSNet / LightMBN and an
untrained ViT backbone). So the ready-to-use appearance axis is:

    osnet_ain : OSNet-AIN x1_0 (MSMT17) — strongest drop-in, the "accuracy" model
    lmbn      : LightMBN (Market1501)   — the plan's edge model
    osnet     : OSNet x0_25 (MSMT17)    — lightweight baseline appearance

Each name maps to a weight file BoxMOT auto-downloads on first use. A tracker
consumes ``build_reid(...)`` — i.e. ``ReID(...).model`` — whose ``get_features``
method returns per-detection embeddings.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# Registry name -> BoxMOT weight file (auto-downloaded from the boxmot release).
REID_WEIGHTS: dict[str, str] = {
    "osnet_ain": "osnet_ain_x1_0_msmt17.pt",  # accuracy
    "lmbn": "lmbn_n_market.pt",  # LightMBN, edge
    "osnet": "osnet_x0_25_msmt17.pt",  # lightweight baseline
}


def build_reid(name: str, device: str = "cuda:0", half: bool = False) -> Any:
    """Build a Re-ID backend model for a registry name. Returns ``ReID(...).model``."""
    if name not in REID_WEIGHTS:
        raise KeyError(f"Unknown reid model '{name}'. Options: {sorted(REID_WEIGHTS)}")
    from boxmot.reid import ReID  # local import: heavy, needs the `track` group

    weight = REID_WEIGHTS[name]
    logger.info(f"Building Re-ID '{name}' ({weight}) on {device}")
    return ReID(path=weight, device=device, half=half).model
