"""Export the detector and Re-ID model to edge-friendly formats.

- Detector (YOLO26): ONNX always; TensorRT ``.engine`` when the ``tensorrt``
  package is installed (Ultralytics builds it on the local GPU).
- Re-ID (LightMBN/OSNet): ONNX via BoxMOT's exporter.

ONNX is the portable baseline (CPU/GPU/edge accelerators via ONNX Runtime);
TensorRT is the GPU-optimized path for NVIDIA edge devices (Jetson etc.).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from loguru import logger

from people_analytics.detection.detector import ensure_weights


def _tensorrt_available() -> bool:
    return importlib.util.find_spec("tensorrt") is not None


def export_detector(
    model: str = "yolo26n.pt",
    weights_dir: str = "weights",
    imgsz: int = 640,
    tensorrt: bool = False,
    half: bool = False,
) -> dict[str, str]:
    """Export the detector to ONNX (and TensorRT if requested + available)."""
    from ultralytics import YOLO

    weights = ensure_weights(model, weights_dir)
    yolo = YOLO(weights)
    out: dict[str, str] = {}

    logger.info(f"Exporting {model} -> ONNX (imgsz={imgsz})")
    out["onnx"] = str(yolo.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True))

    if tensorrt:
        if not _tensorrt_available():
            logger.warning(
                "TensorRT requested but the `tensorrt` package is not installed; "
                "skipping .engine export. Install tensorrt on the target GPU/Jetson, "
                "then re-run with --tensorrt."
            )
        else:
            logger.info(f"Exporting {model} -> TensorRT engine (half={half})")
            out["engine"] = str(yolo.export(format="engine", imgsz=imgsz, half=half))
    return out


def export_reid_onnx(
    reid_name: str = "lmbn",
    out_dir: str = "weights",
    half: bool = False,
) -> str:
    """Export the appearance Re-ID model to ONNX via BoxMOT's exporter."""
    import torch
    from boxmot.reid import ReID
    from boxmot.reid.exporters.onnx_exporter import ensure_onnx_export

    from people_analytics.tracking.reid import REID_WEIGHTS

    weight = REID_WEIGHTS[reid_name]
    backend = ReID(path=weight, device="cpu", half=half).model
    net = backend.model  # underlying nn.Module
    shape = tuple(int(x) for x in backend.input_shape)  # typically (H, W)
    if len(shape) == 2:  # (H, W) -> (N, C, H, W)
        shape = (1, 3, *shape)
    elif len(shape) == 3:  # (C, H, W) -> (N, C, H, W)
        shape = (1, *shape)
    im = torch.zeros(*shape)

    out_path = Path(out_dir) / f"{Path(weight).stem}.onnx"
    logger.info(f"Exporting Re-ID {reid_name} ({weight}) -> ONNX, input {shape}")
    path = ensure_onnx_export(net, im, out_path, dynamic=True, half=half, simplify=True)
    return str(path)
