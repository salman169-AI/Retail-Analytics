"""Detector FPS across backends: PyTorch vs ONNX Runtime, CPU vs GPU.

All rows run the SAME Ultralytics predict pipeline (pre-process + inference +
NMS-free decode) so the comparison is apples-to-apples end-to-end throughput.
TensorRT (`.engine`) is included automatically if it has been exported.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

from people_analytics.edge.export import export_detector
from people_analytics.io import frame_generator


@dataclass
class FpsRow:
    backend: str
    device: str
    fps: float
    ms_per_frame: float


def _load_frames(source: str, n: int, imgsz: int = 640) -> list[np.ndarray]:
    frames = [cv2.resize(f, (imgsz, imgsz)) for f in frame_generator(source, max_frames=n)]
    if not frames:
        raise RuntimeError(f"No frames read from {source}")
    return frames


def _onnx_cuda_usable() -> bool:
    """True only if ONNX Runtime can actually load the CUDA EP (needs cuDNN runtime)."""
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers() and _cuda_ep_loads()
    except Exception:  # noqa: BLE001
        return False


def _cuda_ep_loads() -> bool:
    import onnxruntime as ort
    from onnx import TensorProto, helper

    # Minimal identity model to test whether a CUDA session truly initializes.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    try:
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        return "CUDAExecutionProvider" in sess.get_providers()
    except Exception:  # noqa: BLE001
        return False


def _bench(model_path: str, device: str, frames: list, warmup: int = 5) -> tuple[float, float]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    for f in frames[:warmup]:
        model(f, device=device, verbose=False, imgsz=frames[0].shape[0])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for f in frames:
        model(f, device=device, verbose=False, imgsz=frames[0].shape[0])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    fps = len(frames) / elapsed
    return fps, elapsed / len(frames) * 1000.0


def run_edge_benchmark(
    model: str = "yolo26n.pt",
    weights_dir: str = "weights",
    source: str = "data/samples/people-walking.mp4",
    n_frames: int = 120,
    imgsz: int = 640,
    tensorrt: bool = False,
    out_path: str = "metrics/edge_fps.md",
) -> list[FpsRow]:
    """Export to ONNX (+TRT if asked) and benchmark every available backend."""
    exports = export_detector(model, weights_dir, imgsz=imgsz, tensorrt=tensorrt)
    onnx_path = exports["onnx"]
    pt_path = str(Path(weights_dir) / model)
    frames = _load_frames(source, n_frames, imgsz)
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else "no GPU"

    onnx_gpu = cuda and _onnx_cuda_usable()
    if cuda and not onnx_gpu:
        logger.warning(
            "ONNX Runtime CUDA EP not usable (cuDNN runtime not found) — skipping "
            "ONNX-GPU; on NVIDIA edge, use the TensorRT engine instead."
        )
    plan: list[tuple[str, str, str]] = [("PyTorch", "cpu", pt_path)]
    if cuda:
        plan.append(("PyTorch", "cuda:0", pt_path))
    plan.append(("ONNX Runtime", "cpu", onnx_path))
    if onnx_gpu:
        plan.append(("ONNX Runtime", "cuda:0", onnx_path))
    if "engine" in exports:
        plan.append(("TensorRT", "cuda:0", exports["engine"]))

    rows: list[FpsRow] = []
    for backend, device, path in plan:
        logger.info(f"Benchmarking {backend} on {device} ...")
        try:
            fps, ms = _bench(path, device, frames)
            rows.append(FpsRow(backend, "GPU" if device.startswith("cuda") else "CPU",
                               round(fps, 1), round(ms, 2)))
            logger.info(f"  -> {fps:.1f} FPS ({ms:.2f} ms/frame)")
        except Exception as e:  # noqa: BLE001 - report and continue other backends
            logger.warning(f"  {backend}/{device} failed: {repr(e)[:120]}")

    _write_md(rows, model, imgsz, gpu_name, len(frames), Path(out_path))
    return rows


def _write_md(rows: list[FpsRow], model: str, imgsz: int, gpu: str, n: int, path: Path) -> None:
    base = next((r.fps for r in rows if r.backend == "PyTorch" and r.device == "CPU"), None)
    lines = [
        "# Edge deployment — detector FPS",
        "",
        f"Model: **{model}** @ {imgsz}px · GPU: **{gpu}** · {n} frames · "
        "end-to-end Ultralytics predict (pre-process + inference + decode).",
        "",
        "| Backend | Device | FPS ↑ | ms/frame | Speedup vs PyTorch-CPU |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        speed = f"{r.fps / base:.1f}×" if base else "—"
        lines.append(f"| {r.backend} | {r.device} | **{r.fps}** | {r.ms_per_frame} | {speed} |")
    has_onnx_gpu = any(r.backend == "ONNX Runtime" and r.device == "GPU" for r in rows)
    notes = []
    if not has_onnx_gpu:
        notes.append(
            "ONNX-Runtime **GPU** is omitted: this box lacks the cuDNN 9 runtime ORT "
            "needs, so its CUDA provider falls back to CPU. On a GPU target you deploy "
            "the TensorRT engine, not ONNX-CUDA."
        )
    if not any(r.backend == "TensorRT" for r in rows):
        notes.append(
            "**TensorRT** (`.engine`) isn't measured here (the `tensorrt` package isn't "
            "installed). Export it on the target NVIDIA device (e.g. Jetson) with "
            "`people-analytics edge-bench --tensorrt`, and it appears as another row."
        )
    if notes:
        lines += ["", *[f"> {n}" for n in notes]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote {path}")


def rows_as_dicts(rows: list[FpsRow]) -> list[dict]:
    return [asdict(r) for r in rows]
