"""Command-line entry point.

Phase 0 ships two commands that exercise the scaffold end-to-end without the ML
stack: `info` (print validated config) and `probe` (open a clip and report its
properties via the stub reader). Later phases add `detect`, `track`, `run`, etc.
"""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from people_analytics import __version__
from people_analytics.config import load_config
from people_analytics.data import (
    download_mot_sequence,
    download_sample_clips,
    load_mot_sequence,
)
from people_analytics.io import frame_generator, get_video_info

app = typer.Typer(add_completion=False, help="People counting + dwell-time analytics.")

DEFAULT_CONFIG = "configs/default.yaml"


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"people-analytics {__version__}")


@app.command()
def info(config: str = typer.Option(DEFAULT_CONFIG, help="Path to a YAML config.")) -> None:
    """Load and pretty-print the validated configuration."""
    cfg = load_config(config)
    typer.echo(cfg.model_dump_json(indent=2))


@app.command()
def probe(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to a YAML config."),
    source: str = typer.Option(None, help="Override the video source in the config."),
) -> None:
    """Open the configured clip and report its properties + a frame count."""
    cfg = load_config(config)
    src = source or cfg.io.source
    if not Path(src).exists():
        logger.error(f"Video source not found: {src}")
        raise typer.Exit(code=1)

    vinfo = get_video_info(src)
    logger.info(
        f"{src} | {vinfo.width}x{vinfo.height} @ {vinfo.fps:.2f} fps "
        f"| {vinfo.total_frames} frames"
    )

    read = sum(1 for _ in frame_generator(src, max_frames=cfg.io.max_frames))
    logger.info(f"Read {read} frames through the stub reader.")


@app.command()
def detect(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to a YAML config."),
    source: str = typer.Option(None, help="Override the video source in the config."),
    max_frames: int = typer.Option(None, help="Cap frames processed (quick runs)."),
    output: str = typer.Option(None, help="Output mp4 path (default: outputs/<stem>_detect.mp4)."),
) -> None:
    """Run YOLO26 person detection over a clip; write annotated video + log FPS."""
    import time

    import supervision as sv

    # Heavy imports kept local so `info`/`probe`/`fetch-*` work without the ML stack.
    from people_analytics.detection import PersonDetector

    cfg = load_config(config)
    src = source or cfg.io.source
    if not Path(src).exists():
        logger.error(f"Video source not found: {src}")
        raise typer.Exit(code=1)

    n_frames = max_frames if max_frames is not None else cfg.io.max_frames
    out_dir = Path(cfg.io.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = output or str(out_dir / f"{Path(src).stem}_detect.mp4")

    detector = PersonDetector(cfg.detector)
    vinfo = sv.VideoInfo.from_video_path(src)
    box = sv.BoxAnnotator()
    label = sv.LabelAnnotator(text_scale=max(0.5, vinfo.width / 2000))
    fps_monitor = sv.FPSMonitor()

    det_seconds = 0.0
    frames = 0
    total_persons = 0
    with sv.VideoSink(out_path, vinfo) as sink:
        for frame in frame_generator(src, max_frames=n_frames):
            t0 = time.perf_counter()
            detections = detector.detect(frame)
            det_seconds += time.perf_counter() - t0
            fps_monitor.tick()

            frames += 1
            total_persons += len(detections)
            annotated = box.annotate(frame.copy(), detections)
            labels = [f"person {c:.2f}" for c in detections.confidence]
            annotated = label.annotate(annotated, detections, labels)
            sink.write_frame(annotated)

    detector_fps = frames / det_seconds if det_seconds else 0.0
    logger.info(f"Wrote {out_path} ({frames} frames)")
    logger.info(
        f"Detector-only FPS: {detector_fps:.1f} | end-to-end FPS: {fps_monitor.fps:.1f} "
        f"| avg persons/frame: {total_persons / frames:.1f}"
    )


@app.command()
def track(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to a YAML config."),
    source: str = typer.Option(None, help="Override the video source."),
    tracker: str = typer.Option(None, help="Override tracker (botsort/deepocsort/boosttrack)."),
    reid: str = typer.Option(None, help="Override reid model (osnet_ain/lmbn/osnet)."),
    max_frames: int = typer.Option(None, help="Cap frames processed."),
    output: str = typer.Option(None, help="Output mp4 path."),
) -> None:
    """Detect + track a clip; write annotated video with stable tracker IDs."""
    import supervision as sv

    from people_analytics.detection import PersonDetector
    from people_analytics.tracking import MultiObjectTracker

    cfg = load_config(config)
    src = source or cfg.io.source
    if not Path(src).exists():
        logger.error(f"Video source not found: {src}")
        raise typer.Exit(code=1)

    n_frames = max_frames if max_frames is not None else cfg.io.max_frames
    out_dir = Path(cfg.io.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = output or str(out_dir / f"{Path(src).stem}_track.mp4")

    vinfo = sv.VideoInfo.from_video_path(src)
    detector = PersonDetector(cfg.detector)
    mot = MultiObjectTracker(
        tracker or cfg.tracker.name,
        reid_name=reid or cfg.tracker.reid_model,
        device=cfg.detector.device,
        frame_rate=int(vinfo.fps),
        cfg=cfg.tracker,
    )
    box = sv.BoxAnnotator()
    label = sv.LabelAnnotator(text_scale=max(0.5, vinfo.width / 2000))
    trace = sv.TraceAnnotator(trace_length=int(vinfo.fps))

    seen_ids: set[int] = set()
    frames = 0
    with sv.VideoSink(out_path, vinfo) as sink:
        for frame in frame_generator(src, max_frames=n_frames):
            tracked = mot.update(detector.detect(frame), frame)
            frames += 1
            seen_ids.update(int(i) for i in tracked.tracker_id)
            annotated = box.annotate(frame.copy(), tracked)
            labels = [f"#{int(i)}" for i in tracked.tracker_id]
            annotated = label.annotate(annotated, tracked, labels)
            annotated = trace.annotate(annotated, tracked)
            sink.write_frame(annotated)

    logger.info(f"Wrote {out_path} ({frames} frames)")
    logger.info(f"Tracker: {mot.name} | reid: {mot.reid_name} | unique IDs seen: {len(seen_ids)}")


@app.command()
def benchmark(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to a YAML config."),
    output: str = typer.Option("outputs/benchmark", help="Output dir for MOT files + summary."),
    results_md: str = typer.Option("RESULTS.md", help="Where to write the results table."),
) -> None:
    """Run the full tracker x ReID matrix + ByteTrack baseline; write RESULTS.md."""
    from people_analytics.tracking.benchmark import run_benchmark, write_results_md

    cfg = load_config(config)
    table = run_benchmark(cfg, output)
    from people_analytics.data.mot import MOTSequence

    seq_name = MOTSequence(cfg.eval.sequence_dir).info.name
    winner = write_results_md(table, results_md, seq_name)
    logger.info(f"Winner: {winner.tracker} + {winner.reid} (IDSW={winner.idsw})")


@app.command()
def analyze(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to a YAML config."),
    source: str = typer.Option(None, help="Override the video source."),
    max_frames: int = typer.Option(None, help="Cap frames processed."),
    start_frame: int = typer.Option(None, help="First frame to process (0-based)."),
    heatmap_overlay: bool = typer.Option(False, help="Overlay the heatmap on the video too."),
    privacy: bool = typer.Option(None, help="Override privacy anonymization on/off."),
    no_video: bool = typer.Option(False, help="Skip the annotated MP4 (exports only)."),
) -> None:
    """Full analytics: dwell zones + entry/exit + occupancy + heatmap + CSV/JSON export."""
    from people_analytics.analytics import run_analytics

    cfg = load_config(config)
    if start_frame is not None:
        cfg.io.start_frame = start_frame
    if no_video:
        cfg.analytics.write_video = False
    if privacy is not None:
        cfg.privacy.enabled = privacy
    summary = run_analytics(
        cfg, source=source, max_frames=max_frames, heatmap_overlay=heatmap_overlay
    )
    logger.info(f"Exports: {summary['exports']}")


@app.command()
def tune(config: str = typer.Option(DEFAULT_CONFIG, help="Path to a YAML config.")) -> None:
    """Sweep track_buffer x gallery cosine_thresh to minimize IDSW -> metrics/tuning.md."""
    from people_analytics.tracking.tune import tune as run_tune

    cfg = load_config(config)
    best = run_tune(cfg)["best"]
    logger.info(f"Best IDSW={best['IDSW']} at {best['params']}")


@app.command()
def portfolio(
    config: str = typer.Option(
        "configs/street.yaml",
        help="Scene config (street.yaml is the flagship demo).",
    ),
    rerun: bool = typer.Option(
        False,
        help="Force a fresh analytics pass instead of reusing outputs/analytics/.",
    ),
) -> None:
    """Generate portfolio assets (dwell dashboard, heatmap, summary card, blurb)."""
    from people_analytics.portfolio import build_portfolio

    cfg = load_config(config)
    out = build_portfolio(cfg, rerun=rerun)
    logger.info(f"Portfolio assets written to {out}")


@app.command()
def export(
    config: str = typer.Option("configs/edge.yaml", help="Config (edge profile by default)."),
    tensorrt: bool = typer.Option(False, help="Also export a TensorRT .engine (needs tensorrt)."),
    reid: bool = typer.Option(True, help="Also export the Re-ID model to ONNX."),
) -> None:
    """Export the detector (+ Re-ID) to ONNX / TensorRT for edge deployment."""
    from people_analytics.edge import export_detector, export_reid_onnx

    cfg = load_config(config)
    det = export_detector(cfg.detector.model, cfg.detector.weights_dir, tensorrt=tensorrt)
    for fmt, path in det.items():
        logger.info(f"detector {fmt}: {path}")
    if reid:
        logger.info(f"reid onnx: {export_reid_onnx(cfg.tracker.reid_model)}")


@app.command("edge-bench")
def edge_bench(
    config: str = typer.Option("configs/edge.yaml", help="Config (edge profile by default)."),
    n_frames: int = typer.Option(120, help="Frames to benchmark over."),
    tensorrt: bool = typer.Option(False, help="Include TensorRT if exported."),
) -> None:
    """Benchmark detector FPS: PyTorch vs ONNX Runtime, CPU vs GPU -> metrics/edge_fps.md."""
    from people_analytics.edge.benchmark import run_edge_benchmark

    cfg = load_config(config)
    rows = run_edge_benchmark(
        cfg.detector.model, cfg.detector.weights_dir, cfg.io.source,
        n_frames=n_frames, tensorrt=tensorrt,
    )
    for r in rows:
        logger.info(f"{r.backend} {r.device}: {r.fps} FPS")


@app.command()
def demo(port: int = typer.Option(8501, help="Port for the Streamlit server.")) -> None:
    """Launch the Streamlit demo dashboard (upload/run a clip, view + export results)."""
    import subprocess
    import sys

    dashboard = Path(__file__).parent / "app" / "dashboard.py"
    logger.info(f"Launching dashboard at http://localhost:{port} …")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard), "--server.port", str(port)],
        check=False,
    )


@app.command()
def evaluate(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to a YAML config."),
    out_dir: str = typer.Option("metrics", help="Where to write summary + assets."),
) -> None:
    """Phase 7 proof: TrackEval HOTA/IDF1/IDSW + dwell accuracy + before/after figure & GIF."""
    from people_analytics.eval import run_evaluation

    cfg = load_config(config)
    summary = run_evaluation(cfg, out_dir)
    base = summary["metrics"][summary["baseline_key"]]
    final = summary["metrics"][summary["selected_key"]]
    logger.info(
        f"selected={summary['selected_key']} | "
        f"HOTA {base['HOTA']:.3f}->{final['HOTA']:.3f} | "
        f"IDSW {base['IDSW']}->{final['IDSW']} | "
        f"dwell error {summary['dwell_accuracy']['overall_error_pct']}%"
    )


@app.command("gallery-eval")
def gallery_eval(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to a YAML config."),
) -> None:
    """Run the configured tracker with/without the re-entry gallery; write GALLERY.md."""
    from people_analytics.tracking.benchmark import compare_gallery

    cfg = load_config(config)
    out = compare_gallery(cfg)
    logger.info(f"IDSW {out['before']['idsw']} -> {out['after']['idsw']} (drop {out['idsw_drop']})")


@app.command("fetch-samples")
def fetch_samples(dest: str = typer.Option("data/samples", help="Destination dir.")) -> None:
    """Download the retail/venue sample clips (grocery / market / overhead)."""
    paths = download_sample_clips(dest)
    for p in paths:
        logger.info(f"ready: {p}")


@app.command("fetch-eval")
def fetch_eval(
    sequence: str = typer.Option("MOT20-01", help="MOT sequence name."),
    dest: str = typer.Option("data/eval", help="Destination root dir."),
) -> None:
    """Download a labeled MOT-format eval sequence (ground-truth track IDs)."""
    seq_dir = download_mot_sequence(sequence=sequence, dest_root=dest)
    logger.info(f"eval sequence ready: {seq_dir}")


@app.command("check-eval")
def check_eval(config: str = typer.Option(DEFAULT_CONFIG, help="Path to a YAML config.")) -> None:
    """Load the eval sequence and report frame/ID/box stats (DoD check)."""
    cfg = load_config(config)
    seq_dir = Path(cfg.eval.sequence_dir)
    if not seq_dir.exists():
        logger.error(f"Eval sequence not found: {seq_dir} (run `fetch-eval` first)")
        raise typer.Exit(code=1)

    seq = load_mot_sequence(seq_dir)
    total_boxes = sum(len(v) for v in seq.gt.values())
    logger.info(
        f"{seq.info.name} | {seq.info.im_width}x{seq.info.im_height} "
        f"@ {seq.info.frame_rate:g} fps | {seq.num_frames} frames"
    )
    logger.info(
        f"ground truth: {len(seq.track_ids)} unique track IDs, "
        f"{total_boxes} person boxes across {len(seq.gt)} annotated frames"
    )


MEVA_CONFIG = "configs/meva_candidates.yaml"


def _load_meva(config: str) -> dict:
    import yaml

    return yaml.safe_load(Path(config).read_text(encoding="utf-8"))


@app.command("meva-rank")
def meva_rank(config: str = typer.Option(MEVA_CONFIG, help="MEVA shortlist YAML.")) -> None:
    """Rank every annotated MEVA clip; add people-at-once for the shortlist."""
    from people_analytics.data import meva

    m = _load_meva(config)
    out_dir = Path(m["output_dir"])
    stats = meva.scan_annotations(
        m["annotations"], meva.load_cameras(m["cameras_csv"]), meva.list_s3_clips()
    )
    logger.info(f"all clips -> {meva.write_stats_csv(stats, out_dir / 'clip_stats.csv')}")

    by_clip = {s.clip: s for s in stats}
    shortlist = []
    for role, clips in m["roles"].items():
        for clip in clips:
            s = meva.add_occupancy(m["annotations"], by_clip[clip])
            shortlist.append(s)
            logger.info(
                f"{role:8} {clip}  tracks={s.person_tracks} peak={s.peak_at_once} "
                f"mean={s.mean_at_once} in={s.enters} out={s.exits} buy={s.purchases} "
                f"{s.size_mb}MB"
            )
    logger.info(f"shortlist -> {meva.write_stats_csv(shortlist, out_dir / 'shortlist.csv')}")


@app.command("meva-fetch")
def meva_fetch(config: str = typer.Option(MEVA_CONFIG, help="MEVA shortlist YAML.")) -> None:
    """Download the shortlisted MEVA clips from the public S3 bucket."""
    from people_analytics.data import meva

    m = _load_meva(config)
    s3 = meva.list_s3_clips()
    for clips in m["roles"].values():
        for clip in clips:
            meva.download_clip(s3[clip][0], m["clips_dir"])


@app.command("meva-sheets")
def meva_sheets(
    config: str = typer.Option(MEVA_CONFIG, help="MEVA shortlist YAML."),
    frames: int = typer.Option(4, help="Frames per clip."),
) -> None:
    """One contact sheet per role: `frames` stills spread across each shortlisted clip."""
    from people_analytics.data import meva

    m = _load_meva(config)
    clips_dir = Path(m["clips_dir"])
    for role, clips in m["roles"].items():
        rows = [(clip, clips_dir / f"{clip}.r13.avi") for clip in clips]
        out = meva.contact_sheet(rows, Path(m["output_dir"]) / f"contact_{role}.jpg", n=frames)
        logger.info(f"{role}: {out}")


@app.command("meva-tune")
def meva_tune(
    config: str = typer.Option(..., help="Scene config with meva.tune_clip set."),
    seconds: float = typer.Option(120, help="Length of the busiest window to tune on."),
    variants: str = typer.Option(None, help="Comma-separated subset of variants."),
    tiled: bool = typer.Option(False, help="Force tiled (hybrid) detection on."),
) -> None:
    """Compare tracker/detector variants on a scene's tuning clip (MEVA partial GT)."""
    import json

    from people_analytics.tracking.clip_tune import tune_on_clip

    cfg = load_config(config)
    if tiled:
        cfg.detector.slicer.enabled = True
        cfg.detector.slicer.merge_metric = "ios"
        cfg.detector.slicer.merge_threshold = 0.45
    stem = Path(config).stem
    out_dir = Path("outputs/meva/tune") / stem
    rows = tune_on_clip(cfg, seconds, out_dir, variants.split(",") if variants else None)
    out = Path("metrics") / f"{stem}_tuning.json"
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    keep = [r for r in previous if (r["detector"], r["variant"]) not in
            {(x["detector"], x["variant"]) for x in rows}]
    out.write_text(json.dumps(keep + rows, indent=2), encoding="utf-8")
    logger.info(f"-> {out}")


@app.command()
def activity(
    config: str = typer.Option(..., help="Scene config whose last run to read."),
) -> None:
    """(Re)build the event log (door / zone / queue events) from a run's exports."""
    import json

    from people_analytics.analytics.activity import build_activity, write_activity
    from people_analytics.io import get_video_info

    cfg = load_config(config)
    stem = Path(cfg.io.source).stem
    out_dir = Path(cfg.io.output_dir) / "analytics"
    summary = json.loads((out_dir / f"{stem}_summary.json").read_text(encoding="utf-8"))
    fps = get_video_info(cfg.io.source).fps
    start = summary.get("start_frame", 0)
    rows = build_activity(cfg, out_dir / f"{stem}_events.csv", fps, start + 1,
                          start + summary["frames"])
    path = write_activity(rows, out_dir / f"{stem}_activity.csv")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["event"]] = counts.get(r["event"], 0) + 1
    logger.info(f"{path}: {counts}")


@app.command("meva-eval-doors")
def meva_eval_doors(
    config: str = typer.Option("configs/meva_entrance.yaml", help="Entrance scene config."),
    source: str = typer.Option(None, help="Another clip of the same camera (already run)."),
) -> None:
    """Door IN/OUT vs MEVA enter/exit labels on a clip's activity log."""
    import json

    from people_analytics.eval.meva_doors import evaluate_doors
    from people_analytics.io import get_video_info

    cfg = load_config(config)
    src = source or cfg.io.source
    stem = Path(src).stem
    clip = stem.removesuffix(".r13")
    activity_csv = Path(cfg.io.output_dir) / "analytics" / f"{stem}_activity.csv"
    res = evaluate_doors(activity_csv, cfg.meva.annotations, clip, cfg.doors[0].name,
                         get_video_info(src).fps)
    out = Path("metrics") / "meva_doors_eval.json"
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    rows = [r for r in previous if r["clip"] != clip] + [res]
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info(res)


@app.command()
def composite(
    config: str = typer.Option(..., help="Scene config whose last run to replay."),
    title: str = typer.Option(..., help="Panel title, e.g. 'East doors'."),
    out: str = typer.Option(None, help="Output mp4 (default outputs/meva/composite/<clip>.mp4)."),
    start_s: float = typer.Option(0.0, help="Start of the cut, seconds into the clip."),
    seconds: float = typer.Option(None, help="Length of the cut (default: to the end)."),
    scale: float = typer.Option(0.75, help="Output scale of the 2560x1080 canvas."),
) -> None:
    """Video + stats panel side by side, replayed from a run's exports (no tracking)."""
    from people_analytics.analytics.composite import render_composite

    cfg = load_config(config)
    out = out or str(Path("outputs/meva/composite") / f"{Path(cfg.io.source).stem}.mp4")
    path = render_composite(cfg, title, out, start_s, seconds, scale=scale)
    logger.info(f"-> {path} ({path.stat().st_size / 1e6:.1f} MB)")


@app.command("demo-video")
def demo_video(
    config: str = typer.Option("configs/meva_demo.yaml", help="Shot list YAML."),
) -> None:
    """Assemble the ~60 s portfolio demo from the shot list (replays run exports)."""
    from people_analytics.demo import build_demo

    logger.info(build_demo(config))


@app.command("meva-idsw")
def meva_idsw(
    config: str = typer.Option(..., help="Scene config (its io.source clip is evaluated)."),
    variants: str = typer.Option("base,bytetrack", help="Comma-separated variants."),
) -> None:
    """Identity switches on the scene's own (unseen) clip, full length, vs MEVA's annotations."""
    import json

    from people_analytics.tracking.clip_tune import run_on_clip

    cfg = load_config(config)
    clip = Path(cfg.io.source).name.removesuffix(".r13.avi")
    stem = Path(config).stem
    rows = run_on_clip(cfg, clip, Path("outputs/meva/eval") / stem, variants.split(","))
    out = Path("metrics") / f"{stem}_idsw.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info(f"-> {out}")


@app.command()
def sitemap(
    config: str = typer.Option("configs/meva_sitemap.yaml", help="Site-plan YAML."),
    out_dir: str = typer.Option("outputs/meva/sitemap", help="Output directory."),
    min_track_s: float = typer.Option(3.0, help="Shortest track drawn as a path / counted."),
) -> None:
    """Combined site-map heatmap + walking paths from several cameras' event exports."""
    from people_analytics.analytics.sitemap import render_sitemap

    stats = render_sitemap(config, out_dir, min_track_s)
    logger.info(stats["cameras"])


if __name__ == "__main__":
    app()
