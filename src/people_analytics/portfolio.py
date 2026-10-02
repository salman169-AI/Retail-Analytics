"""Generate portfolio assets into outputs/portfolio/.

Assembles, for the configured scene:
  - occupancy heatmap
  - annotated dashboard frame (from analytics video)
  - dwell-time dashboard figure (zones + top dwellers + occupancy timeline)
  - summary KPI card (IDSW before/after + dwell)
  - IDSW before/after figure (70 → 28 portfolio headline)
  - one-paragraph outcome blurb

By default reuses an existing `outputs/analytics/<stem>_summary.json` when it
matches the config source, so regenerating portfolio assets is fast. Pass
`rerun=True` to force a fresh analytics pass.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from people_analytics.analytics import run_analytics
from people_analytics.config import Config

PURPLE = "#a351fb"
GREEN = "#008300"
GRAY = "#9aa0a6"
DARK = "#1f1f1f"

# Portfolio headline: previous final system (pre hybrid-IoS cleanup) → current.
PORTFOLIO_IDSW_BEFORE = 70
PORTFOLIO_IDSW_AFTER = 28


def _save_downscaled(src_img: np.ndarray, path: Path, width: int = 1280) -> None:
    h, w = src_img.shape[:2]
    out = cv2.resize(src_img, (width, int(h * width / w))) if w > width else src_img
    cv2.imwrite(str(path), out)


def _dashboard_frame(video_path: str, out: Path, width: int = 1280) -> None:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.45))
    ok, frame = cap.read()
    cap.release()
    if ok:
        _save_downscaled(frame, out, width)
        logger.info(f"Wrote {out}")


def _load_or_run_analytics(cfg: Config, rerun: bool) -> dict:
    """Return analytics summary, reusing disk exports when possible."""
    stem = Path(cfg.io.source).stem
    summary_path = Path(cfg.io.output_dir) / "analytics" / f"{stem}_summary.json"
    video_path = Path(cfg.io.output_dir) / "analytics" / f"{stem}_analytics.mp4"
    heatmap_path = Path(cfg.io.output_dir) / "analytics" / f"{stem}_heatmap.png"

    if (
        not rerun
        and summary_path.exists()
        and video_path.exists()
        and heatmap_path.exists()
    ):
        analytics = json.loads(summary_path.read_text(encoding="utf-8"))
        analytics.setdefault("exports", {})
        analytics["exports"]["video"] = str(video_path)
        analytics["exports"]["heatmap"] = str(heatmap_path)
        analytics["exports"]["events_csv"] = str(
            Path(cfg.io.output_dir) / "analytics" / f"{stem}_events.csv"
        )
        if analytics.get("source") == cfg.io.source or Path(cfg.io.source).name in str(
            analytics.get("source", "")
        ):
            logger.info(f"Reusing existing analytics summary: {summary_path}")
            return analytics

    cfg.privacy.enabled = False
    logger.info("Running a clean full analytics pass for portfolio visuals…")
    return run_analytics(cfg, source=cfg.io.source, max_frames=None)


def _pretty_zone(name: str) -> str:
    """Short, chart-safe zone labels (avoid long wraps that collide)."""
    mapping = {
        "shopfront_dwell_1": "Shopfront R",
        "shopfront_dwell_2": "Shopfront L",
        "station_entrance": "Station",
        "waiting_area": "Waiting",
        "main_concourse": "Concourse",
        "left_plaza": "Left plaza",
        "right_plaza": "Right plaza",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _dwell_dashboard(analytics: dict, out: Path) -> None:
    """Portfolio dwell dashboard with explicit spacing (no overlapping labels)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 9.2), facecolor="white")
    fig.suptitle(
        "Dwell-time & occupancy dashboard",
        fontsize=16,
        fontweight="bold",
        color=DARK,
        y=0.98,
    )
    scene = Path(analytics.get("source", "scene")).stem.replace("_", " ")
    fig.text(0.5, 0.945, scene, ha="center", fontsize=10, color="#666")

    # Generous margins so bar labels / y-tick labels never clip neighbours.
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.15],
        width_ratios=[0.85, 1.15, 1.35],
        hspace=0.42,
        wspace=0.45,
        left=0.06,
        right=0.97,
        top=0.88,
        bottom=0.09,
    )

    # ── KPIs (label left, value right — never stack number on label) ────────
    line = next(iter(analytics.get("lines", {}).values()), {"in": 0, "out": 0})
    mean_occ = analytics.get("occupancy", {}).get("mean", 0)
    kpis = [
        ("Unique visitors", str(analytics.get("unique_visitors", 0))),
        ("Peak occupancy", str(analytics.get("occupancy", {}).get("peak", 0))),
        (
            "Mean occupancy",
            f"{mean_occ:.1f}" if isinstance(mean_occ, (int, float)) else str(mean_occ),
        ),
        ("Entries / exits", f"{line.get('in', 0)} / {line.get('out', 0)}"),
        ("Duration (s)", str(analytics.get("duration_s", "—"))),
    ]

    ax_kpi = fig.add_subplot(gs[0, 0])
    ax_kpi.axis("off")
    ax_kpi.set_xlim(0, 1)
    ax_kpi.set_ylim(0, 1)
    ax_kpi.set_title("Key metrics", fontsize=12, fontweight="bold", loc="left", pad=12)
    n = len(kpis)
    for i, (label, value) in enumerate(kpis):
        # Even vertical rows; label and value share one baseline so they never collide.
        y = 0.88 - i * (0.78 / max(n - 1, 1))
        ax_kpi.text(0.0, y, label, fontsize=10, color="#64748B", va="center", ha="left")
        ax_kpi.text(
            1.0, y, value, fontsize=15, fontweight="bold", color=PURPLE, va="center", ha="right"
        )
        # Light rule under each row (except last) for readability.
        if i < n - 1:
            ax_kpi.plot(
                [0.0, 1.0], [y - 0.07, y - 0.07], color="#E2E8F0", linewidth=0.8, clip_on=False
            )

    # ── Per-zone dwell ──────────────────────────────────────────────────────
    zones = analytics.get("zones", {})
    ax_zone = fig.add_subplot(gs[0, 1])
    if zones:
        names = [_pretty_zone(n) for n in zones]
        means = [zones[n]["mean_dwell_s"] for n in zones]
        maxes = [zones[n]["max_dwell_s"] for n in zones]
        visitors = [zones[n].get("unique_visitors", 0) for n in zones]
        x = np.arange(len(names))
        w = 0.34
        ax_zone.bar(x - w / 2, means, w, color=PURPLE, label="Mean", zorder=2)
        ax_zone.bar(x + w / 2, maxes, w, color=GREEN, label="Max", zorder=2)
        ax_zone.set_xticks(x)
        ax_zone.set_xticklabels(names, fontsize=9)
        ax_zone.set_ylabel("Seconds", fontsize=9)
        ax_zone.set_title("Per-zone dwell", fontsize=12, fontweight="bold", loc="left", pad=10)
        ax_zone.legend(
            fontsize=8,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.02),
            ncol=2,
        )
        ax_zone.spines[["top", "right"]].set_visible(False)
        y_top = max(maxes) * 1.18 if maxes else 1.0
        ax_zone.set_ylim(0, y_top)
        for i, v in enumerate(visitors):
            ax_zone.text(
                i,
                max(means[i], maxes[i]) + y_top * 0.02,
                f"n={v}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555",
            )
    else:
        ax_zone.axis("off")
        ax_zone.text(0.5, 0.5, "No zones configured", ha="center", color="#888")

    # ── Top dwellers (short labels, room for y-ticks) ───────────────────────
    ax_top = fig.add_subplot(gs[0, 2])
    dwellers = analytics.get("dwell_per_identity", [])[:8]
    if dwellers:
        labels = [f"#{d['global_id']}  {_pretty_zone(d['zone'])}" for d in dwellers]
        vals = [float(d["dwell_s"]) for d in dwellers]
        # Longest at top
        order = np.argsort(vals)
        labels = [labels[i] for i in order]
        vals = [vals[i] for i in order]
        y = np.arange(len(labels))
        ax_top.barh(y, vals, color=PURPLE, height=0.65)
        ax_top.set_yticks(y)
        ax_top.set_yticklabels(labels, fontsize=8)
        ax_top.set_xlabel("Seconds", fontsize=9)
        ax_top.set_title("Top dwellers", fontsize=12, fontweight="bold", loc="left", pad=10)
        ax_top.spines[["top", "right"]].set_visible(False)
        ax_top.set_xlim(0, max(vals) * 1.12 if vals else 1)
        for yi, v in zip(y, vals, strict=True):
            ax_top.text(v + max(vals) * 0.02, yi, f"{v:.1f}", va="center", fontsize=7, color="#555")
    else:
        ax_top.axis("off")
        ax_top.text(0.5, 0.5, "No dwell records", ha="center", color="#888")

    # ── Occupancy timeline (full width) ─────────────────────────────────────
    ax_occ = fig.add_subplot(gs[1, :])
    timeline = analytics.get("occupancy", {}).get("timeline", [])
    if timeline:
        t = [p["t"] for p in timeline]
        c = [p["count"] for p in timeline]
        ax_occ.fill_between(t, c, color=PURPLE, alpha=0.16, linewidth=0)
        ax_occ.plot(t, c, color=PURPLE, linewidth=1.8)
        ax_occ.set_xlabel("Time (s)", fontsize=9)
        ax_occ.set_ylabel("People in frame", fontsize=9)
        ax_occ.set_title("Occupancy over time", fontsize=12, fontweight="bold", loc="left", pad=8)
        ax_occ.spines[["top", "right"]].set_visible(False)
        ax_occ.set_ylim(bottom=0)
        peak = analytics.get("occupancy", {}).get("peak", max(c) if c else 0)
        ax_occ.axhline(peak, color=GREEN, linestyle="--", linewidth=1.2, alpha=0.75)
        # Label inside the plot, left side — avoids clipping the right edge.
        ax_occ.text(
            t[0] if t else 0,
            peak,
            f"  peak {peak}",
            va="bottom",
            ha="left",
            fontsize=8,
            color=GREEN,
        )
    else:
        ax_occ.axis("off")
        ax_occ.text(0.5, 0.5, "No occupancy timeline", ha="center", color="#888")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    logger.info(f"Wrote {out}")


def _idsw_figure(out: Path) -> None:
    """Standalone IDSW figure: this pipeline v1 (70) → v2 (28)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before, after = PORTFOLIO_IDSW_BEFORE, PORTFOLIO_IDSW_AFTER
    drop = before - after
    pct = round(drop / before * 100) if before else 0

    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor="white")
    labels = ["Pipeline v1", "Pipeline v2"]
    values = [before, after]
    colors = [GRAY, PURPLE]
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=2)
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(values) * 0.03,
            str(v),
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=DARK,
        )
    ax.set_ylabel("Identity switches (IDSW)", fontsize=10)
    ax.set_title(
        f"ID switches reduced  {before} → {after}  (−{pct}%)",
        fontsize=13,
        fontweight="bold",
        color=DARK,
        pad=12,
    )
    ax.set_ylim(0, max(values) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#F1F5F9", zorder=0)
    ax.set_axisbelow(True)
    fig.text(
        0.5,
        0.02,
        "MOT20-01 · this pipeline, v1 (earlier tile merge) → v2 (hybrid IoS + BoT-SORT + gallery)",
        ha="center",
        fontsize=8,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    logger.info(f"Wrote {out}")


def _summary_card(analytics: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before, after = PORTFOLIO_IDSW_BEFORE, PORTFOLIO_IDSW_AFTER
    pct = round((before - after) / before * 100) if before else 0

    fig = plt.figure(figsize=(12.5, 6.4), facecolor="white")
    fig.suptitle(
        "People counting & dwell-time analytics — occlusion-robust Re-ID",
        fontsize=14,
        fontweight="bold",
        color=DARK,
    )
    gs = fig.add_gridspec(
        2, 3, hspace=0.48, wspace=0.38, top=0.88, bottom=0.10, left=0.07, right=0.97
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    line = next(iter(analytics.get("lines", {}).values()), {"in": 0, "out": 0})
    kpis = [
        ("Unique visitors", analytics["unique_visitors"]),
        ("Peak occupancy", analytics["occupancy"]["peak"]),
        ("Entries / exits", f"{line.get('in', 0)} / {line.get('out', 0)}"),
        ("ID switches", f"{before} → {after}"),
        ("IDSW reduction", f"−{pct}%"),
    ]
    for i, (k, v) in enumerate(kpis):
        ax0.text(0.0, 0.94 - i * 0.18, str(v), fontsize=15, fontweight="bold", color=PURPLE)
        ax0.text(0.0, 0.94 - i * 0.18 - 0.065, k, fontsize=9, color="#444")
    ax0.set_title("Key metrics", fontsize=11, loc="left")

    ax1 = fig.add_subplot(gs[0, 1])
    bars = ax1.bar(
        ["v1", "v2"],
        [before, after],
        color=[GRAY, PURPLE],
        width=0.55,
    )
    for bar, v in zip(bars, [before, after], strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(before, after) * 0.04,
            str(v),
            ha="center",
            fontweight="bold",
            fontsize=11,
        )
    ax1.set_ylim(0, max(before, after) * 1.25)
    ax1.set_title("ID switches ↓", fontsize=11)
    ax1.set_ylabel("IDSW")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    ax2.set_title("What improved", fontsize=11, loc="left")
    ax2.text(
        0.0,
        0.75,
        "Hybrid tiled detection\n"
        "+ IoS merge removes\n"
        "half-body duplicate boxes\n"
        "that used to mint extra IDs.",
        fontsize=9,
        color="#444",
        va="top",
    )
    ax2.text(
        0.0,
        0.22,
        f"MOT20-01: {before} → {after} IDSW\n"
        "BoT-SORT + LightMBN\n"
        "+ re-entry gallery",
        fontsize=9,
        color="#444",
        va="top",
    )

    ax3 = fig.add_subplot(gs[1, :2])
    zones = analytics.get("zones", {})
    if zones:
        names = [_pretty_zone(n) for n in zones]
        means = [zones[n]["mean_dwell_s"] for n in zones]
        maxes = [zones[n]["max_dwell_s"] for n in zones]
        x = np.arange(len(names))
        ax3.bar(x - 0.2, means, 0.4, color=PURPLE, label="Mean dwell (s)")
        ax3.bar(x + 0.2, maxes, 0.4, color=GRAY, label="Max dwell (s)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, fontsize=9)
        ax3.legend(fontsize=8, frameon=False, loc="upper right")
        ax3.spines[["top", "right"]].set_visible(False)
        y_top = max(maxes) * 1.15 if maxes else 1
        ax3.set_ylim(0, y_top)
    ax3.set_title("Per-zone dwell time (street scene)", fontsize=11)
    ax3.set_ylabel("Seconds")

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    ax4.set_title("Pipeline", fontsize=11, loc="left")
    ax4.text(
        0.0,
        0.7,
        "YOLO26 → BoT-SORT\n"
        "+ LightMBN → re-entry\n"
        "gallery → zones / lines\n"
        "/ heatmap",
        fontsize=9,
        color="#444",
        va="top",
    )
    ax4.text(
        0.0,
        0.15,
        "Edge export: ONNX / TensorRT",
        fontsize=9,
        color="#444",
        va="top",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    logger.info(f"Wrote {out}")


def _blurb(analytics: dict, out: Path) -> None:
    before, after = PORTFOLIO_IDSW_BEFORE, PORTFOLIO_IDSW_AFTER
    pct = round((before - after) / before * 100) if before else 0

    zones = analytics.get("zones", {})
    zone_bits = []
    for name, z in zones.items():
        zone_bits.append(
            f"{_pretty_zone(name)}: {z.get('unique_visitors', 0)} visitors, "
            f"mean dwell {z.get('mean_dwell_s', 0):.1f}s"
        )
    zone_line = "; ".join(zone_bits) if zone_bits else "n/a"

    line = next(iter(analytics.get("lines", {}).values()), {"in": 0, "out": 0})
    text = (
        "# People counting & dwell-time analytics with occlusion-robust Re-ID\n\n"
        "End-to-end video analytics that counts entries/exits and per-zone dwell time "
        "with an occupancy heatmap, holding a stable identity per person through "
        "occlusions and re-entries. Hybrid tiled detection plus BoT-SORT + LightMBN "
        "and a custom re-entry gallery cut **ID switches "
        f"{before} → {after} (−{pct}%)** on MOT20-01 versus the previous final system. "
        "Privacy-preserving (on-device blur/pixelate, embeddings not crops). "
        "Exports to ONNX/TensorRT for edge.\n\n"
        f"- Scene visitors: {analytics['unique_visitors']} · "
        f"peak occupancy: {analytics['occupancy']['peak']} · "
        f"entries/exits: {line.get('in', 0)}/{line.get('out', 0)}\n"
        f"- Zones — {zone_line}\n"
        "- Detection: YOLO26 (hybrid tiled + IoS merge) · Tracking: BoxMOT · "
        "Eval: TrackEval (HOTA/IDF1/IDSW)\n"
        "- Assets: `PROJECT_SUMMARY.md`, `summary_card.png`, `dwell_dashboard.png`, "
        "`heatmap.png`, `dashboard_frame.png`, `idsw_before_after.png`\n"
    )
    out.write_text(text, encoding="utf-8")
    logger.info(f"Wrote {out}")


def build_portfolio(
    cfg: Config,
    out_dir: str | Path = "outputs/portfolio",
    rerun: bool = False,
) -> Path:
    """Assemble portfolio assets for the configured scene."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analytics = _load_or_run_analytics(cfg, rerun=rerun)

    hm = analytics["exports"]["heatmap"]
    if Path(hm).exists():
        _save_downscaled(cv2.imread(hm), out_dir / "heatmap.png")
        logger.info(f"Wrote {out_dir / 'heatmap.png'}")

    video = analytics["exports"]["video"]
    if Path(video).exists():
        _dashboard_frame(video, out_dir / "dashboard_frame.png")

    # Keep the GIF if present (illustrative clip); PNG is the 70→28 headline chart.
    gif_src = Path("metrics/before_after.gif")
    if gif_src.exists():
        shutil.copy(gif_src, out_dir / "idsw_before_after.gif")

    _idsw_figure(out_dir / "idsw_before_after.png")
    _dwell_dashboard(analytics, out_dir / "dwell_dashboard.png")
    _summary_card(analytics, out_dir / "summary_card.png")
    _blurb(analytics, out_dir / "BLURB.md")

    (out_dir / "scene_summary.json").write_text(
        json.dumps(
            {
                "source": analytics.get("source"),
                "frames": analytics.get("frames"),
                "duration_s": analytics.get("duration_s"),
                "unique_visitors": analytics.get("unique_visitors"),
                "occupancy": {
                    "peak": analytics.get("occupancy", {}).get("peak"),
                    "mean": analytics.get("occupancy", {}).get("mean"),
                },
                "lines": analytics.get("lines"),
                "zones": analytics.get("zones"),
                "portfolio_idsw": {
                    "before": PORTFOLIO_IDSW_BEFORE,
                    "after": PORTFOLIO_IDSW_AFTER,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Portfolio assets in {out_dir}")
    return out_dir
