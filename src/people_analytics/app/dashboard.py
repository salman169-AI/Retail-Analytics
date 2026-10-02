"""Streamlit dashboard for people counting and dwell-time analytics.

Launch: `people-analytics demo`  →  http://localhost:8501
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from people_analytics.analytics import run_analytics
from people_analytics.config import load_config

# ── page + theme ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="People Counting · Dwell Time",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PURPLE = "#7C3AED"
GREEN = "#059669"
SLATE = "#334155"
MUTED = "#64748B"

st.markdown(
    """
    <style>
      /* Tighten default vertical gaps a bit without crushing sections */
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1200px; }
      h1 { font-size: 1.75rem !important; letter-spacing: -0.02em;
           margin-bottom: 0.15rem !important; }
      h2, h3 { letter-spacing: -0.01em; }
      div[data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.85rem 1rem;
      }
      div[data-testid="stMetric"] label { color: #64748B !important; }
      div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.55rem; color: #0F172A;
      }
      /* Tab labels breathe */
      button[data-baseweb="tab"] { font-weight: 600; }
      .pa-muted { color: #64748B; font-size: 0.92rem; margin: 0 0 1rem 0; }
      .pa-section { margin-top: 0.25rem; margin-bottom: 0.75rem; }
      /* Sidebar cleaner */
      section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── helpers ─────────────────────────────────────────────────────────────────
def sample_frames(video_path: str, n: int = 3) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(total * 0.2, total * 0.8, n).astype(int)
    out: list[np.ndarray] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def _download(label: str, path: str, mime: str) -> None:
    p = Path(path)
    if p.exists():
        st.download_button(
            label, p.read_bytes(), file_name=p.name, mime=mime, use_container_width=True
        )


def _find_existing_summary(source: str | None) -> Path | None:
    if not source:
        return None
    candidate = Path("outputs/analytics") / f"{Path(source).stem}_summary.json"
    return candidate if candidate.exists() else None


def _load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    stem = Path(data.get("source", path.stem.replace("_summary", ""))).stem
    exports = data.setdefault("exports", {})
    for key, name in [
        ("video", f"{stem}_analytics.mp4"),
        ("heatmap", f"{stem}_heatmap.png"),
        ("events_csv", f"{stem}_events.csv"),
    ]:
        p = Path("outputs/analytics") / name
        if p.exists():
            exports[key] = str(p)
    return data


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def _zone_dwell_chart(zones: dict) -> alt.Chart:
    rows = []
    for z, stats in zones.items():
        rows.append({"zone": _pretty(z), "metric": "Mean", "seconds": stats["mean_dwell_s"]})
        rows.append({"zone": _pretty(z), "metric": "Max", "seconds": stats["max_dwell_s"]})
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3, size=18)
        .encode(
            x=alt.X(
                "zone:N",
                title=None,
                axis=alt.Axis(labelAngle=0, labelLimit=140, labelPadding=6),
            ),
            xOffset=alt.XOffset("metric:N"),
            y=alt.Y("seconds:Q", title="Dwell (seconds)", scale=alt.Scale(nice=True)),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(domain=["Mean", "Max"], range=[PURPLE, GREEN]),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
            tooltip=[
                alt.Tooltip("zone:N", title="Zone"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("seconds:Q", title="Seconds", format=".1f"),
            ],
        )
        .properties(height=280)
        .configure_axis(labelColor=MUTED, titleColor=SLATE, gridColor="#F1F5F9")
        .configure_view(strokeWidth=0)
    )


def _top_dwellers_chart(records: list[dict], n: int = 10) -> alt.Chart | None:
    if not records:
        return None
    df = pd.DataFrame(records[:n]).copy()
    df["label"] = "#" + df["global_id"].astype(str) + "  ·  " + df["zone"].map(_pretty)
    # Stable order: longest dwell at top
    order = df.sort_values("dwell_s", ascending=True)["label"].tolist()
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3, color=PURPLE, height=16)
        .encode(
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("dwell_s:Q", title="Dwell (seconds)"),
            tooltip=[
                alt.Tooltip("global_id:Q", title="ID"),
                alt.Tooltip("zone:N", title="Zone"),
                alt.Tooltip("dwell_s:Q", title="Seconds", format=".1f"),
            ],
        )
        .properties(height=max(220, 28 * len(df) + 40))
        .configure_axis(labelColor=MUTED, titleColor=SLATE, gridColor="#F1F5F9")
        .configure_view(strokeWidth=0)
    )


def _occupancy_chart(timeline: list[dict]) -> alt.Chart:
    df = pd.DataFrame(timeline)
    peak = int(df["count"].max()) if len(df) else 0
    area = (
        alt.Chart(df)
        .mark_area(line={"color": PURPLE, "strokeWidth": 2}, color=PURPLE, opacity=0.15)
        .encode(
            x=alt.X("t:Q", title="Time (s)"),
            y=alt.Y("count:Q", title="People in frame", scale=alt.Scale(domainMin=0)),
            tooltip=[
                alt.Tooltip("t:Q", title="Time (s)", format=".1f"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"peak": [peak]}))
        .mark_rule(color=GREEN, strokeDash=[6, 4], strokeWidth=1.5)
        .encode(y="peak:Q")
    )
    return (
        (area + rule)
        .properties(height=240)
        .configure_axis(labelColor=MUTED, titleColor=SLATE, gridColor="#F1F5F9")
        .configure_view(strokeWidth=0)
    )


# ── sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("### Controls")
configs = sorted(str(p) for p in Path("configs").glob("*.yaml"))
cfg_default = next(
    (i for i, c in enumerate(configs) if c.replace("\\", "/").endswith("street.yaml")),
    next((i for i, c in enumerate(configs) if c.endswith("default.yaml")), 0),
)
config_path = st.sidebar.selectbox("Config", configs, index=cfg_default)
cfg = load_config(config_path)

samples = sorted(str(p) for p in Path("data/samples").glob("*.mp4"))
cfg_stem = Path(cfg.io.source).stem
default_idx = next((i for i, s in enumerate(samples) if cfg_stem in Path(s).stem), 0)
source = st.sidebar.selectbox("Sample clip", samples, index=default_idx) if samples else None
uploaded = st.sidebar.file_uploader("Or upload video", type=["mp4", "avi", "mov"])

st.sidebar.markdown("---")
max_default = 470 if cfg_stem == "street_scene" else 120
max_frames = st.sidebar.slider("Max frames", 30, 600, min(max_default, 600), 10)
privacy = st.sidebar.checkbox("Blur people (privacy)", value=False)

st.sidebar.caption(
    f"**{cfg.tracker.name}** + **{cfg.tracker.reid_model}**  ·  "
    f"gallery {'on' if cfg.reid_gallery.enabled else 'off'}  ·  "
    f"tiling {'on' if cfg.detector.slicer.enabled else 'off'}"
)

existing = _find_existing_summary(source)
st.sidebar.markdown("---")
run = st.sidebar.button("Run analytics", type="primary", use_container_width=True)
load_btn = False
if existing is not None:
    load_btn = st.sidebar.button(
        "Open last results",
        use_container_width=True,
        help=str(existing),
    )
    st.sidebar.caption(f"Cached: `{existing.name}`")

# ── load / run ──────────────────────────────────────────────────────────────
if load_btn and existing is not None:
    st.session_state["summary"] = _load_summary(existing)
    st.session_state["summary_label"] = f"Loaded from {existing.name}"

if run:
    src = source
    if uploaded is not None:
        tmp = Path(tempfile.gettempdir()) / uploaded.name
        tmp.write_bytes(uploaded.getbuffer())
        src = str(tmp)
    if not src:
        st.error("No video selected.")
    else:
        cfg.privacy.enabled = privacy
        with st.spinner("Detection → tracking → dwell analytics…"):
            st.session_state["summary"] = run_analytics(
                cfg, source=src, max_frames=max_frames
            )
            st.session_state["summary_label"] = f"Ran on {Path(src).name} · max {max_frames} frames"

# Auto-open cached results on first visit so the demo is not a blank page.
if "summary" not in st.session_state and existing is not None:
    st.session_state["summary"] = _load_summary(existing)
    st.session_state["summary_label"] = f"Loaded from {existing.name}"

summary = st.session_state.get("summary")

# ── header ──────────────────────────────────────────────────────────────────
st.title("People counting & dwell-time analytics")
st.markdown(
    '<p class="pa-muted">Stable IDs through occlusion · per-zone dwell · '
    'entry/exit · occupancy heatmap</p>',
    unsafe_allow_html=True,
)

if not summary:
    st.info(
        "Choose a config and clip in the sidebar, then "
        "**Open last results** or **Run analytics**."
    )
    portfolio = Path("outputs/portfolio")
    if (portfolio / "dwell_dashboard.png").exists():
        st.markdown("#### Portfolio preview")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.image(str(portfolio / "dwell_dashboard.png"), use_container_width=True)
            st.caption("Dwell-time dashboard")
        with c2:
            frame = portfolio / "dashboard_frame.png"
            if frame.exists():
                st.image(str(frame), use_container_width=True)
                st.caption("Annotated scene")
    st.stop()

if label := st.session_state.get("summary_label"):
    st.caption(label)

# ── KPIs ────────────────────────────────────────────────────────────────────
line_name, line = next(iter(summary.get("lines", {}).items()), ("line", {"in": 0, "out": 0}))
k1, k2, k3, k4 = st.columns(4, gap="medium")
k1.metric("Unique visitors", summary["unique_visitors"])
k2.metric("Peak occupancy", summary["occupancy"]["peak"])
mean_occ = summary["occupancy"].get("mean")
k3.metric("Mean occupancy", f"{mean_occ:.1f}" if isinstance(mean_occ, (int, float)) else mean_occ)
k4.metric(
    "Entries / exits",
    f"{line.get('in', 0)} / {line.get('out', 0)}",
    help=line_name,
)

# Extra counting lines (if any) sit on their own quiet row.
extra = list(summary.get("lines", {}).items())[1:]
if extra:
    st.markdown("")
    cols = st.columns(len(extra) * 2, gap="medium")
    for i, (name, ln) in enumerate(extra):
        cols[2 * i].metric(f"{_pretty(name)} in", ln.get("in", 0))
        cols[2 * i + 1].metric(f"{_pretty(name)} out", ln.get("out", 0))

st.markdown("")

# ── tabs keep sections from stacking / colliding ────────────────────────────
tab_overview, tab_dwell, tab_scene, tab_export = st.tabs(
    ["Overview", "Dwell analytics", "Scene", "Exports"]
)

# Overview: heatmap + occupancy (the two “whole-scene” views)
with tab_overview:
    left, right = st.columns((1, 1), gap="large")
    with left:
        st.markdown("##### Occupancy heatmap")
        hm = summary["exports"].get("heatmap", "")
        if hm and Path(hm).exists():
            st.image(hm, use_container_width=True)
        else:
            st.warning("Heatmap not found — run analytics first.")
    with right:
        st.markdown("##### Occupancy over time")
        timeline = summary.get("occupancy", {}).get("timeline", [])
        if timeline:
            st.altair_chart(_occupancy_chart(timeline), use_container_width=True)
            st.caption("Dashed green line = peak occupancy.")
        else:
            st.info("No occupancy timeline in this summary.")

    # Zone snapshot table under the charts (full width, no column collision)
    zones = summary.get("zones", {})
    if zones:
        st.markdown("##### Zone summary")
        zdf = pd.DataFrame(
            [
                {
                    "Zone": _pretty(z),
                    "Visitors": stats.get("unique_visitors", 0),
                    "Mean dwell (s)": round(stats.get("mean_dwell_s", 0), 2),
                    "Max dwell (s)": round(stats.get("max_dwell_s", 0), 2),
                }
                for z, stats in zones.items()
            ]
        )
        st.dataframe(zdf, use_container_width=True, hide_index=True)

# Dwell: two charts side by side, raw table collapsed
with tab_dwell:
    zones = summary.get("zones", {})
    col_a, col_b = st.columns((1, 1), gap="large")
    with col_a:
        st.markdown("##### Dwell time per zone")
        if zones:
            st.altair_chart(_zone_dwell_chart(zones), use_container_width=True)
        else:
            st.info("No zones configured for this scene.")
    with col_b:
        st.markdown("##### Top dwellers")
        chart = _top_dwellers_chart(summary.get("dwell_per_identity", []), n=10)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No per-identity dwell records.")

    with st.expander("Full dwell table", expanded=False):
        raw = summary.get("dwell_per_identity", [])
        if raw:
            rdf = pd.DataFrame(raw)
            if "zone" in rdf.columns:
                rdf["zone"] = rdf["zone"].map(_pretty)
            st.dataframe(rdf, use_container_width=True, hide_index=True)
        else:
            st.caption("Empty.")

# Scene: video + optional stills (not three stacked full-bleed images)
with tab_scene:
    video = summary["exports"].get("video", "")
    if video and Path(video).exists():
        st.markdown("##### Annotated video")
        st.video(video)
        st.markdown("##### Sample frames")
        frames = sample_frames(video, n=3)
        if frames:
            fcols = st.columns(len(frames), gap="medium")
            for col, frame in zip(fcols, frames, strict=True):
                col.image(frame, use_container_width=True)
    else:
        st.warning("Annotated video not found — run analytics first.")

# Exports
with tab_export:
    st.markdown("##### Download results")
    st.caption("Files written under `outputs/analytics/` when you run the pipeline.")
    e1, e2, e3, e4 = st.columns(4, gap="medium")
    with e1:
        _download("Annotated video", summary["exports"].get("video", ""), "video/mp4")
    with e2:
        _download("Heatmap PNG", summary["exports"].get("heatmap", ""), "image/png")
    with e3:
        _download("Events CSV", summary["exports"].get("events_csv", ""), "text/csv")
    with e4:
        stem = Path(summary.get("source", "scene")).stem
        _download(
            "Summary JSON",
            f"outputs/analytics/{stem}_summary.json",
            "application/json",
        )

    st.markdown("---")
    st.markdown("##### Scene metadata")
    meta = {
        "Source": summary.get("source"),
        "Frames": summary.get("frames"),
        "Duration (s)": summary.get("duration_s"),
        "Unique visitors": summary.get("unique_visitors"),
        "Peak occupancy": summary.get("occupancy", {}).get("peak"),
        "Mean occupancy": summary.get("occupancy", {}).get("mean"),
    }
    st.json(meta)
