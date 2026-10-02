# Data sources & licenses

Video/label binaries are **gitignored** (large). Regenerate them with:

```bash
people-analytics fetch-samples     # -> data/samples/
people-analytics fetch-eval        # -> data/eval/MOT20-01/
```

## Sample clips (`data/samples/`)

Retail/venue footage from **Roboflow Supervision** public sample assets, chosen to
span the camera angles the project targets:

| File | Scene / angle | Resolution | fps | Frames |
|---|---|---|---|---|
| `grocery-store.mp4` | Eye-level retail aisle | 3840×2160 | 29.97 | 1002 |
| `market-square.mp4` | Elevated, crowded venue | 2160×3840 | 60 | 474 |
| `people-walking.mp4` | Top-down / overhead pedestrians | 1920×1080 | 25 | 341 |

- **Source:** `supervision.assets.VideoAssets`, hosted at `media.roboflow.com`.
- **Use:** provided by Roboflow as public demo/sample media. Used here for
  research/demo purposes; consult Roboflow's terms before any commercial use.
- **Integrity note:** Supervision pins md5 hashes that can go stale against the
  live host, so `fetch-samples` validates each clip by **decoding it** (size +
  first-frame read) rather than by that hash.

## Evaluation sequence (`data/eval/MOT20-01/`)

Ground-truth person boxes **and track IDs** used as the IDSW reference — this is
why no hand-labeling is needed.

- **Dataset:** MOT20 (MOTChallenge), sequence **MOT20-01** — 214 frames @ 25 fps
  (~8.5 s), 1920×1080, 70 unique track IDs, 10,810 annotated person boxes.
- **Layout (MOTChallenge / TrackEval-compatible):**
  `img1/` frames, `gt/gt.txt`, `det/det.txt`, `seqinfo.ini`.
- **Source:** pulled from the HuggingFace mirror
  [`Lekim89/MOT20`](https://huggingface.co/datasets/Lekim89/MOT20) (split
  `ablation`), because the official host `motchallenge.net` is unreachable from
  this environment. Content is identical MOT20 data.
- **License:** MOT20 is released under **CC BY-NC-SA 3.0** (non-commercial).
  Cite: P. Dendorfer et al., *"MOT20: A benchmark for multi object tracking in
  crowded scenes,"* arXiv:2003.09003, 2020.

`gt.txt` columns: `frame, track_id, bb_left, bb_top, bb_width, bb_height, conf,
class, visibility`. For MOT20, `class == 1` is pedestrian and `conf` is a 0/1
"consider" flag; the loader (`people_analytics.data.mot`) filters on both.

## MEVA (`data/meva/`)

Video: MEVA dataset (mevadata.org), CC BY 4.0

- **Dataset:** Multiview Extended Video with Activities (MEVA), Known Facility 1.
  Ground-camera clips (5 min, 30 fps, up to 1920×1080) from 29 cameras at the
  Muscatatuck Urban Training Complex. All actors signed consent forms.
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — commercial
  use allowed with credit. Cite: K. Corona et al., *"MEVA: A Large-Scale Multiview,
  Multimodal Video Dataset for Activity Detection,"* WACV 2021.
- **Layout:**
  - `clips/` — shortlisted `.avi` clips from `s3://mevadata-public-01/drops-123-r13/`
    (anonymous HTTPS; `people-analytics meva-fetch`). The shortlist is
    [`configs/meva_candidates.yaml`](../configs/meva_candidates.yaml).
  - `meva-data-repo/` — `git clone --depth 1 https://gitlab.kitware.com/meva/meva-data-repo`
    (KPF activity/box annotations, camera models). The third-party `contrib/`
    annotations were deleted locally to save ~3.7 GB; they are not used.
  - `kf1-metadata/` — `meva-kf1-metadata.zip` from data.kitware.com (camera list
    CSV, site-map PDF, camera datasheets).

```bash
people-analytics meva-rank     # rank all annotated clips -> outputs/meva/clip_stats.csv
people-analytics meva-fetch    # download the shortlist  -> data/meva/clips/
people-analytics meva-sheets   # contact sheet per role  -> outputs/meva/contact_<role>.jpg
```
