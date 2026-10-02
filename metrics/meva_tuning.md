# MEVA tracker tuning (phase 2)

Generated from `metrics/meva_<scene>_tuning.json` (`people-analytics meva-tune`).

Each scene is tuned on a **different clip from the same camera** than the one used
for the demo and the phase-4 evaluation, on its busiest 120 s by annotation count.
MEVA only boxes people inside an annotated activity, so precision can't be
measured; the two sound numbers are:

- **recall**: share of annotated person boxes covered by our tracks (IoU >= 0.3;
  MEVA boxes are loose).
- **IDSW**: identity switches on annotated people (motmetrics; unannotated people
  can't inflate it).

`ids` is the number of track ids we produced: very few ids for a busy scene means
the re-entry gallery is merging different people. Numbers are for tuning only,
not for the README.

## entrance — tuning clip `2018-03-13.16-30-01.16-35-01.school.G420`, frames 5401–9001

| detector | variant | recall ↑ | IDSW ↓ | ids | ids < 1 s | tracker FPS |
|---|---|---:|---:|---:|---:|---:|
| full | scene config: BoT-SORT + LightMBN + gallery 0.55, conf 0.25, interp on, CMC off | 0.734 | 10 | 47 | 0 | 8.4 |
| full | ByteTrack (motion only) | 0.742 | 24 | 88 | 19 | 93.6 |
| full | gallery off | 0.734 | 8 | 68 | 5 | 14.3 |
| full | Re-ID OSNet-AIN instead of LightMBN | 0.734 | 10 | 16 | 0 | 15.3 |
| full | track_buffer 150 | 0.734 | 10 | 47 | 1 | 8.7 |
| full | detector conf 0.1 | 0.735 | 11 | 51 | 0 | 9.0 |
| full | detector conf 0.4 | 0.736 | 14 | 48 | 1 | 9.0 |
| full | interpolation off | 0.727 | 8 | 47 | 1 | 9.0 |
| full | camera-motion compensation on | 0.734 | 10 | 47 | 0 | 8.1 |
| full | BoT-SORT track_high/new_track 0.4 | 0.758 | 11 | 53 | 2 | 9.0 |
| full | BoT-SORT track_high/new_track 0.3 | 0.775 | 21 | 62 | 6 | 8.9 |
| full | gallery cosine 0.65 **(chosen)** | 0.734 | 8 | 62 | 2 | 9.0 |
| full | BoT-SORT track_high/new_track 0.4 + gallery cosine 0.65 | 0.757 | 11 | 69 | 6 | 9.2 |

## cafe — tuning clip `2018-03-07.16-50-00.16-55-00.school.G421`, frames 3281–6881

| detector | variant | recall ↑ | IDSW ↓ | ids | ids < 1 s | tracker FPS |
|---|---|---:|---:|---:|---:|---:|
| full | scene config: BoT-SORT + LightMBN + gallery 0.55, conf 0.25, interp on, CMC off | 0.855 | 44 | 21 | 2 | 8.2 |
| full | ByteTrack (motion only) | 0.859 | 62 | 126 | 48 | 142.2 |
| full | gallery off | 0.854 | 48 | 63 | 18 | 15.5 |
| full | Re-ID OSNet-AIN instead of LightMBN | 0.855 | 43 | 18 | 1 | 15.4 |
| full | track_buffer 150 | 0.855 | 44 | 21 | 1 | 8.2 |
| full | detector conf 0.1 | 0.865 | 43 | 20 | 2 | 8.2 |
| full | detector conf 0.4 | 0.843 | 45 | 20 | 1 | 8.3 |
| full | interpolation off | 0.846 | 56 | 21 | 2 | 8.3 |
| full | camera-motion compensation on | 0.855 | 44 | 21 | 2 | 7.7 |
| full | BoT-SORT track_high/new_track 0.4 | 0.870 | 47 | 31 | 5 | 8.2 |
| full | BoT-SORT track_high/new_track 0.3 | 0.876 | 64 | 34 | 2 | 8.2 |
| full | gallery cosine 0.65 **(chosen)** | 0.855 | 44 | 31 | 4 | 8.3 |
| full | BoT-SORT track_high/new_track 0.4 + gallery cosine 0.65 | 0.870 | 47 | 42 | 6 | 8.3 |

## gym — tuning clip `2018-03-11.11-50-00.11-55-00.school.G330`, frames 7–3607

| detector | variant | recall ↑ | IDSW ↓ | ids | ids < 1 s | tracker FPS |
|---|---|---:|---:|---:|---:|---:|
| full | scene config: BoT-SORT + LightMBN + gallery 0.55, conf 0.25, interp on, CMC off | 0.172 | 110 | 18 | 0 | 8.2 |
| full | ByteTrack (motion only) | 0.207 | 128 | 208 | 64 | 129.6 |
| full | gallery off | 0.171 | 66 | 113 | 20 | 15.4 |
| full | BoT-SORT track_high/new_track 0.4 | 0.225 | 303 | 21 | 0 | 8.1 |
| full | gallery cosine 0.65 | 0.171 | 85 | 20 | 0 | 8.3 |
| full | gallery off + BoT-SORT track_high/new_track 0.4 | 0.225 | 134 | 200 | 68 | 15.0 |
| tiled | gallery off | 0.243 | 225 | 305 | 90 | 14.6 |
| tiled | gallery off + BoT-SORT track_high/new_track 0.4 **(chosen)** | 0.339 | 356 | 438 | 160 | 14.1 |

## Decisions

- **CMC off everywhere.** Fixed cameras: identical accuracy with it on, and it only
  costs time.
- **LightMBN, not OSNet-AIN.** With OSNet-AIN the gallery collapsed the entrance to
  16 ids in two busy minutes (different people merged).
- **Gallery threshold 0.65 instead of 0.55** at the entrance and café: never more
  identity switches than 0.55 and much less merging. **Gallery off in the gym**: in
  a dense crowd it hands newcomers a neighbour's id (66 IDSW off vs 85–110 on).
- **Detector conf 0.25, track_buffer 60, interpolation on**: conf 0.1/0.4 and a longer
  buffer changed little; switching interpolation off cost 12 IDSW in the café.
- **Gym: tiled detection + tracker gates 0.4.** The tuning clip is a packed crowd
  round a table; untiled recall was 0.17. Tiling + lower gates doubles it (0.34) at
  the cost of fragmented tracks, which the heatmap tolerates. Extra tiled boxes
  were checked by eye (outputs/meva/phase2/gym_tiled_check.jpg).
- **Entrance gates stay at 0.5/0.6**: 0.4 raised recall (0.734 → 0.757) but gave no
  better door counts on the tuning minute (28 IN vs 29 IN, MEVA labels 34).

## Door counter guards (entrance)

Replayed the tracks of frames 5400–7200 of the tuning clip (tracked with the
gallery at its old 0.55 threshold) through the door counter. MEVA labels 34
entries and 0 exits there; 4 of the entries finish after the window:

| settle | cooldown | forget | IN | OUT |
|---:|---:|---:|---:|---:|
| 1 | 0 | 900 | 32 | 8 |
| 1 | 30 | 900 | 29 | 5 |
| 3 | 30 | 900 | 28 | 5 |
| 1 | 30 | 30 | 29 | 1 |
| **3** | **30** | **30** | **28** | **1** |

False OUTs came from (a) legs hidden in a doorway crowd making the box bottom jump
back over both lines, and (b) the gallery giving a newcomer the id of someone who
went in earlier. With the chosen tracker (gallery 0.65) a fresh run of that minute
gave 29 IN / 0 OUT.
