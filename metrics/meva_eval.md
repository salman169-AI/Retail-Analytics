# MEVA evaluation (phase 4)

Every number below comes from a run in this repo; the JSON next to each section
holds the full output. All three test clips are the scenes' **demo clips**, which
were not used for tuning (tuning used a different clip from the same camera, see
`metrics/meva_tuning.md`).

Video: MEVA dataset (mevadata.org), CC BY 4.0

## 1. Door counts vs MEVA's enter/exit labels

Clip `2018-03-09.10-40-01.10-45-01.school.G420` (east doors, 5 min).
`people-analytics meva-eval-doors` → `metrics/meva_doors_eval.json`.

MEVA's `person_enters/exits_scene_through_structure` labels on this camera all
pass through the double doors (checked by drawing every labelled path against the
door lines: `outputs/meva/phase4_label_paths.jpg`), so each label is one true
crossing. A counted crossing is *matched* if it falls inside a label's time span
(±2 s) in the same direction; each label matches at most once.

| Direction | MEVA labels | Counted | Matched | Missed | False counts |
|---|---:|---:|---:|---:|---:|
| IN | 13 | 14 | 13 | 0 | 1 |
| OUT | 59 | 51 | 50 | 9 | 1 |
| **Total** | **72** | **65** | **63** | **9** | **2** |

63 of 72 labelled crossings found (87.5%); 2 of 65 counts were false (3%).

## 2. Identity switches through occlusion

`people-analytics meva-idsw` → `metrics/meva_{cafe,entrance}_idsw.json`. Full
5-minute clips; switches counted (motmetrics) only on people MEVA annotates, so
unannotated people can't inflate them. Same detections for both trackers.

| Clip | Tracker | Recall | ID switches | IDs created |
|---|---|---:|---:|---:|
| Café G421 (busy, groups queueing) | **BoT-SORT + LightMBN + re-entry gallery** | 0.724 | **107** | 58 |
| | ByteTrack (motion only) | 0.745 | 162 | 313 |
| East doors G420 | **BoT-SORT + LightMBN + re-entry gallery** | 0.720 | **53** | 93 |
| | ByteTrack (motion only) | 0.718 | 62 | 179 |

**34% fewer ID switches in the café, 15% fewer at the doors**, at the same recall
(within 2 points), with half as many fragmented identities.

The plan called for counting switches by hand on one busy clip. This uses MEVA's
own person tracks instead: reproducible, but limited to the people MEVA annotates
(those taking part in a labelled activity) and matched at IoU ≥ 0.3 because MEVA
boxes are loose.

## 3. Dwell-time error

15 café visits drawn at random (seeded) from the run's event log, spread over the
queue and the three areas, then timed by hand from frame strips: the true arrival
and departure of the *person*, even when the tracker broke the visit up.
Inputs: `metrics/meva_dwell_sample.csv`, `metrics/meva_dwell_truth.csv` (with a note
per visit); result: `metrics/meva_dwell_eval.json`.

| | |
|---|---:|
| Visits | 15 |
| Median absolute error | **9.3 s** |
| Mean absolute error | 65.8 s |
| Within 2 s | 7 of 15 |

The two numbers tell different stories, and both are true:

- **Visits one track covered from start to end are timed well.** 7 of the 15
  (true length 2.5–103 s) are within 0.6 s.
- **Long stays are badly under-timed.** People queueing or seated for minutes get
  hidden behind others, and their visit breaks into pieces, often under different
  ids. The log then shows several short visits instead of one long one (worst:
  a whole-clip sofa sitter logged as 5 s; a 156 s queue wait logged as 15 s).

Merging pieces of the same id with a longer gap does not fix it (1 s → 3 s gap:
mean 65.8 → 64.3 s, median 9.3 → 9.5 s; `metrics/meva_dwell_gap_sensitivity.json`,
a post-hoc check), because most breaks change id. So **dwell figures for long
stays in a crowd should not be presented as accurate**; the wait estimate shown on
screen (median of recent queue visits) inherits the same downward bias.

Caveats: 2 of the 15 visits were re-drawn because the first draw repeated the same
people; 2 visits are cut by the clip edge (true time censored at 0/300 s).
