# MEVA floor calibration (phase 3)

How each camera's image is mapped to its room floor in metres for the site map
(`analytics/floor.py`, `configs/meva_sitemap.yaml`). All figures below come from
runs in this repo (`people-analytics sitemap` → `outputs/meva/sitemap/site_stats.json`,
plus the checks named).

| Camera | Room | Method | Check |
|---|---|---|---|
| G330, G299 | Gym | MEVA KRTD camera models (shared gym frame, z=0 floor, origin centre court), lens distortion removed | Top-down warps of both cameras independently show the centre circle at the origin and the half-court line at x=0, and the two views line up (`outputs/meva/phase3/top_gym_*.jpg`). |
| G421 | Café | Square-tile grid: grout lines detected, intersections fitted | 63 grid points, 40 inliers, median error 0.012 tiles. The 6 ft glass double door measures 6.09 tiles → 12-inch (0.3048 m) tiles. |
| G420 | Entrance lobby | Two identical mats; depth scale from perpendicular floor directions with the principal point at the image centre | Reprojection 11.7 px RMS. Implied focal length 1484 px vs 1479/1489 px calibrated for G638, the same Reolink model. Mat depth/width 0.595 (a 4×6 ft mat is 0.667). Left 6 ft (1.83 m) double door measures 1.80 m; the right pair 1.52 m (its corners were read less precisely). |

## Limits

- **No floor plan exists for the school.** Each room's floor is measured, but where
  the rooms sit relative to each other follows the MEVA KF1 site map (café north of
  the gym, east doors beside it) and is approximate. The plan says so on its face.
- **Café and lobby ignore lens distortion.** These Reolink cameras have no model of
  their own. Borrowing G638's calibrated lens was tried and made both fits worse
  (café tile inliers 40 → 28; the entrance's implied focal length moved from 1484 to
  1361 px), so it was dropped. Café positions are most reliable near the tiles
  that were fitted (right half of the image) and rougher towards the lounge.
- **Feet off the floor plane** (people sitting on gym bleachers) map to the nearest
  floor spot along the camera ray, i.e. roughly the right place, not exact.
- **Boxes cut by the bottom of the frame are dropped** from the map: their bottom
  edge is the frame border, not the feet.

## Tests

`tests/test_floor.py` checks each method on a synthetic camera with known answers:
the KRTD round trip is exact; the tile fit recovers floor distances within 5% on a
rendered tiled floor; the mats fit recovers the camera's focal length within 2% and
the mat depth within 3 cm.
