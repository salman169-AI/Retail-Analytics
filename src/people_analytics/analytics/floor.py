"""Image -> floor-plane mappings (metres) for the site-map heatmap.

Three ways to get a camera's floor homography, from best to roughest:

- **krtd**: a calibrated camera model (MEVA ships KRTD files for the two gym
  cameras, in a shared frame with z=0 on the floor and the origin at centre
  court). Exact up to the calibration; lens distortion is removed first.
- **tiles**: a square-tile floor. Grout lines are detected, grouped into two
  families, numbered after perspective is removed (where neighbouring lines are
  evenly spaced), and every grid intersection feeds the fit. The tile size sets
  the scale.
- **mats**: two identical rectangular mats side by side (an entrance). The mats
  give the floor's shape except for the depth scale, which a rectangle can't fix;
  that comes from the camera itself: with the principal point at the image centre,
  the along-wall and into-room directions must be perpendicular, which fixes the
  focal length and hence the true depth. The mat width sets the scale.

`tiles` and `mats` treat the camera as a pinhole, so accuracy falls off towards
the image edges. Borrowing the calibrated lens of G638 (the same 1920x1072
Reolink model as G420/G421) was tried and made both fits worse: café tile-grid
inliers fell from 40 to 28, and the focal length the entrance fit implies moved
from 1484 px (matching G638's calibrated 1479/1489) to 1361. These lenses
evidently differ, so no distortion is removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FloorProjector:
    """Maps image points (feet) to floor coordinates in metres."""

    H: np.ndarray  # undistorted image -> floor metres
    K: np.ndarray | None = None
    D: np.ndarray | None = None

    def __call__(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.empty((0, 2))
        if self.K is not None and self.D is not None:
            pts = cv2.undistortPoints(pts, self.K, self.D, P=self.K)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)


def load_krtd(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """K (3x3), R (3x3), t (3,), D (5,) from a KRTD text file."""
    v = [float(x) for x in Path(path).read_text(encoding="utf-8").split()]
    K = np.array(v[:9]).reshape(3, 3)
    R = np.array(v[9:18]).reshape(3, 3)
    t = np.array(v[18:21])
    D = np.array(v[21:])
    return K, R, t, np.pad(D, (0, max(0, 5 - len(D))))


def krtd_projector(path: str | Path) -> FloorProjector:
    """Floor (z=0) projector from a calibrated camera: H = (K [r1 r2 t])^-1."""
    K, R, t, D = load_krtd(path)
    floor_to_image = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return FloorProjector(np.linalg.inv(floor_to_image), K, D)


def right_handed(H: np.ndarray, image_wh: tuple[int, int] = (1920, 1080)) -> np.ndarray:
    """Flip the floor x axis if `H` maps the image to a mirrored floor frame.

    Fitting a homography to tile or mat corners fixes the floor axes only up to
    a reflection. Viewed from above with y forward (into the scene), image-left
    must land on floor-left, as it does for a real calibrated camera; otherwise
    the room would come out mirrored on the site plan.
    """
    w, h = image_wh
    # left, right (same depth), then far and near (same column)
    pts = np.array([[[0.4 * w, 0.75 * h], [0.6 * w, 0.75 * h],
                     [0.5 * w, 0.6 * h], [0.5 * w, 0.9 * h]]])
    a, b, far, near = cv2.perspectiveTransform(pts, H)[0]
    fwd, right = far - near, b - a
    if fwd[0] * right[1] - fwd[1] * right[0] > 0:
        H = np.diag([-1.0, 1.0, 1.0]) @ H
    return H


def median_background(video: str | Path, n: int = 40) -> np.ndarray:
    """Per-pixel median of `n` frames spread over the clip: the scene without people."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for f in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


# --- tiles ---------------------------------------------------------------------


def _grout_segments(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Line segments [x1, y1, x2, y2, angle_deg] along dark grout lines inside `roi`."""
    x0, y0, x1, y1 = roi
    gray = cv2.GaussianBlur(cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (3, 3), 0)
    # Grout is a thin dark line on bright tiles; black-hat isolates exactly that.
    blackhat = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )
    _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=50, minLineLength=60, maxLineGap=8)
    if segs is None:
        return np.empty((0, 5))
    segs = segs[:, 0, :].astype(float) + [x0, y0, x0, y0]
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    return np.column_stack([segs, ang])


def _merge_collinear(segs: np.ndarray, tol_px: float = 6.0) -> list[np.ndarray]:
    """Group near-collinear segments of one family into lines (homogeneous 3-vectors)."""
    theta = np.radians(np.median(segs[:, 4]))
    normal = np.array([-np.sin(theta), np.cos(theta)])
    offsets = ((segs[:, :2] + segs[:, 2:4]) / 2) @ normal
    order = np.argsort(offsets)
    groups, current = [], [order[0]]
    for a, b in zip(order, order[1:], strict=False):
        if offsets[b] - offsets[a] < tol_px:
            current.append(b)
        else:
            groups.append(current)
            current = [b]
    groups.append(current)
    lines = []
    for g in groups:
        length = np.hypot(segs[g, 2] - segs[g, 0], segs[g, 3] - segs[g, 1]).sum()
        if length < 120:  # stray texture, not a grout line
            continue
        pts = np.vstack([segs[g, :2], segs[g, 2:4]])
        centre = pts.mean(axis=0)
        direction = np.linalg.svd(pts - centre)[2][0]
        n = np.array([-direction[1], direction[0]])
        lines.append(np.array([n[0], n[1], -n @ centre]))
    return lines


def _vanishing_point(lines: list[np.ndarray]) -> np.ndarray:
    L = np.array([line / np.linalg.norm(line[:2]) for line in lines])
    return np.linalg.svd(L)[2][-1]


def _line_indices(lines: list[np.ndarray], affine_rectify: np.ndarray) -> np.ndarray:
    """Integer position of each line in its family, counted in whole tiles.

    After removing perspective the family is parallel and evenly spaced, so a
    missed grout line simply shows up as a two-tile gap.
    """
    inv_t = np.linalg.inv(affine_rectify).T  # lines map with the inverse transpose
    L = np.array([inv_t @ line for line in lines])
    L = L / np.linalg.norm(L[:, :2], axis=1)[:, None]
    L[L[:, 0] < 0] *= -1
    offsets = -L[:, 2]
    gaps = np.diff(np.sort(offsets))
    gaps = gaps[gaps > 1e-9]
    unit = np.median(gaps[gaps <= 1.5 * gaps.min()])
    unit = gaps.sum() / np.round(gaps / unit).sum()
    return np.round((offsets - offsets.min()) / unit).astype(int)


def tile_homography(
    background: np.ndarray,
    roi: tuple[int, int, int, int],
    line_angles: tuple[tuple[float, float], tuple[float, float]],
    tile_m: float,
) -> tuple[FloorProjector, dict]:
    """Floor projector from a square-tile floor seen in `roi` of `background`.

    `line_angles` gives the image-angle range (degrees, 0-180) of each grout-line
    family. Returns the projector and fit diagnostics.
    """
    segs = _grout_segments(background, roi)
    families = [
        _merge_collinear(segs[(segs[:, 4] >= lo) & (segs[:, 4] <= hi)]) for lo, hi in line_angles
    ]
    horizon = np.cross(_vanishing_point(families[0]), _vanishing_point(families[1]))
    affine_rectify = np.array([[1, 0, 0], [0, 1, 0], horizon / horizon[2]])
    idx = [_line_indices(f, affine_rectify) for f in families]

    x0, y0, x1, y1 = roi
    src, dst = [], []
    for line_a, i in zip(families[0], idx[0], strict=True):
        for line_b, j in zip(families[1], idx[1], strict=True):
            p = np.cross(line_a, line_b)
            p = p[:2] / p[2]
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                src.append(p)
                dst.append([j, i])
    src, dst = np.array(src), np.array(dst, dtype=float)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 0.15)
    keep = inliers.ravel() == 1
    err = np.hypot(*(cv2.perspectiveTransform(src[None], H)[0] - dst).T)[keep]
    H = right_handed(np.diag([tile_m, tile_m, 1.0]) @ H, background.shape[1::-1])
    return FloorProjector(H / H[2, 2]), {
        "lines": [len(f) for f in families],
        "grid_points": len(src),
        "inliers": int(keep.sum()),
        "median_error_tiles": round(float(np.median(err)), 4),
    }


# --- mats ----------------------------------------------------------------------


def mats_homography(
    corners: np.ndarray,
    mat_width_m: float,
    principal_point: tuple[float, float],
) -> tuple[FloorProjector, dict]:
    """Floor projector from two identical mats side by side along a wall.

    `corners` is 8x2: each mat as [wall-left, wall-right, room-right, room-left].
    The floor frame has x along the wall and y into the room, in metres, with the
    origin at the first mat's wall-left corner (x flipped if needed to keep the
    frame right-handed, see `right_handed`).
    """
    from scipy.optimize import minimize

    corners = np.asarray(corners, dtype=float)

    def world(depth: float, gap: float) -> np.ndarray:
        return np.array([[0, 0], [1, 0], [1, depth], [0, depth],
                         [1 + gap, 0], [2 + gap, 0], [2 + gap, depth], [1 + gap, depth]])

    def reprojection(p: np.ndarray) -> float:
        if p[0] <= 0.05:
            return 1e9
        to_image, _ = cv2.findHomography(world(*p), corners, 0)
        proj = cv2.perspectiveTransform(world(*p)[None], to_image)[0]
        return float(np.mean(np.sum((proj - corners) ** 2, axis=1)))

    # Any depth scale fits equally well (a depth stretch keeps rectangles
    # rectangular); fit the gap with a nominal depth, then fix depth below.
    fit = minimize(reprojection, [0.5, 0.1], method="Nelder-Mead")
    to_image, _ = cv2.findHomography(world(*fit.x), corners, 0)

    c = np.asarray(principal_point, dtype=float)
    v1 = to_image @ [1, 0, 0]
    v2 = to_image @ [0, 1, 0]
    v1, v2 = v1[:2] / v1[2], v2[:2] / v2[2]
    focal = float(np.sqrt(-np.dot(v1 - c, v2 - c)))  # perpendicular directions
    K = np.array([[focal, 0, c[0]], [0, focal, c[1]], [0, 0, 1]])
    M = np.linalg.inv(K) @ to_image
    depth_scale = np.linalg.norm(M[:, 1]) / np.linalg.norm(M[:, 0])

    H = np.diag([mat_width_m, mat_width_m * depth_scale, 1.0]) @ np.linalg.inv(to_image)
    H = right_handed(H / H[2, 2], (int(2 * c[0] + 1), int(2 * c[1] + 1)))
    return FloorProjector(H), {
        "rms_px": round(float(np.sqrt(fit.fun)), 2),
        "focal_px": round(focal, 1),
        "mat_depth_over_width": round(float(fit.x[0] * depth_scale), 3),
        "gap_m": round(float(fit.x[1] * mat_width_m), 3),
    }
