"""Keyboard calibration: maps on-screen pixel coordinates to MIDI pitch.

The calibration is a one-time-per-camera-setup manual step: the user (or a
saved ``calibration.json``) supplies the four corners of the playable key
area in the reference video frame, plus the MIDI pitch of the leftmost and
rightmost key visible. We build a homography from screen space to a
normalized keyboard-space rectangle ``[0,1] x [0,1]`` (x = left->right along
the keys, y = near edge -> far edge / front of keys -> back), then map the
normalized x coordinate to a white-key index and from there to a MIDI pitch.

No cv2 / numpy-heavy perspective libraries are required for the core
homography math: a general 2D projective (planar) homography from 4 point
correspondences is solved directly with numpy linear algebra, so this module
only needs numpy (cv2 is only used, lazily, for optional video-frame
extraction helpers).

White-key layout, one octave (12 semitones, 7 white keys):
    pitch class:  0  1  2  3  4  5  6  7  8  9  10 11
    name:         C  C# D  D# E  F  F# G  G# A  A# B
    white index:  0  -  1  -  2  3  -  4  -  5  -  6

MIDI pitch 21 = A0 (lowest key on 88-key piano), 108 = C8 (highest).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# Pitch classes (0..11) that are white keys, and their index within the
# 7-white-keys-per-octave pattern.
_WHITE_PITCH_CLASSES = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B
_WHITE_INDEX_BY_PITCH_CLASS = {pc: i for i, pc in enumerate(_WHITE_PITCH_CLASSES)}
_IS_WHITE = {0: True, 1: False, 2: True, 3: False, 4: True, 5: True,
             6: False, 7: True, 8: False, 9: True, 10: False, 11: True}


def pitch_to_white_index(pitch: int) -> float:
    """Continuous white-key coordinate for a MIDI pitch.

    White keys map to integers representing each white key's LEFT edge
    (C=0, D=1, E=2, ...) -- so white key C itself spans the interval [0, 1),
    centered at 0.5. Black keys map to a fractional position within that
    interval, biased toward the gap/seam with the NEXT white key (not the
    arithmetic midpoint of the two neighbors' left edges, which would place a
    black key at the CENTER of the lower neighbor's own body instead of in
    the notch between the two keys where it physically sits). The mapping
    stays monotonic and collision-free (the bias factor is strictly between
    0 and 1, so a black key's index never ties with either neighboring white
    key's own index), which `white_index_to_pitch`'s nearest-match search
    depends on.
    """
    octave = pitch // 12
    pc = pitch % 12
    base = octave * 7
    if _IS_WHITE[pc]:
        return base + _WHITE_INDEX_BY_PITCH_CLASS[pc]
    # Black key: `lo` is the left edge of the white key it visually overlaps
    # most (e.g. C for C#), `hi` is the left edge of the next white key (D).
    # Real black keys sit mostly in the gap toward `hi`, not centered on `lo`'s
    # own body -- bias 0.7 of the way from `lo` to `hi` approximates this
    # (a plain 0.5 midpoint, the previous formula, put the marker in the
    # middle of the LOWER white key instead of between the two keys).
    lo_pc = pc - 1
    hi_pc = pc + 1
    lo = base + _WHITE_INDEX_BY_PITCH_CLASS[lo_pc]
    hi = base + _WHITE_INDEX_BY_PITCH_CLASS[hi_pc]
    return lo + 0.7 * (hi - lo)


def white_index_to_pitch(white_index: float) -> int:
    """Inverse of :func:`pitch_to_white_index`: nearest MIDI pitch for a
    continuous white-key coordinate. Used to map a screen position back to
    the nearest pitch (e.g. for preview rendering / debugging)."""
    best_pitch = 0
    best_dist = float("inf")
    lo = int(np.floor(white_index)) * 12 // 7 - 12
    hi = int(np.ceil(white_index)) * 12 // 7 + 12
    for p in range(max(0, lo), min(127, hi) + 1):
        d = abs(pitch_to_white_index(p) - white_index)
        if d < best_dist:
            best_dist = d
            best_pitch = p
    return best_pitch


def undistort_frame(frame, k1: float):
    """Applies a single-parameter approximate radial (barrel/fisheye)
    distortion correction. Uses an approximate camera matrix (fx=fy=max(w,h),
    principal point at image center) since true camera intrinsics aren't
    known -- k1 is meant to be tuned interactively by eye until real straight
    edges (e.g. key-boundary lines) appear straight. k1=0.0 is a no-op
    (returns `frame` unchanged, so existing calibrations/behavior are
    unaffected by default). Uses `cv2.getOptimalNewCameraMatrix(..., alpha=1.0)`
    to compute the output camera matrix, which preserves the FULL original
    field of view (adds black borders rather than cropping/losing corners),
    so all 4 calibration corners stay reachable/clickable regardless of k1
    magnitude."""
    if k1 == 0.0:
        return frame
    import cv2  # lazy import

    h, w = frame.shape[:2]
    f = max(w, h)
    cx, cy = w / 2, h / 2
    camera_matrix = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
    new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1.0
    )
    return cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)


def _projection_error(k1: float, corners, reference_point, reference_pitch: int,
                       low_pitch: int, high_pitch: int, frame_w: int, frame_h: int) -> float:
    """For a candidate k1, undistorts the 4 corners + 1 reference-key click
    (as pixel POINTS, via cv2.undistortPoints -- no image re-rendering needed),
    builds a Calibration from the undistorted corners, and returns how far off
    (in normalized keyboard-x units, 0..1 across the full low/high pitch span)
    the undistorted reference point's projected position is from where
    `reference_pitch` should actually sit. Used as the objective function for
    the 1D k1 search in `_solve_k1_from_reference`."""
    import cv2  # lazy import

    f = max(frame_w, frame_h)
    cx, cy = frame_w / 2, frame_h / 2
    camera_matrix = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
    # cv2.undistortPoints expects points as shape (N, 1, 2) -- N points, each
    # wrapped individually -- NOT (1, N, 2). Passing the wrong convention
    # doesn't raise a shape error; cv2 silently reads it as a single point,
    # which is what caused the original IndexError further down.
    pts = np.asarray(corners + [list(reference_point)], dtype=np.float64).reshape(-1, 1, 2)  # (5, 1, 2)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix).reshape(-1, 2)  # (5, 2)
    undist_corners = undistorted[:4].tolist()
    undist_ref = undistorted[4]
    calib = Calibration(corners=undist_corners, low_pitch=low_pitch, high_pitch=high_pitch)
    u, _v = calib.screen_to_norm(undist_ref)
    expected_u = calib.pitch_to_norm_x(reference_pitch)
    return abs(float(u) - float(expected_u))


def _solve_k1_from_reference(corners, reference_point, reference_pitch: int,
                              low_pitch: int, high_pitch: int, frame_w: int, frame_h: int):
    """Coarse-to-fine 1D grid search (dependency-free -- no scipy) for the k1
    that minimizes `_projection_error`. The k1-vs-error relationship is smooth
    and near-unimodal for the mild-to-moderate barrel/pincushion distortion
    this tool is meant to correct, so a two-pass grid search is sufficient and
    avoids adding scipy as a new dependency. Returns (best_k1, best_error)."""
    best_k1, best_err = 0.0, float("inf")
    for k1 in np.linspace(-0.5, 0.5, 101):  # coarse pass, step ~0.01
        err = _projection_error(float(k1), corners, reference_point, reference_pitch,
                                 low_pitch, high_pitch, frame_w, frame_h)
        if err < best_err:
            best_err, best_k1 = err, float(k1)
    lo, hi = best_k1 - 0.012, best_k1 + 0.012
    for k1 in np.linspace(lo, hi, 49):  # fine pass, step ~0.0005
        err = _projection_error(float(k1), corners, reference_point, reference_pitch,
                                 low_pitch, high_pitch, frame_w, frame_h)
        if err < best_err:
            best_err, best_k1 = err, float(k1)
    return best_k1, best_err


def _solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve a 3x3 projective homography mapping src->dst 2D points, given
    4 OR MORE point correspondences (the classic DLT / direct linear
    transform, solved via SVD). With exactly 4 points this is an exact fit;
    with more than 4 (overdetermined), SVD gives the least-squares
    best-fitting homography over all points. Pure numpy, no cv2 dependency."""
    assert src.shape == dst.shape and src.shape[1] == 2 and src.shape[0] >= 4
    a_rows = []
    for (x, y), (u, v) in zip(src, dst):
        a_rows.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        a_rows.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    a = np.asarray(a_rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return h


def _apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply homography h to Nx2 points, returning Nx2 points."""
    pts = np.atleast_2d(points)
    ones = np.ones((pts.shape[0], 1))
    homog = np.concatenate([pts, ones], axis=1)  # Nx3
    out = homog @ h.T  # Nx3
    out = out[:, :2] / out[:, 2:3]
    return out


def _fit_row_projective(u: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Fit a 1-D projective (rational-linear) map from a normalized
    keyboard-x coordinate ``u`` (0..1 across the low..high pitch span) to
    screen pixels (x, y):

        x(u) = (a*u + b) / (g*u + 1)
        y(u) = (c*u + d) / (g*u + 1)      (shared denominator)

    This is the correct model for the CENTERS of a straight row of keys
    imaged under perspective: the key centers lie on one straight 3-D line,
    which projects to a straight image line with foreshortening captured by
    the shared ``g`` term. Crucially it is well-posed from a *single row* of
    clicked points (unlike a full 2-D homography, whose near<->far axis is
    undetermined when every calibration point sits at the same keyboard
    depth -- the bug this replaces). Needs >=3 points.

    Returns ``[a, b, c, d, g]`` (Python floats) or ``None`` if the fit is
    degenerate / underdetermined.
    """
    u = np.asarray(u, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = u.shape[0]
    if n < 3:
        return None
    # Cross-multiplied (linear in the 5 unknowns) least-squares system:
    #   x*(g*u + 1) = a*u + b  ->  a*u + b       - (u*x)*g = x
    #   y*(g*u + 1) = c*u + d  ->        c*u + d - (u*y)*g = y
    rows = np.zeros((2 * n, 5), dtype=np.float64)
    rhs = np.zeros(2 * n, dtype=np.float64)
    rows[0::2, 0] = u          # a
    rows[0::2, 1] = 1.0        # b
    rows[0::2, 4] = -(u * x)   # g
    rhs[0::2] = x
    rows[1::2, 2] = u          # c
    rows[1::2, 3] = 1.0        # d
    rows[1::2, 4] = -(u * y)   # g
    rhs[1::2] = y
    try:
        coeffs, _res, rank, _sv = np.linalg.lstsq(rows, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 5 or not np.all(np.isfinite(coeffs)):
        return None
    return [float(c) for c in coeffs]


def _row_project(row, u: float):
    """Evaluate the 1-D projective row map at normalized-x ``u`` -> (x, y)."""
    a, b, c, d, g = row
    denom = g * u + 1.0
    if abs(denom) < 1e-9:
        denom = 1e-9 if denom >= 0 else -1e-9
    return (a * u + b) / denom, (c * u + d) / denom


def _row_invert_x(row, x: float) -> float:
    """Invert the row map's x-component: given screen x, return normalized u.
    Solves x = (a*u + b)/(g*u + 1) -> u = (b - x)/(x*g - a)."""
    a, b, _c, _d, g = row
    denom = x * g - a
    if abs(denom) < 1e-9:
        denom = 1e-9 if denom >= 0 else -1e-9
    return (b - x) / denom


def _calib_u_to_screen(calib, u: float):
    """Project a normalized keyboard-x coordinate ``u`` (0..1 across low..high
    pitch) to screen pixels for either calibration mode. Pure row mode places it on
    the calibrated key-center line; depth-aware or corner mode uses the front-of-keys
    depth (v=0.9) where fingers rest. Returns (x, y) floats."""
    if calib.row is not None and calib.row_near is None:
        return _row_project(calib.row, u)
    h_inv = np.linalg.inv(calib._h)
    pt = _apply_homography(h_inv, np.asarray([[u, 0.9]], dtype=np.float64))[0]
    return float(pt[0]), float(pt[1])


def draw_keyboard_overlay(frame, calib, highlight_pitch=None, ghost_pitch=None, height_px: int = 22) -> None:
    """Draw the inferred piano keyboard over ``frame`` (a cv2 BGR image,
    mutated in place): white-key separator ticks, black-key marks, and a note
    label at every C, all positioned by the calibration. Lets the mapping be
    visually verified against the real keys -- in the exported preview video
    and live while calibrating (so points can be dragged until it lines up).

    ``highlight_pitch`` draws a red ring (a hand was detected near this key).
    ``ghost_pitch`` draws a grey ring (MIDI says this note is sounding but no
    hand was detected nearby — ghost note or tracking dropout). Any projection
    failure is swallowed so overlay drawing never crashes the caller."""
    import cv2  # lazy import

    try:
        lo, hi = calib.low_pitch, calib.high_pitch
        lo_wi = pitch_to_white_index(lo)
        hi_wi = pitch_to_white_index(hi)
        span = (hi_wi - lo_wi) if hi_wi != lo_wi else 1.0

        def wi_to_screen(white_index: float):
            u = (white_index - lo_wi) / span
            x, y = _calib_u_to_screen(calib, u)
            return int(round(x)), int(round(y))

        # White-key separators: half-integer white indices between keys.
        first = int(np.floor(lo_wi))
        last = int(np.ceil(hi_wi))
        for k in range(first, last + 1):
            for edge in (k - 0.5, k + 0.5):
                if edge < lo_wi - 0.5 or edge > hi_wi + 0.5:
                    continue
                if calib.row_near is not None:
                    # Depth-aware: draw angled tick from far to near row
                    u = (edge - lo_wi) / span
                    far_pt = _row_project(calib.row, u)
                    near_pt = _row_project(calib.row_near, u)
                    x1, y1 = int(round(far_pt[0])), int(round(far_pt[1]))
                    x2, y2 = int(round(near_pt[0])), int(round(near_pt[1]))
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
                else:
                    # Standard overhead: draw tick at rotation_deg from vertical
                    x, y = wi_to_screen(edge)
                    _rot = math.radians(calib.rotation_deg)
                    _cdx = int(round(height_px * math.sin(_rot)))
                    _cdy = int(round(height_px * math.cos(_rot)))
                    cv2.line(frame, (x - _cdx, y - _cdy), (x + _cdx, y + _cdy), (0, 255, 255), 1, cv2.LINE_AA)

        # Black keys: short thick orange mark at each black key's fractional index,
        # drawn at the same rotation angle as the white-key separators.
        _rot_bk = math.radians(calib.rotation_deg)
        _sin_bk = math.sin(_rot_bk)
        _cos_bk = math.cos(_rot_bk)
        for pitch in range(lo, hi + 1):
            if _IS_WHITE[pitch % 12]:
                continue
            x, y = wi_to_screen(pitch_to_white_index(pitch))
            bk_x1 = int(round(x - height_px * _sin_bk))
            bk_y1 = int(round(y - height_px * _cos_bk))
            bk_x2 = int(round(x - 2 * _sin_bk))
            bk_y2 = int(round(y - 2 * _cos_bk))
            cv2.line(frame, (bk_x1, bk_y1), (bk_x2, bk_y2), (0, 165, 255), 3, cv2.LINE_AA)

        # Note-name label at every C.
        for pitch in range(lo, hi + 1):
            if pitch % 12 != 0:
                continue
            x, y = wi_to_screen(pitch_to_white_index(pitch))
            label = pitch_to_note_name(pitch)
            org = (x - 10, y + height_px + 16)
            cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if highlight_pitch is not None:
            x, y = wi_to_screen(pitch_to_white_index(highlight_pitch))
            cv2.circle(frame, (x, y), 12, (0, 0, 255), 3, cv2.LINE_AA)
        if ghost_pitch is not None:
            x, y = wi_to_screen(pitch_to_white_index(ghost_pitch))
            cv2.circle(frame, (x, y), 12, (160, 160, 160), 2, cv2.LINE_AA)
    except Exception:
        pass


@dataclass
class Calibration:
    """Screen-space corners of the playable key area, in order:
    top-left, top-right, bottom-right, bottom-left (a la OpenCV convention,
    where "top" = far edge of keys / near the music stand, "bottom" = near
    edge / where fingers rest -- but the labeling doesn't matter as long as
    it's consistent), plus the MIDI pitch range shown.
    """

    corners: list  # 4 x [x, y] pixel coordinates, TL, TR, BR, BL
    low_pitch: int
    high_pitch: int
    k1: float = 0.0  # radial (barrel/fisheye) lens-distortion correction coefficient
    # Optional 1-D projective row map [a, b, c, d, g] (see _fit_row_projective).
    # When set, it is the authoritative pixel<->pitch mapping (multi-point /
    # "click the C keys" calibration) and the 4-corner homography is ignored.
    row: list = None
    # Optional second row (front edge of keys) for side-angle cameras. When both
    # row and row_near are set, a 2-D homography is reconstructed from the two
    # rows and stored in _h; this enables depth-aware key tracking.
    row_near: list = None
    _h: np.ndarray = field(default=None, repr=False, compare=False)
    # Rotation of the overlay tick lines from vertical, in degrees (positive = clockwise).
    # Used for angled camera views where the piano is not shot from directly overhead.
    rotation_deg: float = 0.0

    def __post_init__(self) -> None:
        # Case 1: both rows set (side-angle depth calibration) -> build _h from rows
        if self.row is not None and self.row_near is not None:
            src_points = []
            dst_points = []
            # Sample rows at u=0 and u=1 to get 4 corner points
            for u in (0.0, 1.0):
                far_pt = _row_project(self.row, u)
                near_pt = _row_project(self.row_near, u)
                src_points.extend([far_pt, near_pt])
                dst_points.extend([[u, 0], [u, 1]])  # u at far, u at near
            src = np.asarray(src_points, dtype=np.float64)
            dst = np.asarray(dst_points, dtype=np.float64)
            self._h = _solve_homography(src, dst)
        else:
            # Case 2: standard corner-based or single-row homography
            src = np.asarray(self.corners, dtype=np.float64)
            # Normalized keyboard space: x in [0,1] left->right, y in [0,1] far->near
            dst = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
            self._h = _solve_homography(src, dst)

    def screen_to_norm(self, xy: Sequence[float]) -> np.ndarray:
        """Map a screen pixel (x, y) to normalized keyboard space (u, v)."""
        if self.row is not None and self.row_near is None:
            u = _row_invert_x(self.row, float(xy[0]))
            return np.asarray([u, 0.5], dtype=np.float64)
        return _apply_homography(self._h, np.asarray([xy], dtype=np.float64))[0]

    def norm_to_pitch(self, u: float) -> int:
        """Map a normalized keyboard-space x coordinate (0..1) to the
        nearest MIDI pitch in [low_pitch, high_pitch]."""
        lo_wi = pitch_to_white_index(self.low_pitch)
        hi_wi = pitch_to_white_index(self.high_pitch)
        wi = lo_wi + u * (hi_wi - lo_wi)
        pitch = white_index_to_pitch(wi)
        return int(np.clip(pitch, self.low_pitch, self.high_pitch))

    def pitch_to_norm_x(self, pitch: int) -> float:
        """Map a MIDI pitch to its normalized x position (0..1), inverse-ish
        of norm_to_pitch. Used to place the "expected key" marker in preview
        rendering, and to find the screen location for matching."""
        lo_wi = pitch_to_white_index(self.low_pitch)
        hi_wi = pitch_to_white_index(self.high_pitch)
        wi = pitch_to_white_index(pitch)
        if hi_wi == lo_wi:
            return 0.5
        return float((wi - lo_wi) / (hi_wi - lo_wi))

    def pitch_to_screen(self, pitch: int, v: float = 0.9) -> np.ndarray:
        """Map a MIDI pitch (+ a normalized depth v, default near the front
        of the keys where fingers rest) back to screen pixel coordinates.

        In pure row-map mode (no row_near) the marker sits on the clicked
        key-center line, so ``v`` is ignored. In depth-aware mode (row_near
        set), v is used to interpolate between the two rows."""
        u = self.pitch_to_norm_x(pitch)
        if self.row is not None and self.row_near is None:
            x, y = _row_project(self.row, u)
            return np.asarray([x, y], dtype=np.float64)
        h_inv = np.linalg.inv(self._h)
        return _apply_homography(h_inv, np.asarray([[u, v]], dtype=np.float64))[0]

    def screen_to_pitch(self, xy: Sequence[float]) -> int:
        u, _v = self.screen_to_norm(xy)
        return self.norm_to_pitch(float(np.clip(u, 0.0, 1.0)))

    def screen_to_white_index(self, xy: Sequence[float]) -> float:
        """Continuous white-key index (in `pitch_to_white_index` units) for a
        screen pixel, using ONLY the horizontal keyboard coordinate -- i.e.
        depth-invariant. This is the correct space for finger<->key matching:
        a finger identifies a key by its position ALONG the keyboard, not by
        how far down the key's depth it happens to sit. Matching in raw 2-D
        pixels instead penalizes the natural depth spread of fingers and
        collapses confidence for every note."""
        u = float(self.screen_to_norm(xy)[0])
        lo_wi = pitch_to_white_index(self.low_pitch)
        hi_wi = pitch_to_white_index(self.high_pitch)
        return lo_wi + u * (hi_wi - lo_wi)

    def to_dict(self) -> dict:
        d = {
            "corners": [list(map(float, c)) for c in self.corners],
            "low_pitch": self.low_pitch,
            "high_pitch": self.high_pitch,
            "k1": self.k1,
        }
        if self.row is not None:
            d["row"] = [float(c) for c in self.row]
        if self.row_near is not None:
            d["row_near"] = [float(c) for c in self.row_near]
        if self.rotation_deg != 0.0:
            d["rotation_deg"] = self.rotation_deg
        return d

    @staticmethod
    def from_dict(d: dict) -> "Calibration":
        return Calibration(
            corners=d["corners"],
            low_pitch=d["low_pitch"],
            high_pitch=d["high_pitch"],
            k1=d.get("k1", 0.0),
            row=d.get("row"),
            row_near=d.get("row_near"),
            rotation_deg=d.get("rotation_deg", 0.0),
        )


def load_calibration(path: str) -> Calibration:
    import json

    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return Calibration.from_dict(d)


def save_calibration(path: str, calib: Calibration) -> None:
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(calib.to_dict(), f, indent=2)


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_to_note_name(pitch: int) -> str:
    """Standard `C4`/`A0`-style note name for a MIDI pitch (pitch 60 = C4,
    pitch 21 = A0), using the common convention where middle C (60) is in
    octave 4."""
    octave = pitch // 12 - 1
    name = _NOTE_NAMES[pitch % 12]
    return f"{name}{octave}"


def _draw_calibration_overlay(disp, points: list, low_pitch: int, high_pitch: int) -> None:
    """Draw a live "what you get" keyboard overlay on ``disp`` (a cv2 BGR
    image, mutated in place) from the 4 currently-clicked corners.

    Reuses :class:`Calibration`'s own homography (built from ``points`` in
    the exact same TL/TR/BR/BL -> normalized (0,0)/(1,0)/(1,1)/(0,1)
    convention used by the real analysis pipeline) so the overlay is
    guaranteed geometrically consistent with actual note matching. Any
    projection failure (e.g. degenerate/collinear corners) is swallowed --
    the caller still gets the dots + instructions even if the overlay can't
    be drawn this frame.
    """
    import cv2  # lazy import (this module is only reached from interactive_calibrate)

    try:
        calib = Calibration(corners=points, low_pitch=low_pitch, high_pitch=high_pitch)
        h_inv = np.linalg.inv(calib._h)

        def norm_to_screen(u: float, v: float) -> tuple:
            pt = _apply_homography(h_inv, np.asarray([[u, v]], dtype=np.float64))[0]
            return (int(round(pt[0])), int(round(pt[1])))

        # Scope quad: TL -> TR -> BR -> BL -> TL, in a bright color.
        quad = [tuple(map(int, p)) for p in points]
        cv2.polylines(disp, [np.asarray(quad, dtype=np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)

        lo_wi = pitch_to_white_index(low_pitch)
        hi_wi = pitch_to_white_index(high_pitch)

        # Per-white-key boundary lines, from the low white index to the high
        # white index + 1 (so the last key's trailing edge is also drawn).
        first_white = int(np.floor(lo_wi))
        last_white = int(np.ceil(hi_wi)) + 1
        for wi in range(first_white, last_white + 1):
            u = (wi - lo_wi) / (hi_wi - lo_wi) if hi_wi != lo_wi else 0.5
            if u < -0.02 or u > 1.02:
                continue
            top = norm_to_screen(u, 0.0)
            bot = norm_to_screen(u, 1.0)
            cv2.line(disp, top, bot, (0, 255, 255), 1, cv2.LINE_AA)

        # Black keys: bright orange outline (not a dark fill -- a dark fill would
        # be invisible against real black keys, which are already near-black) at
        # each black key's projected fractional position.
        black_w = 0.4  # fraction of one white-key unit, in normalized-x units
        for pitch in range(low_pitch, high_pitch + 1):
            if _IS_WHITE[pitch % 12]:
                continue
            wi = pitch_to_white_index(pitch)
            u = (wi - lo_wi) / (hi_wi - lo_wi) if hi_wi != lo_wi else 0.5
            du = black_w / (hi_wi - lo_wi) / 2.0 if hi_wi != lo_wi else 0.02
            corners_norm = [(u - du, 0.0), (u + du, 0.0), (u + du, 0.62), (u - du, 0.62)]
            quad_pts = np.asarray([norm_to_screen(uu, vv) for uu, vv in corners_norm], dtype=np.int32)
            cv2.polylines(disp, [quad_pts], True, (0, 165, 255), 2, cv2.LINE_AA)

        # Endpoint labels: low_pitch and high_pitch note names, distinct color.
        for pitch, u in ((low_pitch, 0.0), (high_pitch, 1.0)):
            pos = norm_to_screen(u, 0.5)
            label = pitch_to_note_name(pitch)
            cv2.putText(disp, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(disp, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
    except Exception:
        # Never let overlay drawing crash calibration -- dots + instructions
        # (drawn by the caller) are enough to keep going without it.
        pass


@dataclass
class CalibPoint:
    """A single calibration point: pixel coordinates + the white key pitch it
    represents. Used by the multi-point calibration flow in
    `interactive_calibrate`, where the user clicks the center of each white
    key they can clearly see and specifies which key it is."""

    pixel_xy: tuple  # (x, y) in video pixel space
    pitch: int  # MIDI pitch (0..127), must be a white key

    def __post_init__(self) -> None:
        if self.pitch % 12 not in _WHITE_PITCH_CLASSES:
            raise ValueError(f"{self.pitch} is not a white key")


def _show_proceed_dialog(title: str, message: str) -> None:
    """Show a Tkinter dialog with a Proceed button to confirm next step."""
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    dialog = tk.Toplevel(root)
    dialog.title(title)
    dialog.resizable(False, False)

    # Message
    label = tk.Label(dialog, text=message, font=("Arial", 12), padx=20, pady=20)
    label.pack()

    # Proceed button
    def on_proceed():
        dialog.destroy()
        root.destroy()

    btn = tk.Button(dialog, text="✓ Proceed", font=("Arial", 12), padx=20, pady=10, command=on_proceed)
    btn.pack(pady=10)

    dialog.wait_window()


def select_calibration_frame(video_path: str) -> np.ndarray:
    """Let the user scrub through the video to find a good frame for calibration
    (skip black/faded frames at the start). Returns the selected frame as a numpy BGR array."""
    import cv2  # lazy import

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 10000  # fallback for streams

    current_idx = 0
    window = "Frame scrubber: arrow keys to scrub, SPACE to confirm, ESC to cancel"

    MAX_DISPLAY_DIM = 2450  # 1.75x the old 1400 max
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame")

    h, w = frame.shape[:2]
    scale = min(1.0, MAX_DISPLAY_DIM / max(w, h))
    if scale < 1.0:
        frame_display = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
    else:
        frame_display = frame.copy()

    print(f"\n>>> Calibration frame selector window opening...", flush=True)
    print(f">>> Window: '{window}'", flush=True)
    print(f">>> If window doesn't appear, check taskbar or use Alt+Tab", flush=True)
    print(f">>> Click this window or press ESC to auto-select first frame", flush=True)
    print(f">>> If no input after 60 seconds, will auto-proceed.\n", flush=True)

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)  # Use AUTOSIZE and NORMAL flags
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)  # Try to make window stay on top
    selected_frame = None

    import time
    start_time = time.time()
    timeout_sec = 60  # Auto-proceed if no input for 60 seconds

    while True:
        disp = frame_display.copy()
        text = f"Frame {current_idx}/{total_frames} | Arrows: ±10 | PgUp/PgDn: ±150 | SPACE: confirm | P: proceed | ESC: auto"
        cv2.putText(disp, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(disp, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window, disp)
        key = cv2.waitKeyEx(30)
        key_ascii = key & 0xFF if key != -1 else -1

        # Check for timeout (auto-proceed if no input)
        if time.time() - start_time > timeout_sec:
            print(f"\n>>> No input for {timeout_sec}s - auto-selecting first non-black frame...")
            for i in range(total_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, test_frame = cap.read()
                if not ok:
                    break
                if cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY).mean() > 20:
                    selected_frame = test_frame
                    print(f"✓ Auto-selected frame {i}")
                    break
            if selected_frame is None:
                selected_frame = frame
            break

        if key == -1:
            continue

        # Arrow keys to scrub (±10 frames)
        if key in {65361, 81, 2424832}:  # LEFT
            current_idx = max(0, current_idx - 10)
        elif key in {65363, 83, 2555904}:  # RIGHT
            current_idx = min(total_frames - 1, current_idx + 10)
        # Page Up/Down for larger jumps (±150 frames)
        elif key in {65365, 2555904}:  # PAGEUP / Alternative
            current_idx = max(0, current_idx - 150)
        elif key in {65366, 2621440}:  # PAGEDOWN
            current_idx = min(total_frames - 1, current_idx + 150)
        elif key_ascii == 32:  # SPACE: confirm this frame
            selected_frame = frame
            break
        elif key_ascii == 112 or key_ascii == 80:  # P or p: proceed with current frame
            selected_frame = frame
            print(f"✓ Proceeding with frame {current_idx}")
            break
        elif key_ascii == 27:  # ESC: auto-find first non-black frame
            for i in range(total_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, test_frame = cap.read()
                if not ok:
                    break
                # Simple heuristic: frame is "non-black" if mean pixel value > 20
                if cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY).mean() > 20:
                    selected_frame = test_frame
                    print(f"Auto-selected frame {i} (first non-black)")
                    break
            if selected_frame is None:
                selected_frame = frame  # fallback to current
            break

        # Seek to new frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        ok, frame = cap.read()
        if ok:
            if scale < 1.0:
                frame_display = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
            else:
                frame_display = frame.copy()

    cap.release()
    cv2.destroyWindow(window)

    # Show confirmation dialog with Proceed button
    _show_proceed_dialog("Calibration Complete", "Frame calibration done.\nReady to proceed to hand color picking?")

    return selected_frame


def interactive_calibrate(
    reference_frame, low_pitch: int, high_pitch: int,
    use_cv_calibration: bool = True, seed_calibration: Optional["Calibration"] = None,
) -> Calibration:
    """Let the user click the centers of white keys across the playable
    keyboard area (left to right). For each click, the user specifies which
    white key it is (via a small Tkinter key-picker popup), and the system
    accumulates points into a list of `CalibPoint`. Once 4+ points are
    placed, the overlay shows a live least-squares-fitted homography; the
    user presses ESC to finish once satisfied (8+ points recommended for a
    good fit across severe perspective/lens distortion).

    An overlay is pre-seeded (shown immediately; ESC alone accepts it -- no
    clicking needed) from one of two sources:
      - ``seed_calibration``, if the caller already resolved one (Phase B:
        extract_fingering.py's per-source persistence + agreement check
        decides between a saved calibration and a fresh CV read before
        calling this function, and prints its own reasoning); or
      - otherwise, when ``use_cv_calibration`` (default on), this function
        runs the CV black-key detector (keyboard_cv.detect_keyboard_
        calibration) itself on the reference frame.
    Clicking A/C/G keys always overrides either seed: once >=3 points are
    placed, the row is refit from those clicks instead, exactly as in
    pure-manual mode. Pressing 'b' clears any clicks and reverts to the
    seed overlay. No seed (failed/low-confidence detection, or none given)
    falls back to requiring manual clicks, same as before this feature
    existed.

    Lazy-imports cv2 -- only called when actually calibrating from live
    video, never from selftest.py.
    """
    import cv2  # lazy import

    auto_calib = seed_calibration
    if auto_calib is None and use_cv_calibration:
        try:
            from keyboard_cv import detect_keyboard_calibration
            auto_calib = detect_keyboard_calibration(reference_frame, low_pitch, high_pitch)
        except Exception:
            auto_calib = None
        print("CV auto-calibration found a confident keyboard fit -- ESC to accept, or click A/C/G keys to override."
              if auto_calib is not None else
              "CV auto-calibration didn't find a confident keyboard fit -- click white key centers manually.")

    # Guard against the window manager / OpenCV backend silently displaying
    # the window at a different pixel size than `reference_frame`'s true
    # array resolution (e.g. a 4K frame shown shrunk to fit the screen). If
    # that happens, a click at a visually-correct spot lands at the wrong
    # pixel in the underlying array -- producing exactly the "roughly right
    # spacing but systematically offset" symptom this feature exists to fix.
    # Rather than trust cv2/Qt/GTK's window-resize-to-callback-coordinate
    # mapping (backend/version-specific), we do our own explicit, controlled
    # downscale of the array before showing it, and our own explicit upscale
    # of the final points before returning -- so the window is always
    # AUTOSIZE-created and pixel-exact to whatever array we hand it, and no
    # window-manager-driven resize ever occurs.
    MAX_DISPLAY_DIM = 2450
    h, w = reference_frame.shape[:2]
    scale = min(1.0, MAX_DISPLAY_DIM / max(w, h))
    if scale < 1.0:
        print(
            f"Video frame is {w}x{h}, displaying scaled to fit your screen "
            f"(points will be mapped back to full resolution automatically)."
        )

    # The auto row is fit in FULL-resolution pixel space (detected straight off
    # `reference_frame`); the live preview draws on the display-scaled `disp`
    # image, so a scaled-down copy is needed just for that preview (x,y scale
    # linearly with a shared denominator -- only a,b,c,d scale, g doesn't).
    auto_row_display = None
    if auto_calib is not None and auto_calib.row is not None:
        a, b, c, d, g = auto_calib.row
        auto_row_display = [a * scale, b * scale, c * scale, d * scale, g]

    points: list = []  # accumulated CalibPoint (key centers), in display-scaled pixel space
    points_near: list = []  # accumulated CalibPoint (front-edge clicks) for side-angle mode
    front_edge_mode = False  # toggle via 'f' key
    selected_idx = None  # index into the active points list (points or points_near)
    dragging_idx = None  # index into the active points list being mouse-dragged
    mouse_pos = (0, 0)
    octave_offset = 0  # shift entire keyboard overlay up/down by octaves (</>)
    overlay_rotation = 0.0  # tick rotation in degrees from vertical ([/] keys), for angled cameras
    window = "Calibrate: click white key centers (left to right), specify pitch, press ESC to finish"

    def _rebuild_display_frame():
        """Apply display-scale-safety downscale (no lens correction for
        multi-point -- the fitted homography itself absorbs mild lens
        distortion when enough points are spread across the keyboard)."""
        if scale < 1.0:
            uh, uw = reference_frame.shape[:2]
            return cv2.resize(reference_frame, (int(round(uw * scale)), int(round(uh * scale))))
        return reference_frame.copy()

    display_frame = _rebuild_display_frame()
    frame_h, frame_w = display_frame.shape[:2]

    lo_wi = pitch_to_white_index(low_pitch)
    hi_wi = pitch_to_white_index(high_pitch)

    def _norm_u(pitch: int) -> float:
        return (pitch_to_white_index(pitch) - lo_wi) / (hi_wi - lo_wi) if hi_wi != lo_wi else 0.5

    def _fit_row(pts):
        """Fit the 1-D projective row map (normalized-x -> pixel) from the
        clicked key-center points. Returns [a,b,c,d,g] or None (<3 points /
        degenerate). Operates in the given pixel space (caller supplies either
        display-scaled or full-res points)."""
        if len(pts) < 3:
            return None
        u = np.asarray([_norm_u(p.pitch) for p in pts], dtype=np.float64)
        x = np.asarray([p.pixel_xy[0] for p in pts], dtype=np.float64)
        y = np.asarray([p.pixel_xy[1] for p in pts], dtype=np.float64)
        return _fit_row_projective(u, x, y)

    DRAG_RADIUS_SQ = 20 * 20  # generous grab radius so points are easy to seize

    def _get_active_points():
        """Return the currently-active points list (points or points_near)."""
        return points_near if front_edge_mode else points

    def _nearest_point(x, y):
        """Index of the point within grab radius of (x, y), or None."""
        active = _get_active_points()
        best, best_d = None, DRAG_RADIUS_SQ
        for i, p in enumerate(active):
            dx, dy = p.pixel_xy[0] - x, p.pixel_xy[1] - y
            d = dx * dx + dy * dy
            if d <= best_d:
                best, best_d = i, d
        return best

    def on_click(event, x, y, flags, userdata):  # noqa: ANN001
        nonlocal selected_idx, dragging_idx, mouse_pos
        if event == cv2.EVENT_LBUTTONDOWN:
            # Grab the nearest existing point to drag it; else place a new one.
            hit = _nearest_point(x, y)
            if hit is not None:
                dragging_idx = hit
                selected_idx = hit
                return
            _open_key_picker(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click deletes the nearest point (remove a misplaced one).
            hit = _nearest_point(x, y)
            if hit is not None:
                active = _get_active_points()
                active.pop(hit)
                selected_idx = (len(active) - 1) if active else None
        elif event == cv2.EVENT_MOUSEMOVE:
            mouse_pos = (x, y)
            if dragging_idx is not None:
                cx = min(max(x, 0), frame_w - 1)
                cy = min(max(y, 0), frame_h - 1)
                active = _get_active_points()
                active[dragging_idx].pixel_xy = (cx, cy)
        elif event == cv2.EVENT_LBUTTONUP:
            if dragging_idx is not None:
                dragging_idx = None

    def _open_key_picker(click_x: float, click_y: float):
        """Pop up A / C / G buttons; octave is inferred from click position."""
        nonlocal selected_idx
        import tkinter as tk
        from tkinter import ttk

        # Estimate which octave the click landed in by interpolating click_x
        # against the known keyboard x-span.
        frac = max(0.0, min(1.0, click_x / max(frame_w - 1, 1)))
        est_wi = lo_wi + frac * (hi_wi - lo_wi)

        # Pitch class offsets within an octave (C=0, G=7, A=9)
        PC = {"C": 0, "G": 7, "A": 9}

        def _nearest_pitch(pc_name: str) -> int:
            """MIDI pitch for pitch class nearest to est_wi within the keyboard range."""
            pc = PC[pc_name]
            best, best_d = low_pitch, float("inf")
            for p in range(low_pitch, high_pitch + 1):
                if p % 12 == pc:
                    d = abs(pitch_to_white_index(p) - est_wi)
                    if d < best_d:
                        best_d, best = d, p
            return best

        root = tk.Tk()
        root.withdraw()
        picker = tk.Toplevel(root)
        picker.title("Select key")

        picker_w, picker_h = 200, 80
        picker.update_idletasks()
        screen_w = picker.winfo_screenwidth()
        screen_h = picker.winfo_screenheight()
        x = max(0, (screen_w - picker_w) // 2)
        y = max(0, (screen_h - picker_h) // 2)
        picker.geometry(f"{picker_w}x{picker_h}+{x}+{y}")

        style = ttk.Style(picker)
        style.configure("Key.TButton", font=("Arial", 14))

        result = {"pitch": None}

        def on_select(pc_name: str):
            result["pitch"] = _nearest_pitch(pc_name)
            picker.destroy()

        frame = ttk.Frame(picker)
        frame.pack(padx=10, pady=10, fill="both", expand=True)
        row = ttk.Frame(frame)
        row.pack(fill="x")
        for name in ["A", "C", "G"]:
            ttk.Button(
                row, text=name, width=4, command=lambda n=name: on_select(n),
                style="Key.TButton",
            ).pack(side="left", padx=4, pady=2)

        picker.protocol("WM_DELETE_WINDOW", picker.destroy)
        picker.wait_window()
        root.destroy()

        if result["pitch"] is not None:
            try:
                pt = CalibPoint(pixel_xy=(float(click_x), float(click_y)), pitch=result["pitch"])
                active = _get_active_points()
                active.append(pt)
                selected_idx = len(active) - 1
            except ValueError as e:
                print(f"Invalid key selection: {e}")

    def _draw_wrapped_hint(disp, segments, x, y, max_width, font_scale=0.7, line_height=26):
        """Draws '|'-joined hint segments as cv2 text, wrapping onto extra
        lines (packed greedily by measured pixel width) instead of one long
        line -- cv2.putText doesn't wrap or clip safely on its own, so a
        long instruction string simply runs off the edge of a modestly-
        sized video frame and becomes unreadable, rather than erroring or
        visibly truncating with an ellipsis."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = []
        current = ""
        for seg in segments:
            candidate = seg if not current else f"{current} | {seg}"
            (text_w, _), _ = cv2.getTextSize(candidate, font, font_scale, 1)
            if text_w > max_width and current:
                lines.append(current)
                current = seg
            else:
                current = candidate
        if current:
            lines.append(current)
        for i, line in enumerate(lines):
            ly = y + i * line_height
            cv2.putText(disp, line, (x, ly), font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(disp, line, (x, ly), font, font_scale, (0, 255, 255), 1, cv2.LINE_AA)
        return len(lines)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)

    ARROW_LEFT = {65361, 81, 2424832}
    ARROW_UP = {65362, 82, 2490368}
    ARROW_RIGHT = {65363, 83, 2555904}
    ARROW_DOWN = {65364, 84, 2621440}
    ALL_ARROWS = ARROW_LEFT | ARROW_UP | ARROW_RIGHT | ARROW_DOWN
    TAB_KEY_ASCII = 9

    while True:
        display_frame = _rebuild_display_frame()
        disp = display_frame.copy()

        # Draw key-center points (green/red)
        for i, p in enumerate(points):
            px, py = int(p.pixel_xy[0]), int(p.pixel_xy[1])
            if i == selected_idx and not front_edge_mode:
                cv2.circle(disp, (px, py), 10, (0, 0, 255), 2)  # red for selected
                cv2.line(disp, (px - 14, py), (px + 14, py), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(disp, (px, py - 14), (px, py + 14), (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(disp, (px, py), 6, (0, 255, 0), -1)  # green for unselected
            cv2.putText(disp, pitch_to_note_name(p.pitch), (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

        # Draw front-edge points (cyan/red)
        for i, p in enumerate(points_near):
            px, py = int(p.pixel_xy[0]), int(p.pixel_xy[1])
            if i == selected_idx and front_edge_mode:
                cv2.circle(disp, (px, py), 10, (0, 0, 255), 2)  # red for selected
                cv2.line(disp, (px - 14, py), (px + 14, py), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(disp, (px, py - 14), (px, py + 14), (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(disp, (px, py), 6, (255, 255, 0), -1)  # cyan for unselected
            cv2.putText(disp, pitch_to_note_name(p.pitch), (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1)

        # Draw the live inferred keyboard (full white/black key overlay).
        # Manual clicks always override once there are enough to fit (>=3);
        # with none placed, fall back to the auto CV detection (if any) so
        # its overlay is what's shown by default.
        row_fit = _fit_row(points) if points else auto_row_display
        row_near_fit = _fit_row(points_near) if points_near else None
        if row_fit is not None or row_near_fit is not None:
            # Apply octave offset to the keyboard range
            shifted_low = max(0, min(127, low_pitch + octave_offset * 12))
            shifted_high = max(0, min(127, high_pitch + octave_offset * 12))
            preview_calib = Calibration(
                corners=[[0, 0], [1, 0], [1, 1], [0, 1]],
                low_pitch=shifted_low, high_pitch=shifted_high,
                row=row_fit, row_near=row_near_fit,
                rotation_deg=overlay_rotation,
            )
            draw_keyboard_overlay(disp, preview_calib)

        # Instructions
        auto_hint = " | b: revert to auto" if auto_calib is not None else ""
        mode_hint = " [FRONT-EDGE MODE]" if front_edge_mode else ""
        finish_hint = "ESC: finish" if (points and len(points) >= 4) or (not points and auto_calib is not None) else "ESC: finish (need 4+)"
        octave_hint = f" | Octave: {octave_offset:+d}" if octave_offset != 0 else ""
        rot_hint = f" | Rotation: {overlay_rotation:+.0f}°" if overlay_rotation != 0.0 else ""
        hint_segments = [
            f"Click white keys ({len(points)} center, {len(points_near)} front)",
            "f: toggle front-edge",
            "L-drag: move",
            "R-click / Del / x: remove",
            "Tab: select",
            "Arrows: nudge",
            f"< / >: shift octave{auto_hint}",
            "[ / ]: rotate overlay",
            f"U: undo",
            f"{finish_hint}{mode_hint}{octave_hint}{rot_hint}",
        ]
        _draw_wrapped_hint(disp, hint_segments, 20, 40, frame_w - 40)

        cv2.imshow(window, disp)
        key = cv2.waitKeyEx(30)
        key_ascii = key & 0xFF if key != -1 else -1

        if key == -1:
            continue
        if key_ascii == 27:  # ESC: finish
            if points:
                if len(points) >= 4:
                    break
                print(f"Need at least 4 center points, have {len(points)}. Keep clicking.")
                continue
            if auto_calib is not None:
                break  # accept the pure-auto overlay as-is
            print("Need at least 4 center points (no auto detection available). Keep clicking.")
            continue
        if key_ascii in (ord("f"), ord("F")):
            # Toggle front-edge mode
            front_edge_mode = not front_edge_mode
            selected_idx = None
            continue
        if key_ascii in (ord("b"), ord("B")) and auto_calib is not None:
            # Revert to the pure auto overlay, discarding manual clicks.
            points.clear()
            points_near.clear()
            selected_idx = None
            continue
        if key_ascii in (ord("u"), ord("U")):
            # Undo: pop from the active list
            active = _get_active_points()
            if active:
                active.pop()
                selected_idx = (len(active) - 1) if active else None
            continue
        if key_ascii in (ord("<"), ord(",")) and not points:
            # Shift octave down (< or ,) -- only when no manual clicks yet
            octave_offset = max(-2, octave_offset - 1)
            continue
        if key_ascii in (ord(">"), ord(".")) and not points:
            # Shift octave up (> or .) -- only when no manual clicks yet
            octave_offset = min(2, octave_offset + 1)
            continue
        if key_ascii == ord("["):
            # Rotate overlay 5° counter-clockwise (tilt left)
            overlay_rotation = max(-45.0, overlay_rotation - 5.0)
            continue
        if key_ascii == ord("]"):
            # Rotate overlay 5° clockwise (tilt right)
            overlay_rotation = min(45.0, overlay_rotation + 5.0)
            continue
        if key_ascii in (255, 8, ord("x"), ord("X")) and selected_idx is not None:
            # Delete / Backspace / x: remove the currently-selected point.
            active = _get_active_points()
            if active:
                active.pop(selected_idx)
                selected_idx = (len(active) - 1) if active else None
            continue
        active = _get_active_points()
        if key_ascii == TAB_KEY_ASCII and active:
            selected_idx = (selected_idx + 1) % len(active) if selected_idx is not None else 0
            continue
        if key in ALL_ARROWS and selected_idx is not None and active:
            x, y = active[selected_idx].pixel_xy
            if key in ARROW_LEFT:
                x -= 1
            elif key in ARROW_RIGHT:
                x += 1
            elif key in ARROW_UP:
                y -= 1
            elif key in ARROW_DOWN:
                y += 1
            x = min(max(x, 0), frame_w - 1)
            y = min(max(y, 0), frame_h - 1)
            active[selected_idx].pixel_xy = (x, y)
            continue

    cv2.destroyWindow(window)

    if not points and auto_calib is not None:
        # User accepted the pure-auto overlay with no manual clicks --
        # auto_calib's row is already in full-resolution pixel space
        # (detected straight off `reference_frame`), so no rescale needed.
        # Preserve any rotation the user dialed in with [ / ].
        if overlay_rotation != 0.0:
            from dataclasses import replace as _dc_replace
            return _dc_replace(auto_calib, rotation_deg=overlay_rotation)
        return auto_calib

    # `points` and `points_near` live in display-scaled pixel space. Convert back
    # to true full-resolution pixel space, then fit the 1-D projective row maps
    # (the real calibration). Corners are kept only as a serialized placeholder.
    class _P:
        __slots__ = ("pixel_xy", "pitch")

        def __init__(self, xy, pitch):
            self.pixel_xy = xy
            self.pitch = pitch

    full_pts = [_P((p.pixel_xy[0] / scale, p.pixel_xy[1] / scale), p.pitch) for p in points]
    full_pts_near = [_P((p.pixel_xy[0] / scale, p.pixel_xy[1] / scale), p.pitch) for p in points_near] if points_near else []

    row_final = _fit_row(full_pts)
    if row_final is None:
        raise RuntimeError(
            "Calibration failed: could not fit a keyboard mapping from the "
            f"{len(points)} clicked center point(s). Click at least 3 white keys "
            "spread across the keyboard (more is better) and retry."
        )

    row_near_final = None
    if full_pts_near:
        row_near_final = _fit_row(full_pts_near)
        if row_near_final is None:
            print(f"Warning: could not fit front-edge row from {len(points_near)} points; "
                  "using center-only calibration.")

    calib = Calibration(
        corners=[[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
        low_pitch=low_pitch,
        high_pitch=high_pitch,
        k1=0.0,
        row=row_final,
        row_near=row_near_final,
        rotation_deg=overlay_rotation,
    )
    return calib
