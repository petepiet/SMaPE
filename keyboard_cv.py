"""CV auto-calibration: detect a piano keyboard's key layout directly from a
single reference video frame, producing the SAME ``Calibration.row`` output
that manual C/G-key clicking produces in ``keyboard.py``. Used to pre-seed
``interactive_calibrate`` with a working overlay so most videos need only an
ESC to accept, with manual clicks always available to override/refine.

Algorithm (see ai/tasks/005-cv-calibration-batch/PLAN.md Phase A):
  1. Find a horizontal scanline that passes through the black keys (cv2,
     not unit-tested -- mirrors hands.py's cv2 paths).
  2. Extract each black key's x-centroid on that scanline (cv2).
  3. Classify the gaps between consecutive black keys as "small" (within a
     2-group {C#,D#} or 3-group {F#,G#,A#}) or "large" (crossing an E-F or
     B-C boundary, where no black key sits) -- PURE, numpy-only.
  4. Group black keys by those gaps, then align the detected group-size
     sequence (e.g. [2,3,2,3,...]) against the sequence expected for the
     given [low_pitch, high_pitch] range to assign each detected black key
     its absolute MIDI pitch -- PURE.
  5. Feed the resulting (pitch, x, y) anchors into keyboard.py's EXISTING
     ``_fit_row_projective`` -- reuse, don't reinvent -- producing a
     ``Calibration`` with a ``row``, identical in shape to the manual flow.

Steps 3-4 have no cv2 dependency and are unit-tested in selftest.py with
synthetic centroid data. Steps 1-2 and the top-level ``detect_keyboard_
calibration`` need cv2 and are only exercised against real frames (verified
by eye against downloaded videos), same as hands.py/keyboard.py's cv2 code.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from keyboard import Calibration, _fit_row_projective, pitch_to_white_index

# Pitch classes (mod 12) of the two black-key group shapes.
_TWO_GROUP_PCS = (1, 3)        # C#, D#
_THREE_GROUP_PCS = (6, 8, 10)  # F#, G#, A#
_BLACK_PCS = set(_TWO_GROUP_PCS) | set(_THREE_GROUP_PCS)

MIN_CONFIDENCE = 0.6


# --------------------------------------------------------------------------
# Pure logic (no cv2) -- unit-tested in selftest.py
# --------------------------------------------------------------------------

def classify_gap_sizes(gaps: np.ndarray) -> np.ndarray:
    """Classifies each consecutive black-key gap as "large" (crossing an E-F
    or B-C boundary) vs "small" (within a 2- or 3- black-key group).

    In an idealized top-down view, small gaps (within a 2- or 3- black-key
    group) and large gaps (crossing an E-F or B-C boundary) differ by a
    clean 2x ratio in white-key-index units. Real photographed keyboards
    don't preserve that exact ratio (measured on real footage, small gaps
    were ~37-43px and large ~57-67px -- a ~1.5x ratio, not 2x) but DO show a
    clearly bimodal distribution. So rather than assume a ratio, this finds
    the split point that maximizes between-cluster variance (1-D Otsu,
    exhaustive over sorted split points) -- the standard way to threshold a
    bimodal distribution. A naive "biggest single jump between consecutive
    sorted values" was tried first and failed on real footage: one stray
    wide gap (e.g. a mis-detected blob at the keyboard's edge) creates an
    isolated outlier whose neighboring jump is the largest in the array
    but does NOT sit at the true small/large boundary. Otsu's variance
    criterion weighs both cluster sizes and means, so a lone outlier can't
    hijack the split the way a single max-jump search can.

    Returns a bool array (True = large), same length as ``gaps``. Empty or
    single-element input can't be classified relatively; returns all-False
    (the caller's group/alignment step naturally produces low confidence
    from too few points regardless).
    """
    gaps = np.asarray(gaps, dtype=np.float64)
    n = gaps.size
    if n < 2:
        return np.zeros(gaps.shape, dtype=bool)
    sorted_gaps = np.sort(gaps)
    best_threshold, best_variance = None, -1.0
    for i in range(1, n):
        left, right = sorted_gaps[:i], sorted_gaps[i:]
        w0, w1 = i / n, (n - i) / n
        variance = w0 * w1 * (left.mean() - right.mean()) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = (sorted_gaps[i - 1] + sorted_gaps[i]) / 2.0
    return gaps > best_threshold


def group_black_keys(is_large_gap: np.ndarray, n: int) -> list:
    """Splits ``n`` consecutive black-key indices into groups, cutting at
    each large gap. ``is_large_gap[i]`` is the gap between key i and i+1.
    Returns a list of index-lists (each a contiguous run of small gaps)."""
    groups = []
    current = [0]
    for i, large in enumerate(is_large_gap):
        if large:
            groups.append(current)
            current = [i + 1]
        else:
            current.append(i + 1)
    groups.append(current)
    return groups


def expected_black_key_groups(low_pitch: int, high_pitch: int) -> list:
    """The sequence of black-key groups (each a list of MIDI pitches) that
    are visible within [low_pitch, high_pitch], in ascending pitch order.
    A range that starts/ends mid-group yields a partial (size-1) group at
    that end, matching what a real partial-keyboard crop would show (e.g.
    the lone A#0 at the very bottom of an 88-key piano)."""
    groups: list = []
    current: list = []
    current_key = None
    for pitch in range(low_pitch, high_pitch + 1):
        pc = pitch % 12
        if pc not in _BLACK_PCS:
            continue
        key = (pitch // 12, "two" if pc in _TWO_GROUP_PCS else "three")
        if key != current_key:
            if current:
                groups.append(current)
            current = [pitch]
            current_key = key
        else:
            current.append(pitch)
    if current:
        groups.append(current)
    return groups


def assign_group_pitches(
    detected_sizes: Sequence[int], low_pitch: int, high_pitch: int
) -> Optional[list]:
    """Aligns the detected black-key group-size sequence against the
    sequence expected for [low_pitch, high_pitch], and returns, per
    detected group, the list of MIDI pitches it corresponds to (or
    ``None`` for a group whose size didn't match at the best alignment --
    ambiguous, dropped rather than guessed, per the "flag more than drop"
    philosophy).

    The alignment slides in BOTH directions: detected groups may start
    partway into the expected range (camera doesn't show the low end), but
    may also carry spurious leading/trailing groups the expected sequence
    has no counterpart for at all (e.g. a cabinet edge mis-detected as a
    lone black key just past the real keyboard edge) -- both were observed
    on real footage. The score is normalized by the OVERLAPPING span only,
    so junk hanging off either end doesn't dilute the confidence of a
    correct interior alignment.

    Returns ``None`` if no alignment overlaps enough (<3 groups) or matches
    well enough (<60% of the overlap by size) to trust at all.
    """
    expected_groups = expected_black_key_groups(low_pitch, high_pitch)
    expected_sizes = [len(g) for g in expected_groups]
    n_det, n_exp = len(detected_sizes), len(expected_sizes)
    if n_det == 0 or n_exp == 0:
        return None

    best_offset, best_score, best_overlap = None, -1, 0
    for offset in range(-(n_det - 1), n_exp):
        score, overlap = 0, 0
        for i in range(n_det):
            j = offset + i
            if 0 <= j < n_exp:
                overlap += 1
                if detected_sizes[i] == expected_sizes[j]:
                    score += 1
        if overlap == 0:
            continue
        if score > best_score or (score == best_score and overlap > best_overlap):
            best_score, best_offset, best_overlap = score, offset, overlap

    if best_offset is None or best_overlap < 3:
        return None
    if best_score / best_overlap < MIN_CONFIDENCE:
        return None

    result = []
    for i in range(n_det):
        j = best_offset + i
        if 0 <= j < n_exp and detected_sizes[i] == expected_sizes[j]:
            result.append(expected_groups[j])
        else:
            result.append(None)
    return result


def build_anchor_points(
    groups: Sequence[Sequence[int]],
    group_pitches: Sequence[Optional[Sequence[int]]],
    centroids_x: Sequence[float],
    centroids_y: Sequence[float],
    low_pitch: int,
    high_pitch: int,
):
    """Builds the (u, x, y) anchor arrays ``_fit_row_projective`` expects,
    from detected black-key groups matched to absolute pitches. Groups with
    no pitch assignment (ambiguous, see `assign_group_pitches`) contribute
    no anchors. ``u`` uses the same normalized-x convention as the rest of
    the calibration pipeline (0..1 across [low_pitch, high_pitch] in
    white-key-index space)."""
    lo_wi = pitch_to_white_index(low_pitch)
    hi_wi = pitch_to_white_index(high_pitch)
    span = (hi_wi - lo_wi) if hi_wi != lo_wi else 1.0
    us, xs, ys = [], [], []
    for idxs, pitches in zip(groups, group_pitches):
        if pitches is None:
            continue
        for idx, pitch in zip(idxs, pitches):
            us.append((pitch_to_white_index(pitch) - lo_wi) / span)
            xs.append(centroids_x[idx])
            ys.append(centroids_y[idx])
    return np.asarray(us, dtype=np.float64), np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


# --------------------------------------------------------------------------
# cv2 detection (not covered by selftest.py -- verified against real frames)
# --------------------------------------------------------------------------

def _row_dark_spans(gray, y: int, min_w: float, max_w: float, dark_thresh: int):
    """Contiguous dark (< dark_thresh) x-spans on row ``y`` whose width is
    within [min_w, max_w] -- candidate black-key cross-sections."""
    row = gray[y]
    dark = row < dark_thresh
    spans = []
    start = None
    w = len(row)
    for x in range(w):
        if dark[x] and start is None:
            start = x
        elif not dark[x] and start is not None:
            spans.append((start, x))
            start = None
    if start is not None:
        spans.append((start, w))
    return [(a, b) for a, b in spans if min_w <= (b - a) <= max_w]


def _iou_at(gray, y: int, dy: int, dark_thresh: int) -> float:
    """Vertical stability of the dark mask between row y and row y+dy.
    Real black keys are near-vertical rectangles, so their dark mask barely
    shifts a few rows down; piano-interior strings run diagonally and
    fan out, so their dark mask shifts a lot. This is what separates a true
    black-key row from a string-fan false positive above the keybed."""
    h = gray.shape[0]
    if y + dy >= h:
        return 0.0
    a = gray[y] < dark_thresh
    b = gray[y + dy] < dark_thresh
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union > 0 else 0.0


def _find_black_key_row(gray, dark_thresh: int = 90) -> Optional[int]:
    """Finds the y row most likely to be slicing through the black keys.
    Searches the lower ~60% of the frame (keys are always in the lower
    portion of these overhead shots) for rows with many (>=15), regularly
    sized (low coefficient-of-variation) dark blobs, then picks among those
    the row with the highest vertical stability (see `_iou_at`) -- the
    signal that rejects piano-interior string fans, which can otherwise
    produce MORE numerous and even more "regular-looking" blobs than the
    real keys."""
    h, w = gray.shape
    lo, hi = int(h * 0.35), int(h * 0.95)
    min_w, max_w = max(2, w // 300), w // 12
    candidates = []
    for y in range(lo, hi):
        spans = _row_dark_spans(gray, y, min_w, max_w, dark_thresh)
        if len(spans) < 15:
            continue
        widths = np.array([b - a for a, b in spans], dtype=np.float64)
        cv = float(widths.std() / widths.mean()) if widths.mean() > 0 else float("inf")
        if cv > 0.55:
            continue
        iou = _iou_at(gray, y, 15, dark_thresh)
        candidates.append((y, len(spans), cv, iou))
    if not candidates:
        return None
    candidates.sort(key=lambda r: (-r[3], -r[1], r[2]))
    return candidates[0][0]


def _black_key_centroids(gray, y: int, dark_thresh: int = 90):
    """(x_centroid, y) pairs for each black-key blob on row ``y``, left to
    right."""
    h, w = gray.shape
    min_w, max_w = max(2, w // 300), w // 12
    spans = _row_dark_spans(gray, y, min_w, max_w, dark_thresh)
    return [((a + b) / 2.0, float(y)) for a, b in spans]


FULL_KEYBOARD_LOW = 21   # A0
FULL_KEYBOARD_HIGH = 108  # C8


def _detect_for_range(centroids_x, centroids_y, groups, detected_sizes, low_pitch, high_pitch):
    """One anchoring attempt for a given [low_pitch, high_pitch] window --
    factored out of `detect_keyboard_calibration` so it can be tried twice
    (full keyboard first, then the caller's requested range) without
    redoing the cv2 row/blob detection."""
    group_pitches = assign_group_pitches(detected_sizes, low_pitch, high_pitch)
    if group_pitches is None:
        return None
    u, x, y = build_anchor_points(groups, group_pitches, centroids_x, centroids_y, low_pitch, high_pitch)
    if u.shape[0] < 3:
        return None
    row_fit = _fit_row_projective(u, x, y)
    if row_fit is None:
        return None
    return row_fit


def detect_keyboard_calibration(
    frame, low_pitch: int, high_pitch: int
) -> Optional[Calibration]:
    """Detects the keyboard's key layout in ``frame`` (a cv2 BGR image) and
    returns a `Calibration` with a `row` fit -- the same output shape as
    manual multi-point clicking -- or ``None`` if detection isn't confident
    enough (caller falls back to manual calibration).

    Absolute-octave anchoring is tried against the FULL 88-key range
    (A0..C8) FIRST, regardless of the caller's requested [low_pitch,
    high_pitch] (typically the narrower range of pitches actually present
    in the MIDI) -- and, if that succeeds, the returned Calibration's own
    low_pitch/high_pitch are widened to the full keyboard too (nothing
    downstream depends on them matching the MIDI's note range; the row
    fit and its (low,high) domain must simply agree with each other,
    which they do here). This matters because the 2-/3- black-key group
    pattern REPEATS every octave: matching a short expected window (e.g.
    just the MIDI's 2-3 octave range) against a full keyboard's worth of
    detected groups is ambiguous -- any octave-shifted alignment scores
    identically -- and silently locked onto the wrong octave on real
    footage (confirmed: overlay's C-labels landed a full octave off from
    the real keys, on a video clearly showing the entire keyboard). The
    full 88-key range doesn't have that ambiguity: its expected group
    sequence starts with a UNIQUE partial group (the lone A#0, size 1,
    with every other group size 2 or 3), which anchors the phase
    absolutely. Only falls back to the caller's narrower range if the
    full-keyboard attempt fails -- e.g. a genuinely partial/cropped shot
    that doesn't show both keyboard ends.
    """
    import cv2  # lazy import

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    row_y = _find_black_key_row(gray)
    if row_y is None:
        return None

    centroids = _black_key_centroids(gray, row_y)
    if len(centroids) < 3:
        return None
    centroids_x = [c[0] for c in centroids]
    centroids_y = [c[1] for c in centroids]

    gaps = np.diff(np.asarray(centroids_x, dtype=np.float64))
    is_large = classify_gap_sizes(gaps)
    groups = group_black_keys(is_large, len(centroids))
    detected_sizes = [len(g) for g in groups]

    row_fit = _detect_for_range(centroids_x, centroids_y, groups, detected_sizes, FULL_KEYBOARD_LOW, FULL_KEYBOARD_HIGH)
    if row_fit is not None:
        used_low, used_high = FULL_KEYBOARD_LOW, FULL_KEYBOARD_HIGH
    else:
        row_fit = _detect_for_range(centroids_x, centroids_y, groups, detected_sizes, low_pitch, high_pitch)
        if row_fit is None:
            return None
        used_low, used_high = low_pitch, high_pitch

    return Calibration(
        corners=[[0, 0], [frame.shape[1], 0], [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]],
        low_pitch=used_low,
        high_pitch=used_high,
        k1=0.0,
        row=row_fit,
    )
