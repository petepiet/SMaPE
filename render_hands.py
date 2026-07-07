"""Synthesia-style render support: the video contributes exactly ONE thing
-- which hand plays each
already-transcribed note, read from the lit key's colour. Audio (via
--transcribe) remains the sole source of pitch/timing/pedal; this module
never reads note onsets/offsets from pixels.

Split the same way as keyboard.py/keyboard_cv.py and hands.py/reconcile.py:
pure numpy logic (clustering, lit detection, MIDI-duration trimming) is
fully covered by selftest.py; cv2 video sampling is not (verified against
real render footage instead).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Pure logic -- unit-tested in selftest.py
# --------------------------------------------------------------------------

# Below this inter-cluster centre RGB distance (0..255-per-channel scale),
# the lit colours don't look like a genuine two-hand split (e.g. a
# single-colour render theme, or a one-hand piece) -- don't invent one.
SEPARATION_CONFIDENT_THRESHOLD = 60.0

# Below this hue separation (chord length on unit circle, 0..2 scale),
# don't invent a two-hand split. 0.5 ≈ 29° hue difference; distinct hand
# hues (e.g. green vs blue) are 120°+ apart (chord ≥ 1.7); same-hue tint/
# shade noise is a few degrees (chord ≈ 0.01–0.05).
HUE_SEPARATION_CONFIDENT_THRESHOLD = 0.5

# Euclidean RGB distance (0..255 scale per channel) beyond which a sampled
# key patch counts as "lit" relative to its unlit baseline.
LIT_THRESHOLD = 40.0

# Consecutive not-lit samples required before declaring the key released --
# absorbs a single noisy/compression-artifact sample, mirrors reconcile.py's
# FINGER_LEAVE_GAP_FRAMES.
LIT_END_GAP_SAMPLES = 2

# Revert a lit-based trim if it would leave the note shorter than this --
# likely a sampling/compression glitch, not a real early release. Mirrors
# reconcile.py's MIN_TRIMMED_DURATION_SEC.
MIN_TRIMMED_DURATION_SEC = 0.05


def _rgb_to_saturation(colors: Sequence[Sequence[float]]) -> np.ndarray:
    """Compute saturation for each RGB colour. Returns array of (0..1) values."""
    colors = np.asarray(colors, dtype=np.float64)
    max_c = np.max(colors, axis=1)
    min_c = np.min(colors, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sat = (max_c - min_c) / max_c
    sat[max_c == 0] = 0.0  # Handle black (max_c=0)
    return sat


def _rgb_to_hue_vec(colors: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert RGB colours to unit-magnitude hue vectors (2D), invariant under
    tinting (light rendering) and shading (dark rendering). Returns (N, 2)
    array where each row is [cos(hue), sin(hue)]. Same hue, different
    brightness → same vector. Different hues → separated vectors.

    Formula: hue = atan2(sqrt(3)*(g-b), 2*r-g-b), then return [cos(h), sin(h)].
    This is the standard geometry for RGB↔HSV conversion.
    """
    colors = np.asarray(colors, dtype=np.float64)
    n = colors.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)

    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    hue = np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)
    return np.column_stack([np.cos(hue), np.sin(hue)])


def cluster_hand_colors(hue_vecs: Sequence[Sequence[float]], n_iter: int = 20):
    """2-means clustering of hue vectors (unit-magnitude 2D points on the
    unit circle) -- deterministic, no random seed. Returns ``(labels, separation)``.

    Separation is the Euclidean distance between cluster centres on the hue circle
    (chord length, 0..2 scale). Used to decide if this is a genuine two-hand split
    or just colour noise.

    Degenerate inputs (0 or 1 colours) return trivial results.
    """
    hue_vecs = np.asarray(hue_vecs, dtype=np.float64)
    n = hue_vecs.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), 0.0
    if n == 1:
        return np.zeros(1, dtype=int), 0.0

    # k-means++ initialization: pick first centre, then the farthest point
    centers = [hue_vecs[0]]
    d = np.linalg.norm(hue_vecs - centers[0], axis=1)
    centers.append(hue_vecs[int(np.argmax(d))])
    centers = np.array(centers, dtype=np.float64)

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        # Assign each point to nearest centre
        distances = np.array([np.linalg.norm(hue_vecs - c, axis=1) for c in centers])
        new_labels = np.argmin(distances, axis=0)
        converged = np.array_equal(new_labels, labels)
        labels = new_labels
        # Update centres
        for k in range(2):
            mask = labels == k
            if mask.any():
                centers[k] = hue_vecs[mask].mean(axis=0)
        if converged:
            break

    separation = float(np.linalg.norm(centers[0] - centers[1]))
    return labels, separation


def is_separation_confident(separation: float, threshold: float = SEPARATION_CONFIDENT_THRESHOLD) -> bool:
    return separation >= threshold


def is_separation_confident_hue(separation: float, threshold: float = HUE_SEPARATION_CONFIDENT_THRESHOLD) -> bool:
    return separation >= threshold


def assign_clusters_to_hands(labels: np.ndarray, pitches: Sequence[float], flip: bool = False) -> dict:
    """Maps each cluster id (0/1) to 'L'/'R': the cluster whose notes have
    the LOWER mean pitch is the left hand (a pianist's left hand plays the
    lower register). ``flip`` swaps the mapping -- mirrors the existing
    ``--flip-handedness`` for the real-hands MediaPipe path, for render
    themes where this convention doesn't hold."""
    pitches = np.asarray(pitches, dtype=np.float64)
    means = {}
    for k in (0, 1):
        mask = labels == k
        means[k] = float(pitches[mask].mean()) if mask.any() else float("inf")
    left_cluster = 0 if means[0] <= means[1] else 1
    right_cluster = 1 - left_cluster
    return {left_cluster: "R", right_cluster: "L"} if flip else {left_cluster: "L", right_cluster: "R"}


def assign_hands_for_notes(colors: Sequence[Sequence[float]], pitches: Sequence[float], flip: bool = False):
    """High-level entry point: clusters `colors` (converted to hue space) and
    assigns each note a hand. Returns ``(hands: list[str], confidence: float, degenerate: bool)``.

    Hue clustering is invariant under tinting/shading, so light-blue (white key)
    and dark-blue (black key) of the same hand collapse to the same hue, solving
    the Synthesia 4-color render problem as a 2-color problem.

    Near-achromatic colours (saturation < 0.05) have unreliable hue values and
    are treated as degenerate. When clustering doesn't look like a genuine split
    (low hue separation), doesn't invent one: ALL notes get the same hand based
    on overall register. `confidence` is 0.0 in the degenerate case.
    """
    if len(colors) == 0:
        return [], 0.0, True

    # Reject near-achromatic colours (hue is undefined at low saturation)
    saturations = _rgb_to_saturation(colors)
    mean_saturation = float(np.mean(saturations))
    if mean_saturation < 0.05:
        median_pitch = float(np.median(np.asarray(pitches, dtype=np.float64)))
        hand = "L" if median_pitch < 60 else "R"
        return [hand] * len(colors), 0.0, True

    hue_vecs = _rgb_to_hue_vec(colors)
    labels, separation = cluster_hand_colors(hue_vecs)
    if not is_separation_confident_hue(separation):
        median_pitch = float(np.median(np.asarray(pitches, dtype=np.float64)))
        hand = "L" if median_pitch < 60 else "R"
        return [hand] * len(colors), 0.0, True

    mapping = assign_clusters_to_hands(labels, pitches, flip=flip)
    hands = [mapping[int(label)] for label in labels]
    # Monotonic 0..1 squashing of the (0..2) hue-separation chord distance --
    # exact curve doesn't matter, only that more separation -> higher confidence.
    # Scaled so the confidence threshold itself sits mid-range.
    confidence = separation / (separation + HUE_SEPARATION_CONFIDENT_THRESHOLD)
    return hands, float(confidence), False


# Hue-agreement tolerance for colors_agree() (degrees). Converted internally
# to a chord-length threshold on the unit hue circle (see that function's
# docstring for the conversion) -- picked looser than
# HUE_SEPARATION_CONFIDENT_THRESHOLD's ~29 degrees since this is checking
# "is it still roughly the same lit-key colour", not "are two hands
# distinguishable" -- some render-to-render compression/gamma drift is
# expected even for the identical theme.
COLORS_AGREE_TOLERANCE_DEG = 25.0


def colors_agree(saved: dict, sampled_rgbs: Sequence[Sequence[float]], tolerance_deg: float = COLORS_AGREE_TOLERANCE_DEG) -> bool:
    """Whether a NEW video's sampled lit-key colours still look like the SAME
    colour theme as ``saved`` (a per-channel LH/RH reference dict loaded from
    color_store.py) -- the corroboration check that lets a batch run reuse a
    saved colour pick instead of re-opening `interactive_pick_hand_colors` for
    every video from the same channel.

    Deliberately an easier, more robust problem than from-scratch clustering
    (`cluster_hand_colors`/`assign_hands_for_notes`): rather than discovering
    two clusters from ``sampled_rgbs`` alone, this matches each sampled colour
    against two ALREADY-KNOWN target hues (the saved LH_white/RH_white), so a
    single matching sample for each hand is enough to confirm agreement.

    Keys the check on the WHITE-key references only (LH_white, RH_white) --
    the black-key variants are often just a darker/tinted derivative of the
    same hue and, per HAND_COLOR_LABELS' docstring, sometimes missing/assumed
    entirely; white is the more reliably user-picked, less noisy anchor.

    Hue distance: `_rgb_to_hue_vec` maps each RGB colour to a unit vector
    [cos(hue), sin(hue)] on the hue circle. Two colours' hue difference
    (in degrees) relates to the Euclidean (chord) distance between their unit
    vectors by the standard chord-length identity
        chord = 2 * sin(delta_degrees / 2 * pi/180)
    so a `tolerance_deg` angular tolerance becomes a chord-distance threshold
    via that same formula -- one consistent scheme, computed once here rather
    than converting each sample's chord distance back to degrees.

    Returns False (cannot confirm agreement) if ``saved`` is missing either
    LH_white or RH_white, or if ``sampled_rgbs`` is empty -- "not confirmed"
    is always the safe default; a caller treating that as "re-pick" never
    silently trusts a stale/incomplete cache entry.
    """
    if not sampled_rgbs:
        return False
    if "LH_white" not in saved or "RH_white" not in saved:
        return False
    if saved["LH_white"] is None or saved["RH_white"] is None:
        return False

    chord_tolerance = 2.0 * math.sin(math.radians(tolerance_deg) / 2.0)

    sampled_hue_vecs = _rgb_to_hue_vec(list(sampled_rgbs))
    target_hue_vecs = _rgb_to_hue_vec([saved["LH_white"], saved["RH_white"]])

    for target_vec in target_hue_vecs:
        distances = np.linalg.norm(sampled_hue_vecs - target_vec, axis=1)
        if not np.any(distances <= chord_tolerance):
            return False
    return True


# Reference-color labels for manual hand-color picking (interactive_pick_hand_colors
# below) and the hand each maps to. A render themes its lit keys in exactly these
# four roles -- LH/RH, white/black key -- so this is the complete set.
HAND_COLOR_LABELS = ["LH_white", "LH_black", "RH_white", "RH_black"]
_LABEL_TO_HAND = {"LH_white": "L", "LH_black": "L", "RH_white": "R", "RH_black": "R"}


def assign_hands_from_reference_colors(
    colors: Sequence[Optional[Sequence[float]]], reference_colors: dict, flip: bool = False,
):
    """Assigns each note a hand by nearest-match against manually-picked
    reference colors (see `interactive_pick_hand_colors`) -- an alternative
    to `assign_hands_for_notes`'s automatic clustering, for renders where
    that clustering finds the wrong split (e.g. groups by lit-vs-unlit
    brightness instead of by hand color, producing an implausibly skewed
    hand distribution despite a "confident" separation score).

    `reference_colors` is a dict with a subset of `HAND_COLOR_LABELS` as
    keys (at least one "LH_*" and one "RH_*" required -- a missing black-
    key variant is NOT auto-filled here; callers should fall back to that
    hand's white-key color themselves, mirroring this module's other
    "don't guess, leave it to the caller" conventions, e.g.
    `assign_hands_for_notes`'s degenerate-case handling above). Matching is
    plain Euclidean RGB distance -- no hue conversion -- since exact
    hand-picked colors don't need tint-invariance the way automatic
    clustering does.

    Returns ``(hands: list[str], confidences: list[float])`` -- one
    confidence PER NOTE (unlike `assign_hands_for_notes`'s single global
    score), since exact-color nearest-match naturally supports it:
    confidence is how much closer the winning hand's nearest reference is
    than the nearest opposing-hand reference (0..1; ~0.5 = roughly
    equidistant/ambiguous, ~1.0 = decisively one hand's color). A note with
    no sampled color (``None``) gets confidence 0.0.
    """
    hand_for_label = dict(_LABEL_TO_HAND)
    if flip:
        hand_for_label = {label: ("R" if hand == "L" else "L") for label, hand in hand_for_label.items()}

    ref_items = [
        (hand_for_label[label], np.asarray(rgb, dtype=np.float64))
        for label, rgb in reference_colors.items()
        if rgb is not None and label in hand_for_label
    ]
    if not ref_items:
        return ["L"] * len(colors), [0.0] * len(colors)

    hands: list = []
    confidences: list = []
    for color in colors:
        if color is None:
            hands.append(ref_items[0][0])
            confidences.append(0.0)
            continue
        c = np.asarray(color, dtype=np.float64)
        dists = sorted(
            ((hand, float(np.linalg.norm(c - rgb))) for hand, rgb in ref_items),
            key=lambda hd: hd[1],
        )
        best_hand, best_dist = dists[0]
        opposite_dists = [d for hand, d in dists if hand != best_hand]
        if opposite_dists:
            opp_dist = min(opposite_dists)
            total = opp_dist + best_dist
            confidence = (opp_dist / total) if total > 0 else 1.0
        else:
            confidence = 1.0  # only one hand has any reference colors at all
        hands.append(best_hand)
        confidences.append(float(confidence))
    return hands, confidences


def lit_delta(sample_rgb: Sequence[float], baseline_rgb: Sequence[float]) -> float:
    """Euclidean RGB distance between a sampled key patch and that key's
    unlit baseline colour."""
    a = np.asarray(sample_rgb, dtype=np.float64)
    b = np.asarray(baseline_rgb, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def is_lit(sample_rgb: Sequence[float], baseline_rgb: Sequence[float], threshold: float = LIT_THRESHOLD) -> bool:
    return lit_delta(sample_rgb, baseline_rgb) > threshold


def lit_end_time(
    samples: Sequence[tuple],
    baseline_rgb: Sequence[float],
    onset_sec: float,
    threshold: float = LIT_THRESHOLD,
    gap_samples: int = LIT_END_GAP_SAMPLES,
) -> Optional[float]:
    """First time >= `onset_sec` at which, for `gap_samples` consecutive
    samples, the key reads as NOT lit -- the render's ground truth for key
    release (mirrors reconcile.py's `finger_leave_time`, same gap-tolerant
    walk). ``samples`` is a time-sorted ``[(time_sec, (r,g,b)), ...]``
    covering at least ``onset_sec`` onward. Returns ``None`` if the key
    stays lit through all samples, or there are none at/after `onset_sec` --
    the caller must leave that note's duration untouched (flag more than
    guess), never invent a release time.

    Before counting a not-lit run toward release, this waits for the key to
    actually be OBSERVED lit at least once from `onset_sec` onward. A
    render's visual glow isn't always frame-perfectly synced to the audio
    onset it's paired with (discovered via real R3 validation: the very
    first post-onset sample(s) can still be mid-attack-ramp, reading as
    not-lit) -- without this guard, that brief pre-glow gap looked
    indistinguishable from an instant release AT onset. If the key is never
    observed lit at all in the sampled window, this returns None rather than
    reporting that spurious near-onset "release" -- same undeterminable-case
    handling as no samples at all."""
    start_idx = None
    for i, (t, _rgb) in enumerate(samples):
        if t >= onset_sec:
            start_idx = i
            break
    if start_idx is None:
        return None

    lit_seen_idx = None
    for i in range(start_idx, len(samples)):
        if is_lit(samples[i][1], baseline_rgb, threshold):
            lit_seen_idx = i
            break
    if lit_seen_idx is None:
        return None

    gap = 0
    for i in range(lit_seen_idx, len(samples)):
        t, rgb = samples[i]
        if is_lit(rgb, baseline_rgb, threshold):
            gap = 0
        else:
            gap += 1
            if gap > gap_samples:
                return samples[i - gap_samples][0]
    return None


def trim_note_durations(notes: Sequence, lit_end_times: Sequence[Optional[float]], min_duration_sec: float = MIN_TRIMMED_DURATION_SEC):
    """Rewrites each note's `duration_sec` to end at its lit-key release
    instead of Kong's audio-detected (often pedal-inflated) offset -- the
    fix for "no longer-than-necessary notes" while a separate sustain-pedal
    CC64 track (untouched by this) carries the actual pedal information.

    `lit_end_times` is parallel to `notes`: the absolute lit-key-release
    time, or None if it couldn't be determined for that note (occluded
    sampling, ran out of video, etc.) -- that note's original duration is
    left untouched rather than guessed. Never EXTENDS a note past its
    audio-detected duration (`min(original, lit-based)`), and reverts the
    trim entirely if it would leave the note shorter than
    `min_duration_sec` (likely a sampling glitch, not a real early
    release), mirroring reconcile.py's de-pedal guardrails on the
    real-hands path.
    """
    trimmed = []
    for note, lit_end in zip(notes, lit_end_times):
        if lit_end is None:
            trimmed.append(note)
            continue
        candidate_duration = lit_end - note.start_sec
        new_duration = min(note.duration_sec, candidate_duration)
        if new_duration < min_duration_sec:
            trimmed.append(note)
            continue
        trimmed.append(replace(note, duration_sec=new_duration))
    return trimmed


# --------------------------------------------------------------------------
# cv2 video sampling -- not covered by selftest.py, verified against real
# render footage
# --------------------------------------------------------------------------

def _sample_patch(frame_bgr, x: float, y: float, radius: int = 3):
    """Mean (r, g, b) of a small square patch centered at (x, y) in a cv2
    BGR frame, clamped to the frame bounds. Averaging a few pixels (not
    just one) absorbs compression noise/anti-aliasing at the key edge."""
    h, w = frame_bgr.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
    y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return (0.0, 0.0, 0.0)
    mean_bgr = patch.reshape(-1, 3).mean(axis=0)
    return (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))


def sample_key_baselines(video_path: str, calib, pitches: Sequence[int], n_samples: int = 8, patch_radius: int = 3) -> dict:
    """Per-pitch unlit baseline colour: the MEDIAN (robust to a minority of
    lit samples, unlike a mean) of that key's patch colour at `n_samples`
    times spread evenly across the whole video. A handful of seek+read
    calls (not a full decode) -- cheap relative to the note-sampling pass
    below.
    """
    import cv2  # lazy import

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps if fps > 0 else 0.0
    if duration <= 0:
        cap.release()
        return {p: (128.0, 128.0, 128.0) for p in pitches}

    sample_times = [duration * (i + 1) / (n_samples + 1) for i in range(n_samples)]
    samples_per_pitch = {p: [] for p in pitches}
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        for p in pitches:
            x, y = calib.pitch_to_screen(p)
            samples_per_pitch[p].append(_sample_patch(frame, x, y, patch_radius))
    cap.release()

    baselines = {}
    for p in pitches:
        arr = np.asarray(samples_per_pitch[p], dtype=np.float64)
        baselines[p] = tuple(np.median(arr, axis=0)) if arr.size else (128.0, 128.0, 128.0)
    return baselines


def sample_notes(video_path: str, calib, notes: Sequence, fps: float = 30.0, lookahead_sec: float = 4.0, patch_radius: int = 3):
    """ONE sequential decode pass over `video_path` (no per-note seeks --
    real renders can have thousands of notes, and seeking that many times
    would be far slower than one forward walk). For each note, samples its
    key's calibrated pixel patch on every sampled frame from just before
    its onset through `lookahead_sec` later, which is enough forward
    context for `lit_end_time` to find the release even for a long
    sustained note.

    Returns a list (parallel to `notes`) of dicts:
      {'onset_rgb': (r,g,b) | None, 'samples': [(time_sec, (r,g,b)), ...]}
    `onset_rgb` is the first sample at/after the note's onset (feeds hand
    clustering); `samples` feeds `lit_end_time` (release detection).
    """
    import cv2  # lazy import

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    # Sanity check: if src_fps is 0 or way too low, use the requested fps
    if src_fps <= 1.0:
        src_fps = fps
    frame_stride = max(1, round(src_fps / fps))

    per_note_samples = [[] for _ in notes]
    max_end = max((n.start_sec + lookahead_sec for n in notes), default=0.0)
    # Safeguard: never read more than 10x the expected frames (catches broken files
    # where read() returns ok=True forever, or variable-framerate YouTube videos
    # that confuse the fps detection).
    max_frames = int(max_end * src_fps * 1.5) + 1000

    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_stride == 0:
            t = idx / src_fps
            if t > max_end:
                break
            for i, note in enumerate(notes):
                if note.start_sec - 0.05 <= t <= note.start_sec + lookahead_sec:
                    x, y = calib.pitch_to_screen(note.pitch)
                    per_note_samples[i].append((t, _sample_patch(frame, x, y, patch_radius)))
        idx += 1
    cap.release()

    results = []
    for samples in per_note_samples:
        onset_rgb = samples[0][1] if samples else None
        results.append({"onset_rgb": onset_rgb, "samples": samples})
    return results


def _show_proceed_dialog(title: str, message: str) -> None:
    """Show a Tkinter dialog with a Proceed button to confirm next step."""
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    dialog = tk.Toplevel(root)
    dialog.title(title)
    dialog.resizable(False, False)

    # Message
    label = tk.Label(dialog, text=message, font=("Arial", 11), padx=20, pady=20)
    label.pack()

    # Proceed button
    def on_proceed():
        dialog.destroy()
        root.destroy()

    btn = tk.Button(dialog, text="✓ Proceed to Analysis", font=("Arial", 12), padx=20, pady=10, command=on_proceed)
    btn.pack(pady=10)

    dialog.wait_window()


def interactive_pick_hand_colors(video_path: str) -> dict:
    """Let the user scrub through `video_path` and click directly on lit keys
    to manually specify the four hand/key-type reference colors (see
    HAND_COLOR_LABELS), for renders where `assign_hands_for_notes`'s automatic
    hue clustering finds the wrong split (see module docstring). Mirrors
    keyboard.py's `select_calibration_frame` (scrub loop) + `interactive_
    calibrate`'s `_open_key_picker` (click -> Tkinter label popup) patterns.

    ESC finishes, but only once at least one LH_* and one RH_* label has been
    picked (either key-type variant satisfies either hand) -- otherwise prints
    a message and keeps looping, mirroring this module's "don't guess" spirit:
    a hand with zero reference colors can't be matched against at all.

    Returns a dict with a SUBSET of HAND_COLOR_LABELS as keys -> (r,g,b)
    tuples (only the labels actually picked; missing black-key variants are
    the caller's responsibility to fall back, per assign_hands_from_reference_
    colors' docstring).
    """
    import cv2  # lazy import

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 10000  # fallback for streams

    MAX_DISPLAY_DIM = 2450  # same display-scale-safety pattern as keyboard.py
    current_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame")

    h, w = frame.shape[:2]
    scale = min(1.0, MAX_DISPLAY_DIM / max(w, h))

    def _display_of(full_frame):
        if scale < 1.0:
            return cv2.resize(full_frame, (int(round(w * scale)), int(round(h * scale))))
        return full_frame.copy()

    picks: dict = {}  # label -> (r, g, b)
    window = "Pick hand colors: arrows scrub, click a lit key, ESC when done (LH+RH required)"

    def _open_label_picker(rgb):
        """Tkinter popup with the 4 HAND_COLOR_LABELS as buttons -- same
        root.withdraw()/Toplevel/wait_window()/root.destroy() shape as
        keyboard.py's _open_key_picker. Returns the chosen label, or None
        if the popup was closed without a choice."""
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.withdraw()
        picker = tk.Toplevel(root)
        picker.title("Which key/hand did you click?")

        # Fit to screen: leave 80px margin on each side
        screen_w = picker.winfo_screenwidth()
        screen_h = picker.winfo_screenheight()
        margin = 80
        max_w = max(500, screen_w - 2 * margin)
        max_h = max(120, screen_h - 2 * margin)
        picker.geometry(f"{max_w}x{max_h}")

        # Center on screen
        picker.update_idletasks()
        x = max(0, (screen_w - max_w) // 2)
        y = max(0, (screen_h - max_h) // 2)
        picker.geometry(f"{max_w}x{max_h}+{x}+{y}")

        style = ttk.Style(picker)
        style.configure("Big.TButton", font=("Arial", 16))

        hexcode = "#%02x%02x%02x" % tuple(int(round(c)) for c in rgb)
        swatch = tk.Frame(picker, bg=hexcode, width=60, height=60)
        swatch.pack(pady=(12, 6))
        ttk.Label(picker, text=f"Sampled colour: {hexcode}").pack()

        result = {"label": None}

        def on_select(label: str):
            result["label"] = label
            picker.destroy()

        frame_inner = ttk.Frame(picker)
        frame_inner.pack(padx=18, pady=18, fill="both", expand=True)
        for label in HAND_COLOR_LABELS:
            ttk.Button(
                frame_inner, text=label, width=12, command=lambda l=label: on_select(l),
                style="Big.TButton",
            ).pack(side="left", padx=6, pady=6)

        picker.protocol("WM_DELETE_WINDOW", picker.destroy)
        picker.wait_window()
        root.destroy()
        return result["label"]

    def on_click(event, x, y, flags, userdata):  # noqa: ANN001
        nonlocal frame
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Map display-scaled click coords back to full-resolution pixel space
        # before sampling -- same reasoning as interactive_calibrate's docstring
        # re: window-manager resize mismatches.
        full_x, full_y = x / scale, y / scale
        rgb = _sample_patch(frame, full_x, full_y, radius=3)
        label = _open_label_picker(rgb)
        if label is not None:
            picks[label] = rgb

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)

    ARROW_LEFT = {65361, 81, 2424832}
    ARROW_RIGHT = {65363, 83, 2555904}

    import time
    start_time = time.time()
    timeout_sec = 60  # Auto-proceed with defaults if no input

    while True:
        disp = _display_of(frame)
        y = 40
        cv2.putText(disp, f"Frame {current_idx}/{total_frames} | Arrows: ±10 | PgUp/PgDn: ±150 | click a lit key | P: proceed | ESC: done",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(disp, f"Frame {current_idx}/{total_frames} | Arrows: ±10 | PgUp/PgDn: ±150 | click a lit key | P: proceed | ESC: done",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        for label in HAND_COLOR_LABELS:
            y += 32
            rgb = picks.get(label)
            status = "#%02x%02x%02x" % tuple(int(round(c)) for c in rgb) if rgb else "not set"
            color_bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0])) if rgb else (128, 128, 128)
            cv2.putText(disp, f"{label}: {status}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(disp, f"{label}: {status}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 1, cv2.LINE_AA)

        cv2.imshow(window, disp)
        key = cv2.waitKeyEx(30)

        # Check for timeout (auto-proceed with defaults if no input)
        if time.time() - start_time > timeout_sec:
            print(f"\n>>> No colors picked for {timeout_sec}s - using default colors...", flush=True)
            # Use default cyan for left, magenta for right
            if not picks:
                picks = {
                    "LH_white": (100, 150, 220),  # cyan-ish for left
                    "RH_white": (200, 50, 200),   # magenta for right
                }
            break

        if key == -1:
            continue
        key_ascii = key & 0xFF

        if key in ARROW_LEFT:
            current_idx = max(0, current_idx - 10)
        elif key in ARROW_RIGHT:
            current_idx = min(total_frames - 1, current_idx + 10)
        elif key in {65365, 2555904}:  # PAGEUP
            current_idx = max(0, current_idx - 150)
        elif key in {65366, 2621440}:  # PAGEDOWN
            current_idx = min(total_frames - 1, current_idx + 150)
        elif key_ascii == 112 or key_ascii == 80:  # P or p: proceed
            if not picks:
                print(">>> No colors picked yet - using defaults (LH white: cyan, RH white: magenta)", flush=True)
                picks = {
                    "LH_white": (100, 150, 220),  # cyan
                    "RH_white": (200, 50, 200),   # magenta
                }
            print(f"✓ Proceeding with colors: {picks}", flush=True)
            break
        elif key_ascii == 27:  # ESC
            hands_picked = {_LABEL_TO_HAND[label] for label in picks}
            if "L" in hands_picked and "R" in hands_picked:
                break
            print("Need at least one LH_* and one RH_* colour picked before finishing "
                  f"(have: {sorted(picks.keys()) or 'none'}).")
            continue
        else:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        ok, new_frame = cap.read()
        if ok:
            frame = new_frame

    cap.release()
    cv2.destroyWindow(window)

    # Show confirmation dialog with Proceed button
    _show_proceed_dialog("Hand Colors Complete", f"Hand colors saved:\n{picks}\n\nReady to proceed to analysis?")

    return picks
