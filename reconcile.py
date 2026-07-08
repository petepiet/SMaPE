"""Phase D: cross-checks AMT-transcribed notes against the video's own hand
tracking, to flag/drop ghost notes and trim pedal-inflated durations.

Motivation (from a real comparison against a commercial AMT tool's export,
same source video): audio-only heuristics like a minimum velocity or note
duration cannot distinguish "confidently wrong" from "confidently right" --
a run against real data showed the disputed notes were NOT quiet or short
(median velocity 55/127, median duration 0.86s), so no volume/length
threshold could ever touch them. The video gives independent evidence audio
alone cannot: a ghost note has no finger on the key; a real one does.

Conservative by design: flag more than you drop, and never judge a note
whose onset falls in an occluded region of the video (no fingertip tracked
nearby at all) -- audio wins when the video has no evidence either way.

Pure python/numpy -- no cv2/mediapipe -- fully covered by selftest.py using
synthetic FingertipFrame data (the same convention hands.py/match.py use).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from keyboard import pitch_to_white_index

# Farther than this (in "key" units -- the same white-key-index space
# match.py matches in) from every tracked fingertip at onset = no support:
# a ghost-note candidate.
MAX_SUPPORT_DIST_KEYS = 2.5

# Only DROP (not just flag) a no-support note when it's ALSO this quiet.
# Loud, no-support notes are flagged but kept -- the video may simply have
# missed/misidentified the finger, which is common and not on its own
# reason to delete audio evidence.
GHOST_DROP_VELOCITY = 30

# No fingertip tracked within this many seconds of the onset at all = the
# video has no evidence either way here; skip all video-based judgements.
OCCLUSION_WINDOW_SEC = 0.15

# "The key is still covered" threshold for the de-pedal finger-leave scan --
# deliberately looser than MAX_SUPPORT_DIST_KEYS since any finger (not just
# the exact matched one) resting near the key counts as "still down".
FINGER_NEAR_KEYS = 1.5

# Consecutive sampled frames with no finger near the key before considering
# it "left" -- absorbs a single dropped/noisy tracking frame.
FINGER_LEAVE_GAP_FRAMES = 2

# The finger must leave at least this long before the audio-reported offset
# to bother trimming (a few ms of noise isn't worth acting on).
DEPEDAL_MIN_LEAD_SEC = 0.15

# Revert a de-pedal trim if it would leave the note shorter than this --
# likely tracking jitter, not a real early release.
MIN_TRIMMED_DURATION_SEC = 0.12


@dataclass
class ReconcileResult:
    flags: list = field(default_factory=list)
    # Set only when de-pedal trimmed: the ORIGINAL (pre-trim) duration, so
    # the caller can keep it alongside the new (trimmed) one.
    audio_duration_sec: float | None = None
    # Set only when de-pedal trimmed: the new, trimmed duration to apply.
    trimmed_duration_sec: float | None = None


def _frame_times(frames) -> list:
    """Precompute sorted frame times once, for bisect-based window lookups
    across many notes (O(log n) instead of an O(n) scan per note)."""
    return [f.time_sec for f in frames]


def is_occluded(frame_times: list, frames: list, t: float, window_sec: float = OCCLUSION_WINDOW_SEC) -> bool:
    """True if no fingertip at all was tracked within `window_sec` of `t` --
    i.e. the video gives no evidence either way at this moment. Uses
    `fingertip_positions()` (the duck-typed interface match.py/sync.py also
    rely on), not FingertipFrame's own `.hands` attribute, so this works
    with any frame-like object exposing that method (including selftest.py's
    synthetic frames, which don't have `.hands`)."""
    lo = bisect.bisect_left(frame_times, t - window_sec)
    hi = bisect.bisect_right(frame_times, t + window_sec)
    for f in frames[lo:hi]:
        for _ in f.fingertip_positions():
            return False
    return True


def finger_leave_time(
    frame_times: list,
    frames: list,
    calib,
    onset_sec: float,
    pitch: int,
    near_keys: float = FINGER_NEAR_KEYS,
    gap_frames: int = FINGER_LEAVE_GAP_FRAMES,
):
    """First time >= `onset_sec` at which, for `gap_frames` consecutive
    sampled frames, no fingertip (any hand/finger -- not necessarily the one
    matched to this note) is within `near_keys` of `pitch`'s key. Returns
    None if the key stays covered through the end of tracking (or there are
    no frames at/after `onset_sec` at all)."""
    start_idx = bisect.bisect_left(frame_times, onset_sec)
    key_x = pitch_to_white_index(pitch)
    gap = 0
    for i in range(start_idx, len(frames)):
        f = frames[i]
        near = any(
            abs(calib.screen_to_white_index((x, y)) - key_x) <= near_keys
            for _h, _fi, x, y in f.fingertip_positions()
        )
        if near:
            gap = 0
        else:
            gap += 1
            if gap > gap_frames:
                return frames[i - gap_frames].time_sec
    return None


def reconcile_note(
    velocity: int,
    duration_sec: float,
    video_onset_sec: float,
    pitch: int,
    match_distance_keys,
    frame_times: list,
    frames: list,
    calib,
    pedal_segments=None,
) -> ReconcileResult:
    """Reconciles one transcribed note against the video's hand tracking.

    `match_distance_keys` is the matcher's own `MatchResult.distance` (see
    match.py) -- already computed in the SAME white-key-index space this
    module works in, so no re-computation of "nearest fingertip" is needed
    for the ghost-note check; it's reused directly. `pedal_segments` (if
    given) is a list of (start_sec, end_sec) sustain-pedal-down intervals in
    the same time base as `video_onset_sec`.
    """
    result = ReconcileResult()
    occluded = is_occluded(frame_times, frames, video_onset_sec)
    if occluded:
        return result  # no video evidence here -- audio wins, untouched

    if match_distance_keys is not None and match_distance_keys > MAX_SUPPORT_DIST_KEYS:
        result.flags.append("no-finger-support")
        if velocity < GHOST_DROP_VELOCITY:
            result.flags.append("dropped")

    if pedal_segments:
        audio_offset_sec = video_onset_sec + duration_sec
        # <= e, not < e: a pedal-inflated note's audio-detected offset IS the
        # moment the pedal releases (that's WHY it's inflated) -- it lands
        # exactly at or a hair before the segment end, not strictly inside it.
        in_pedal = any(s <= audio_offset_sec <= e for s, e in pedal_segments)
        if in_pedal:
            leave = finger_leave_time(frame_times, frames, calib, video_onset_sec, pitch)
            if leave is not None and (audio_offset_sec - leave) >= DEPEDAL_MIN_LEAD_SEC:
                trimmed_duration_sec = leave - video_onset_sec
                if trimmed_duration_sec >= MIN_TRIMMED_DURATION_SEC:
                    result.flags.append("depedaled")
                    result.audio_duration_sec = duration_sec
                    result.trimmed_duration_sec = trimmed_duration_sec

    return result


def confidence_by_window(notes_with_confidence, window_sec: float = 10.0) -> list:
    """Buckets (start_sec, confidence) pairs into `window_sec`-wide windows
    and returns [(window_start_sec, mean_confidence, note_count), ...],
    sorted by window. A declining trend across windows over a long song is
    the signature of video/MIDI sync drift (a single constant offset can't
    fix it) rather than a uniformly-bad calibration -- see README 'Sync'.
    """
    buckets: dict = {}
    for start_sec, confidence in notes_with_confidence:
        w = int(start_sec // window_sec)
        buckets.setdefault(w, []).append(confidence)
    return [
        (w * window_sec, sum(vals) / len(vals), len(vals))
        for w, vals in sorted(buckets.items())
    ]


# Below this fraction of sampled frames showing BOTH hands, the tracking is
# assumed to have silently dropped one hand (rather than the pianist actually
# playing one-handed): on a real two-handed cover video where MediaPipe's
# detector sat just below threshold, the dual-hand rate measured 2% -- while
# lowering min_hand_confidence recovered 90%. A genuinely one-handed piece
# would also trip this, which is fine: the retry at a lower threshold is
# harmless there (it just confirms what's visible).
DUAL_HAND_RATE_MIN = 0.2

# The confidence extract_fingering.py retries tracking at when the dual-hand
# rate comes back below DUAL_HAND_RATE_MIN. Measured 93% dual-hand detection
# at 0.15 on the same video that gave 2% at 0.5.
RETRY_HAND_CONFIDENCE = 0.15


def dual_hand_rate(frames) -> float:
    """Fraction of frames in which BOTH hands ('L' and 'R') have at least one
    tracked point. The cheap post-tracking health check for the most damaging
    silent failure mode: MediaPipe dropping one hand entirely (every note then
    matches the surviving hand's label and the other hand's notes all come
    back no-finger-support with ~0 confidence). Duck-typed on
    `fingertip_positions()` like everything else here, so selftest.py can
    feed it synthetic frames. Returns 0.0 for an empty frame list."""
    if not frames:
        return 0.0
    both = 0
    for f in frames:
        hands = {h for h, _fi, _x, _y in f.fingertip_positions()}
        if "L" in hands and "R" in hands:
            both += 1
    return both / len(frames)


# Hands closer together than this (white-key units) make the "nearest tracked
# fingertip" check in reconcile_note ambiguous between hands -- fingertips
# from BOTH hands then compete within the same few-key radius, well under
# MAX_SUPPORT_DIST_KEYS, in a way register-separated playing never does.
CLOSE_HANDS_THRESHOLD_KEYS = 3.0

# Below this mean hand-to-hand distance (white keys), extract_fingering.py
# prints an upfront warning that no-finger-support is likely to run high.
# Chosen well above CLOSE_HANDS_THRESHOLD_KEYS: even a piece that dips close
# together occasionally (chord unisons, hand-offs) shouldn't trip this --
# only a piece where the hands stay close together ON AVERAGE should.
HAND_SPREAD_WARN_THRESHOLD_KEYS = 4.0


@dataclass
class HandSpreadReport:
    frames_with_both_hands: int
    mean_distance_keys: float
    median_distance_keys: float
    # Fraction of dual-hand frames closer than CLOSE_HANDS_THRESHOLD_KEYS.
    frac_close: float


def analyze_hand_spread(frames, calib) -> HandSpreadReport | None:
    """Diagnostic: how far apart the two hands stay, across frames where both
    are tracked simultaneously. Predicts no-finger-support risk BEFORE the
    (expensive) matching pass runs: a piece where both hands work in the same
    narrow register (chordal comping, hands close together or crossing --
    common in funk/soul keys parts) pushes fingertips from BOTH hands into the
    same few-key radius, confusing the "nearest tracked fingertip within
    MAX_SUPPORT_DIST_KEYS" check far more than register-separated playing
    (bass LH / melody RH) ever does. Printed by extract_fingering.py right
    after hand-tracking finishes, before the (much slower) matching pass.

    Distance is measured between each hand's fingertip CENTROID (mean of
    whichever fingertips were tracked that frame), in the same depth-invariant
    white-key-index space `screen_to_white_index` uses for matching.

    Returns None if no frame has both hands tracked at once (nothing to
    compare -- e.g. a one-handed piece, or hands never both in frame).
    """
    distances = []
    for frame in frames:
        by_hand: dict = {}
        for hand, _finger, x, y in frame.fingertip_positions():
            by_hand.setdefault(hand, []).append(calib.screen_to_white_index((x, y)))
        left = by_hand.get('L')
        right = by_hand.get('R')
        if left and right:
            l_center = sum(left) / len(left)
            r_center = sum(right) / len(right)
            distances.append(abs(r_center - l_center))

    if not distances:
        return None

    distances.sort()
    n = len(distances)
    median = distances[n // 2] if n % 2 else (distances[n // 2 - 1] + distances[n // 2]) / 2
    close_count = sum(1 for d in distances if d < CLOSE_HANDS_THRESHOLD_KEYS)
    return HandSpreadReport(
        frames_with_both_hands=n,
        mean_distance_keys=sum(distances) / n,
        median_distance_keys=median,
        frac_close=close_count / n,
    )
