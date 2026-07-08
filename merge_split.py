"""Merge-split: recover the second hand when MediaPipe merges two adjacent hands.

The hands-close-together failure mode (a ballad where both hands play in the
same mid-register): MediaPipe's palm detector sees the two touching hands as
ONE palm, returns a single hand, and every note the other hand played comes
back no-finger-support. Neither the crop nor CLAHE helps -- it's a detection
limit on adjacent hands, not a visibility problem.

The fix: when only one hand is tracked but the MIDI notes sounding at that
instant clearly split into a LOW cluster and a HIGH cluster (a gap wider than
one hand can span), we know two hands are present. We compute a **split point**
between the clusters and synthesize a coarse observation for the missing hand
at the centroid of the uncovered cluster, labelled the opposite hand -- enough
for the matcher to give those notes the correct L/R, which is the goal.

Split point, strongest signal first:
  1. MIDI cluster gap -- the midpoint (in white-key space) of the widest gap
     between the sounding pitches, when that gap marks two hand-sized groups.
  2. Last known L/R positions -- for continuity / tie-breaking when the MIDI
     is ambiguous (few notes sounding).

Pure python (no cv2/numpy) so the split logic is cheap and unit-testable; the
optional HSV blob signal is a separate, video-tuned layer added on top later.
"""

from __future__ import annotations

from dataclasses import dataclass

# A hand spans ~1 octave, ~1.5 with a stretch. A gap between sounding-note
# groups wider than this means the notes need two hands, not one.
SPLIT_GAP_SEMITONES = 15
# Each side of the split must itself be within one hand's reach to be credible
# as a single hand (else it's three+ voices / a glissando, not a clean split).
MAX_SIDE_SPAN_SEMITONES = 20


@dataclass
class SplitResult:
    split_white_index: float   # boundary in pitch_to_white_index units
    split_screen_x: float      # boundary x in screen pixels (via calib), or nan
    low_pitches: list          # sounding pitches assigned to the LOW (LH) side
    high_pitches: list         # ...to the HIGH (RH) side


def sounding_pitches(midi_notes, t_video: float, offset_sec: float, pad: float = 0.03):
    """Pitches of notes sounding at video time `t_video` (start..end inclusive,
    with a small pad). `midi_notes` need `.pitch`/`.start_sec`/`.duration_sec`."""
    out = []
    for n in midi_notes:
        s = float(n.start_sec) + offset_sec
        e = s + max(float(n.duration_sec), 0.0)
        if s - pad <= t_video <= e + pad:
            out.append(int(n.pitch))
    return out


def compute_split(pitches, calib=None,
                  split_gap: int = SPLIT_GAP_SEMITONES,
                  max_side_span: int = MAX_SIDE_SPAN_SEMITONES):
    """Determine a two-hand split point from the pitches sounding at one
    instant. Returns a SplitResult, or None if the notes don't clearly need
    two hands (span fits one hand, or no clean gap).

    The split is placed at the WIDEST gap between adjacent sorted pitches, if
    that gap is at least `split_gap` semitones and each resulting side spans no
    more than `max_side_span` (so it reads as two hand-sized groups).
    """
    from keyboard import pitch_to_white_index  # lazy

    ps = sorted(set(int(p) for p in pitches))
    if len(ps) < 2:
        return None
    if ps[-1] - ps[0] <= split_gap:
        return None  # everything fits within one hand's reach -> one hand

    # Widest gap between adjacent pitches.
    best_gap, cut = -1, 1
    for i in range(1, len(ps)):
        g = ps[i] - ps[i - 1]
        if g > best_gap:
            best_gap, cut = g, i
    if best_gap < split_gap:
        return None

    low, high = ps[:cut], ps[cut:]
    if (low[-1] - low[0]) > max_side_span or (high[-1] - high[0]) > max_side_span:
        return None  # a side is too wide to be a single hand

    split_pitch = (low[-1] + high[0]) / 2.0
    split_wi = pitch_to_white_index(int(round(split_pitch)))

    split_x = float("nan")
    if calib is not None:
        try:
            from keyboard import _calib_u_to_screen  # lazy
            u = calib.pitch_to_norm_x(int(round(split_pitch)))
            split_x = float(_calib_u_to_screen(calib, u)[0])
        except Exception:
            pass

    return SplitResult(split_white_index=split_wi, split_screen_x=split_x,
                       low_pitches=low, high_pitches=high)


def _cluster_centroid_screen(pitches, calib):
    """Screen (x, y) at the centroid pitch of a cluster, via calib. None on
    failure or missing calib."""
    if calib is None or not pitches:
        return None
    try:
        from keyboard import _calib_u_to_screen  # lazy
        c = sum(pitches) / len(pitches)
        u = calib.pitch_to_norm_x(int(round(c)))
        x, y = _calib_u_to_screen(calib, u)
        return float(x), float(y)
    except Exception:
        return None


@dataclass
class SplitStats:
    one_hand_frames: int = 0     # frames with exactly one tracked hand
    split_candidates: int = 0    # ...where MIDI indicated two hands
    hands_synthesized: int = 0   # coarse missing-hand observations added

    def summary(self) -> str:
        if self.split_candidates == 0:
            return "merge-split: no merged-hand frames found"
        return (
            f"merge-split: recovered {self.hands_synthesized} merged second-hand(s) "
            f"of {self.split_candidates} candidate frame(s)"
        )


def split_merged_hands(frames, calib, midi_notes, offset_sec: float = 0.0) -> SplitStats:
    """For frames with exactly one tracked hand where the sounding MIDI notes
    split into two hand-sized clusters, synthesize a coarse observation for the
    missing hand at the uncovered cluster's centroid (opposite L/R label).
    Mutates `frames`; returns SplitStats. The caller should re-run
    `fix_handedness_continuity` afterwards.

    Requires calibration (for the key->screen projection) and MIDI notes.
    """
    from hands import HandObservation, _obs_screen_xy  # lazy

    stats = SplitStats()
    if not frames or calib is None or not midi_notes:
        return stats

    for frame in frames:
        if len(frame.hands) != 1:
            continue
        stats.one_hand_frames += 1
        pitches = sounding_pitches(midi_notes, frame.time_sec, offset_sec)
        split = compute_split(pitches, calib)
        if split is None:
            continue
        stats.split_candidates += 1

        obs = frame.hands[0]
        xy = _obs_screen_xy(obs)
        if xy is None:
            continue
        try:
            tracked_wi = calib.screen_to_white_index(xy)
        except Exception:
            continue

        # Which side does the tracked hand sit on? Fill the OTHER side.
        tracked_is_low = tracked_wi <= split.split_white_index
        missing_pitches = split.high_pitches if tracked_is_low else split.low_pitches
        centroid = _cluster_centroid_screen(missing_pitches, calib)
        if centroid is None:
            continue

        # The tracked hand keeps its own MediaPipe label; the synthesized hand
        # is the opposite. (fix_handedness_continuity re-orders L/R by geometry
        # afterwards, so a wrong guess here is corrected.)
        missing_hand = "R" if obs.hand == "L" else "L"
        synth = HandObservation(hand=missing_hand, fingertips=[],
                                wrist=centroid, palm=centroid)
        frame.hands.append(synth)
        stats.hands_synthesized += 1

    return stats
