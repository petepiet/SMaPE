"""Post-tracking reconstruction pass (pipeline stages 3 + 4).

This sits BETWEEN hand tracking (``hands.extract_fingertip_frames`` +
``fix_handedness_continuity``) and note<->finger matching. It does NOT
invent new landmarks (that would be full trajectory reconstruction, a
separate, riskier stage); it only *removes* points that are physically
implausible, so the existing gap-tolerant ``interpolate_fingertips`` bridges
them instead of the matcher trusting a glitch.

Two stages, deliberately conservative:

Stage 3 -- kinematic outlier rejection
    A fingertip cannot teleport. For each (hand, finger) trajectory we look
    for an isolated *spike*: a point that jumps far from BOTH its temporal
    neighbours while those neighbours agree with each other (it darts out and
    snaps back within a couple of frames). That is the signature of a
    single-frame detection glitch, not real motion -- a genuine fast run
    moves monotonically, it does not return to where it started one frame
    later. Sustained motion is never touched.

Stage 4 -- MIDI onset protection (correction constraint)
    The strongest ground truth available: a MIDI onset at pitch P at time t
    means SOME finger really was on key P's position at t. So a point that
    stage 3 would call a "spike" but which actually lands on a key being
    played at that instant is a real, fast press -- MIDI *protects* it from
    removal. This keeps the pass from eating exactly the fast staccato notes
    it should preserve. When no MIDI is supplied the stage-3 thresholds alone
    apply (still safe: a wrongly-removed point only degrades to a gap).

Why removal-only (no injection): today a missing point -> no-finger-support
-> low confidence -> a *visible* failure. A wrongly *invented* point ->
high confidence -> a *silent* failure. Removing a glitch degrades gracefully
to the first case; inventing a point risks the second. So this pass only ever
subtracts.

Measured in depth-invariant white-key-index units (via
``calib.screen_to_white_index``) so thresholds are resolution- and
zoom-independent and live in the same space the matcher uses. Falls back to
raw pixel-x when no calibration is available. Pure python -- no numpy, no cv2
-- so it is cheap (well under a second for a few thousand frames) and unit
-testable without the heavy tracking stack.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Stage 3 (kinematic) tuning -------------------------------------------
# A spike must jump more than this many white keys away from BOTH neighbours.
# ~3 white keys ≈ nearly half an octave in a single sampled frame: real finger
# motion at 30 fps rarely does that AND reverses next frame.
SPIKE_KEYS = 3.0
# ...and the two neighbours must agree with each other to within this fraction
# of the smaller jump -- i.e. the point darts out and comes back. Prevents
# flagging the genuine start/end of a fast run (where neighbours disagree).
SPAN_RATIO = 0.5
# Neighbours are only comparable if both sit within this time gap of the
# suspect point; across a longer dropout the "spike" test is meaningless.
NEIGHBOR_GAP_SEC = 0.12

# --- Stage 4 (MIDI protection) tuning -------------------------------------
# A suspect point is protected if a MIDI onset falls within this time window...
PRESS_WINDOW_SEC = 0.08
# ...and that onset's key is within this many white keys of the suspect point.
PRESS_KEYS = 1.5


@dataclass
class ReconstructStats:
    frames: int = 0
    trajectory_points: int = 0
    spikes_found: int = 0
    removed: int = 0
    midi_protected: int = 0

    def summary(self) -> str:
        if self.spikes_found == 0:
            return f"reconstruction: no kinematic outliers in {self.trajectory_points} tracked points"
        prot = (
            f", {self.midi_protected} kept (real presses confirmed by MIDI)"
            if self.midi_protected
            else ""
        )
        return (
            f"reconstruction: removed {self.removed} teleporting outlier(s) "
            f"of {self.trajectory_points} tracked points{prot}"
        )


def _wi(calib, x: float, y: float) -> float:
    """Depth-invariant white-key index for a pixel, or raw x without calib."""
    if calib is None:
        return x
    try:
        return calib.screen_to_white_index((x, y))
    except Exception:
        return x


def _build_onset_index(midi_notes, offset_sec: float):
    """Sorted list of (video_time_sec, expected_white_index) MIDI onsets.

    Requires each note to expose ``.pitch`` and ``.start_sec``. Returns [] if
    no usable notes were supplied.
    """
    if not midi_notes:
        return []
    from keyboard import pitch_to_white_index  # lazy: avoids import at module load

    onsets = []
    for n in midi_notes:
        try:
            onsets.append((float(n.start_sec) + offset_sec, pitch_to_white_index(n.pitch)))
        except Exception:
            continue
    onsets.sort(key=lambda o: o[0])
    return onsets


def _is_midi_protected(onsets, t: float, wi: float,
                       window: float, keys: float) -> bool:
    """True if a MIDI onset near time ``t`` expects a key near white-index
    ``wi`` -- i.e. the suspect point is really a fast press, not a glitch."""
    if not onsets:
        return False
    import bisect

    times = [o[0] for o in onsets]
    lo = bisect.bisect_left(times, t - window)
    hi = bisect.bisect_right(times, t + window)
    for i in range(lo, hi):
        if abs(onsets[i][1] - wi) <= keys:
            return True
    return False


def reconstruct_frames(
    frames,
    calib=None,
    midi_notes=None,
    offset_sec: float = 0.0,
    spike_keys: float = SPIKE_KEYS,
    span_ratio: float = SPAN_RATIO,
    neighbor_gap_sec: float = NEIGHBOR_GAP_SEC,
    press_window_sec: float = PRESS_WINDOW_SEC,
    press_keys: float = PRESS_KEYS,
) -> ReconstructStats:
    """Remove kinematic outlier fingertips in place, protecting real presses
    with MIDI onsets. Mutates ``frames`` (drops offending ``Fingertip``s from
    their ``HandObservation.fingertips``) and returns a ``ReconstructStats``.

    ``frames`` is a list of ``hands.FingertipFrame`` (time-ordered).
    ``midi_notes`` is optional; when given, each must expose ``.pitch`` and
    ``.start_sec`` (seconds). ``offset_sec`` maps MIDI time to video time
    (video_time = midi_time + offset), matching the rest of the pipeline.
    """
    stats = ReconstructStats(frames=len(frames))
    if not frames:
        return stats

    onsets = _build_onset_index(midi_notes, offset_sec)

    # Group every tracked fingertip into per-(hand, finger) trajectories,
    # preserving frame order. Each entry keeps a direct reference to the
    # HandObservation + Fingertip so a flagged point can be removed later.
    trajectories: dict = {}
    for frame in frames:
        for obs in frame.hands:
            for tip in obs.fingertips:
                key = (obs.hand, tip.finger)
                trajectories.setdefault(key, []).append(
                    (frame.time_sec, _wi(calib, tip.x, tip.y), obs, tip)
                )

    to_remove = []  # (obs, tip) pairs
    for pts in trajectories.values():
        stats.trajectory_points += len(pts)
        # Interior points only: a spike needs a neighbour on each side.
        for i in range(1, len(pts) - 1):
            t_prev, wi_prev, _, _ = pts[i - 1]
            t_cur, wi_cur, obs, tip = pts[i]
            t_next, wi_next, _, _ = pts[i + 1]

            # Neighbours must be temporally close for the test to be valid.
            if (t_cur - t_prev) > neighbor_gap_sec or (t_next - t_cur) > neighbor_gap_sec:
                continue

            d_prev = abs(wi_cur - wi_prev)
            d_next = abs(wi_cur - wi_next)
            d_span = abs(wi_prev - wi_next)

            # Spike: far from both neighbours, which themselves agree.
            if (
                d_prev > spike_keys
                and d_next > spike_keys
                and d_span <= span_ratio * min(d_prev, d_next)
            ):
                stats.spikes_found += 1
                if _is_midi_protected(onsets, t_cur, wi_cur, press_window_sec, press_keys):
                    stats.midi_protected += 1
                    continue
                to_remove.append((obs, tip))

    for obs, tip in to_remove:
        try:
            obs.fingertips.remove(tip)
            stats.removed += 1
        except ValueError:
            pass  # already gone (shouldn't happen -- each tip is unique)

    return stats
