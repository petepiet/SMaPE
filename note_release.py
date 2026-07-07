"""Note release correction: computes, for every MATCHED note, a physical
key-release time (`corrected_end`) distinct from the audible sound end.

Motivation: audio transcription (Kong AMT) reports note-offs that are too
late whenever sustain/pedal resonance inflates the perceived duration of a
note. That inflated duration is *correct* as a description of the SOUND
(when the ear stops hearing the note) but *wrong* as a description of the
KEY (when the finger actually left the key). This module derives the
latter -- the physical key-release time -- using geometric rules (what a
human hand can physically do: it can't still be pressing a key when the
same finger is already starting the next note) plus, where available and
trusted, the video's own finger tracking.

All times here are in the MIDI/note timeline (`note.start_sec`), NOT video
time -- callers are responsible for any offset translation before/after
calling into this module.

Per matched note:
    start        = note.start_sec
    sound_end    = start + note.duration_sec   (unchanged audio offset;
                   may be pedal-inflated -- that is correct, it's a real
                   description of the sound)
    corrected_end = the physical key-release time (this module's output)
    key_duration  = corrected_end - start
    sound_duration = sound_end - start

Invariant: key_duration <= sound_duration always holds, because
`original_end_sec` (== sound_end) is itself one of the candidate caps this
module chooses from (i.e. corrected_end can never exceed it), and sustain
pedal never enters the corrected_end computation at all.

corrected_end is the SMALLEST of the available "caps" below. On (near-)
ties in value, the cap with the higher priority (lower rank number) wins,
and its reason string is attributed as `release_reason`:

    rank 2  REASON_SAME_FINGER  next start of the next note played by the
            SAME (hand, finger), minus a small margin -- a real hand
            physically cannot still be holding this key down once that
            same finger is already pressing the next key.
    rank 3  REASON_SAME_PITCH   next start of the next note at the SAME
            pitch (a restrike) minus the same margin -- the key must have
            come back up before it can be struck again.
    rank 4  REASON_VISUAL       a TRUSTED visual finger-leave time, if the
            caller supplies one (may be None -- this module never computes
            visual evidence itself, only consumes it).
    rank 5  REASON_ORIGINAL     start + note.duration_sec, i.e. sound_end --
            the fallback when no geometric or visual evidence disagrees
            with the audio.
    rank 6  REASON_FALLBACK     start + FALLBACK_DURATION_SEC, used ONLY
            when no other cap is available at all (e.g. no MIDI duration
            and no visual evidence).

`REASON_FINGER_LEFT` ("finger_left_key") is reserved for a future,
per-finger-specific visual release reason distinct from the current single
collapsed REASON_VISUAL bucket -- nothing emits it yet.

Pure python, no cv2/mediapipe/mido dependency -- fully unit-testable like
reconcile.py.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default guard margin (seconds) subtracted from a "next note" start time
# when using it as a cap: the key must be released strictly before the next
# onset, not merely by the time it arrives.
DEFAULT_MARGIN_SEC = 0.015

# Never report a key duration shorter than this -- a near-zero or negative
# corrected duration is almost certainly a tracking/geometry artifact, not a
# real instantaneous key press.
MIN_KEY_DURATION_SEC = 0.03

# Used only when NO other cap is available at all.
FALLBACK_DURATION_SEC = 0.3

REASON_SAME_FINGER = "same_finger_new_note"
REASON_SAME_PITCH = "same_pitch_restrike"
REASON_VISUAL = "visual_key_release"
REASON_FINGER_LEFT = "finger_left_key"  # reserved for future per-finger visual tracking
REASON_ORIGINAL = "original_midi_offset"
REASON_FALLBACK = "fallback_estimate"


@dataclass
class ReleaseInput:
    start_sec: float
    original_end_sec: "float | None"
    hand: "str | None"
    finger: "int | None"
    pitch: int
    visual_release_sec: "float | None" = None


@dataclass
class ReleaseResult:
    corrected_end_sec: float
    reason: str


def _next_greater_starts(keys, starts) -> list:
    """For each index i, returns the smallest starts[j] with starts[j] >
    starts[i] among indices sharing the same key (keys[i] == keys[j]), or
    None if there is no such index. `keys[i]` of None means "no group"
    (skipped entirely -- result is None for that index).

    Implemented by grouping indices by key, sorting each group's (start,
    idx) pairs, then scanning for the smallest strictly-greater start per
    entry (duplicate/equal starts never cap each other).
    """
    n = len(keys)
    result = [None] * n
    groups: dict = {}
    for i, k in enumerate(keys):
        if k is None:
            continue
        groups.setdefault(k, []).append(i)

    for k, idxs in groups.items():
        pairs = sorted((starts[i], i) for i in idxs)
        distinct_starts = sorted(set(s for s, _ in pairs))
        for s, i in pairs:
            # Smallest distinct start strictly greater than this note's own.
            nxt = None
            for ds in distinct_starts:
                if ds > s:
                    nxt = ds
                    break
            result[i] = nxt
    return result


def compute_corrected_ends(
    inputs: list,
    margin_sec: float = DEFAULT_MARGIN_SEC,
    min_key_duration_sec: float = MIN_KEY_DURATION_SEC,
) -> list:
    """Returns a list of ReleaseResult parallel to `inputs` -- one physical
    key-release estimate per note. See module docstring for the algorithm
    and cap priority order."""
    n = len(inputs)
    starts = [inp.start_sec for inp in inputs]

    finger_keys = [
        (inp.hand, inp.finger) if inp.hand is not None and inp.finger is not None else None
        for inp in inputs
    ]
    pitch_keys = [inp.pitch for inp in inputs]

    next_same_finger_start = _next_greater_starts(finger_keys, starts)
    next_same_pitch_start = _next_greater_starts(pitch_keys, starts)

    results = []
    for i, inp in enumerate(inputs):
        start = inp.start_sec
        candidates = []

        if next_same_finger_start[i] is not None:
            candidates.append((next_same_finger_start[i] - margin_sec, 2, REASON_SAME_FINGER))
        if next_same_pitch_start[i] is not None:
            candidates.append((next_same_pitch_start[i] - margin_sec, 3, REASON_SAME_PITCH))
        if inp.visual_release_sec is not None:
            candidates.append((inp.visual_release_sec, 4, REASON_VISUAL))
        if inp.original_end_sec is not None:
            candidates.append((inp.original_end_sec, 5, REASON_ORIGINAL))

        if not candidates:
            candidates.append((start + FALLBACK_DURATION_SEC, 6, REASON_FALLBACK))

        val, _rank, reason = min(candidates, key=lambda c: (c[0], c[1]))

        floor = start + min_key_duration_sec
        corrected = max(val, floor)
        if inp.original_end_sec is not None:
            corrected = min(corrected, inp.original_end_sec)
        corrected = max(corrected, start)

        results.append(ReleaseResult(corrected_end_sec=corrected, reason=reason))

    return results
