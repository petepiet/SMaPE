"""Voice separation seeded by the video's confident hand labels.

The hardest hand-assignment case is two hands playing close together in the
same register: neither pitch (register clustering) nor space (blob) can split
them, because the notes overlap and the hands touch. What DOES survive is the
musical role -- the left hand plays a bass line with a recurring rhythm/pattern
while the right hand plays melody. When the hands are far apart the video reads
them reliably; we learn each hand's signature from those confident passages and
apply it during the ambiguous (low-confidence) ones.

Three signals, all MIDI-space (no video tuning), combined per ambiguous note:

  1. Register fit -- distance to the LOCAL LH vs RH pitch centroid, learned
     from confident notes in a time window around the note (so it follows
     register shifts and section changes, not a fixed middle-C).
  2. Pitch recurrence -- this exact pitch's LH-vs-RH history in the window. A
     bass ostinato keeps hitting the same notes, so an ambiguous G2 that the
     LH repeatedly played nearby is probably the LH again. This is the
     "same pattern when they come back together" idea, made concrete.
  3. Lowest-voice bias -- at a given onset the lowest sounding note leans LH
     (the bass), the way a human reads a score.

Strong video evidence always wins: only notes below `min_video_conf` are
re-considered, and only reassigned when the combined signal is decisive. Pure
python, unit-testable, complements the register-only `midi_hand_prior` and the
(future) spatial blob split.
"""

from __future__ import annotations

# Below this video confidence a note is "ambiguous" and eligible for voice
# reasoning; at or above it the video is trusted and the note anchors the model.
MIN_VIDEO_CONF = 0.5
# Time window (each side) over which to learn the local per-hand signature.
WINDOW_SEC = 3.0
# Notes starting within this of each other count as one onset (for lowest-voice).
ONSET_TOL_SEC = 0.03
# Combined preference must exceed this (|.|) to overrule -- keeps it decisive.
DECISION_MARGIN = 0.35

# Signal weights (register + recurrence + lowest-voice), tuned conservative.
W_REGISTER = 1.0
W_RECURRENCE = 1.0
W_LOWEST = 0.5


def _onset_groups(notes, tol: float = ONSET_TOL_SEC):
    """Map note index -> the min pitch among notes sharing its onset (within
    tol). Used for the lowest-voice bias."""
    order = sorted(range(len(notes)), key=lambda i: notes[i].start_sec)
    lowest_at = {}
    i = 0
    n = len(order)
    while i < n:
        j = i
        t0 = notes[order[i]].start_sec
        grp = []
        while j < n and notes[order[j]].start_sec - t0 <= tol:
            grp.append(order[j]); j += 1
        mn = min(int(notes[k].pitch) for k in grp)
        for k in grp:
            lowest_at[k] = mn
        i = j
    return lowest_at


def refine_hands_by_voice(
    results,
    notes,
    min_video_conf: float = MIN_VIDEO_CONF,
    window_sec: float = WINDOW_SEC,
    decision_margin: float = DECISION_MARGIN,
) -> int:
    """Re-assign low-confidence notes' hand using a locally-learned, video-
    seeded voice model. `results` maps note-index -> a match object with
    `.hand` and `.confidence`; `notes` is parallel-indexable with `.pitch` /
    `.start_sec`. Mutates `results` in place; returns the number of changes.
    """
    if not results:
        return 0

    # Confident (video-reliable) anchors: (time, pitch, hand).
    anchors = []
    for idx, m in results.items():
        if m is not None and m.confidence >= min_video_conf and m.hand in ("L", "R"):
            anchors.append((float(notes[idx].start_sec), int(notes[idx].pitch), m.hand))
    if len(anchors) < 4:
        return 0
    anchors.sort()
    times = [a[0] for a in anchors]

    lowest_at = _onset_groups(notes)
    import bisect

    changed = 0
    for idx, m in results.items():
        if m is None or m.confidence >= min_video_conf:
            continue  # trust the video
        t = float(notes[idx].start_sec)
        pitch = int(notes[idx].pitch)

        lo = bisect.bisect_left(times, t - window_sec)
        hi = bisect.bisect_right(times, t + window_sec)
        L_pitches = [anchors[i][1] for i in range(lo, hi) if anchors[i][2] == "L"]
        R_pitches = [anchors[i][1] for i in range(lo, hi) if anchors[i][2] == "R"]
        if not L_pitches or not R_pitches:
            continue  # need both hands represented locally to disambiguate

        # 1. Register fit: closer centroid preferred, normalised to [-1, 1]
        #    (+ = leans L).
        lc = sum(L_pitches) / len(L_pitches)
        rc = sum(R_pitches) / len(R_pitches)
        dL = abs(pitch - lc)
        dR = abs(pitch - rc)
        register_pref = (dR - dL) / (dR + dL + 1e-6)

        # 2. Pitch recurrence: did THIS pitch recur under L or R nearby?
        nL = sum(1 for p in L_pitches if abs(p - pitch) <= 1)
        nR = sum(1 for p in R_pitches if abs(p - pitch) <= 1)
        recurrence_pref = (nL - nR) / (nL + nR + 1)

        # 3. Lowest-voice bias: the lowest note at this onset leans L.
        lowest_pref = 1.0 if lowest_at.get(idx, pitch) == pitch else 0.0

        pref = (W_REGISTER * register_pref
                + W_RECURRENCE * recurrence_pref
                + W_LOWEST * lowest_pref) / (W_REGISTER + W_RECURRENCE + W_LOWEST)

        target = None
        if pref >= decision_margin:
            target = "L"
        elif pref <= -decision_margin:
            target = "R"
        if target is not None and target != m.hand:
            m.hand = target
            changed += 1

    return changed
