"""Final neighbour-consistency cleanup on the per-note hand assignments.

`fix_handedness_continuity` smooths L/R on the tracking FRAMES; this smooths the
final per-NOTE assignments, catching the leftover visible errors: an isolated
note whose hand disagrees with every nearby note (in pitch and time) while its
own match confidence is low -- e.g. a treble G4 assigned to the left hand at
~0 confidence while everything around it is the right hand. Those show up as
wrong-coloured notes in Symplethesia.

Deliberately conservative: only LOW-confidence notes are eligible, and only
when the surrounding notes agree overwhelmingly on the OTHER hand. Genuine hand
crossings (which come back with real confidence, and often aren't isolated) are
left alone. Pure python, unit-testable.
"""

from __future__ import annotations

# Only reconsider a note whose match confidence is at or below this.
MAX_CONF = 0.35
# Neighbours: notes within this time and pitch distance.
TIME_WINDOW_SEC = 0.6
PITCH_WINDOW = 7
# Need at least this many neighbours, and the opposite hand must outnumber the
# same hand by at least this ratio, to flip.
MIN_NEIGHBOURS = 3
OPP_RATIO = 4.0


def smooth_note_hands(
    results,
    notes,
    max_conf: float = MAX_CONF,
    time_window: float = TIME_WINDOW_SEC,
    pitch_window: int = PITCH_WINDOW,
    min_neighbours: int = MIN_NEIGHBOURS,
    opp_ratio: float = OPP_RATIO,
) -> int:
    """Flip isolated low-confidence notes to match their neighbours. `results`
    maps note-index -> a match with `.hand`/`.confidence`; `notes` is parallel-
    indexable with `.pitch`/`.start_sec`. Mutates `results`; returns #flips.

    Decisions use the ORIGINAL hands (snapshot first) so one flip can't cascade
    into its neighbour within the same pass.
    """
    if not results:
        return 0

    idxs = [i for i in results if results[i] is not None]
    # Snapshot (time, pitch, hand) so flips don't chain within one pass.
    info = {i: (float(notes[i].start_sec), int(notes[i].pitch), results[i].hand) for i in idxs}
    order = sorted(idxs, key=lambda i: info[i][0])
    times = [info[i][0] for i in order]
    import bisect

    flips = 0
    for i in idxs:
        m = results[i]
        if m.confidence > max_conf or m.hand not in ("L", "R"):
            continue
        t0, p0, h0 = info[i]
        lo = bisect.bisect_left(times, t0 - time_window)
        hi = bisect.bisect_right(times, t0 + time_window)
        same = opp = 0
        other = "R" if h0 == "L" else "L"
        for k in range(lo, hi):
            j = order[k]
            if j == i:
                continue
            tj, pj, hj = info[j]
            if abs(pj - p0) > pitch_window:
                continue
            if hj == h0:
                same += 1
            elif hj == other:
                opp += 1
        if opp >= min_neighbours and opp >= opp_ratio * (same + 1) - 1 and opp > same:
            m.hand = other
            flips += 1
    return flips
