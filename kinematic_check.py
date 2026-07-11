"""Physical-reachability check on the per-note hand assignments.

One hand cannot be in two places at once: if two notes are assigned to the SAME
hand within a very short window (200ms) but more than 1.5 octaves apart, the
hand would have had to teleport -- so one of the two notes belongs to the OTHER
hand. This catches assignment errors the register/voice/neighbour passes miss,
e.g. a stray treble note pinned to the left hand while the left hand is playing
bass at the same instant.

Reports every violation (time + the two pitches) and, conservatively, fixes the
clear ones: the note that DOESN'T fit that hand's trajectory (its neighbour on
that hand is far away too) and whose match confidence is low is flipped to the
other hand. A genuine fast leap -- where the hand's own preceding/following
notes line up with the jump -- is reported but left alone. Pure python,
unit-testable.
"""

from __future__ import annotations

# Max semitones one hand can plausibly span between two notes 200ms apart.
MAX_LEAP_SEMITONES = 18   # 1.5 octaves
WINDOW_SEC = 0.2
# Only auto-fix a note at or below this confidence; above it, report only.
MAX_CONF_TO_FIX = 0.35


def check_hand_leaps(
    results,
    notes,
    max_leap: int = MAX_LEAP_SEMITONES,
    window_sec: float = WINDOW_SEC,
    max_conf_to_fix: float = MAX_CONF_TO_FIX,
):
    """Return (violations, flips). `violations` is a list of dicts
    {time, pitch_a, pitch_b, hand, fixed} for every same-hand >max_leap jump
    within window_sec; `flips` is how many were auto-corrected. Mutates
    `results` (hand of the flipped notes). `results` maps idx -> match with
    `.hand`/`.confidence`; `notes` is parallel-indexable with
    `.pitch`/`.start_sec`.
    """
    idxs = [i for i in results if results[i] is not None and results[i].hand in ("L", "R")]
    if len(idxs) < 2:
        return [], 0

    # Per-hand timelines (index lists sorted by time).
    by_hand = {"L": [], "R": []}
    for i in idxs:
        by_hand[results[i].hand].append(i)
    for h in by_hand:
        by_hand[h].sort(key=lambda i: notes[i].start_sec)

    def _fits_outer(hand_seq, pos, direction, pitch):
        """Does `pitch` reach this hand's note one step further out (in
        `direction`: -1 = earlier, +1 = later) from `pos`? Only the note
        BEYOND the leap pair tells us whether `pitch` fits this hand's own
        line -- the immediate neighbour on the leap side IS the far note we're
        comparing against, so checking it would be trivially false."""
        q = pos + direction
        if 0 <= q < len(hand_seq):
            return abs(int(notes[hand_seq[q]].pitch) - pitch) <= max_leap
        return False

    violations = []
    flips = 0
    for hand, seq in by_hand.items():
        other = "R" if hand == "L" else "L"
        for k in range(len(seq) - 1):
            ia, ib = seq[k], seq[k + 1]
            ta, tb = float(notes[ia].start_sec), float(notes[ib].start_sec)
            if tb - ta > window_sec:
                continue
            pa, pb = int(notes[ia].pitch), int(notes[ib].pitch)
            if abs(pb - pa) <= max_leap:
                continue

            # Decide the outlier: the note whose OTHER same-hand neighbour is
            # also far (doesn't fit the hand's line), preferring to move the
            # lower-confidence one. `k`/`k+1` are each other's neighbour, so
            # check the note one further out on each side.
            a_fits = _fits_outer(seq, k, -1, pa)      # A vs its earlier neighbour seq[k-1]
            b_fits = _fits_outer(seq, k + 1, +1, pb)  # B vs its later neighbour seq[k+2]
            if a_fits and not b_fits:
                suspect = ib
            elif b_fits and not a_fits:
                suspect = ia
            else:
                # ambiguous -> the lower-confidence note is the suspect
                suspect = ia if results[ia].confidence <= results[ib].confidence else ib

            fixed = False
            if results[suspect].confidence <= max_conf_to_fix:
                results[suspect].hand = other
                flips += 1
                fixed = True
            violations.append({
                "time": round(ta, 1), "pitch_a": pa, "pitch_b": pb,
                "hand": hand, "conf_a": round(results[ia].confidence, 2) if not fixed or suspect != ia else None,
                "fixed": fixed,
            })

    return violations, flips
