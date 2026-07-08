"""MIDI-only hand prior (unified hand-assignment system, layer 1).

Independent of the video: from the MIDI alone, decide which hand most likely
played each note, using register clustering. On a two-handed piano piece the
left hand sits low and the right hand high; where they overlap, notes group
into chords/clusters that a single hand can span (<= ~1.5 octaves) while a
wider spread is two hands. This gives a strong, cheap prior that fixes the most
common video failure: when both hands are close together the nearest-fingertip
matcher assigns notes to the wrong side ("notes bleed across the hand
boundary"). The prior is fused with the video result conservatively -- strong
video evidence always wins; only weak/ambiguous video defers to this.

Pure python (no numpy/cv2) so it is cheap and unit-testable. Notes only need
`.pitch` (int) and `.start_sec` (float); they are assumed time-sorted (as
`MidiData.notes` is).
"""

from __future__ import annotations

from dataclasses import dataclass

# A hand comfortably spans about an octave; up to ~1.5 octaves with a stretch.
# A cluster wider than this is almost certainly two hands, not one.
MAX_HAND_SPAN = 18  # semitones (1.5 octaves)

# Simultaneity window: notes starting within this of each other are treated as
# one cluster (chord / rolled chord) for the low/high split.
CLUSTER_WINDOW_SEC = 0.06

# How fast the running per-hand pitch centroids follow new evidence (EMA).
CENTROID_EMA = 0.25

# Half-width (semitones) over which prior strength ramps 0->1 with distance
# from the current hand split. A note right on the split is ambiguous
# (strength 0); an octave away is certain (strength 1).
STRENGTH_SPAN = 12.0


@dataclass
class HandPrior:
    hand: str      # 'L' or 'R'
    strength: float  # 0..1 confidence that this note belongs to `hand`


def _cluster_indices(notes, window_sec: float):
    """Yield lists of note indices grouped by onset simultaneity (within
    `window_sec` of the group's first onset). `notes` time-sorted."""
    group: list = []
    group_start = None
    for i, n in enumerate(notes):
        t = float(n.start_sec)
        if group and group_start is not None and (t - group_start) <= window_sec:
            group.append(i)
        else:
            if group:
                yield group
            group = [i]
            group_start = t
    if group:
        yield group


def compute_hand_prior(
    notes,
    max_hand_span: int = MAX_HAND_SPAN,
    window_sec: float = CLUSTER_WINDOW_SEC,
    centroid_ema: float = CENTROID_EMA,
    strength_span: float = STRENGTH_SPAN,
) -> list:
    """Return a list of HandPrior (or None) parallel to `notes`.

    For each simultaneity cluster: if the pitch spread exceeds `max_hand_span`
    it is split into a low (LH) and high (RH) group at the widest internal gap;
    otherwise the whole cluster goes to whichever running hand-centroid it sits
    nearer. Running centroids and a split pitch track the piece so the boundary
    follows hand crossings and register shifts instead of a fixed middle-C.
    """
    n = len(notes)
    out: list = [None] * n
    if n == 0:
        return out

    # Seed centroids from the overall pitch range so the first few notes get a
    # sensible split even before much history exists.
    pitches_all = [int(x.pitch) for x in notes]
    lo, hi = min(pitches_all), max(pitches_all)
    lh_centroid = float(lo + (hi - lo) * 0.25)
    rh_centroid = float(lo + (hi - lo) * 0.75)
    if rh_centroid - lh_centroid < 1.0:
        lh_centroid, rh_centroid = lh_centroid - 6.0, rh_centroid + 6.0

    def split_pitch() -> float:
        return (lh_centroid + rh_centroid) / 2.0

    def _strength(pitch: float) -> float:
        d = abs(pitch - split_pitch())
        return max(0.0, min(1.0, d / strength_span))

    for group in _cluster_indices(notes, window_sec):
        ps = sorted((int(notes[i].pitch), i) for i in group)
        pitch_vals = [p for p, _ in ps]
        span = pitch_vals[-1] - pitch_vals[0]

        if span > max_hand_span and len(ps) >= 2:
            # Two hands: cut at the widest gap between adjacent pitches.
            best_gap, cut = -1, 1
            for k in range(1, len(ps)):
                g = pitch_vals[k] - pitch_vals[k - 1]
                if g > best_gap:
                    best_gap, cut = g, k
            low = ps[:cut]
            high = ps[cut:]
            low_mean = sum(p for p, _ in low) / len(low)
            high_mean = sum(p for p, _ in high) / len(high)
            lh_centroid += centroid_ema * (low_mean - lh_centroid)
            rh_centroid += centroid_ema * (high_mean - rh_centroid)
            for p, i in low:
                out[i] = HandPrior("L", _strength(p))
            for p, i in high:
                out[i] = HandPrior("R", _strength(p))
        else:
            # One hand: assign the whole cluster to the nearer centroid.
            c = sum(pitch_vals) / len(pitch_vals)
            if abs(c - lh_centroid) <= abs(c - rh_centroid):
                hand = "L"
                lh_centroid += centroid_ema * (c - lh_centroid)
            else:
                hand = "R"
                rh_centroid += centroid_ema * (c - rh_centroid)
            for p, i in ps:
                out[i] = HandPrior(hand, _strength(p))

    return out


def fuse_prior_into_results(
    results: dict,
    notes,
    priors: list,
    min_video_conf: float = 0.5,
    min_prior_strength: float = 0.5,
) -> int:
    """Override each match's hand with the MIDI prior ONLY where the video
    evidence is weak (`confidence < min_video_conf`) AND the prior is strong
    (`strength >= min_prior_strength`) AND they disagree. Strong video always
    wins -- this just stops ambiguous, hands-close notes from bleeding onto the
    wrong side. Mutates `results` in place; returns the number of corrections.

    `results` maps note-index -> a match object with `.hand` and `.confidence`.
    `priors` is the list from `compute_hand_prior`, parallel to `notes`.
    """
    corrected = 0
    for idx, m in results.items():
        if m is None or idx >= len(priors):
            continue
        prior = priors[idx]
        if prior is None:
            continue
        if m.confidence >= min_video_conf:
            continue  # trust the video
        if prior.strength < min_prior_strength:
            continue  # prior too weak to overrule anything
        if prior.hand == m.hand:
            continue
        m.hand = prior.hand
        corrected += 1
    return corrected
