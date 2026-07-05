"""Matching: for each MIDI note onset, pick which hand+finger played it by
finding the nearest fingertip (in screen space, at the synced video time) to
the key's expected screen location (from keyboard.py calibration).

Pure numpy logic -- no cv2/mediapipe -- fully covered by selftest.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Plausibility gate: if the nearest fingertip is farther than this many
# "key widths" (normalized units, see keyboard.Calibration) from the
# expected key location, treat the match as low-confidence but still return
# the best guess (confidence will reflect the distance).
DEFAULT_MAX_NORM_DIST = 0.35


@dataclass
class Candidate:
    hand: str  # 'L' or 'R'
    finger: int  # 1..5
    x: float
    y: float


@dataclass
class MatchResult:
    hand: str
    finger: int
    confidence: float
    distance: float


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    return float(np.hypot(ax - bx, ay - by))


def confidence_from_distance(distance: float, scale: float = 60.0) -> float:
    """Map a pixel distance to a 0..1 confidence score: 0 distance -> 1.0,
    decaying smoothly, asymptoting to 0 for large distances. `scale` is the
    characteristic distance (pixels) at which confidence drops to ~1/e."""
    return float(np.exp(-max(0.0, distance) / max(1e-6, scale)))


def match_note(
    key_xy,
    candidates,
    used: "set | None" = None,
    scale: float = 60.0,
):
    """Pick the candidate fingertip nearest `key_xy` (a (x, y) tuple), among
    `candidates` (list[Candidate]), excluding any (hand, finger) pairs
    already in `used` (for chord conflict resolution -- see
    `resolve_chord_conflicts`).

    Returns a MatchResult, or None if no candidates are available.
    """
    kx, ky = key_xy
    best = None
    best_dist = float("inf")
    for c in candidates:
        if used is not None and (c.hand, c.finger) in used:
            continue
        d = _distance(kx, ky, c.x, c.y)
        if d < best_dist:
            best_dist = d
            best = c
    if best is None:
        return None
    return MatchResult(hand=best.hand, finger=best.finger, confidence=confidence_from_distance(best_dist, scale), distance=best_dist)


@dataclass
class NoteToMatch:
    note_id: object  # opaque identifier (e.g. onset_tick, or index)
    key_xy: tuple  # expected (x, y) screen location for this note's pitch


def resolve_chord_conflicts(notes, candidates, scale: float = 60.0):
    """Match a group of simultaneous notes (a chord) against the same set of
    fingertip candidates, ensuring no (hand, finger) pair is assigned to more
    than one note. Greedy nearest-first assignment: repeatedly pick the
    globally closest (note, candidate) pair, assign it, remove both from
    further consideration, and repeat.

    `notes` is a list[NoteToMatch]. Returns dict[note_id -> MatchResult].
    """
    pending_notes = list(notes)
    pending_candidates = list(candidates)
    results: dict = {}

    while pending_notes and pending_candidates:
        best_note_idx = None
        best_cand_idx = None
        best_dist = float("inf")
        for ni, note in enumerate(pending_notes):
            kx, ky = note.key_xy
            for ci, c in enumerate(pending_candidates):
                d = _distance(kx, ky, c.x, c.y)
                if d < best_dist:
                    best_dist = d
                    best_note_idx = ni
                    best_cand_idx = ci
        if best_note_idx is None:
            break
        note = pending_notes.pop(best_note_idx)
        cand = pending_candidates.pop(best_cand_idx)
        results[note.note_id] = MatchResult(
            hand=cand.hand, finger=cand.finger, confidence=confidence_from_distance(best_dist, scale), distance=best_dist
        )

    # Any notes left over (more notes than available fingertip candidates)
    # get no match.
    return results


def group_simultaneous(note_times, tolerance_sec: float = 0.03):
    """Group indices of `note_times` (a sequence of onset times, seconds)
    into chords: consecutive notes within `tolerance_sec` of the group's
    first note's time are grouped together. Returns a list of lists of
    indices (indices refer to positions in the original `note_times`, and
    are grouped preserving input order per group)."""
    order = sorted(range(len(note_times)), key=lambda i: note_times[i])
    groups: list = []
    current: list = []
    group_start_time = None
    for i in order:
        t = note_times[i]
        if current and group_start_time is not None and (t - group_start_time) <= tolerance_sec:
            current.append(i)
        else:
            if current:
                groups.append(current)
            current = [i]
            group_start_time = t
    if current:
        groups.append(current)
    return groups
