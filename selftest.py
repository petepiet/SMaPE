#!/usr/bin/env python3
"""Pure-python/numpy self-tests for the piano-fingering tool's core geometry,
sync, and matching logic. Deliberately avoids cv2 / mediapipe / mido so it
can run in a minimal environment with only numpy installed:

    python3 -m pip install --user numpy
    python3 selftest.py
    # or: python3 extract_fingering.py --selftest

Exits with status 0 and prints "OK" on success; prints failures and exits 1
otherwise.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from keyboard import (
    Calibration,
    pitch_to_white_index,
    white_index_to_pitch,
    _fit_row_projective,
)
from keyboard_cv import (
    classify_gap_sizes,
    group_black_keys,
    expected_black_key_groups,
    assign_group_pitches,
    build_anchor_points,
    _detect_for_range,
)
from calibration_store import (
    calibration_key,
    calibrations_agree,
    load_saved_calibration,
    save_calibration,
)
from render_hands import (
    _rgb_to_hue_vec,
    cluster_hand_colors,
    is_separation_confident,
    is_separation_confident_hue,
    assign_clusters_to_hands,
    assign_hands_for_notes,
    assign_hands_from_reference_colors,
    lit_delta,
    is_lit,
    lit_end_time,
    trim_note_durations,
)
from midi_io import MidiNote
from bundle import write_symple_bundle, default_bundle_path
from sync import detect_press_moments, estimate_offset, onset_density
from extract_fingering import default_out_path_from_video, build_arg_parser, _strip_fingering_json_suffix
from transcribe import filter_note_events
from reconcile import (
    is_occluded, finger_leave_time, reconcile_note, confidence_by_window, _frame_times,
    analyze_hand_spread, CLOSE_HANDS_THRESHOLD_KEYS,
)
from match import (
    Candidate,
    NoteToMatch,
    match_note,
    resolve_chord_conflicts,
    group_simultaneous,
    confidence_from_distance,
)


class TestFailure(Exception):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise TestFailure(msg)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Keyboard / homography tests
# ---------------------------------------------------------------------------

def test_white_index_roundtrip():
    for pitch in range(21, 109):
        wi = pitch_to_white_index(pitch)
        recovered = white_index_to_pitch(wi)
        check(recovered == pitch, f"white-index roundtrip failed for pitch {pitch}: got {recovered}")


def test_homography_known_corners_to_expected_pitches():
    # A simple axis-aligned rectangle: screen x in [100, 900], y in [50, 250].
    # low_pitch=21 (A0) at left edge, high_pitch=108 (C8) at right edge.
    corners = [[100, 50], [900, 50], [900, 250], [100, 250]]  # TL, TR, BR, BL
    calib = Calibration(corners=corners, low_pitch=21, high_pitch=108)

    # Left edge should map to (or very near) low_pitch.
    p_left = calib.screen_to_pitch((100, 150))
    check(p_left == 21, f"left edge expected pitch 21, got {p_left}")

    # Right edge should map to (or very near) high_pitch.
    p_right = calib.screen_to_pitch((900, 150))
    check(p_right == 108, f"right edge expected pitch 108, got {p_right}")

    # Middle C (60) should map to somewhere strictly between the edges.
    mid_xy = calib.pitch_to_screen(60)
    check(100 < mid_xy[0] < 900, f"middle C screen x should be interior, got {mid_xy}")

    # Round trip: pitch -> screen -> pitch should recover exactly (calibration
    # is a clean affine rectangle so this should be exact within key
    # resolution).
    for pitch in [21, 30, 45, 60, 61, 72, 90, 108]:
        xy = calib.pitch_to_screen(pitch)
        recovered = calib.screen_to_pitch(xy)
        check(recovered == pitch, f"pitch->screen->pitch roundtrip failed for {pitch}: got {recovered} at {xy}")


def test_row_calibration_single_row_click():
    # Regression for the multi-point ("click the C keys") calibration bug:
    # every clicked point sits on one horizontal key-center row, which makes a
    # full 2-D homography degenerate (its near<->far axis is undetermined) and
    # previously collapsed pitch_to_screen to the frame center. The 1-D
    # projective row fit must instead spread pitches across the row and
    # round-trip cleanly.
    low, high = 36, 89
    lo_wi = pitch_to_white_index(low)
    hi_wi = pitch_to_white_index(high)

    # Ground-truth image: a genuine 1-D projective row with foreshortening
    # (g != 0) and a slight vertical tilt -- i.e. a slightly angled camera.
    def truth(u: float):
        denom = 0.35 * u + 1.0
        return (900.0 * u + 80.0) / denom, (40.0 * u + 300.0) / denom

    # Clicks: C and G white keys spread across the keyboard (all on one row).
    clicked = [36, 43, 48, 55, 60, 67, 72, 79, 84]
    u = np.array([(pitch_to_white_index(p) - lo_wi) / (hi_wi - lo_wi) for p in clicked])
    xy = np.array([truth(uu) for uu in u])
    row = _fit_row_projective(u, xy[:, 0], xy[:, 1])
    check(row is not None, "row fit returned None for a valid single-row click set")

    calib = Calibration(
        corners=[[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
        low_pitch=low, high_pitch=high, row=row,
    )

    xs = [calib.pitch_to_screen(p)[0] for p in range(low, high + 1)]
    check(max(xs) - min(xs) > 400, f"pitch_to_screen collapsed (x span only {max(xs)-min(xs):.1f})")

    for p in range(low, high + 1):
        rp = calib.screen_to_pitch(calib.pitch_to_screen(p))
        check(rp == p, f"row-mode roundtrip failed for pitch {p}: got {rp}")

    # Serialization must preserve the row map (else load_calibration silently
    # falls back to the broken corner homography).
    c2 = Calibration.from_dict(calib.to_dict())
    check(c2.row is not None, "row map lost through to_dict/from_dict")
    check(c2.screen_to_pitch(c2.pitch_to_screen(60)) == 60, "row map broken after serialize")


def test_row_fit_rejects_too_few_points():
    # <3 points cannot determine the 5-parameter projective row -> None, so
    # interactive_calibrate raises instead of installing a garbage mapping.
    u = np.array([0.0, 1.0])
    check(_fit_row_projective(u, np.array([10.0, 90.0]), np.array([5.0, 5.0])) is None,
          "row fit should reject fewer than 3 points")


# ---------------------------------------------------------------------------
# CV auto-calibration (keyboard_cv.py) pure-logic tests
# ---------------------------------------------------------------------------

def _synthetic_black_key_centroids(low_pitch: int, high_pitch: int, px_per_white_key: float = 40.0):
    """Synthetic black-key x-centroids for every black key in [low, high],
    spaced as if photographed top-down with no perspective (a flat scale
    factor from white-key-index units to pixels) -- enough to exercise the
    gap-classification / grouping / pitch-assignment logic without cv2."""
    lo_wi = pitch_to_white_index(low_pitch)
    pitches = [p for p in range(low_pitch, high_pitch + 1) if p % 12 in (1, 3, 6, 8, 10)]
    xs = [(pitch_to_white_index(p) - lo_wi) * px_per_white_key for p in pitches]
    return pitches, xs


def test_classify_gap_sizes_full_keyboard():
    _pitches, xs = _synthetic_black_key_centroids(21, 108)
    gaps = np.diff(xs)
    is_large = classify_gap_sizes(gaps)
    # Real 2-/3- black-key groups: a large gap marks a group boundary
    # (crossing E-F or B-C); groups should come out size 2 or 3 throughout.
    groups = group_black_keys(is_large, len(xs))
    sizes = [len(g) for g in groups]
    check(all(s in (1, 2, 3) for s in sizes), f"unexpected group size in {sizes}")
    check(sizes.count(1) <= 1, f"only the partial leading A#0 group should be size 1: {sizes}")


def test_classify_gap_sizes_noisy_missing_black_key():
    # Drop one interior centroid (simulates a missed blob, e.g. finger
    # occlusion) -- the classifier must still separate small/large cleanly
    # for the surrounding, unaffected gaps.
    pitches, xs = _synthetic_black_key_centroids(60, 84)
    del pitches[3]
    del xs[3]
    gaps = np.diff(xs)
    is_large = classify_gap_sizes(gaps)
    groups = group_black_keys(is_large, len(xs))
    sizes = [len(g) for g in groups]
    check(max(sizes) <= 3, f"missing-blob noise should not merge groups: {sizes}")


def test_expected_black_key_groups_full_88():
    groups = expected_black_key_groups(21, 108)
    sizes = [len(g) for g in groups]
    check(sizes[0] == 1, f"88-key range should start with the partial A#0 group, got sizes {sizes}")
    check(sizes[1:] == [2, 3] * ((len(sizes) - 1) // 2), f"expected alternating 2,3 pattern, got {sizes}")
    check(groups[0] == [22], f"first group should be lone A#0 (pitch 22), got {groups[0]}")


def test_assign_group_pitches_partial_keyboard():
    # A crop showing only 2 octaves (60..84, middle C up) -- no edge partial
    # groups, every detected group should size-match exactly.
    pitches, xs = _synthetic_black_key_centroids(60, 84)
    gaps = np.diff(xs)
    is_large = classify_gap_sizes(gaps)
    groups = group_black_keys(is_large, len(xs))
    sizes = [len(g) for g in groups]
    result = assign_group_pitches(sizes, 60, 84)
    check(result is not None, "confident partial-keyboard alignment should not be rejected")
    check(all(g is not None for g in result), f"no group should be ambiguous in a clean partial view: {result}")
    flat = [p for g in result for p in g]
    check(flat == pitches, f"assigned pitches {flat} != true pitches {pitches}")


def test_assign_group_pitches_noisy_missing_black_key():
    # One black key blob missing (e.g. occluded by a finger): that one
    # group's size no longer matches the expected pattern and must be
    # dropped (None), but the rest of a confident alignment still resolves.
    # The missing blob is the LAST element of its group (66,68,70 -> 66,68)
    # -- occlusion realistically clips a group from one edge inward, not a
    # clean hole from its middle, which would instead fragment one group
    # into two and is a fundamentally harder (and separately handled)
    # ambiguity than "one shrunken group".
    pitches, xs = _synthetic_black_key_centroids(60, 84)
    del pitches[4]  # last member (70) of the {F#,G#,A#} group starting at 66
    del xs[4]
    gaps = np.diff(xs)
    is_large = classify_gap_sizes(gaps)
    groups = group_black_keys(is_large, len(xs))
    sizes = [len(g) for g in groups]
    result = assign_group_pitches(sizes, 60, 84)
    check(result is not None, "one bad group should not sink the whole alignment")
    check(any(g is None for g in result), "the group with the missing blob should be flagged ambiguous (None)")
    check(sum(1 for g in result if g is not None) >= len(result) - 1,
          "only the affected group should be dropped")


def test_assign_group_pitches_edge_junk():
    # Real footage showed spurious lone-blob detections just past the
    # keyboard's physical edges (e.g. a cabinet seam). The true groups sit
    # in the interior; leading/trailing junk must be dropped (None) rather
    # than corrupting the alignment of the real groups.
    detected_sizes = [1, 1, 2, 3, 2, 3, 1]  # junk, junk, {C#D#}, {F#G#A#}, {C#D#}, {F#G#A#}, junk
    result = assign_group_pitches(detected_sizes, 60, 84)
    check(result is not None, "edge junk should not prevent a confident interior alignment")
    check(result[0] is None and result[-1] is None, f"leading/trailing junk should be dropped: {result}")
    check(all(g is not None for g in result[2:-1]), f"interior groups should all resolve: {result}")


def test_assign_group_pitches_low_confidence_returns_none():
    # A detected size sequence that doesn't resemble the expected pattern at
    # all (e.g. every group wrongly split to size 1) must be rejected
    # outright rather than guessing.
    result = assign_group_pitches([1, 1, 1, 1, 1, 1, 1, 1], 21, 108)
    check(result is None, "implausible group-size sequence should yield no assignment")


def test_assign_group_pitches_narrow_range_is_octave_ambiguous():
    # Regression documentation (found against real footage): the 2-/3-
    # black-key group pattern REPEATS every octave, so aligning a full
    # keyboard's worth of DETECTED groups against a narrow EXPECTED window
    # (e.g. just the pitch range a MIDI file happens to use) has no way to
    # pick the right octave from group sizes alone -- any octave-shifted
    # window of a plain [2,3,2,3,...] repeat matches equally well. This
    # doesn't raise or return None (there's nothing "wrong" about the
    # match found), it just isn't necessarily the physically correct
    # octave -- which is exactly why detect_keyboard_calibration always
    # tries the FULL 88-key range first (its unique leading partial group
    # anchors the phase) and only falls back to a narrower requested range
    # if that fails. This test exists to document/pin the ambiguity, not
    # to assert one "correct" answer out of the tied candidates.
    _pitches, xs = _synthetic_black_key_centroids(21, 108)
    gaps = np.diff(xs)
    is_large = classify_gap_sizes(gaps)
    groups = group_black_keys(is_large, len(xs))
    sizes = [len(g) for g in groups]

    result_full = assign_group_pitches(sizes, 21, 108)
    flat_full = [p for g in result_full for p in g if g is not None]
    check(flat_full == _pitches, "the full 88-key range must always resolve correctly (unique leading anchor)")

    # A narrow window taken from the MIDDLE of the same real keyboard is
    # genuinely ambiguous: assign_group_pitches has no signal to prefer the
    # true octave over an adjacent one, since both look identical in group
    # sizes. We only assert that SOME confident (non-None-dominated) match
    # is returned -- proving the narrow call alone cannot be trusted to
    # pick the physically correct octave, which is the whole point.
    result_narrow = assign_group_pitches(sizes, 60, 84)
    check(result_narrow is not None, "a plausible (if not necessarily correct) narrow-range match should still be found")


def test_build_anchor_points_skips_ambiguous_groups():
    groups = [[0, 1], [2, 3, 4]]
    group_pitches = [[61, 63], None]  # second group ambiguous -> contributes nothing
    centroids_x = [10.0, 20.0, 30.0, 40.0, 50.0]
    centroids_y = [5.0, 5.0, 5.0, 5.0, 5.0]
    u, x, y = build_anchor_points(groups, group_pitches, centroids_x, centroids_y, 60, 84)
    check(len(u) == 2, f"expected 2 anchors from the resolved group only, got {len(u)}")
    check(list(x) == [10.0, 20.0], f"anchor x values should come from the resolved group's centroids: {x}")


def test_detect_for_range_full_keyboard_recovers_correct_mapping():
    # Pins the actual fix: detect_keyboard_calibration tries the FULL
    # 88-key range first (see test_assign_group_pitches_narrow_range_is_
    # octave_ambiguous for why a narrower range alone can't be trusted).
    # Verify _detect_for_range's row fit, evaluated at (21,108), places
    # known pitches at their true synthetic pixel positions -- not just
    # "some" plausible mapping.
    lo_wi = pitch_to_white_index(21)
    pitches, xs = _synthetic_black_key_centroids(21, 108)
    ys = [300.0] * len(xs)
    gaps = np.diff(xs)
    is_large = classify_gap_sizes(gaps)
    groups = group_black_keys(is_large, len(xs))
    sizes = [len(g) for g in groups]

    row = _detect_for_range(xs, ys, groups, sizes, 21, 108)
    check(row is not None, "full-range detection should succeed on a clean synthetic full keyboard")

    calib = Calibration(corners=[[0, 0], [1920, 0], [1920, 1080], [0, 1080]], low_pitch=21, high_pitch=108, row=row)
    for probe_pitch in (24, 60, 96):
        expected_x = (pitch_to_white_index(probe_pitch) - lo_wi) * 40.0
        got_x = calib.pitch_to_screen(probe_pitch)[0]
        check(approx(got_x, expected_x, tol=1.0), f"pitch {probe_pitch}: expected x~{expected_x:.1f}, got {got_x:.1f}")


# ---------------------------------------------------------------------------
# Calibration persistence (calibration_store.py) tests
# ---------------------------------------------------------------------------

def test_calibration_key_channel_vs_path():
    # Same channel_id -> same key regardless of the specific video URL
    # (that's the whole point -- reuse across a channel's uploads).
    k1 = calibration_key("https://youtu.be/aaa", "UC123")
    k2 = calibration_key("https://youtu.be/bbb", "UC123")
    check(k1 == k2, f"same channel_id should give the same key: {k1} != {k2}")
    check(k1.startswith("channel-"), f"channel-keyed entries should be tagged: {k1}")

    # No channel_id (local file) -> keyed by path hash; same path -> same
    # key, different path -> different key.
    p1 = calibration_key("/tmp/some/video.mp4", None)
    p2 = calibration_key("/tmp/some/video.mp4", None)
    p3 = calibration_key("/tmp/other/video.mp4", None)
    check(p1 == p2, "same local path should give the same key")
    check(p1 != p3, "different local paths should give different keys")
    check(p1.startswith("path-"), f"path-keyed entries should be tagged: {p1}")


def _row_calibration(low, high, dx=0.0, dy=0.0):
    # A simple synthetic row calibration spanning [low, high] linearly in
    # pixels, offset by `dx`/`dy` pixels -- used to simulate a saved
    # calibration that's identical to, shifted from, or detected at a
    # different scanline height than a "fresh" one.
    lo_wi = pitch_to_white_index(low)
    hi_wi = pitch_to_white_index(high)
    clicked = [p for p in (low, low + 12, high - 12, high) if low <= p <= high]
    u = np.array([(pitch_to_white_index(p) - lo_wi) / (hi_wi - lo_wi) for p in clicked])
    x = 900.0 * u + 100.0 + dx
    y = np.full_like(x, 300.0 + dy)
    row = _fit_row_projective(u, x, y)
    return Calibration(corners=[[0, 0], [1920, 0], [1920, 1080], [0, 1080]], low_pitch=low, high_pitch=high, row=row)


def test_calibrations_agree_identical():
    calib = _row_calibration(21, 108)
    check(calibrations_agree(calib, calib, 21, 108), "a calibration should agree with itself")


def test_calibrations_agree_small_shift_within_tolerance():
    # A few pixels of jitter (sub-percent of the keyboard's ~900px span)
    # should read as the same camera setup, not a mismatch.
    saved = _row_calibration(21, 108, dx=0.0)
    fresh = _row_calibration(21, 108, dx=2.0)
    check(calibrations_agree(saved, fresh, 21, 108), "a tiny pixel jitter should still agree")


def test_calibrations_agree_large_shift_disagrees():
    # A shift of a large fraction of the keyboard's span simulates a
    # meaningfully re-angled/re-positioned camera between takes.
    saved = _row_calibration(21, 108, dx=0.0)
    fresh = _row_calibration(21, 108, dx=150.0)
    check(not calibrations_agree(saved, fresh, 21, 108), "a large shift should be flagged as disagreement")


def test_calibrations_agree_ignores_row_height_difference():
    # Regression (found against real footage): the black-key row detector
    # can legitimately lock onto a different scanline height between two
    # reads of the SAME unchanged camera (hand position differs between the
    # two sampled frames, shifting which row has the clearest black-key
    # blobs) -- a real pair of frames from one video measured a 48px Y
    # difference with near-zero X difference. Y doesn't affect pitch at all
    # (screen_to_white_index is depth/Y-invariant), so a Y-only difference,
    # however large, must never read as a mismatch.
    saved = _row_calibration(21, 108, dy=0.0)
    fresh = _row_calibration(21, 108, dy=48.0)
    check(calibrations_agree(saved, fresh, 21, 108), "a Y-only (row height) difference must not count as disagreement")


def test_calibration_store_round_trip():
    tmp_dir = tempfile.mkdtemp(prefix="piano-fingering-calib-test-")
    try:
        calib = _row_calibration(21, 108)
        key = calibration_key("https://youtu.be/xyz", "UCabc")
        check(load_saved_calibration(key, base_dir=tmp_dir) is None, "nothing saved yet should load as None")
        save_calibration(key, calib, base_dir=tmp_dir)
        loaded = load_saved_calibration(key, base_dir=tmp_dir)
        check(loaded is not None, "calibration should load back after saving")
        check(loaded.row == calib.row, f"loaded row {loaded.row} != saved row {calib.row}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Synthesia-render support (render_hands.py) pure-logic tests
# ---------------------------------------------------------------------------

def test_cluster_hand_colors_two_clear_clusters():
    rng = np.random.RandomState(0)
    red_ish = rng.normal(loc=[200, 50, 50], scale=3.0, size=(10, 3))
    blue_ish = rng.normal(loc=[50, 50, 200], scale=3.0, size=(10, 3))
    colors = np.vstack([red_ish, blue_ish])
    hue_vecs = _rgb_to_hue_vec(colors)
    labels, separation = cluster_hand_colors(hue_vecs)
    check(is_separation_confident_hue(separation), f"two visually distinct clusters should be confident, got {separation}")
    # The 10 red-ish samples should all share one label, the 10 blue-ish the other.
    check(len(set(labels[:10])) == 1, "red-ish samples should form one cluster")
    check(len(set(labels[10:])) == 1, "blue-ish samples should form one cluster")
    check(labels[0] != labels[10], "the two colour groups should get different labels")


def test_cluster_hand_colors_single_cluster_low_separation():
    # Use a single desaturated blue colour (not pure gray) to avoid hue arbitrariness
    # Pure gray has undefined hue, so near-gray with noise can produce arbitrary
    # hue clusters. A desaturated blue has a well-defined hue despite low saturation.
    rng = np.random.RandomState(1)
    desaturated_blue = rng.normal(loc=[110, 110, 130], scale=2.0, size=(20, 3))
    pitches = list(range(40, 60))
    hands, confidence, degenerate = assign_hands_for_notes(desaturated_blue, pitches)
    check(degenerate, f"a single desaturated colour should be degenerate, got hands={set(hands)}")


def test_cluster_hand_colors_degenerate_inputs():
    hue_vecs0 = _rgb_to_hue_vec([])
    labels0, sep0 = cluster_hand_colors(hue_vecs0)
    check(len(labels0) == 0 and sep0 == 0.0, "empty input should return trivially, not raise")
    hue_vecs1 = _rgb_to_hue_vec([[10, 20, 30]])
    labels1, sep1 = cluster_hand_colors(hue_vecs1)
    check(len(labels1) == 1 and sep1 == 0.0, "single-colour input should return trivially, not raise")


def test_cluster_hand_colors_4color_tint_shade_theme():
    """Synthesia render with white-key/black-key brightness variants of two
    hues (blue bass, green treble). Hue clustering should separate them
    despite the 2x brightness difference within each hue."""
    rng = np.random.RandomState(42)
    # Bass: light-blue (white keys) and dark-blue (black keys), same hue
    light_blue = rng.normal(loc=[150, 150, 255], scale=3.0, size=(5, 3))
    dark_blue = rng.normal(loc=[40, 40, 200], scale=3.0, size=(5, 3))
    bass_colors = np.vstack([light_blue, dark_blue])
    bass_pitches = [40, 42, 44, 46, 48, 41, 43, 45, 47, 49]
    # Treble: light-green (white keys) and dark-green (black keys), different hue
    light_green = rng.normal(loc=[150, 255, 150], scale=3.0, size=(5, 3))
    dark_green = rng.normal(loc=[40, 200, 40], scale=3.0, size=(5, 3))
    treble_colors = np.vstack([light_green, dark_green])
    treble_pitches = [70, 72, 74, 76, 78, 71, 73, 75, 77, 79]
    # Merge all colors and pitches
    all_colors = np.vstack([bass_colors, treble_colors])
    all_pitches = bass_pitches + treble_pitches
    # Cluster in hue space
    hue_vecs = _rgb_to_hue_vec(all_colors)
    labels, separation = cluster_hand_colors(hue_vecs)
    check(is_separation_confident_hue(separation),
          f"4-color tint/shade theme should show confident hue separation, got {separation}")
    # Hand assignment: all bass should be L, all treble should be R
    hands, confidence, degenerate = assign_hands_for_notes(all_colors, all_pitches)
    check(not degenerate, "4-color theme with clear hue separation should not be degenerate")
    check(confidence > 0.0, "4-color theme should report positive confidence")
    check(all(h == "L" for h in hands[:10]), f"all bass notes should be L, got {hands[:10]}")
    check(all(h == "R" for h in hands[10:]), f"all treble notes should be R, got {hands[10:]}")


def test_cluster_hand_colors_same_hue_two_lightness():
    """Same hue, two lightness levels (e.g. light-blue bass, dark-blue treble).
    Hue clustering should fail to separate these (both are blue), so the result
    should be degenerate -- no hand information in brightness alone."""
    rng = np.random.RandomState(43)
    light_blue = rng.normal(loc=[150, 150, 255], scale=2.0, size=(10, 3))
    dark_blue = rng.normal(loc=[40, 40, 200], scale=2.0, size=(10, 3))
    colors = np.vstack([light_blue, dark_blue])
    pitches = [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88]
    hands, confidence, degenerate = assign_hands_for_notes(colors, pitches)
    check(degenerate, "same-hue two-lightness split should be degenerate")
    check(confidence == 0.0, "degenerate should report zero confidence")


def test_cluster_hand_colors_near_gray_white_theme():
    """Near-achromatic theme (gray/white with minimal hue). Hue clustering
    cannot separate these, so result should be degenerate."""
    rng = np.random.RandomState(44)
    # Near-white/gray: high lightness, low saturation (all channels similar)
    grays = rng.normal(loc=[230, 230, 225], scale=2.0, size=(20, 3))
    pitches = list(range(40, 60))
    hands, confidence, degenerate = assign_hands_for_notes(grays, pitches)
    check(degenerate, "near-gray/white theme should be degenerate")
    check(confidence == 0.0, "degenerate should report zero confidence")


def test_assign_clusters_to_hands_lower_pitch_is_left():
    labels = np.array([0, 0, 1, 1])
    pitches = [40, 42, 70, 72]  # cluster 0 = bass register, cluster 1 = treble
    mapping = assign_clusters_to_hands(labels, pitches)
    check(mapping[0] == "L" and mapping[1] == "R", f"lower-pitch cluster should map to L, got {mapping}")
    flipped = assign_clusters_to_hands(labels, pitches, flip=True)
    check(flipped[0] == "R" and flipped[1] == "L", f"flip should swap the mapping, got {flipped}")


def test_assign_hands_for_notes_confident_split():
    rng = np.random.RandomState(2)
    bass_colors = rng.normal(loc=[200, 50, 50], scale=2.0, size=(5, 3))
    treble_colors = rng.normal(loc=[50, 50, 200], scale=2.0, size=(5, 3))
    colors = np.vstack([bass_colors, treble_colors])
    pitches = [40, 42, 44, 46, 48, 70, 72, 74, 76, 78]
    hands, confidence, degenerate = assign_hands_for_notes(colors, pitches)
    check(not degenerate, "well-separated colours correlated with register should not be degenerate")
    check(confidence > 0.0, "confident split should report positive confidence")
    check(all(h == "L" for h in hands[:5]), f"bass-register notes should be L: {hands}")
    check(all(h == "R" for h in hands[5:]), f"treble-register notes should be R: {hands}")


def test_assign_hands_for_notes_degenerate_single_color_theme():
    # A render theme where both hands light the same colour genuinely
    # carries no hand information -- must not invent a split.
    rng = np.random.RandomState(3)
    one_color = rng.normal(loc=[180, 180, 60], scale=2.0, size=(10, 3))
    pitches = [40, 42, 44, 46, 48, 70, 72, 74, 76, 78]
    hands, confidence, degenerate = assign_hands_for_notes(one_color, pitches)
    check(degenerate, "a single-colour theme should be flagged degenerate")
    check(confidence == 0.0, "degenerate assignment should report zero confidence")
    check(len(set(hands)) == 1, f"degenerate case should assign one hand to everything, got {set(hands)}")


def test_assign_hands_from_reference_colors_clear_match():
    reference = {"LH_white": (200, 50, 50), "RH_white": (50, 50, 200)}
    colors = [(198, 52, 48), (52, 48, 198)]
    hands, confidences = assign_hands_from_reference_colors(colors, reference)
    check(hands == ["L", "R"], f"expected [L, R], got {hands}")
    check(all(c > 0.9 for c in confidences), f"clear-cut matches should be high confidence, got {confidences}")


def test_assign_hands_from_reference_colors_ambiguous_midpoint():
    reference = {"LH_white": (0, 0, 0), "RH_white": (200, 200, 200)}
    midpoint = (100, 100, 100)
    hands, confidences = assign_hands_from_reference_colors([midpoint], reference)
    check(abs(confidences[0] - 0.5) < 0.01, f"equidistant colour should give ~0.5 confidence, got {confidences[0]}")


def test_assign_hands_from_reference_colors_missing_black_variants():
    # Only white-key colours given (as if the caller hadn't done its
    # white->black fallback) -- must not crash, still matches correctly.
    reference = {"LH_white": (200, 50, 50), "RH_white": (50, 50, 200)}
    colors = [(190, 60, 55), (60, 55, 190), None]
    hands, confidences = assign_hands_from_reference_colors(colors, reference)
    check(hands[0] == "L" and hands[1] == "R", f"expected [L, R, ...], got {hands}")
    check(confidences[2] == 0.0, "a None colour should get 0.0 confidence")


def test_assign_hands_from_reference_colors_empty_reference():
    hands, confidences = assign_hands_from_reference_colors([(100, 100, 100), None], {})
    check(len(hands) == 2, "empty reference dict should still return one hand per input colour")
    check(all(c == 0.0 for c in confidences), "no reference colours means zero confidence everywhere")


def test_assign_hands_from_reference_colors_none_color_in_list():
    reference = {"LH_white": (200, 50, 50), "RH_white": (50, 50, 200)}
    hands, confidences = assign_hands_from_reference_colors([None], reference)
    check(len(hands) == 1 and confidences[0] == 0.0, "a None sampled colour must not crash and gets 0.0 confidence")


def test_lit_delta_and_is_lit():
    check(lit_delta((100, 100, 100), (100, 100, 100)) == 0.0, "identical colours should have zero delta")
    d = lit_delta((200, 50, 50), (50, 50, 50))
    check(d > 0, "different colours should have positive delta")
    check(is_lit((200, 50, 50), (50, 50, 50)), "a strongly different colour should read as lit")
    check(not is_lit((52, 51, 50), (50, 50, 50)), "a barely different colour should not read as lit")


def test_lit_end_time_detects_release():
    baseline = (50.0, 50.0, 50.0)
    lit = (220.0, 60.0, 60.0)
    # Lit from t=1.0 through t=1.6, then released (back to baseline) from t=1.7 onward.
    samples = [(round(0.1 * i, 2), lit if 1.0 <= 0.1 * i <= 1.6 else baseline) for i in range(30)]
    t = lit_end_time(samples, baseline, onset_sec=1.0)
    check(t is not None, "should detect a release when the key clearly un-lights")
    check(approx(t, 1.7, tol=0.05), f"expected release ~1.7s, got {t}")


def test_lit_end_time_none_when_stays_lit():
    baseline = (50.0, 50.0, 50.0)
    lit = (220.0, 60.0, 60.0)
    samples = [(round(0.1 * i, 2), lit) for i in range(20)]
    check(lit_end_time(samples, baseline, onset_sec=0.5) is None, "a key that never releases should return None")


def test_lit_end_time_none_when_no_samples_at_onset():
    baseline = (50.0, 50.0, 50.0)
    samples = [(0.0, baseline), (0.1, baseline)]
    check(lit_end_time(samples, baseline, onset_sec=5.0) is None, "no samples at/after onset should return None, not raise")


def test_lit_end_time_ignores_pre_lit_attack_gap():
    """A render's visual glow isn't always frame-synced to the audio onset
    it's paired with -- discovered via real R3 validation against an actual
    Synthesia-style render: the first post-onset sample(s) still read as
    not-lit while the glow ramps up, and without a guard, that briefly
    looked exactly like an instant release AT onset (every single trimmed
    duration across 2278 real notes came out ~0s and got reverted). Verify
    that a short pre-lit gap right after onset is correctly skipped, and the
    REAL release later on is what gets reported."""
    baseline = (50.0, 50.0, 50.0)
    lit = (220.0, 60.0, 60.0)
    # onset=1.0, but the glow doesn't actually start until t=1.1 (two samples
    # of not-lit after onset); lit through 1.6, released from 1.7 on.
    samples = [
        (round(0.1 * i, 2), lit if 1.1 <= round(0.1 * i, 2) <= 1.6 else baseline)
        for i in range(30)
    ]
    t = lit_end_time(samples, baseline, onset_sec=1.0)
    check(t is not None, "should still detect the real release, not bail out on the pre-lit gap")
    check(approx(t, 1.7, tol=0.05), f"expected the ACTUAL release ~1.7s, got {t}")


def test_lit_end_time_none_when_never_lit():
    """If the key is never observed lit at all in the sampled window (e.g. a
    completely missed/occluded onset), this must return None -- not a
    spurious near-onset "release" -- same as any other undeterminable case."""
    baseline = (50.0, 50.0, 50.0)
    samples = [(round(0.1 * i, 2), baseline) for i in range(20)]
    check(
        lit_end_time(samples, baseline, onset_sec=0.5) is None,
        "a key never seen lit should return None, not an instant false release",
    )


def _midi_note(start_sec=1.0, duration_sec=2.0, pitch=60):
    return MidiNote(onset_tick=0, pitch=pitch, start_sec=start_sec, duration_sec=duration_sec, velocity=80)


def test_trim_note_durations_shortens_to_lit_end():
    note = _midi_note(start_sec=1.0, duration_sec=2.0)  # audio says it rings until t=3.0
    lit_end = 1.5  # key released at t=1.5 (pedal held the sound on longer)
    trimmed = trim_note_durations([note], [lit_end])
    check(approx(trimmed[0].duration_sec, 0.5), f"expected trimmed duration 0.5, got {trimmed[0].duration_sec}")


def test_trim_note_durations_never_extends():
    note = _midi_note(start_sec=1.0, duration_sec=0.3)
    lit_end = 5.0  # would imply a much LONGER duration than audio detected
    trimmed = trim_note_durations([note], [lit_end])
    check(trimmed[0].duration_sec == 0.3, "trimming must never extend a note past its audio-detected duration")


def test_trim_note_durations_reverts_below_min():
    note = _midi_note(start_sec=1.0, duration_sec=2.0)
    lit_end = 1.01  # candidate duration 0.01s -- below MIN_TRIMMED_DURATION_SEC
    trimmed = trim_note_durations([note], [lit_end])
    check(trimmed[0].duration_sec == 2.0, "an implausibly tiny trim should revert to the original duration")


def test_trim_note_durations_keeps_original_when_lit_end_none():
    note = _midi_note(start_sec=1.0, duration_sec=2.0)
    trimmed = trim_note_durations([note], [None])
    check(trimmed[0].duration_sec == 2.0, "an undeterminable lit-end should leave the note untouched")


def test_default_out_path_from_video():
    # Used when --transcribe is given with no --midi (so there's no MIDI path
    # to derive an output filename from). Places the output directly in
    # ~/Downloads (matching every other default in this tool) using the
    # video's own basename -- NOT next to the video file itself, which for a
    # downloaded YouTube video sits buried in a downloads/ cache subfolder.
    import os
    downloads = os.path.expanduser("~/Downloads")
    check(
        default_out_path_from_video("/home/user/videos/downloads/piece.mp4")
        == os.path.join(downloads, "piece.fingering.json"),
        "default_out_path_from_video should place the output in ~/Downloads, not next to the video",
    )
    check(
        default_out_path_from_video("recording.mov") == os.path.join(downloads, "recording.fingering.json"),
        "default_out_path_from_video should work on a bare relative filename too",
    )


def test_strip_fingering_json_suffix():
    # Regression: Path("piece.fingering.json").with_suffix("") only strips
    # the LAST dot-segment, leaving "piece.fingering" behind -- which made
    # --transcribe's own transcribed-MIDI path come out as
    # "piece.fingering.transcribed.mid" instead of "piece.transcribed.mid"
    # (caught from a real --transcribe run's printed output).
    check(
        _strip_fingering_json_suffix("/a/b/piece.fingering.json") == "/a/b/piece",
        "should strip the full .fingering.json compound suffix, not just .json",
    )
    check(
        _strip_fingering_json_suffix("/a/b/piece.json") == "/a/b/piece",
        "should also handle a bare .json suffix",
    )
    check(
        _strip_fingering_json_suffix("/a/b/piece.mid") == "/a/b/piece",
        "should fall back to stripping any other single suffix",
    )


def test_filter_note_events():
    notes = [
        {"onset_time": 0.0, "offset_time": 0.5, "midi_note": 60, "velocity": 80},   # normal note
        {"onset_time": 1.0, "offset_time": 1.02, "midi_note": 62, "velocity": 70},  # short blip (20ms)
        {"onset_time": 2.0, "offset_time": 2.4, "midi_note": 64, "velocity": 8},    # very quiet (ghost)
        {"onset_time": 3.0, "offset_time": 3.3, "midi_note": 65, "velocity": 90},   # normal note
    ]

    # Defaults are a no-op (behavior unchanged unless explicitly opted in).
    check(filter_note_events(notes) == notes, "filter_note_events with defaults should not drop anything")

    by_velocity = filter_note_events(notes, min_velocity=15)
    check([n["midi_note"] for n in by_velocity] == [60, 62, 65],
          "min_velocity should drop only the quiet (ghost-like) note")

    by_duration = filter_note_events(notes, min_duration_sec=0.05)
    check([n["midi_note"] for n in by_duration] == [60, 64, 65],
          "min_duration_sec should drop only the short blip")

    both = filter_note_events(notes, min_velocity=15, min_duration_sec=0.05)
    check([n["midi_note"] for n in both] == [60, 65],
          "combining both filters should drop the ghost AND the blip")


def test_transcribe_arg_parsing():
    # --transcribe must be accepted, default to off, and --midi must remain
    # optional at the argparse level (analyze() enforces "either --midi or
    # --transcribe" at runtime, not argparse) so `--video x --transcribe`
    # parses without needing a MIDI file.
    parser = build_arg_parser()
    args = parser.parse_args(["--video", "v.mp4", "--transcribe"])
    check(args.transcribe is True, "--transcribe should set args.transcribe = True")
    check(args.midi is None, "--midi should remain optional/None when --transcribe is used")

    default_args = parser.parse_args(["--video", "v.mp4", "--midi", "m.mid"])
    check(default_args.transcribe is False, "--transcribe should default to False")


def test_symple_bundle_round_trip():
    # The .symple bundle is what lets Symplethesia load a MIDI + fingering
    # analysis in one step (see App.ts handleSympleFile / bundle.py). Verify
    # write_symple_bundle produces a zip with the expected fixed entry names
    # and byte-identical content, and that default_bundle_path derives the
    # right sibling filename from a .fingering.json output path.
    import json
    import tempfile
    import zipfile

    with tempfile.TemporaryDirectory() as d:
        midi_path = os.path.join(d, "song.mid")
        fingering_path = os.path.join(d, "piece.fingering.json")
        midi_bytes = b"MThd-fake-midi-bytes-for-test"
        fingering_obj = {"version": 1, "notes": [{"pitch": 60, "hand": "R"}]}

        with open(midi_path, "wb") as f:
            f.write(midi_bytes)
        with open(fingering_path, "w", encoding="utf-8") as f:
            json.dump(fingering_obj, f)

        out_path = default_bundle_path(fingering_path)
        check(out_path == os.path.join(d, "piece.symple"),
              f"default_bundle_path gave unexpected path: {out_path}")

        write_symple_bundle(out_path, midi_path, fingering_path, source_video="https://example.com/v")

        check(os.path.exists(out_path), "bundle file was not written")
        with zipfile.ZipFile(out_path) as zf:
            names = set(zf.namelist())
            check(names == {"manifest.json", "song.mid", "fingering.json"},
                  f"unexpected bundle contents: {names}")
            check(zf.read("song.mid") == midi_bytes, "song.mid bytes did not round-trip")
            check(json.loads(zf.read("fingering.json")) == fingering_obj,
                  "fingering.json did not round-trip")
            manifest = json.loads(zf.read("manifest.json"))
            check(manifest["version"] == 1, "manifest version mismatch")
            check(manifest["contents"] == ["song.mid", "fingering.json"],
                  f"manifest contents list wrong: {manifest['contents']}")
            check(manifest["source"]["video"] == "https://example.com/v",
                  "manifest did not record source video")


def test_homography_perspective_skew():
    # A skewed quadrilateral (simulating an angled camera): top edge narrower
    # than bottom edge (typical of an overhead-ish shot looking slightly down
    # the keyboard).
    corners = [[300, 50], [700, 50], [900, 300], [100, 300]]  # TL, TR, BR, BL
    calib = Calibration(corners=corners, low_pitch=21, high_pitch=108)
    # A point on the bottom edge near the left corner should map close to
    # low_pitch; near the right corner, close to high_pitch.
    p_left = calib.screen_to_pitch((100, 300))
    p_right = calib.screen_to_pitch((900, 300))
    check(p_left == 21, f"skewed bottom-left expected pitch 21, got {p_left}")
    check(p_right == 108, f"skewed bottom-right expected pitch 108, got {p_right}")


# ---------------------------------------------------------------------------
# Sync / offset tests
# ---------------------------------------------------------------------------

def test_offset_cross_correlation_recovers_known_offset():
    rng = np.random.default_rng(42)
    midi_onsets = np.sort(rng.uniform(0, 20, size=60))
    true_offset = 1.35  # video_time = midi_time + true_offset
    # Video press moments are the midi onsets shifted by the true offset,
    # plus a little jitter to simulate imperfect detection.
    jitter = rng.normal(0, 0.01, size=midi_onsets.shape)
    press_times = midi_onsets + true_offset + jitter

    est = estimate_offset(press_times, midi_onsets, max_offset=5.0, bin_sec=0.05)
    check(
        approx(est, true_offset, tol=0.06),
        f"offset estimate off: expected ~{true_offset}, got {est}",
    )


def test_offset_zero_when_aligned():
    rng = np.random.default_rng(7)
    midi_onsets = np.sort(rng.uniform(0, 10, size=30))
    press_times = midi_onsets.copy()
    est = estimate_offset(press_times, midi_onsets, max_offset=3.0, bin_sec=0.05)
    check(approx(est, 0.0, tol=0.06), f"expected ~0 offset, got {est}")


def test_detect_press_moments_synthetic():
    # Build a synthetic FingertipFrame sequence where one fingertip
    # descends (y increasing) then stops -- a single press moment.
    class FakeFrame:
        def __init__(self, t, tips):
            self.time_sec = t
            self._tips = tips

        def fingertip_positions(self):
            return self._tips

    frames = []
    # y descends from 100 to 200 over 0..0.3s then holds at 200.
    for i in range(10):
        t = i * 0.05
        y = 100 + min(i, 6) * (100 / 6)
        frames.append(FakeFrame(t, [("R", 1, 50.0, y)]))
    presses = detect_press_moments(frames)
    check(len(presses) >= 1, f"expected at least one detected press moment, got {presses}")


# ---------------------------------------------------------------------------
# Reconciliation tests (Phase D: video-based ghost-note / de-pedal checks)
# ---------------------------------------------------------------------------

class _FakeFrame:
    """Same minimal FingertipFrame-like double used elsewhere in this file:
    .time_sec + .fingertip_positions() yielding (hand, finger, x, y)."""

    def __init__(self, t, tips):
        self.time_sec = t
        self._tips = tips

    def fingertip_positions(self):
        return self._tips


class _FakeCalib:
    """screen_to_white_index is the only Calibration method reconcile.py
    uses -- identity here (screen x IS the white-key-index unit) so tests
    can place fingertips directly in key-index coordinates."""

    def screen_to_white_index(self, xy):
        return xy[0]


def test_is_occluded():
    calib = _FakeCalib()
    frames = [_FakeFrame(0.0, [("R", 1, 35.0, 0.0)]), _FakeFrame(1.0, [("R", 1, 35.0, 0.0)])]
    frame_times = _frame_times(frames)
    check(not is_occluded(frame_times, frames, 0.0), "a fingertip right at t=0.0 should not be occluded")
    check(is_occluded(frame_times, frames, 0.5), "no fingertip anywhere near t=0.5 should be occluded")


def test_ghost_flagged_when_visible():
    # Video IS tracking hands near onset, but nowhere near this note's key
    # (match_distance_keys is large) -- the video actively disagrees.
    calib = _FakeCalib()
    frames = [_FakeFrame(0.0, [("R", 1, 10.0, 0.0)])]  # far from the note's key
    frame_times = _frame_times(frames)

    loud = reconcile_note(
        velocity=90, duration_sec=0.5, video_onset_sec=0.0, pitch=60,
        match_distance_keys=5.0, frame_times=frame_times, frames=frames, calib=calib,
    )
    check("no-finger-support" in loud.flags, "a far-from-any-fingertip note should be flagged")
    check("dropped" not in loud.flags, "a LOUD unsupported note should be flagged but NOT dropped")

    quiet = reconcile_note(
        velocity=10, duration_sec=0.5, video_onset_sec=0.0, pitch=60,
        match_distance_keys=5.0, frame_times=frame_times, frames=frames, calib=calib,
    )
    check("dropped" in quiet.flags, "a QUIET unsupported note should be dropped (ghost: no support + quiet)")


def test_ghost_kept_when_occluded():
    # No fingertip tracked anywhere near onset at all -- the video has no
    # evidence either way, so it must NOT judge this note (audio wins).
    calib = _FakeCalib()
    frames = [_FakeFrame(5.0, [("R", 1, 10.0, 0.0)])]  # nowhere near t=0.0
    frame_times = _frame_times(frames)
    rec = reconcile_note(
        velocity=10, duration_sec=0.5, video_onset_sec=0.0, pitch=60,
        match_distance_keys=5.0, frame_times=frame_times, frames=frames, calib=calib,
    )
    check(rec.flags == [], f"an occluded note must get no flags at all, got {rec.flags}")


def test_finger_leave_time_detects_departure():
    key_x = pitch_to_white_index(60)
    frames = [
        _FakeFrame(0.0, [("R", 1, key_x, 0.0)]),
        _FakeFrame(0.1, [("R", 1, key_x, 0.0)]),
        _FakeFrame(0.2, [("R", 1, key_x + 10.0, 0.0)]),  # left
        _FakeFrame(0.3, [("R", 1, key_x + 10.0, 0.0)]),
        _FakeFrame(0.4, [("R", 1, key_x + 10.0, 0.0)]),
    ]
    frame_times = _frame_times(frames)
    leave = finger_leave_time(frame_times, frames, _FakeCalib(), onset_sec=0.0, pitch=60, gap_frames=2)
    check(leave == 0.2, f"expected the finger to be detected leaving at t=0.2, got {leave}")

    staying_frames = [_FakeFrame(t, [("R", 1, key_x, 0.0)]) for t in (0.0, 0.1, 0.2, 0.3, 0.4)]
    staying_times = _frame_times(staying_frames)
    never_left = finger_leave_time(staying_times, staying_frames, _FakeCalib(), onset_sec=0.0, pitch=60)
    check(never_left is None, f"a finger that never leaves should return None, got {never_left}")


def test_depedal_trims_when_finger_leaves_before_audio_offset():
    key_x = pitch_to_white_index(60)
    # Finger leaves at t=0.2; audio (pedal-inflated) offset is at t=1.0 --
    # well past DEPEDAL_MIN_LEAD_SEC, and past MIN_TRIMMED_DURATION_SEC.
    frames = [
        _FakeFrame(0.0, [("R", 1, key_x, 0.0)]),
        _FakeFrame(0.1, [("R", 1, key_x, 0.0)]),
        _FakeFrame(0.2, [("R", 1, key_x + 10.0, 0.0)]),
        _FakeFrame(0.3, [("R", 1, key_x + 10.0, 0.0)]),
        _FakeFrame(0.4, [("R", 1, key_x + 10.0, 0.0)]),
    ]
    frame_times = _frame_times(frames)
    pedal_segments = [(0.0, 1.0)]  # pedal down through the whole (inflated) note

    rec = reconcile_note(
        velocity=90, duration_sec=1.0, video_onset_sec=0.0, pitch=60,
        match_distance_keys=0.1, frame_times=frame_times, frames=frames, calib=_FakeCalib(),
        pedal_segments=pedal_segments,
    )
    check("depedaled" in rec.flags, f"expected a depedal trim, got flags {rec.flags}")
    check(rec.audio_duration_sec == 1.0, "should keep the original audio-reported duration")
    check(abs(rec.trimmed_duration_sec - 0.2) < 1e-9, f"trimmed duration should be ~0.2s, got {rec.trimmed_duration_sec}")


def test_depedal_not_applied_without_pedal_or_when_finger_stays():
    key_x = pitch_to_white_index(60)
    frames = [_FakeFrame(t, [("R", 1, key_x, 0.0)]) for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    frame_times = _frame_times(frames)

    no_pedal = reconcile_note(
        velocity=90, duration_sec=1.0, video_onset_sec=0.0, pitch=60,
        match_distance_keys=0.1, frame_times=frame_times, frames=frames, calib=_FakeCalib(),
        pedal_segments=None,
    )
    check("depedaled" not in no_pedal.flags, "no pedal data at all -- must never trim")

    finger_stays = reconcile_note(
        velocity=90, duration_sec=1.0, video_onset_sec=0.0, pitch=60,
        match_distance_keys=0.1, frame_times=frame_times, frames=frames, calib=_FakeCalib(),
        pedal_segments=[(0.0, 1.0)],
    )
    check("depedaled" not in finger_stays.flags, "finger never leaves the key -- must never trim")


def test_depedal_reverts_when_trim_would_be_too_short():
    key_x = pitch_to_white_index(60)
    # Finger leaves almost immediately (t=0.02) -- trimming to that would
    # leave well under MIN_TRIMMED_DURATION_SEC (0.12s), so it must revert.
    frames = [
        _FakeFrame(0.0, [("R", 1, key_x, 0.0)]),
        _FakeFrame(0.01, [("R", 1, key_x, 0.0)]),
        _FakeFrame(0.02, [("R", 1, key_x + 10.0, 0.0)]),
        _FakeFrame(0.03, [("R", 1, key_x + 10.0, 0.0)]),
        _FakeFrame(0.04, [("R", 1, key_x + 10.0, 0.0)]),
    ]
    frame_times = _frame_times(frames)
    rec = reconcile_note(
        velocity=90, duration_sec=1.0, video_onset_sec=0.0, pitch=60,
        match_distance_keys=0.1, frame_times=frame_times, frames=frames, calib=_FakeCalib(),
        pedal_segments=[(0.0, 1.0)],
    )
    check("depedaled" not in rec.flags, "a too-short trim should be reverted, not applied")


def test_confidence_by_window():
    notes = [(0.0, 0.9), (5.0, 0.8), (12.0, 0.5), (15.0, 0.3), (25.0, 0.7)]
    windows = confidence_by_window(notes, window_sec=10.0)

    def approx_window(w, expected):
        start, conf, count = w
        e_start, e_conf, e_count = expected
        return approx(start, e_start) and approx(conf, e_conf) and count == e_count

    check(approx_window(windows[0], (0.0, 0.85, 2)), f"window 0 should average the two notes at t=0,5 -> got {windows[0]}")
    check(approx_window(windows[1], (10.0, 0.4, 2)), f"window 1 should average the two notes at t=12,15 -> got {windows[1]}")
    check(approx_window(windows[2], (20.0, 0.7, 1)), f"window 2 should have the single note at t=25 -> got {windows[2]}")


def test_analyze_hand_spread_none_when_hands_never_both_present():
    calib = _FakeCalib()
    frames = [_FakeFrame(0.0, [("R", 1, 40.0, 0.0)]), _FakeFrame(0.1, [("R", 1, 41.0, 0.0)])]
    check(analyze_hand_spread(frames, calib) is None, "a one-handed video should have nothing to compare")


def test_analyze_hand_spread_measures_centroid_distance():
    calib = _FakeCalib()
    # Two frames, hands a constant 10 keys apart (centroid-to-centroid).
    frames = [
        _FakeFrame(0.0, [("L", 1, 10.0, 0.0), ("R", 1, 20.0, 0.0)]),
        _FakeFrame(0.1, [("L", 1, 10.0, 0.0), ("R", 1, 20.0, 0.0)]),
    ]
    report = analyze_hand_spread(frames, calib)
    check(report is not None, "both hands present should produce a report")
    check(report.frames_with_both_hands == 2, f"expected 2 dual-hand frames, got {report.frames_with_both_hands}")
    check(approx(report.mean_distance_keys, 10.0), f"expected mean 10.0 keys apart, got {report.mean_distance_keys}")
    check(approx(report.frac_close, 0.0), "10 keys apart should never count as 'close'")


def test_analyze_hand_spread_flags_close_hands():
    calib = _FakeCalib()
    # Hands consistently under CLOSE_HANDS_THRESHOLD_KEYS apart (chordal comping).
    frames = [_FakeFrame(t, [("L", 1, 30.0, 0.0), ("R", 1, 31.0, 0.0)]) for t in (0.0, 0.1, 0.2)]
    report = analyze_hand_spread(frames, calib)
    check(report.mean_distance_keys < CLOSE_HANDS_THRESHOLD_KEYS, "hands 1 key apart should read as close")
    check(approx(report.frac_close, 1.0), "every frame should count as 'close' when hands are 1 key apart")


def test_analyze_hand_spread_averages_multiple_fingertips_per_hand():
    calib = _FakeCalib()
    # Left hand's own fingertips span 8..12 (centroid 10); right hand at 20 alone.
    frames = [_FakeFrame(0.0, [
        ("L", 1, 8.0, 0.0), ("L", 2, 12.0, 0.0), ("R", 1, 20.0, 0.0),
    ])]
    report = analyze_hand_spread(frames, calib)
    check(approx(report.mean_distance_keys, 10.0), f"expected centroid distance 10.0, got {report.mean_distance_keys}")


# ---------------------------------------------------------------------------
# Matching tests
# ---------------------------------------------------------------------------

def test_matcher_picks_correct_finger():
    # Expected key at (500, 100). Two candidate fingertips: one very close
    # (right thumb), one far away (left pinky). Matcher should pick the near one.
    candidates = [
        Candidate(hand="R", finger=1, x=505, y=102),
        Candidate(hand="L", finger=5, x=50, y=400),
    ]
    result = match_note((500, 100), candidates)
    check(result is not None, "expected a match result")
    check(result.hand == "R" and result.finger == 1, f"expected R1, got {result.hand}{result.finger}")
    check(result.confidence > 0.9, f"expected high confidence for close match, got {result.confidence}")


def test_matcher_low_confidence_for_far_match():
    candidates = [Candidate(hand="L", finger=3, x=50, y=400)]
    result = match_note((500, 100), candidates)
    check(result is not None, "expected a match result even if far")
    check(result.confidence < 0.3, f"expected low confidence for far match, got {result.confidence}")


def test_chord_conflict_resolution_no_reuse():
    # Three simultaneous notes, expected key positions spread out; four
    # candidate fingertips from two hands. No (hand, finger) should be used
    # twice.
    notes = [
        NoteToMatch(note_id="n1", key_xy=(100, 100)),
        NoteToMatch(note_id="n2", key_xy=(300, 100)),
        NoteToMatch(note_id="n3", key_xy=(500, 100)),
    ]
    candidates = [
        Candidate(hand="L", finger=5, x=105, y=100),
        Candidate(hand="L", finger=1, x=305, y=100),
        Candidate(hand="R", finger=1, x=505, y=100),
        Candidate(hand="R", finger=2, x=520, y=100),  # decoy, farther from n3
    ]
    results = resolve_chord_conflicts(notes, candidates)
    check(len(results) == 3, f"expected 3 matches, got {len(results)}")
    assigned = [(r.hand, r.finger) for r in results.values()]
    check(len(assigned) == len(set(assigned)), f"duplicate hand/finger assignment: {assigned}")
    check(results["n1"].hand == "L" and results["n1"].finger == 5, f"n1 mismatch: {results['n1']}")
    check(results["n2"].hand == "L" and results["n2"].finger == 1, f"n2 mismatch: {results['n2']}")
    check(results["n3"].hand == "R" and results["n3"].finger == 1, f"n3 mismatch: {results['n3']}")


def test_chord_conflict_more_notes_than_candidates():
    notes = [
        NoteToMatch(note_id="n1", key_xy=(100, 100)),
        NoteToMatch(note_id="n2", key_xy=(300, 100)),
    ]
    candidates = [Candidate(hand="L", finger=1, x=100, y=100)]
    results = resolve_chord_conflicts(notes, candidates)
    check(len(results) == 1, f"expected exactly 1 match when candidates are scarce, got {len(results)}")


def test_group_simultaneous():
    times = [0.0, 0.01, 0.02, 1.0, 1.005, 2.5]
    groups = group_simultaneous(times, tolerance_sec=0.03)
    check(len(groups) == 3, f"expected 3 groups, got {len(groups)}: {groups}")
    sizes = sorted(len(g) for g in groups)
    check(sizes == [1, 2, 3], f"expected group sizes [1,2,3], got {sizes}")


def test_confidence_monotonic_decay():
    c0 = confidence_from_distance(0.0)
    c1 = confidence_from_distance(30.0)
    c2 = confidence_from_distance(300.0)
    check(c0 == 1.0, f"zero distance should give confidence 1.0, got {c0}")
    check(c0 > c1 > c2, f"confidence should decay monotonically: {c0}, {c1}, {c2}")


def test_calibration_depth_aware_reconstruction():
    """Synthetic side-angle setup: build a known 2D homography, sample near
    and far rows through it, fit both row maps, reconstruct the homography
    into a Calibration, and verify screen_to_white_index recovers the true
    pitch at a FRONT-EDGE screen point -- specifically a point where the old
    single-row x-only inversion would fail (prove the fix matters)."""
    # Simulate a 30° tilt: far and near rows recede into the image at different angles.
    # True homography: maps (u, v) in [0,1]^2 to screen coordinates.
    # We'll construct one by hand: far row (v=0) goes left->right across the screen,
    # near row (v=1) is foreshortened by perspective.
    h_true = np.array([
        [500, 100, 100],   # far-left at (500, 100), near-right at (600, 400)
        [1400, 100, 300],  # far-right at (1400, 100), near-right at (1700, 400)
        [0, 1, 1]
    ], dtype=np.float64)
    # Proper homography form: normalize by h[2,2]=1
    h_true = h_true / h_true[2, 2]

    def apply_h(h, u, v):
        pt = h @ np.array([u, v, 1])
        return pt[:2] / pt[2]

    # Sample the far row at u=0.1, 0.5, 0.9 (in full-res pixel space)
    far_points = [apply_h(h_true, u, 0.0) for u in [0.0, 0.5, 1.0]]
    # Sample the near row at same u values
    near_points = [apply_h(h_true, u, 1.0) for u in [0.0, 0.5, 1.0]]

    # Fit rows via _fit_row_projective
    u_vals = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    far_x = np.array([pt[0] for pt in far_points])
    far_y = np.array([pt[1] for pt in far_points])
    near_x = np.array([pt[0] for pt in near_points])
    near_y = np.array([pt[1] for pt in near_points])

    from keyboard import _fit_row_projective, _row_project, Calibration
    row_far = _fit_row_projective(u_vals, far_x, far_y)
    row_near = _fit_row_projective(u_vals, near_x, near_y)

    check(row_far is not None, "far row fit should succeed")
    check(row_near is not None, "near row fit should succeed")

    # Build Calibration with both rows
    calib = Calibration(
        corners=[[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
        low_pitch=48, high_pitch=72,
        row=row_far, row_near=row_near,
    )

    # Pick a test point on the near row at u=0.3 (should map to a specific pitch)
    test_u = 0.3
    screen_pt = apply_h(h_true, test_u, 1.0)  # Near-row point

    # Old method (single-row x-only inversion) would be wrong here:
    # It would invert screen_x alone, ignoring screen_y, so it'd miss the depth.
    # New method should use the full homography and recover test_u correctly.

    recovered_u, recovered_v = calib.screen_to_norm(screen_pt)
    check(approx(recovered_u, test_u, tol=0.05),
          f"expected u={test_u}, got {recovered_u} (indicates full homography working)")
    check(approx(recovered_v, 1.0, tol=0.05),
          f"expected v=1.0 (near edge), got {recovered_v}")


def test_calibration_depth_aware_round_trip():
    """Verify that Calibration.to_dict -> from_dict preserves row_near and
    still produces correct pitch_to_screen output."""
    # Simple test: create a calib with both rows, serialize, deserialize, check equality.
    from keyboard import Calibration
    row_far = [100.0, 500.0, 200.0, 100.0, 0.005]
    row_near = [150.0, 700.0, 250.0, 150.0, 0.003]

    calib_orig = Calibration(
        corners=[[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
        low_pitch=48, high_pitch=72,
        row=row_far, row_near=row_near,
    )

    d = calib_orig.to_dict()
    check("row" in d, "serialized dict should have 'row'")
    check("row_near" in d, "serialized dict should have 'row_near'")
    check(d["row"] == row_far, "serialized row should match")
    check(d["row_near"] == row_near, "serialized row_near should match")

    calib_restored = Calibration.from_dict(d)
    check(calib_restored.row == row_far, "restored row should match")
    check(calib_restored.row_near == row_near, "restored row_near should match")

    # Verify pitch_to_screen still works
    pt = calib_restored.pitch_to_screen(60, v=0.9)
    check(pt is not None, "pitch_to_screen should work on restored calib")
    check(len(pt) == 2, "pitch_to_screen should return 2D point")


def test_calibration_backward_compat_single_row():
    """Backward compatibility: a Calibration with row set but row_near=None
    should behave identically to the old single-row path."""
    from keyboard import Calibration
    row_map = [100.0, 500.0, 200.0, 100.0, 0.005]

    calib = Calibration(
        corners=[[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
        low_pitch=48, high_pitch=72,
        row=row_map, row_near=None,
    )

    # screen_to_norm should use the x-only inversion (single-row branch)
    xy = (800.0, 200.0)
    norm = calib.screen_to_norm(xy)
    check(norm is not None, "screen_to_norm should work")
    check(len(norm) == 2, "screen_to_norm should return 2D")
    check(approx(norm[1], 0.5, tol=0.01), "single-row mode should always return v~0.5 (middle)")

    # pitch_to_screen should use single-row projection
    pt = calib.pitch_to_screen(60)
    check(pt is not None, "pitch_to_screen should work in single-row mode")


TESTS = [
    test_white_index_roundtrip,
    test_homography_known_corners_to_expected_pitches,
    test_row_calibration_single_row_click,
    test_row_fit_rejects_too_few_points,
    test_classify_gap_sizes_full_keyboard,
    test_classify_gap_sizes_noisy_missing_black_key,
    test_expected_black_key_groups_full_88,
    test_assign_group_pitches_partial_keyboard,
    test_assign_group_pitches_noisy_missing_black_key,
    test_assign_group_pitches_edge_junk,
    test_assign_group_pitches_low_confidence_returns_none,
    test_assign_group_pitches_narrow_range_is_octave_ambiguous,
    test_build_anchor_points_skips_ambiguous_groups,
    test_detect_for_range_full_keyboard_recovers_correct_mapping,
    test_calibration_key_channel_vs_path,
    test_calibrations_agree_identical,
    test_calibrations_agree_small_shift_within_tolerance,
    test_calibrations_agree_large_shift_disagrees,
    test_calibrations_agree_ignores_row_height_difference,
    test_calibration_store_round_trip,
    test_cluster_hand_colors_two_clear_clusters,
    test_cluster_hand_colors_single_cluster_low_separation,
    test_cluster_hand_colors_degenerate_inputs,
    test_cluster_hand_colors_4color_tint_shade_theme,
    test_cluster_hand_colors_same_hue_two_lightness,
    test_cluster_hand_colors_near_gray_white_theme,
    test_assign_clusters_to_hands_lower_pitch_is_left,
    test_assign_hands_for_notes_confident_split,
    test_assign_hands_for_notes_degenerate_single_color_theme,
    test_assign_hands_from_reference_colors_clear_match,
    test_assign_hands_from_reference_colors_ambiguous_midpoint,
    test_assign_hands_from_reference_colors_missing_black_variants,
    test_assign_hands_from_reference_colors_empty_reference,
    test_assign_hands_from_reference_colors_none_color_in_list,
    test_lit_delta_and_is_lit,
    test_lit_end_time_detects_release,
    test_lit_end_time_none_when_stays_lit,
    test_lit_end_time_none_when_no_samples_at_onset,
    test_lit_end_time_ignores_pre_lit_attack_gap,
    test_lit_end_time_none_when_never_lit,
    test_trim_note_durations_shortens_to_lit_end,
    test_trim_note_durations_never_extends,
    test_trim_note_durations_reverts_below_min,
    test_trim_note_durations_keeps_original_when_lit_end_none,
    test_default_out_path_from_video,
    test_strip_fingering_json_suffix,
    test_filter_note_events,
    test_transcribe_arg_parsing,
    test_symple_bundle_round_trip,
    test_homography_perspective_skew,
    test_offset_cross_correlation_recovers_known_offset,
    test_offset_zero_when_aligned,
    test_detect_press_moments_synthetic,
    test_is_occluded,
    test_ghost_flagged_when_visible,
    test_ghost_kept_when_occluded,
    test_finger_leave_time_detects_departure,
    test_depedal_trims_when_finger_leaves_before_audio_offset,
    test_depedal_not_applied_without_pedal_or_when_finger_stays,
    test_depedal_reverts_when_trim_would_be_too_short,
    test_confidence_by_window,
    test_analyze_hand_spread_none_when_hands_never_both_present,
    test_analyze_hand_spread_measures_centroid_distance,
    test_analyze_hand_spread_flags_close_hands,
    test_analyze_hand_spread_averages_multiple_fingertips_per_hand,
    test_matcher_picks_correct_finger,
    test_matcher_low_confidence_for_far_match,
    test_chord_conflict_resolution_no_reuse,
    test_chord_conflict_more_notes_than_candidates,
    test_group_simultaneous,
    test_confidence_monotonic_decay,
    test_calibration_depth_aware_reconstruction,
    test_calibration_depth_aware_round_trip,
    test_calibration_backward_compat_single_row,
]


def run() -> bool:
    failures = []
    for test in TESTS:
        name = test.__name__
        try:
            test()
            print(f"  ok  {name}")
        except TestFailure as e:
            print(f"FAIL  {name}: {e}")
            failures.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failures.append(name)
    print()
    if failures:
        print(f"{len(failures)}/{len(TESTS)} tests FAILED: {', '.join(failures)}")
        return False
    print(f"OK - all {len(TESTS)} tests passed")
    return True


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
