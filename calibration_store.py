"""Per-source calibration persistence (Phase B of
ai/tasks/005-cv-calibration-batch/PLAN.md): remembers a confirmed
`Calibration` keyed by camera setup ("this channel", "this local file") so
a later video from the same setup can default to reusing it instead of
re-detecting/re-clicking from scratch.

No cv2 dependency -- safe to import from selftest.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import numpy as np

from keyboard import Calibration, pitch_to_white_index

CALIBRATION_DIR = os.path.expanduser("~/.piano-fingering/calibrations")

# Default agreement tolerance: max allowed disagreement at any sampled C,
# as a fraction of one white-key width (resolution-independent).
DEFAULT_AGREEMENT_TOLERANCE = 0.15


def calibration_key(video: str, channel_id: Optional[str]) -> str:
    """The storage key for a video's camera setup: the yt-dlp `channel_id`
    when known (stable across that channel's uploads, and matches how a
    user thinks about it -- "the Maximizer videos"), else a short hash of
    the local file path (so at least re-running the exact same local file
    reuses its own calibration)."""
    if channel_id:
        return f"channel-{channel_id}"
    digest = hashlib.sha1(os.path.abspath(video).encode("utf-8")).hexdigest()[:16]
    return f"path-{digest}"


def _path_for_key(key: str, base_dir: Optional[str] = None) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return os.path.join(base_dir or CALIBRATION_DIR, f"{safe}.json")


def load_saved_calibration(key: str, base_dir: Optional[str] = None) -> Optional[Calibration]:
    """``base_dir`` overrides CALIBRATION_DIR -- used by selftest.py to
    exercise the round-trip against a throwaway temp directory instead of
    the real ``~/.piano-fingering/calibrations``."""
    path = _path_for_key(key, base_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return Calibration.from_dict(json.load(f))
    except Exception:
        # A corrupt/unreadable cache entry should never block calibration --
        # just behave as if nothing were saved for this key.
        return None


def save_calibration(key: str, calib: Calibration, base_dir: Optional[str] = None) -> None:
    target_dir = base_dir or CALIBRATION_DIR
    os.makedirs(target_dir, exist_ok=True)
    with open(_path_for_key(key, base_dir), "w", encoding="utf-8") as f:
        json.dump(calib.to_dict(), f, indent=2)


def calibrations_agree(
    saved: Calibration,
    fresh: Calibration,
    low_pitch: int,
    high_pitch: int,
    tolerance: float = DEFAULT_AGREEMENT_TOLERANCE,
) -> bool:
    """Whether ``saved`` still looks like a good fit for the same camera
    setup as ``fresh`` (an independently-detected calibration for a NEW
    video from that setup) -- the corroboration check that keeps a stale
    saved calibration from being silently reused after the camera gets
    bumped or re-angled between takes.

    Compares only the X (along-the-keyboard) component of where each C
    projects, since that's the only component that actually determines
    pitch (screen_to_white_index is deliberately depth/Y-invariant -- see
    keyboard.py). The row detector can legitimately lock onto a different
    scanline height between two reads of the SAME unchanged camera (hand
    position differs frame to frame, shifting which row has the clearest
    black-key blobs) -- that changes each anchor's Y a lot while leaving X
    essentially untouched, and comparing full (X, Y) distance would flag
    that as a mismatch even though the pitch mapping is identical. The
    tolerance is in one white-key-width units (estimated from ``fresh``,
    since it reflects THIS video's actual resolution/framing) -- relative
    and resolution-independent, since the two videos may not even share a
    resolution.
    """
    cs = [p for p in range(low_pitch, high_pitch + 1) if p % 12 == 0]
    if not cs:
        return False

    lo_wi = pitch_to_white_index(low_pitch)
    hi_wi = pitch_to_white_index(high_pitch)
    if hi_wi == lo_wi:
        return False
    x_lo = float(fresh.pitch_to_screen(low_pitch)[0])
    x_hi = float(fresh.pitch_to_screen(high_pitch)[0])
    white_key_px = abs(x_hi - x_lo) / (hi_wi - lo_wi)
    if white_key_px <= 0:
        return False

    for pitch in cs:
        sx = float(saved.pitch_to_screen(pitch)[0])
        fx = float(fresh.pitch_to_screen(pitch)[0])
        if abs(sx - fx) > tolerance * white_key_px:
            return False
    return True
