"""Per-channel hand-color persistence: remembers the user's picked left-hand
(LH) and right-hand (RH) reference colors keyed by camera setup ("this channel",
"this local file") so a later Synthesia-render video from the same setup can
reuse them instead of re-picking from scratch.

No cv2 dependency -- safe to import from selftest.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

COLORS_DIR = os.path.expanduser("~/.piano-fingering/colors")


def color_key(video: str, channel_id: Optional[str]) -> str:
    """The storage key for a video's hand colors: the yt-dlp `channel_id`
    when known (stable across that channel's uploads, and matches how a
    user thinks about it -- "the Maximizer videos"), else a short hash of
    the local file path (so at least re-running the exact same local file
    reuses its own colors)."""
    if channel_id:
        return f"channel-{channel_id}"
    digest = hashlib.sha1(os.path.abspath(video).encode("utf-8")).hexdigest()[:16]
    return f"path-{digest}"


def _path_for_key(key: str, base_dir: Optional[str] = None) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return os.path.join(base_dir or COLORS_DIR, f"{safe}.json")


def load_saved_colors(key: str, base_dir: Optional[str] = None) -> Optional[dict]:
    """Load saved hand colors for a key. ``base_dir`` overrides COLORS_DIR --
    used by selftest.py to exercise the round-trip against a throwaway temp
    directory instead of the real ``~/.piano-fingering/colors``.

    Returns the color dict, or None if the file does not exist OR is
    corrupt/unreadable (wrap in try/except and return None on any exception,
    exactly like load_saved_calibration).
    """
    path = _path_for_key(key, base_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # A corrupt/unreadable cache entry should never block color picking --
        # just behave as if nothing were saved for this key.
        return None


def save_colors(key: str, colors: dict, base_dir: Optional[str] = None) -> None:
    """Save hand colors for a key.

    The colors dict shape is:
        {"LH_white": [r, g, b], "LH_black": [r, g, b],
         "RH_white": [r, g, b], "RH_black": [r, g, b]}
    where each value is a list of 3 ints (0-255). Store/load it verbatim as
    JSON -- do NOT validate, transform, or add missing keys (a caller decides that).
    """
    target_dir = base_dir or COLORS_DIR
    os.makedirs(target_dir, exist_ok=True)
    with open(_path_for_key(key, base_dir), "w", encoding="utf-8") as f:
        json.dump(colors, f, indent=2)
