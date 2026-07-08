"""MediaPipe Hands wrapper: extracts per-frame hand landmarks from video.

All heavy imports (cv2, mediapipe) are done lazily inside functions so this
module can be imported (e.g. for its data classes) without either package
installed. `selftest.py` never calls the functions in this file that need
cv2/mediapipe -- it only exercises `sync.py` / `match.py` / `keyboard.py`
with synthetic `FingertipFrame` data built by hand.

MediaPipe Hands landmark indices we care about (fingertips):
    4  = thumb tip
    8  = index tip
    12 = middle tip
    16 = ring tip
    20 = pinky tip
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field

FINGERTIP_LANDMARKS = {1: 4, 2: 8, 3: 12, 4: 16, 5: 20}  # finger number -> landmark index


@contextlib.contextmanager
def _suppress_native_stderr():
    """Redirects the OS-level stderr file descriptor to /dev/null for the
    duration of the block.

    mediapipe's C++ backend (absl/glog) prints a burst of native log lines
    -- a one-time "WARNING: All log messages before absl::InitializeLog()
    is called are written to STDERR" notice, EGL/GL context init, "Created
    TensorFlow Lite XNNPACK delegate", "Feedback manager requires a model
    with a single signature inference", etc. -- the moment a HandLandmarker
    is actually constructed (`HandLandmarker.create_from_options`), not at
    `import mediapipe`. These are printed via raw fprintf/glog before (or
    entirely outside of) any severity filter Python could configure --
    env vars like GLOG_minloglevel/TF_CPP_MIN_LOG_LEVEL do NOT suppress
    them on this mediapipe build (verified empirically). A raw fd-level
    redirect around just the `create_from_options` call is the only
    reliable way to catch them, confirmed by testing with/without it."""
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


@dataclass
class Fingertip:
    finger: int  # 1..5 (thumb..pinky)
    x: float  # pixel x
    y: float  # pixel y


@dataclass
class HandObservation:
    hand: str  # 'L' or 'R'
    fingertips: list  # list[Fingertip], one per finger 1..5 (may be partial)


@dataclass
class FingertipFrame:
    """All hands observed in a single video frame."""

    time_sec: float
    hands: list = field(default_factory=list)  # list[HandObservation]

    def fingertip_positions(self):
        """Yields (hand, finger, x, y) for every fingertip in this frame."""
        for obs in self.hands:
            for tip in obs.fingertips:
                yield obs.hand, tip.finger, tip.x, tip.y


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def _model_path():
    """Returns the cached local path to the HandLandmarker `.task` model,
    relative to this file's own directory (not cwd). Lazy-imports pathlib
    only (already stdlib, no heavy deps)."""
    from pathlib import Path

    return Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def _ensure_model_downloaded(model_path) -> None:
    """Downloads the HandLandmarker model to `model_path` if it doesn't
    already exist, using stdlib urllib (no new required dependency)."""
    import urllib.request

    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmark model (one-time, ~10MB) from {MODEL_URL} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
    except Exception as exc:
        raise RuntimeError(
            "Failed to download the MediaPipe hand landmark model automatically "
            f"({exc}). Please download it manually from:\n    {MODEL_URL}\n"
            f"and place it at:\n    {model_path}"
        ) from exc


def extract_fingertip_frames(
    video_path: str,
    fps: float = 30.0,
    max_hands: int = 2,
    flip_handedness: bool = False,
    k1: float = 0.0,
    min_hand_confidence: float = 0.5,
):
    """Runs MediaPipe's HandLandmarker (Tasks API) over the video and
    returns a list of `FingertipFrame`, one per sampled frame at the
    requested fps.

    `min_hand_confidence` sets both `min_hand_detection_confidence` and
    `min_hand_presence_confidence` (the library default for both is 0.5).
    Lower this if the preview/output shows one hand frequently missing
    entirely even though it's clearly visible and playing -- a real,
    observed failure mode on at least one monochrome/black-and-white source
    video (MediaPipe's hand detector was trained mostly on color imagery and
    partly relies on skin-tone cues, which desaturated footage lacks). This
    trades a higher false-detection rate for fewer missed real hands; 0.3 is
    a reasonable first thing to try. `min_tracking_confidence` is left at
    the library default -- this is about a hand not being detected AT ALL,
    not about noisy tracking of an already-detected hand.

    Lazy-imports cv2 and mediapipe. Not used by selftest.py.
    """
    import cv2  # lazy import
    import mediapipe as mp  # lazy import
    from keyboard import undistort_frame  # lazy import (no circular import risk)

    model_path = _model_path()
    _ensure_model_downloaded(model_path)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=min_hand_confidence,
        min_hand_presence_confidence=min_hand_confidence,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    frame_stride = max(1, round(src_fps / fps))

    frames: list = []
    with _suppress_native_stderr():
        landmarker_ctx = HandLandmarker.create_from_options(options)
    with landmarker_ctx as landmarker:
        idx = 0
        last_timestamp_ms = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if k1 != 0.0:
                frame = undistort_frame(frame, k1)
            if idx % frame_stride == 0:
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                time_sec = idx / src_fps
                timestamp_ms = int(round(time_sec * 1000))
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                observations: list = []
                if result.hand_landmarks and result.handedness:
                    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                        # MediaPipe reports handedness from the camera's
                        # perspective assuming a mirror-selfie view; for an
                        # overhead shot of the performer's own hands (not
                        # mirrored) the raw label matches physical hand.
                        label = handedness[0].category_name  # 'Left' or 'Right'
                        hand_code = "L" if label == "Left" else "R"
                        if flip_handedness:
                            hand_code = "R" if hand_code == "L" else "L"
                        tips = []
                        for finger, lm_idx in FINGERTIP_LANDMARKS.items():
                            lm = landmarks[lm_idx]
                            tips.append(Fingertip(finger=finger, x=lm.x * w, y=lm.y * h))
                        observations.append(HandObservation(hand=hand_code, fingertips=tips))
                frames.append(FingertipFrame(time_sec=time_sec, hands=observations))
            idx += 1
    cap.release()
    return frames


# Maximum gap between two detection frames to still interpolate between them.
# Beyond this, the hand moved too far or was occluded too long to trust the
# straight-line path (octave jumps, hand leaving frame, etc.).
_MAX_INTERP_GAP_SEC = 0.20

# Maximum distance from a single detection frame to still carry its position
# forward or backward in time when the other side has no detection.
# Covers brief dropouts (2-4 frames at 25-30 fps) without bridging real gaps.
_MAX_CARRY_SEC = 0.15


def interpolate_fingertips(frames, time_sec: float,
                           max_interp_gap: float = _MAX_INTERP_GAP_SEC,
                           max_carry: float = _MAX_CARRY_SEC):
    """Interpolate fingertip positions to time_sec from the nearest frames
    that actually contain hand detections (not just the nearest frames by
    time, which are usually empty).

    - Both sides detected within max_interp_gap → linear interpolation.
    - Only one side detected within max_carry → carry that position.
    - Gap larger than the applicable limit → return [] (no candidates).

    Pure-python/numpy-free; used with real MediaPipe frames and in selftest.py.
    """
    if not frames:
        return []

    # Binary search: find the insertion point for time_sec in frames[].time_sec.
    lo, hi = 0, len(frames)
    while lo < hi:
        mid = (lo + hi) // 2
        if frames[mid].time_sec <= time_sec:
            lo = mid + 1
        else:
            hi = mid
    # frames[lo-1].time_sec <= time_sec < frames[lo].time_sec
    pivot = lo  # first frame strictly after time_sec

    # Walk backward from pivot to find nearest frame WITH detections at/before time_sec.
    before = None
    for i in range(pivot - 1, -1, -1):
        if frames[i].hands:
            before = frames[i]
            break
        if time_sec - frames[i].time_sec > max_interp_gap:
            break  # too far back, no point continuing

    # Walk forward from pivot to find nearest frame WITH detections after time_sec.
    after = None
    for i in range(pivot, len(frames)):
        if frames[i].hands:
            after = frames[i]
            break
        if frames[i].time_sec - time_sec > max_interp_gap:
            break  # too far ahead

    def _lerp(fa, fb):
        span = fb.time_sec - fa.time_sec
        t = 0.0 if span <= 0 else (time_sec - fa.time_sec) / span
        a_map = {(h, f): (x, y) for h, f, x, y in fa.fingertip_positions()}
        b_map = {(h, f): (x, y) for h, f, x, y in fb.fingertip_positions()}
        out = []
        for key in set(a_map) | set(b_map):
            if key in a_map and key in b_map:
                (ax, ay), (bx, by) = a_map[key], b_map[key]
                out.append((key[0], key[1], ax + (bx - ax) * t, ay + (by - ay) * t))
            elif key in a_map:
                out.append((key[0], key[1], *a_map[key]))
            else:
                out.append((key[0], key[1], *b_map[key]))
        return out

    if before is not None and after is not None:
        if after.time_sec - before.time_sec <= max_interp_gap:
            return _lerp(before, after)
        # Gap too large: use whichever side is closer, within carry limit.
        dist_b = time_sec - before.time_sec
        dist_a = after.time_sec - time_sec
        if dist_b <= dist_a and dist_b <= max_carry:
            return list(before.fingertip_positions())
        if dist_a < dist_b and dist_a <= max_carry:
            return list(after.fingertip_positions())
        return []

    if before is not None and time_sec - before.time_sec <= max_carry:
        return list(before.fingertip_positions())
    if after is not None and after.time_sec - time_sec <= max_carry:
        return list(after.fingertip_positions())
    return []
