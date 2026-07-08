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
WRIST_LANDMARK = 0
PALM_LANDMARKS = [5, 9, 13, 17]  # MCP knuckles — stable base row, one per finger


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
    wrist: object = None   # (x, y) of landmark 0 — most stable hand point
    palm: object = None    # (x, y) of average MCP knuckles (lm 5,9,13,17)


@dataclass
class FingertipFrame:
    """All hands observed in a single video frame."""

    time_sec: float
    hands: list = field(default_factory=list)  # list[HandObservation]

    def fingertip_positions(self):
        """Yields (hand, finger, x, y) for every tracked point in this frame.

        finger 1-5 = fingertips (thumb→pinky)
        finger 0   = wrist (most stable, used as fallback for hand assignment)
        finger 6   = palm center / MCP knuckles (best hand-region estimate)
        """
        for obs in self.hands:
            for tip in obs.fingertips:
                yield obs.hand, tip.finger, tip.x, tip.y
            if obs.wrist is not None:
                yield obs.hand, 0, obs.wrist[0], obs.wrist[1]
            if obs.palm is not None:
                yield obs.hand, 6, obs.palm[0], obs.palm[1]


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


def _keyboard_crop_box(calib, frame_h: int, frame_w: int):
    """Return (y1, y2, x1, x2) covering the keyboard band plus hand headroom.

    MediaPipe internally downscales its input to ~192 px; hands in a full
    piano-video frame are tiny and routinely missed. Cropping to just the
    keyboard region + margin makes the hands 3-4× larger to the detector,
    dramatically improving detection rate especially at the start of the
    video and during octave jumps where the hand briefly leaves the centre.
    Falls back to the full frame if no row-map calibration is available.
    """
    try:
        from keyboard import _row_project  # noqa: PLC0415
    except ImportError:
        return 0, frame_h, 0, frame_w

    if calib is None or calib.row is None:
        return 0, frame_h, 0, frame_w

    ys = [_row_project(calib.row, u)[1] for u in (0.0, 0.25, 0.5, 0.75, 1.0)]
    if calib.row_near is not None:
        ys += [_row_project(calib.row_near, u)[1] for u in (0.0, 0.25, 0.5, 0.75, 1.0)]

    key_y_min = min(ys)
    key_y_max = max(ys)

    # Depending on the camera setup the hand BODY (palm + wrist) can sit either
    # above the key line (camera behind the player, hands reaching in from the
    # far/music-stand side) OR below it (overhead camera in front, hands coming
    # from the near/player edge -- fingers reach up onto the keys while palms and
    # wrists fall below). MediaPipe's first stage is a PALM detector, so a crop
    # that clips the palm/wrist drops detection to zero even when the fingers are
    # perfectly visible. Empirically, cropping to just the key strip on an
    # overhead shot yields 0 hands; extending past the palms recovers both. So
    # give generous headroom on BOTH sides -- the modest loss of zoom costs far
    # less than losing a whole hand.
    margin_above = min(int(frame_h * 0.40), 400)
    margin_below = min(int(frame_h * 0.35), 350)

    y1 = max(0, int(key_y_min) - margin_above)
    y2 = min(frame_h, int(key_y_max) + margin_below)
    return y1, y2, 0, frame_w


def fix_handedness_continuity(frames):
    """Post-processing pass that fixes isolated L/R label swaps in two steps.

    1. Geometric swap: if both hands are detected and L wrist is to the right
       of R wrist, the labels are geometrically impossible — swap them.
    2. Temporal vote: a single-hand frame whose label disagrees with both of
       its neighbours (within a ±3-frame window) is likely a spurious flip;
       relabel it to match the majority.

    Piano hands essentially never cross, so these rules are very low-risk.
    """
    def _hand_x(obs):
        if obs.wrist is not None:
            return obs.wrist[0]
        return obs.fingertips[0].x if obs.fingertips else 0.0

    n = len(frames)

    # Rule 1 — geometric consistency in two-hand frames
    for frame in frames:
        if len(frame.hands) != 2:
            continue
        by_label = {obs.hand: obs for obs in frame.hands}
        obs_l = by_label.get('L')
        obs_r = by_label.get('R')
        if obs_l and obs_r and _hand_x(obs_l) > _hand_x(obs_r):
            obs_l.hand, obs_r.hand = 'R', 'L'

    # Rule 2 — temporal vote for isolated single-hand mislabels
    for i in range(n):
        frame = frames[i]
        if len(frame.hands) != 1:
            continue
        obs = frame.hands[0]
        ox = _hand_x(obs)
        same = flip = 0
        for j in range(max(0, i - 3), min(n, i + 4)):
            if j == i:
                continue
            for other in frames[j].hands:
                if abs(_hand_x(other) - ox) < 100:   # within 100 px → same hand
                    if other.hand == obs.hand:
                        same += 1
                    else:
                        flip += 1
        if flip > same * 2:                            # overwhelming counter-evidence
            obs.hand = 'R' if obs.hand == 'L' else 'L'

    return frames


def _make_clahe():
    """CLAHE operator for local-contrast enhancement of MediaPipe's input.
    Modest clip limit so it lifts detail in dark/low-contrast footage (a common
    cause of missed hands) without amplifying compression noise into false
    texture. Reused across frames (creating one per frame is wasteful)."""
    import cv2  # lazy
    return cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))


def _clahe_enhance(bgr, clahe):
    """Apply CLAHE to the L (lightness) channel of a BGR image and return BGR.
    Only lightness is touched, so hue/skin-tone cues MediaPipe relies on are
    preserved. Enhancing only affects detection quality -- landmark coordinates
    are normalised, so this never shifts the mapping back to pixels."""
    import cv2  # lazy
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _result_to_observations(result, cx: float, cy: float, cw: float, ch: float,
                            flip_handedness: bool) -> list:
    """Convert a MediaPipe HandLandmarker result into HandObservations, mapping
    each landmark's crop-normalised (0-1) coordinates back to full-frame pixels
    using the crop origin (cx, cy) and size (cw, ch). Shared by the main
    tracking loop and the targeted MIDI-anchored recovery pass so both build
    observations identically."""
    observations: list = []
    if not (result.hand_landmarks and result.handedness):
        return observations
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        # MediaPipe reports handedness from the camera's perspective assuming a
        # mirror-selfie view; for an overhead shot of the performer's own hands
        # (not mirrored) the raw label matches the physical hand.
        label = handedness[0].category_name  # 'Left' or 'Right'
        hand_code = "L" if label == "Left" else "R"
        if flip_handedness:
            hand_code = "R" if hand_code == "L" else "L"

        def _px(lm):
            return cx + lm.x * cw, cy + lm.y * ch

        tips = []
        for finger, lm_idx in FINGERTIP_LANDMARKS.items():
            ax, ay = _px(landmarks[lm_idx])
            tips.append(Fingertip(finger=finger, x=ax, y=ay))
        wrist = _px(landmarks[WRIST_LANDMARK])
        plms = [landmarks[i] for i in PALM_LANDMARKS]
        palm = (
            cx + sum(p.x * cw for p in plms) / len(plms),
            cy + sum(p.y * ch for p in plms) / len(plms),
        )
        observations.append(HandObservation(
            hand=hand_code, fingertips=tips, wrist=wrist, palm=palm,
        ))
    return observations


def extract_fingertip_frames(
    video_path: str,
    fps: float = 30.0,
    max_hands: int = 2,
    flip_handedness: bool = False,
    k1: float = 0.0,
    min_hand_confidence: float = 0.3,
    calib=None,
    live_frame_path: Optional[str] = None,
    enhance_contrast: bool = True,
):
    """Runs MediaPipe's HandLandmarker (Tasks API) over the video and
    returns a list of `FingertipFrame`, one per sampled frame at the
    requested fps.

    `min_hand_confidence` sets both `min_hand_detection_confidence` and
    `min_hand_presence_confidence`. The default here is 0.3, deliberately
    below the library's own 0.5: on the typical overhead piano shot (keys
    at the bottom of the frame, hands only partly visible) 0.5 silently
    drops the SECOND hand almost entirely -- measured on a real cover
    video: both hands seen in 2% of frames at 0.5 vs 90% at 0.3, with
    handedness labels still correct. Lower further (e.g. 0.15) if a
    clearly visible hand still goes missing; raise toward 0.5 if ghost
    hands appear.

    `calib` (optional Calibration): when provided, each frame is cropped to
    the keyboard band + hand headroom before MediaPipe sees it.  MediaPipe
    internally downscales input to ~192 px; in a full piano-video frame the
    hands are tiny.  The crop makes them 3-4× larger to the detector,
    fixing most "hands not recognised at the start / after octave jumps"
    failures.  Landmark coordinates are mapped back to full-frame pixels
    automatically so the rest of the pipeline is unaffected.

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

    # Compute keyboard crop box once from the video's declared dimensions.
    # undistort_frame preserves the frame size, so these stay valid throughout.
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    crop_y1, crop_y2, crop_x1, crop_x2 = _keyboard_crop_box(calib, orig_h, orig_w)
    crop_h = crop_y2 - crop_y1
    crop_w = crop_x2 - crop_x1
    using_crop = (crop_y1 > 0 or crop_y2 < orig_h or crop_x1 > 0 or crop_x2 < orig_w)

    clahe = _make_clahe() if enhance_contrast else None

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
                time_sec = idx / src_fps
                timestamp_ms = int(round(time_sec * 1000))
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                # Enhance the MediaPipe input only (not `frame`, which feeds the
                # live view and stays the true image). Crop first, enhance the
                # smaller region, then convert to RGB.
                mp_bgr = frame[crop_y1:crop_y2, crop_x1:crop_x2] if using_crop else frame
                if clahe is not None:
                    mp_bgr = _clahe_enhance(mp_bgr, clahe)
                mp_input = cv2.cvtColor(mp_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_input)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                observations = _result_to_observations(
                    result, crop_x1, crop_y1, crop_w, crop_h, flip_handedness
                )
                frames.append(FingertipFrame(time_sec=time_sec, hands=observations))
                if live_frame_path:
                    _disp = frame.copy()
                    # Yellow rectangle = the region actually fed to MediaPipe
                    # (the keyboard crop). Makes it obvious when the crop is
                    # clipping the hands -- the usual cause of missed detections.
                    if using_crop:
                        cv2.rectangle(_disp, (crop_x1, crop_y1), (crop_x2 - 1, crop_y2 - 1),
                                      (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(_disp, "analysis area", (crop_x1 + 6, max(18, crop_y1 + 18)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    for _obs in observations:
                        _color = (50, 100, 255) if _obs.hand == "R" else (50, 210, 100)
                        for _tip in _obs.fingertips:
                            cv2.circle(_disp, (int(_tip.x), int(_tip.y)), 9, _color, -1, cv2.LINE_AA)
                            cv2.circle(_disp, (int(_tip.x), int(_tip.y)), 9, (255, 255, 255), 1, cv2.LINE_AA)
                        if _obs.wrist:
                            _wx, _wy = int(_obs.wrist[0]), int(_obs.wrist[1])
                            cv2.putText(_disp, _obs.hand, (_wx - 12, _wy - 14),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
                            cv2.putText(_disp, _obs.hand, (_wx - 12, _wy - 14),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, _color, 1, cv2.LINE_AA)
                    cv2.putText(_disp, f"t={time_sec:.1f}s  {len(observations)} hand(s)",
                                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(_disp, f"t={time_sec:.1f}s  {len(observations)} hand(s)",
                                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
                    _lh, _lw = _disp.shape[:2]
                    if _lw > 640:
                        _ls = 640 / _lw
                        _disp = cv2.resize(_disp, (640, max(1, int(_lh * _ls))))
                    try:
                        cv2.imwrite(live_frame_path, _disp, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    except Exception:
                        pass
            idx += 1
    cap.release()
    return frames


def _obs_screen_xy(obs):
    """Representative screen point for a hand observation: palm (best hand-
    region estimate), else wrist, else the mean of its fingertips."""
    if obs.palm is not None:
        return float(obs.palm[0]), float(obs.palm[1])
    if obs.wrist is not None:
        return float(obs.wrist[0]), float(obs.wrist[1])
    if obs.fingertips:
        n = len(obs.fingertips)
        return (sum(t.x for t in obs.fingertips) / n,
                sum(t.y for t in obs.fingertips) / n)
    return None


@dataclass
class RecoverStats:
    problem_frames: int = 0      # frames with <2 hands AND an uncovered sounding note
    reran: int = 0               # frames actually re-run through MediaPipe
    hands_recovered: int = 0     # new hand observations merged in
    frames_now_dual: int = 0     # problem frames that reached 2 hands after recovery

    def summary(self) -> str:
        if self.problem_frames == 0:
            return "MIDI recovery: no frames needed a second look"
        return (
            f"MIDI recovery: recovered {self.hands_recovered} missing hand(s) "
            f"across {self.reran} re-run frame(s) "
            f"({self.frames_now_dual}/{self.problem_frames} now show both hands)"
        )


def _find_recovery_targets(frames, calib, notes_span, cover_keys: float):
    """Return [(frame, x_min, x_max, had_one_hand_label|None), ...] for frames
    with <2 tracked hands where a sounding MIDI note's key is uncovered by any
    tracked hand. ``notes_span`` is [(start_v, end_v, wi, screen_x), ...] sorted
    by start. Pure (no video/MediaPipe) so it is unit-testable."""
    targets = []
    for frame in frames:
        if len(frame.hands) >= 2:
            continue
        t = frame.time_sec
        tracked_wi = []
        for obs in frame.hands:
            xy = _obs_screen_xy(obs)
            if xy is not None:
                try:
                    tracked_wi.append(calib.screen_to_white_index(xy))
                except Exception:
                    pass
        unc_x = []
        for start_v, end_v, wi, sx in notes_span:
            if start_v - 0.03 <= t <= end_v + 0.03:
                if all(abs(wi - twi) > cover_keys for twi in tracked_wi):
                    unc_x.append(sx)
        if not unc_x:
            continue
        had_label = frame.hands[0].hand if len(frame.hands) == 1 else None
        targets.append((frame, min(unc_x), max(unc_x), had_label))
    return targets


def recover_missing_hands(
    video_path: str,
    frames,
    calib,
    midi_notes,
    offset_sec: float = 0.0,
    fps: float = 30.0,
    min_hand_confidence: float = 0.3,
    k1: float = 0.0,
    flip_handedness: bool = False,
    cover_keys: float = 5.0,
    margin_keys: float = 4.0,
    dedup_px: float = 60.0,
    max_recover_frames: "int | None" = None,
    enhance_contrast: bool = True,
) -> RecoverStats:
    """Targeted second MediaPipe pass, anchored on MIDI (pipeline stage: hand
    recovery). For each sampled frame where fewer than two hands were tracked
    but a MIDI note is sounding in a register with NO tracked hand nearby, crop
    tightly around that register and re-run MediaPipe (IMAGE mode) there -- the
    same crop trick the initial pass uses, but aimed where a hand is known to
    be. Recovered hands are merged into ``frames`` (mutated in place).

    Why MIDI as the anchor (not a colour/blob tracker): a sounding note is
    ground truth that a hand is on that key at that instant, and it is only at
    onsets that hand assignment needs to be right -- so MIDI is both a more
    reliable and a more relevant position prior than a same-colour skin blob
    that fails exactly on hand-crossing. See the project note on the hand-
    assignment goal.

    Requires calibration with a usable key<->screen projection and MIDI notes
    exposing ``.pitch``/``.start_sec``/``.duration_sec``. Returns RecoverStats.
    The caller should re-run ``fix_handedness_continuity`` afterwards to sort
    the (now more complete) L/R labels.
    """
    stats = RecoverStats()
    if not frames or calib is None or calib.row is None or not midi_notes:
        return stats

    import cv2  # lazy
    import mediapipe as mp  # lazy
    from keyboard import (  # lazy: no circular import at module load
        undistort_frame, _calib_u_to_screen, pitch_to_white_index,
    )

    # Sounding notes as (start_v, end_v, wi, screen_x). screen_x locates the
    # crop; wi tests whether a tracked hand already covers the note.
    span = pitch_to_white_index(calib.high_pitch) - pitch_to_white_index(calib.low_pitch)
    px_per_key = None
    notes_span = []
    for n in midi_notes:
        try:
            u = calib.pitch_to_norm_x(n.pitch)
            sx, _sy = _calib_u_to_screen(calib, u)
            start_v = float(n.start_sec) + offset_sec
            end_v = start_v + max(float(n.duration_sec), 0.0)
            notes_span.append((start_v, end_v, pitch_to_white_index(n.pitch), float(sx)))
        except Exception:
            continue
    if not notes_span:
        return stats
    notes_span.sort()

    # Pixel width of the crop margin, derived from the keyboard's own scale.
    try:
        u_lo = 0.0
        u_hi = min(1.0, 1.0 / span) if span else 0.05
        x_lo, _ = _calib_u_to_screen(calib, u_lo)
        x_hi, _ = _calib_u_to_screen(calib, u_hi)
        px_per_key = abs(float(x_hi) - float(x_lo)) or 40.0
    except Exception:
        px_per_key = 40.0
    margin_px = max(60.0, margin_keys * px_per_key)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return stats
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    band_y1, band_y2, _bx1, _bx2 = _keyboard_crop_box(calib, orig_h, orig_w)

    # Identify problem frames + the screen-x span of their uncovered notes.
    targets = _find_recovery_targets(frames, calib, notes_span, cover_keys)
    stats.problem_frames = len(targets)
    if not targets:
        cap.release()
        return stats
    if max_recover_frames is not None and len(targets) > max_recover_frames:
        targets = targets[:max_recover_frames]

    clahe = _make_clahe() if enhance_contrast else None
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(_model_path())),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,  # a wide crop over a 0-hand frame may contain both hands
        min_hand_detection_confidence=min_hand_confidence,
        min_hand_presence_confidence=min_hand_confidence,
    )
    with _suppress_native_stderr():
        landmarker_ctx = mp.tasks.vision.HandLandmarker.create_from_options(options)
    with landmarker_ctx as landmarker:
        for frame, x_min, x_max, had_label in targets:
            src_idx = int(round(frame.time_sec * src_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
            ok, raw = cap.read()
            if not ok:
                continue
            if k1 != 0.0:
                raw = undistort_frame(raw, k1)
            cx1 = max(0, int(x_min - margin_px))
            cx2 = min(orig_w, int(x_max + margin_px))
            cy1, cy2 = band_y1, band_y2
            if cx2 - cx1 < 20 or cy2 - cy1 < 20:
                continue
            crop = raw[cy1:cy2, cx1:cx2]
            if clahe is not None:
                crop = _clahe_enhance(crop, clahe)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            recovered = _result_to_observations(
                result, cx1, cy1, cx2 - cx1, cy2 - cy1, flip_handedness
            )
            if not recovered:
                continue
            stats.reran += 1

            # Existing hands' x positions, to skip re-detections of a hand we
            # already have (same physical hand, not a recovery).
            existing_x = []
            for obs in frame.hands:
                xy = _obs_screen_xy(obs)
                if xy is not None:
                    existing_x.append(xy[0])

            added = False
            for obs in recovered:
                oxy = _obs_screen_xy(obs)
                if oxy is None:
                    continue
                if any(abs(oxy[0] - ex) < dedup_px for ex in existing_x):
                    continue  # already tracked -- not a recovery
                # If the frame already had exactly one hand, the FIRST recovered
                # hand is the OTHER hand by construction (it fills an uncovered
                # region a same-labelled hand couldn't). Force the opposite
                # label so the later geometric pass can order L/R correctly; any
                # further recovered hand keeps MediaPipe's own label.
                if had_label is not None and not added:
                    obs.hand = "R" if had_label == "L" else "L"
                frame.hands.append(obs)
                existing_x.append(oxy[0])
                stats.hands_recovered += 1
                added = True
            if added and len(frame.hands) >= 2:
                stats.frames_now_dual += 1
    cap.release()
    return stats


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
