"""Blob-tracker: coarse two-hand positions from skin segmentation (layer 2).

Independent of MediaPipe. Where MediaPipe's palm detector merges or drops a hand
(the two-hands-close-together failure that defeats crop, register prior and
voice reasoning alike), a dumb-but-robust skin blob still shows *where the two
hands are*. That coarse position is exactly what the matcher needs to give the
missing hand's notes the right L/R -- and it produces the balanced hand labels
that the video-seeded voice separation needs to work.

Pipeline role:
  1. Segment skin (YCrCb -- more lighting-invariant than HSV, so less per-video
     tuning) inside the keyboard crop band; the keys are black/white, so skin
     blobs there are hands.
  2. Up to two largest blobs -> coarse hand centroids, assigned L/R by x with
     temporal continuity to the previous frame.
  3. When the two hands touch into ONE wide blob, split it at the vertical
     valley in the skin mask (falling back to the midpoint of the last known
     L/R centroids) -- the spatial split point.
  4. Fuse: for frames where MediaPipe tracked fewer than two hands but the blob
     tracker saw the missing side, synthesize a coarse observation there.

Opt-in (skin segmentation needs per-video tuning; a wrong range adds noise), so
default OFF -- enable with --blob-recover. The mask->blobs and split logic are
pure numpy and unit-tested; the video loop uses cv2.
"""

from __future__ import annotations

from dataclasses import dataclass

# YCrCb skin gate -- a widely-used, fairly lighting-invariant default. Y is left
# wide open; the chroma bounds do the work. Tune Cr/Cb per video if needed.
DEFAULT_SKIN_LO = (0, 133, 77)
DEFAULT_SKIN_HI = (255, 173, 127)

# A blob smaller than this fraction of the crop area is noise, not a hand.
MIN_BLOB_AREA_FRAC = 0.01
# A single blob wider than this fraction of the crop width is probably two
# merged hands and gets split.
MERGED_WIDTH_FRAC = 0.42


def _skin_mask(bgr, lo=DEFAULT_SKIN_LO, hi=DEFAULT_SKIN_HI):
    """Binary skin mask (uint8 0/255) for a BGR image, via YCrCb thresholding
    plus a small open/close to drop speckle and close key-shadow gaps."""
    import cv2
    import numpy as np
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, np.array(lo, np.uint8), np.array(hi, np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _blobs_from_mask(mask, min_area: int):
    """Connected components of a binary mask as (cx, cy, area, x, y, w, h),
    sorted largest-first, keeping only those >= min_area. Pure numpy/cv2."""
    import cv2
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):  # 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        blobs.append((float(cent[i][0]), float(cent[i][1]), area, x, y, w, h))
    blobs.sort(key=lambda b: b[2], reverse=True)
    return blobs


def _split_column(mask, x0: int, x1: int):
    """Column x (absolute) of the vertical valley in a merged blob's x-range:
    the column with the fewest skin pixels, which is the seam between the two
    touching hands. Returns None if the range is too narrow."""
    import numpy as np
    if x1 - x0 < 8:
        return None
    col = mask[:, x0:x1].sum(axis=0).astype(np.float64)
    # Ignore the outer 20% each side so the valley is interior, not an edge.
    m = len(col)
    lo = int(m * 0.2); hi = int(m * 0.8)
    if hi - lo < 3:
        return None
    j = int(np.argmin(col[lo:hi])) + lo
    return x0 + j


def assign_and_split(blobs, mask, last_l_x, last_r_x,
                     merged_width_frac=MERGED_WIDTH_FRAC):
    """Turn detected blobs into (L_xy, R_xy) coarse hand centroids (either may
    be None). Two blobs -> assign by x with continuity to the last positions.
    One wide blob -> split at the mask valley (or the last-positions midpoint).
    Pure logic (numpy mask + blob tuples) -- unit-testable without video.
    """
    mask_w = mask.shape[1]
    if not blobs:
        return None, None

    if len(blobs) >= 2:
        b0, b1 = blobs[0], blobs[1]
        a = (b0[0], b0[1]); b = (b1[0], b1[1])
        left, right = (a, b) if a[0] <= b[0] else (b, a)
        return left, right

    # Single blob: merged (wide) -> split; else assign to the nearer side.
    cx, cy, area, x, y, w, h = blobs[0]
    if w >= merged_width_frac * mask_w:
        split_x = _split_column(mask, x, x + w)
        if split_x is None and last_l_x is not None and last_r_x is not None:
            split_x = (last_l_x + last_r_x) / 2.0
        if split_x is None:
            split_x = x + w / 2.0
        left = ((x + split_x) / 2.0, cy)
        right = ((split_x + x + w) / 2.0, cy)
        return left, right

    # A lone narrow blob -> whichever side it is nearer to (by last positions).
    if last_l_x is not None and last_r_x is not None:
        if abs(cx - last_l_x) <= abs(cx - last_r_x):
            return (cx, cy), None
        return None, (cx, cy)
    # No history: call the left half L, right half R.
    return ((cx, cy), None) if cx < mask_w / 2 else (None, (cx, cy))


@dataclass
class BlobStats:
    frames_scanned: int = 0
    missing_side_frames: int = 0   # MediaPipe < 2 hands but blob saw the other
    hands_synthesized: int = 0

    def summary(self) -> str:
        if self.frames_scanned == 0:
            return "blob-recover: not run (no calibration crop)"
        if self.hands_synthesized == 0:
            return f"blob-recover: scanned {self.frames_scanned} frame(s), nothing to add"
        return (f"blob-recover: added {self.hands_synthesized} coarse hand(s) from skin blobs "
                f"across {self.missing_side_frames} frame(s)")


def blob_recover_hands(video_path, frames, calib, fps=30.0, k1=0.0,
                       skin_lo=DEFAULT_SKIN_LO, skin_hi=DEFAULT_SKIN_HI,
                       min_blob_area_frac=MIN_BLOB_AREA_FRAC) -> BlobStats:
    """For frames where MediaPipe tracked fewer than two hands, use the skin
    blob on the MISSING side (splitting a merged blob if needed) to synthesize a
    coarse HandObservation there, so those notes get the right L/R. Mutates
    `frames`; returns BlobStats. Caller should re-run fix_handedness_continuity.

    Needs a calibration crop (to know the keyboard band and the screen<->key
    projection). Skin range is tunable per video.
    """
    from hands import _keyboard_crop_box, _obs_screen_xy, HandObservation  # lazy
    from keyboard import undistort_frame  # lazy

    stats = BlobStats()
    if not frames or calib is None or calib.row is None:
        return stats

    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return stats
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    y1, y2, x1, x2 = _keyboard_crop_box(calib, orig_h, orig_w)
    min_area = int(max(1, (y2 - y1) * (x2 - x1) * min_blob_area_frac))

    last_l_x = last_r_x = None
    for frame in frames:
        if len(frame.hands) >= 2:
            # keep continuity from real detections
            xs = sorted(_obs_screen_xy(o)[0] for o in frame.hands if _obs_screen_xy(o))
            if len(xs) >= 2:
                last_l_x, last_r_x = xs[0], xs[-1]
            continue
        stats.frames_scanned += 1
        src_idx = int(round(frame.time_sec * src_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
        ok, raw = cap.read()
        if not ok:
            continue
        if k1 != 0.0:
            raw = undistort_frame(raw, k1)
        crop = raw[y1:y2, x1:x2]
        mask = _skin_mask(crop, skin_lo, skin_hi)
        blobs = _blobs_from_mask(mask, min_area)
        # convert last positions into crop-local x for the split logic
        ll = (last_l_x - x1) if last_l_x is not None else None
        lr = (last_r_x - x1) if last_r_x is not None else None
        left, right = assign_and_split(blobs, mask, ll, lr)
        if left is None and right is None:
            continue

        # Which side is MediaPipe missing?
        tracked_x = [_obs_screen_xy(o)[0] for o in frame.hands if _obs_screen_xy(o)]
        blob_pts = {"L": left, "R": right}
        for side, pt in blob_pts.items():
            if pt is None:
                continue
            full_xy = (x1 + pt[0], y1 + pt[1])
            # skip if a tracked hand already sits near this blob
            if any(abs(full_xy[0] - tx) < 80 for tx in tracked_x):
                continue
            synth = HandObservation(hand=side, fingertips=[], wrist=full_xy, palm=full_xy)
            frame.hands.append(synth)
            stats.hands_synthesized += 1
            stats.missing_side_frames += 1
            break  # add at most one missing hand per frame
    cap.release()
    return stats
