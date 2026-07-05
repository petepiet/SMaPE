"""Video<->MIDI sync: a single global time offset (same performance, so no
time-warping is needed -- just video_time = midi_time + offset).

Three ways to get the offset:
  1. Manual: the user passes ``--offset SECONDS`` directly.
  2. Audio (default, most reliable): extract the video's audio track and run
     onset detection (librosa) to find the first significant piano sound,
     then align it with the MIDI's first note. See
     ``estimate_offset_from_audio``.
  3. Press-moments (legacy fallback): detect "press moments" in the video
     (fingertip downward-velocity minima, i.e. moments a fingertip stops
     descending -- a reasonable proxy for a key press without needing to see
     the actual key depress), build a density signal over time, and
     cross-correlate it against the density of MIDI note onsets to find the
     lag that best aligns the two. This method is noisy (it depends on
     accurate hand tracking) and is kept only as a fallback.

The press-moments path (`onset_density`, `detect_press_moments`,
`estimate_offset`) is pure numpy -- no cv2/mediapipe -- so it's fully
covered by selftest.py. `estimate_offset_from_audio` lazily imports
librosa/subprocess-ffmpeg so the module still imports fine without them.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np


def onset_density(times: np.ndarray, duration: float, bin_sec: float = 0.05) -> np.ndarray:
    """Histogram of event `times` (seconds) into `bin_sec`-wide bins spanning
    [0, duration]. Returns a 1D float array (one bin count per bin)."""
    n_bins = max(1, int(np.ceil(duration / bin_sec)))
    hist = np.zeros(n_bins, dtype=np.float64)
    for t in times:
        if t < 0:
            continue
        b = int(t / bin_sec)
        if 0 <= b < n_bins:
            hist[b] += 1.0
    return hist


def detect_press_moments(fingertip_frames, bin_sec: float = 0.05):
    """Given a time-ordered list of FingertipFrame (from hands.py), detect
    "press moments": times at which some fingertip's downward (y-increasing,
    assuming a roughly overhead/oblique camera where the piano surface is
    "down" in image space) velocity crosses from positive (descending) to
    non-positive (stopped/rising) -- a local minimum of a descent, which is
    a reasonable proxy for the finger reaching the key.

    Returns a sorted list of times (seconds).
    """
    # Track each (hand, finger) trajectory across frames.
    tracks: dict = {}
    for frame in fingertip_frames:
        for hand, finger, x, y in frame.fingertip_positions():
            tracks.setdefault((hand, finger), []).append((frame.time_sec, y))

    press_times: list = []
    for key, samples in tracks.items():
        samples.sort(key=lambda s: s[0])
        if len(samples) < 3:
            continue
        times = np.array([s[0] for s in samples])
        ys = np.array([s[1] for s in samples])
        vy = np.gradient(ys, times)
        # A press moment: velocity was positive (moving down) just before,
        # and is <= 0 now (stopped or reversing) -- i.e. a local max in y
        # preceded by descent.
        for i in range(1, len(vy) - 1):
            if vy[i - 1] > 0 and vy[i] <= 0:
                press_times.append(float(times[i]))
    press_times.sort()
    return press_times


def estimate_offset(
    press_times,
    midi_onset_times,
    max_offset: float = 5.0,
    bin_sec: float = 0.05,
) -> float:
    """Cross-correlate the density of detected video press moments against
    the density of MIDI onsets to find the offset that best aligns them.

    Convention: video_time = midi_time + offset (offset may be negative if
    the video starts after the MIDI's t=0, i.e. video lags).

    We search offsets in [-max_offset, max_offset] at bin_sec resolution and
    pick the one maximizing the dot-product (cross-correlation) of the two
    density histograms after shifting the MIDI density by that offset.
    """
    press_times = np.asarray(sorted(press_times), dtype=np.float64)
    midi_times = np.asarray(sorted(midi_onset_times), dtype=np.float64)
    if len(press_times) == 0 or len(midi_times) == 0:
        return 0.0

    duration = max(press_times.max(), midi_times.max()) + max_offset + 1.0
    video_density = onset_density(press_times, duration, bin_sec)
    midi_density = onset_density(midi_times, duration, bin_sec)

    n_shift = int(round(max_offset / bin_sec))
    best_score = -np.inf
    best_offset = 0.0
    for shift_bins in range(-n_shift, n_shift + 1):
        shifted = np.roll(midi_density, shift_bins)
        if shift_bins > 0:
            shifted[:shift_bins] = 0.0
        elif shift_bins < 0:
            shifted[shift_bins:] = 0.0
        score = float(np.dot(video_density, shifted))
        if score > best_score:
            best_score = score
            best_offset = shift_bins * bin_sec
    return best_offset


# Alias: the legacy/fallback method, named to match the newer
# `estimate_offset_from_audio` for symmetry. `estimate_offset` itself is left
# untouched (selftest.py imports and calls it directly by that name).
def estimate_offset_from_press_moments(fingertip_frames, midi_data, max_offset: float = 5.0, bin_sec: float = 0.05) -> float:
    """Fallback offset estimator: cross-correlate detected fingertip "press
    moments" against MIDI onset density (see module docstring). Thin wrapper
    around `detect_press_moments` + `estimate_offset` that takes the same
    kind of arguments (`fingertip_frames`, `midi_data`) as
    `estimate_offset_from_audio`, so callers can switch between the two
    with a single `if`.
    """
    press_times = detect_press_moments(fingertip_frames)
    midi_onset_times = [n.start_sec for n in midi_data.notes]
    return estimate_offset(press_times, midi_onset_times, max_offset=max_offset, bin_sec=bin_sec)


def extract_audio_wav(video_path: str) -> str:
    """Extracts `video_path`'s audio track to a temp mono WAV via the system
    `ffmpeg` binary, returning the WAV's path. The caller owns cleanup (the
    file and its containing temp dir) -- see the `finally` blocks of callers
    for the pattern. Factored out of `estimate_offset_from_audio` so
    `transcribe.py`'s AMT step can reuse the exact same extraction instead of
    inventing a second ffmpeg-invocation convention.
    """
    tmp_dir = tempfile.mkdtemp(prefix="piano_fingering_audio_")
    audio_path = os.path.join(tmp_dir, "audio.wav")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "22050",
        "-q:a", "9",
        audio_path,
    ]
    try:
        try:
            proc = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Install with: sudo apt install ffmpeg (Linux) "
                "or brew install ffmpeg (Mac)"
            )

        if proc.returncode != 0 or not os.path.exists(audio_path):
            raise RuntimeError(
                "Failed to extract audio from video with ffmpeg.\n"
                f"Command: {' '.join(ffmpeg_cmd)}\n"
                f"stderr:\n{proc.stderr}\n"
                "If this video has no audio track, re-run with "
                "--sync-method press-moments instead."
            )
        return audio_path
    except Exception:
        # Never leave an empty temp dir behind on failure (nothing to remove
        # from it -- ffmpeg didn't produce audio_path in any failure path above).
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
        raise


def estimate_offset_from_audio(video_path: str, midi_data, onset_threshold: float = 0.3) -> float:
    """Extract audio from `video_path`, detect the first significant piano
    sound onset, and calculate the sync offset based on when the MIDI's
    first note starts.

    Returns: offset_sec (float), where video_time = midi_time + offset_sec.

    This is far more reliable than cross-correlating hand-tracking "press
    moments" (see `estimate_offset_from_press_moments`) because it listens
    to the actual performance audio instead of inferring key-presses from
    fingertip motion.

    Lazily imports `librosa` (audio onset detection) and shells out to the
    system `ffmpeg` binary (audio extraction), so this function -- and only
    this function -- requires those to be installed. The rest of `sync.py`
    stays pure-numpy/import-light for selftest.py.
    """
    if not midi_data.notes:
        raise ValueError(
            "MIDI has no notes -- cannot estimate audio offset (nothing to "
            "align the detected audio onset against)."
        )
    midi_first_note_time = midi_data.notes[0].start_sec

    import librosa  # lazy import

    audio_path = None
    try:
        audio_path = extract_audio_wav(video_path)
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        onset_times = librosa.onset.onset_detect(
            y=y, sr=sr, units="time", backtrack=False
        )
        onset_strengths = librosa.onset.onset_strength(y=y, sr=sr)

        if len(onset_times) == 0:
            raise RuntimeError(
                "No audio onsets detected in the extracted audio -- the video "
                "may have no audio track, or the audio may be silent/too quiet. "
                "Try --sync-method press-moments instead."
            )

        # Filter for the first *significant* onset: skip onsets whose local
        # onset-strength envelope value is below `onset_threshold` fraction
        # of the envelope's peak, to avoid triggering on faint noise/hiss
        # before the first real piano note.
        hop_length = 512  # librosa default for onset_strength
        strength_times = librosa.frames_to_time(
            np.arange(len(onset_strengths)), sr=sr, hop_length=hop_length
        )
        peak_strength = float(np.max(onset_strengths)) if len(onset_strengths) else 0.0
        min_strength = onset_threshold * peak_strength

        audio_first_onset_time = None
        for t in onset_times:
            idx = int(np.argmin(np.abs(strength_times - t)))
            if onset_strengths[idx] >= min_strength:
                audio_first_onset_time = float(t)
                break
        if audio_first_onset_time is None:
            # All onsets were below threshold (shouldn't normally happen);
            # fall back to the very first detected onset.
            audio_first_onset_time = float(onset_times[0])

        offset = audio_first_onset_time - midi_first_note_time
        print(
            f"Detected audio onset at {audio_first_onset_time:.3f}s, "
            f"MIDI starts at {midi_first_note_time:.3f}s, "
            f"offset = {offset:.3f}s"
        )
        return offset
    finally:
        if audio_path is not None:
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                os.rmdir(os.path.dirname(audio_path))
            except OSError:
                pass


def apply_offset(midi_time_sec: float, offset_sec: float) -> float:
    """video_time = midi_time + offset."""
    return midi_time_sec + offset_sec
