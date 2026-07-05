#!/usr/bin/env python3
"""CLI: analyze a near-overhead piano performance video, in sync with its
exact-same-performance MIDI, to determine which hand/finger played each MIDI
note. Writes a JSON file consumable by Symplethesia's dev-only importer.

Typical usage:

    python3 extract_fingering.py \\
        --video performance.mp4 --midi performance.mid \\
        --low-pitch 21 --high-pitch 108 \\
        --out fingering.json

Every run recalibrates from scratch: no calibration.json is ever loaded or
saved. You'll first scrub to a good (non-black/faded) reference frame, then
click white key centers across the playable keyboard area to fit the
calibration for that video.

Run `python3 extract_fingering.py --selftest` (or `python3 selftest.py`) to
run the pure-numpy self-tests, which require only `numpy` (no cv2 /
mediapipe / mido / yt-dlp).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Line-buffer stdout even when it's a pipe (e.g. run as the GUI's subprocess).
# Without this, prints from third-party libraries (like Kong's per-segment
# "Segment X / Y" transcription progress) sit invisible in Python's block
# buffer for the entire run, making long phases look hung.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Silence noisy-but-harmless native-library logging that looks alarming but
# reflects nothing actually wrong (the pipeline works fine underneath it).
# Harmless to set unconditionally even when --selftest never imports the
# libraries these apply to, since these are just env var writes.
#
# mediapipe/absl/glog's own burst of native log lines at HandLandmarker
# creation (EGL/GL init, "Feedback manager requires a model with a single
# signature", the absl::InitializeLog() bootstrap notice, etc.) do NOT
# respect these env vars on the mediapipe build this tool has been tested
# against (verified empirically -- GLOG_minloglevel/TF_CPP_MIN_LOG_LEVEL
# made no difference) -- see hands.py's `_suppress_native_stderr`, a raw
# fd-level redirect around that specific call, for the fix that actually
# works. These two are kept anyway as a harmless defensive no-op in case a
# different mediapipe/TensorFlow build DOES respect them.
os.environ.setdefault("GLOG_minloglevel", "2")  # mediapipe/absl (glog): errors only
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TensorFlow Lite: errors only
# cv2's bundled Qt backend separately warns about a missing font directory
# ("QFontDatabase: Cannot find font directory .../cv2/qt/fonts") when it
# spins up a Qt platform plugin -- unverified whether this env var actually
# suppresses it (not exercised by an interactive cv2 GUI window in testing),
# but it's the standard mechanism for silencing Qt logging categories and is
# harmless if it turns out to be a no-op here too.
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

DOWNLOADS_DIR = os.path.expanduser("~/Downloads")


def default_out_path(midi_path: str) -> str:
    """Default --out path when not specified: same directory and basename as
    the MIDI file, with a `.fingering.json` suffix instead of `.mid`/`.midi`.

    e.g. "/home/user/songs/piece.mid" -> "/home/user/songs/piece.fingering.json"
    """
    midi_p = Path(midi_path)
    return str(midi_p.with_suffix("")) + ".fingering.json"


def default_out_path_from_video(video_path: str) -> str:
    """Default --out path when transcribing (no --midi to derive a name
    from): the video's own basename (its downloaded title, for a URL input
    -- see `_download_if_url`'s title-based outtmpl), placed directly in
    ~/Downloads -- NOT the `downloads/` cache subfolder the raw downloaded
    video itself lives in, so the actual deliverables (fingering JSON,
    .symple bundle, transcribed MIDI) are easy to find rather than buried
    next to an intermediate cache file."""
    basename = Path(video_path).with_suffix("").name
    return os.path.join(DOWNLOADS_DIR, basename + ".fingering.json")


def _strip_fingering_json_suffix(path: str) -> str:
    """Strips a trailing `.fingering.json` (or bare `.json`) from `path`.

    `Path.with_suffix("")` only strips the LAST dot-segment, so on a path
    like ".../piece.fingering.json" it leaves ".../piece.fingering" behind
    (".json" removed, ".fingering" not) -- silently producing filenames like
    "piece.fingering.transcribed.mid" instead of "piece.transcribed.mid".
    Mirrors bundle.py's `default_bundle_path`, which already gets this right.
    """
    for suffix in (".fingering.json", ".json"):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return str(Path(path).with_suffix(""))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", help="Path to a local video file, or a YouTube (or yt-dlp-supported) URL")
    p.add_argument("--midi", help="Path to the exact-same-performance MIDI file (omit when using --transcribe)")
    p.add_argument(
        "--transcribe",
        action="store_true",
        default=False,
        help="Transcribe the MIDI from --video's own audio (Kong high-resolution piano transcription model) "
        "instead of requiring --midi. Needs the extra 'transcribe' dependencies -- see README.",
    )
    p.add_argument(
        "--midi-only",
        action="store_true",
        default=False,
        help="Transcribe the MIDI from --video's own audio and stop there -- no calibration, no hand tracking, "
        "no fingering/hand analysis. Implies --transcribe. Still writes the fingering JSON/.symple bundle "
        "(with an empty notes list and \"midiOnly\": true) so the transcribed MIDI stays one-step-importable.",
    )
    p.add_argument(
        "--onset-threshold",
        type=float,
        default=None,
        help="--transcribe only: raise the model's onset-confidence cutoff (library default 0.3) so it's "
        "pickier about declaring a note -- fewer ghost notes, at the risk of losing quiet/ambiguous real "
        "ones. Omit to leave the library default alone. See README 'Reducing ghost notes'.",
    )
    p.add_argument(
        "--min-velocity",
        type=int,
        default=0,
        help="--transcribe only: drop transcribed notes quieter than this (0-127 scale) -- ghost notes "
        "(pedal resonance/harmonics) tend to be quiet. Default 0 = no filtering.",
    )
    p.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="--transcribe only: drop transcribed notes shorter than this (seconds) -- tracking blips tend "
        "to be very short. Default 0.0 = no filtering.",
    )
    p.add_argument(
        "--no-reconcile",
        action="store_true",
        default=False,
        help="Skip video-based reconciliation (ghost-note flag/drop + sustain-pedal de-pedal trim). On by "
        "default: it's conservative (flags more than it drops, never judges occluded video) but costs "
        "nothing to disable if you want the raw matcher output.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: <midi-directory>/<midi-basename>.fingering.json)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for fingering.json and .symple bundle (default: same as --out or --midi directory)",
    )
    p.add_argument("--offset", type=float, default=None, help="Manual video-vs-MIDI offset in seconds (video_time = midi_time + offset). Overrides auto-estimate.")
    # Keyboard range is auto-derived from the MIDI by default. With C/G-key
    # calibration it no longer affects the mapping (only the overlay extent),
    # so these are optional overrides, not something you normally set.
    p.add_argument("--low-pitch", type=int, default=None, help="(optional) leftmost overlay key; default auto from MIDI")
    p.add_argument("--high-pitch", type=int, default=None, help="(optional) rightmost overlay key; default auto from MIDI")
    p.add_argument("--calibration", default="calibration.json", help="Unused placeholder, kept for CLI compatibility -- calibration is never loaded from or saved to disk; every run recalibrates interactively")
    p.add_argument(
        "--preview-seconds",
        type=float,
        default=20.0,
        help="For a --video URL: how many seconds of the start of the video to fetch up-front (a small, fast "
        "download) so the interactive calibration window can open almost immediately, instead of waiting "
        "minutes for the full video+audio download and ffmpeg mux to finish first. The full download still "
        "happens (in the background, overlapping with the time you spend calibrating) and the pipeline "
        "continues with it once both finish. Tradeoff: 'scrub to a good frame' can only scrub within this "
        "preview window, not the whole video -- 45s (default) is chosen to skip past channel "
        "intro while still being a fast download. Set to 0 to disable and always wait for the full download "
        "first (the old sequential behavior). Ignored for a local file --video (no download to hide behind). "
        "With --transcribe (no MIDI yet at calibration time), calibration uses the full 88-key range "
        "(or --low-pitch/--high-pitch if given) instead of the MIDI-derived range -- equivalent in practice, "
        "since CV detection tries the full keyboard first anyway and manual C/G clicks carry their own pitches.",
    )
    p.add_argument("--no-cv-calibration", action="store_true", default=False, help="Skip automatic CV keyboard detection; go straight to manual C/G-key clicking")
    p.add_argument("--preview", action="store_true", help="Render a preview video with detected fingertips + synced key highlight, for eyeballing alignment")
    p.add_argument("--no-align", action="store_true", default=False, help="Skip the interactive video/MIDI alignment step (before hand tracking)")
    p.add_argument("--fps", type=float, default=30.0, help="Frame sampling rate for hand tracking (default 30)")
    p.add_argument("--flip-handedness", action="store_true", default=False, help="Swap detected L/R hand labels (use if --preview shows hands swapped)")
    p.add_argument(
        "--min-hand-confidence",
        type=float,
        default=0.5,
        help="MediaPipe hand-detection confidence threshold (default 0.5, the library default). Lower this "
        "(e.g. 0.3) if --preview shows a hand frequently missing entirely while clearly visible and playing "
        "-- a real failure mode observed on monochrome/black-and-white source video.",
    )
    p.add_argument("--confidence-threshold", type=float, default=0.0, help="Drop matches below this confidence (default 0.0 = keep all)")
    p.add_argument(
        "--sync-method",
        choices=["audio", "press-moments"],
        default="audio",
        help="How to auto-estimate the video/MIDI offset when --offset is not given: "
        "'audio' (default) extracts the video's audio and detects the first piano "
        "onset; 'press-moments' cross-correlates hand-tracking press moments "
        "against MIDI onset density (legacy fallback).",
    )
    p.add_argument("--selftest", action="store_true", help="Run the pure-numpy self-tests and exit (no cv2/mediapipe needed)")
    p.add_argument("--no-bundle", action="store_true", default=False, help="Don't write the .symple bundle (MIDI + fingering JSON) alongside the output JSON")
    p.add_argument("--artist", type=str, default="", help="Song artist/performer (included in .symple bundle metadata)")
    p.add_argument("--title", type=str, default="", help="Song title (included in .symple bundle metadata)")
    p.add_argument("--genre", type=str, default="", help="Song genre (included in .symple bundle metadata)")
    p.add_argument("--difficulty", type=str, default="", choices=["", "easy", "intermediate", "advanced", "expert"], help="Song difficulty level (included in .symple bundle metadata)")
    p.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Synthesia-render mode: use lit-key color clustering for hand assignment instead of hand tracking. "
        "Requires --transcribe or --midi (no hand tracking needed). --midi must have the notes; --transcribe extracts them from audio.",
    )
    p.add_argument(
        "--flip-render-hands",
        action="store_true",
        default=False,
        help="--render only: flip left/right hand assignment from the color clustering (use if hands are swapped).",
    )
    p.add_argument(
        "--pick-hand-colors",
        action="store_true",
        default=False,
        help="Manually click-pick the LH/RH white/black key lit colors instead of relying on automatic hue "
        "clustering (implies --render). Use when clustering produces an implausible hand split (e.g. one "
        "hand gets nearly the whole keyboard) despite reporting high confidence.",
    )
    return p


def _strip_youtube_params(url: str) -> str:
    """Strip YouTube Radio/playlist parameters that cause yt-dlp to download
    the entire mix instead of just the single video. Handles both long-form
    (youtube.com/watch?v=ID) and short-form (youtu.be/ID) URLs.
    Returns the cleaned URL, or the original if not a YouTube URL."""
    if not ("youtube.com" in url or "youtu.be" in url):
        return url
    # For long-form: keep only ?v=ID (strip everything after first &)
    if "youtube.com" in url and "&" in url:
        url = url.split("&")[0]
    # For short-form: strip everything after ? or &
    if "youtu.be" in url:
        if "?" in url:
            url = url.split("?")[0]
        elif "&" in url:
            url = url.split("&")[0]
    return url


def _download_if_url(video: str, download_dir: str) -> tuple:
    """Returns (video_path, audio_path, channel_id). audio_path is a
    separately-downloaded audio-only file (see below) or None for a local
    file, in which case callers should fall back to treating video_path as
    also being the audio source (it presumably already has an embedded
    audio track). channel_id is yt-dlp's stable per-channel identifier when
    ``video`` is a URL (used to key Phase B's per-source calibration cache
    -- see calibration_store.py), or None for a local file (calibration_store
    falls back to hashing the path)."""
    if not (video.startswith("http://") or video.startswith("https://")):
        return video, None, None

    # Clean YouTube URLs: strip Radio/playlist parameters that cause
    # yt-dlp to download hours of content instead of the single video
    video = _strip_youtube_params(video)

    import yt_dlp  # lazy import

    os.makedirs(download_dir, exist_ok=True)
    # The video's own title (not its cryptic id) so the downloaded file --
    # and anything later derived from its name, like the transcribed MIDI
    # and fingering JSON's default --out path -- reads as a normal filename.
    # yt-dlp sanitizes illegal filesystem characters automatically; ".200B"
    # caps it at 200 bytes so an unusually long title plus our own
    # ".fingering.json"/".transcribed.mid" suffixes still fit comfortably
    # under typical filesystem filename limits (255 bytes on ext4).
    out_template = os.path.join(download_dir, "%(title).200B.%(ext)s")
    # YouTube often serves AV1-encoded mp4/webm streams, which many OpenCV/ffmpeg
    # builds cannot decode (fails silently: cv2.VideoCapture reads zero frames).
    # H.264 (avc1) is decodable everywhere, so prefer it explicitly and fall back
    # to whatever's available only if no avc1 stream exists.
    #
    # Resolution is capped at 1080p: the CV keyboard detector and hand-
    # tracking only need enough pixels to resolve key boundaries and
    # fingertips clearly (a keyboard band a few hundred pixels tall is
    # already precise enough), so a 4K/8K source buys nothing here but
    # multiplies download time.
    #
    # Video and audio are downloaded as TWO SEPARATE files (no ffmpeg mux)
    # rather than yt-dlp's usual "bestvideo+bestaudio" merge: nothing in this
    # tool ever plays the source video's own audio back to the user (hand
    # tracking/calibration/render-mode color sampling only read video
    # frames; audio-based sync and --transcribe only read the audio) --
    # video and audio streams share the same timeline from t=0 regardless
    # of whether they're physically combined into one container, so muxing
    # them bought nothing here but the mux step's own time and a redundant
    # temp copy of the video's bytes. (A single progressive video+audio
    # stream was tried too, as a different way to avoid a mux -- but
    # YouTube only ever serves ONE progressive tier, hard-capped at 360p,
    # a bad trade for a tool whose whole job is precise pixel positions;
    # this keeps 1080p by fetching the two adaptive streams separately.)
    base_opts = {
        "outtmpl": out_template,
        "quiet": False,
        # Keeps download progress bars (quiet=False, above) but drops
        # yt-dlp's own WARNING-level notices -- e.g. the "No supported
        # JavaScript runtime could be found" nag, which fires on every
        # extract_info call and is harmless here (basic format resolution
        # doesn't need it; only some signature-decryption edge cases would).
        # Real failures still raise as exceptions regardless of this flag.
        "no_warnings": True,
        # A pasted URL often carries playlist/radio context (e.g. "&list=...
        # &start_radio=1" from clicking a video inside a playlist or Up Next
        # autoplay). Without this, yt-dlp downloads the ENTIRE playlist/radio
        # instead of just the one video the URL points at. noplaylist forces
        # "just this one video" regardless of what else is in the URL --
        # more robust than stripping specific query params by hand, since
        # YouTube mixes several playlist-related params (list, start_radio,
        # index, ...) in varying combinations.
        "noplaylist": True,
    }

    video_opts = dict(base_opts, format=(
        "bestvideo[height<=1080][vcodec^=avc1][ext=mp4]/"
        "best[vcodec^=avc1][ext=mp4]/"  # progressive (already has audio) fallback
        "bestvideo[ext=mp4]/mp4/best"
    ))
    with yt_dlp.YoutubeDL(video_opts) as ydl:
        info = ydl.extract_info(video, download=True)
        video_path = ydl.prepare_filename(info)
    channel_id = info.get("channel_id") or info.get("uploader_id")

    # Prefer MP3; explicitly skip M4A to avoid FixupM4a hanging delays.
    # yt-dlp's M4A container fixing post-processor can hang indefinitely.
    # If MP3 unavailable, fall back to any audio format except M4A.
    audio_opts = dict(base_opts, format="best[ext=mp3]/bestaudio[ext=mp3]/bestaudio[ext!=m4a]/bestaudio")
    with yt_dlp.YoutubeDL(audio_opts) as ydl:
        audio_info = ydl.extract_info(video, download=True)
        audio_path = ydl.prepare_filename(audio_info)

    return video_path, audio_path, channel_id


def _download_preview(video_url: str, download_dir: str, seconds: float = 45) -> tuple:
    """Fetches only the first ``seconds`` of ``video_url`` -- a small, fast
    download meant purely to unblock the interactive calibration window
    (`_get_or_create_calibration`/`select_calibration_frame`) while the real,
    full-resolution `_download_if_url` download (minutes: two HTTP streams
    plus an ffmpeg mux) runs in a background thread. See `analyze()`.

    Uses the same video-only format selector as `_download_if_url` (H.264/
    avc1 preference, 1080p cap -- see that function's comment for why) so
    the preview frame is representative of what the full video will look
    like when scrubbing for a calibration frame. No audio is fetched here
    at all -- this clip is only ever used for a calibration video FRAME,
    never for audio (audio-based sync/--transcribe always read from the
    full download's separate audio file, never this throwaway clip).
    `force_keyframes_at_cuts` is deliberately NOT set: it forces yt-dlp to
    re-encode around the cut point for a frame-accurate trim, which is slow
    -- overkill for a throwaway preview that only needs to be "roughly the
    first N seconds", not exact.

    Returns (preview_path, channel_id). Returns (None, None) on ANY
    failure: `download_ranges` isn't supported by every yt-dlp extractor/
    site, and this is purely an optimization -- a failure here must never
    break a video that would otherwise download fine; callers fall back to
    today's plain sequential download.
    """
    try:
        import yt_dlp  # lazy import, matching _download_if_url

        os.makedirs(download_dir, exist_ok=True)
        # Fixed, throwaway name (not the title-based template used for the
        # real download) -- nothing downstream depends on this file's name,
        # and "overwrites" below means a stale preview.mp4 from a previous
        # run's URL never gets mistaken for this run's.
        out_template = os.path.join(download_dir, "preview.%(ext)s")
        ydl_opts = {
            "format": (
                "bestvideo[height<=1080][vcodec^=avc1][ext=mp4]/"
                "best[vcodec^=avc1][ext=mp4]/"
                "bestvideo[ext=mp4]/mp4/best"
            ),
            "outtmpl": out_template,
            "quiet": False,
            "no_warnings": True,  # see _download_if_url's base_opts comment
            # Same playlist-context guard as _download_if_url: a pasted URL
            # may carry "&list=...&start_radio=1" etc from a playlist/Up
            # Next click, and without this yt-dlp would fetch the whole
            # playlist/radio instead of just this one video.
            "noplaylist": True,
            # The whole point of this function: only fetch the first
            # `seconds` seconds instead of the entire video.
            "download_ranges": yt_dlp.utils.download_range_func(None, [(0, seconds)]),
            # A stale preview.mp4 from a *different* URL run earlier must
            # never be silently reused for this one.
            "overwrites": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            path = ydl.prepare_filename(info)
        if not os.path.exists(path):
            return None, None
        channel_id = info.get("channel_id") or info.get("uploader_id")
        return path, channel_id
    except Exception:
        # Ranged/preview download isn't supported everywhere -- degrade
        # gracefully to the normal full-sequential-download path.
        return None, None


def _first_frame(video_path: str):
    import cv2  # lazy import

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(
            f"Could not read first frame from {video_path}\n"
            "This usually means OpenCV/ffmpeg on this machine can't decode the "
            "video's codec (commonly AV1 -- look for 'AV1 decoding' / 'Failed to "
            "get pixel format' earlier in the output). Fixes:\n"
            "  - If this came from --video <url>, just re-run: the downloader now "
            "requests H.264 (avc1) instead of AV1, which decodes everywhere. "
            "Delete the old file in downloads/ first so it re-downloads.\n"
            "  - For a local file, re-encode it to H.264, e.g.:\n"
            f"      ffmpeg -i {video_path} -c:v libx264 -c:a aac fixed.mp4"
        )
    return frame


def _get_or_create_calibration(
    video_path: str, calib_path: str, low_pitch: int, high_pitch: int,
    use_cv_calibration: bool = True, calib_key: str = None,
):
    """Interactively (re-)calibrates this video, seeded from whichever of a
    per-source saved calibration (``calib_key``, see calibration_store.py --
    Phase B) or a fresh CV read is more trustworthy this run:
      - both agree (same camera setup, unchanged) -> reuse the saved one;
      - saved exists but disagrees, or fresh is low-confidence -> show the
        fresh read instead (a stale cache is never silently trusted);
      - no saved entry -> the fresh read (or manual, if that's also empty).
    Manual C/G clicks always override the seed either way. Confirming a
    calibration updates the store for ``calib_key`` (if given) so the next
    video from this source benefits.

    ``--calibration`` (``calib_path``) is legacy CLI-compatibility cruft --
    calibration is never loaded from/saved to that literal path.
    """
    from keyboard import select_calibration_frame, interactive_calibrate

    print("Scrub to a good frame, then confirm or click white key centers.")
    frame = select_calibration_frame(video_path)

    seed = None
    if use_cv_calibration:
        from keyboard_cv import detect_keyboard_calibration
        from calibration_store import load_saved_calibration, calibrations_agree

        saved = load_saved_calibration(calib_key) if calib_key else None
        try:
            fresh = detect_keyboard_calibration(frame, low_pitch, high_pitch)
        except Exception:
            fresh = None

        if saved is not None and fresh is not None and calibrations_agree(saved, fresh, low_pitch, high_pitch):
            print("Saved calibration for this source matches a fresh CV read -- reusing it.")
            seed = saved
        elif saved is not None:
            reason = "camera angle looks different from the saved calibration" if fresh is not None \
                else "CV detection is low-confidence this time"
            print(f"{reason} for this source -- showing a fresh detection instead of the saved one. "
                  "Confirm it, or click C/G keys to correct manually.")
            seed = fresh
        else:
            seed = fresh
            print("CV auto-calibration found a confident keyboard fit -- ESC to accept, or click C/G keys to override."
                  if seed is not None else
                  "CV auto-calibration didn't find a confident keyboard fit -- click white key centers manually.")

    calib = interactive_calibrate(frame, low_pitch, high_pitch, use_cv_calibration=False, seed_calibration=seed)

    if use_cv_calibration and calib_key:
        from calibration_store import save_calibration
        save_calibration(calib_key, calib)

    return calib


def _render_midi_wav(notes, offset_sec: float, duration_sec: float, path: str, sr: int = 44100) -> bool:
    """Synthesize the MIDI notes as simple piano-like tones at their onsets,
    shifted by ``offset_sec`` so they land on the video timeline. Written as a
    16-bit PCM WAV. Dependency-light (numpy + stdlib ``wave``): lets you HEAR
    the MIDI over the preview/aligner and audibly verify sync.

    "Piano-like" here = additive harmonic stack (fundamental + 4 overtones,
    higher harmonics decaying faster), a ~3 ms percussive attack, exponential
    decay that is faster for higher pitches (like real strings), note length
    from the MIDI duration (capped), and amplitude from MIDI velocity. Not a
    sampled piano, but far closer than a sine beep. Returns True on success."""
    import wave
    import numpy as np

    n = max(1, int(round(duration_sec * sr)) + sr // 2)
    buf = np.zeros(n, dtype=np.float64)
    HARMONICS = (1.0, 0.55, 0.32, 0.18, 0.09)
    for note in notes:
        t = note.start_sec + offset_sec
        i0 = int(round(t * sr))
        if i0 < 0 or i0 >= n:
            continue
        freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
        dur = float(min(max(note.duration_sec, 0.25), 2.5))
        blen = min(int(dur * sr), n - i0)
        if blen <= 0:
            continue
        ts = np.arange(blen) / sr
        # String-like decay: higher notes die faster; each harmonic k decays
        # a bit faster still (k * 0.7 extra rate).
        base_rate = 2.5 + freq / 400.0
        seg = np.zeros(blen, dtype=np.float64)
        for k, amp in enumerate(HARMONICS, start=1):
            fk = freq * k
            if fk >= sr / 2:
                break
            seg += amp * np.exp(-ts * (base_rate + k * 0.7)) * np.sin(2.0 * np.pi * fk * ts)
        # ~3 ms attack ramp to avoid a click, gentle velocity scaling.
        atk = max(1, int(0.003 * sr))
        seg[:atk] *= np.linspace(0.0, 1.0, atk)
        vel = getattr(note, "velocity", 90) or 90
        seg *= 0.25 + 0.75 * (vel / 127.0)
        buf[i0:i0 + blen] += seg

    peak = float(np.max(np.abs(buf)))
    if peak > 0:
        buf = buf / peak * 0.9
    pcm = (buf * 32767.0).astype(np.int16)
    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        return True
    except Exception as e:
        print(f"  (could not write preview audio: {e})")
        return False


def _mux_audio(video_path: str, audio_path: str, out_path: str) -> bool:
    """Mux ``audio_path`` into ``video_path`` -> ``out_path`` via ffmpeg
    (video copied, audio AAC). Returns True on success; False if ffmpeg is
    missing or fails (caller then keeps the silent video)."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-i", audio_path,
             "-c:v", "copy", "-c:a", "aac", "-shortest", out_path],
            check=True,
        )
        return True
    except Exception as e:
        print(f"  (ffmpeg mux failed, keeping silent preview: {e})")
        return False


def run_preview(video_path: str, calib, midi_data, offset_sec: float, fingertip_frames, fps: float) -> None:
    import cv2  # lazy import
    from hands import interpolate_fingertips
    from keyboard import undistort_frame, draw_keyboard_overlay

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = "preview.mp4"
    video_only_path = "preview_video.mp4"
    writer = cv2.VideoWriter(video_only_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))

    # Build a quick lookup of MIDI notes by nearest video time for overlay.
    note_video_times = [(n.start_sec + offset_sec, n.pitch) for n in midi_data.notes]
    note_video_times.sort()

    idx = 0
    ni = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = undistort_frame(frame, calib.k1)
        t = idx / src_fps
        # Advance ni to the note whose video time is nearest t (within 0.15s).
        while ni + 1 < len(note_video_times) and note_video_times[ni + 1][0] <= t:
            ni += 1
        current_pitch = None
        if note_video_times and abs(note_video_times[ni][0] - t) < 0.15:
            current_pitch = note_video_times[ni][1]

        # Inferred keyboard overlay (verifies calibration against the real
        # keys); the currently-sounding note is ringed in red.
        draw_keyboard_overlay(frame, calib, highlight_pitch=current_pitch)

        for hand, finger, x, y in interpolate_fingertips(fingertip_frames, t):
            color = (255, 0, 0) if hand == "L" else (0, 255, 0)
            cv2.circle(frame, (int(x), int(y)), 6, color, -1)
            cv2.putText(frame, f"{hand}{finger}", (int(x) + 6, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

    # Render the MIDI as beeps aligned to the video timeline and mux it in, so
    # the preview has sound (and sync can be verified by ear).
    import os as _os

    duration_sec = idx / src_fps if src_fps else 0.0
    audio_path = "preview_audio.wav"
    muxed = False
    if _render_midi_wav(midi_data.notes, offset_sec, duration_sec, audio_path):
        muxed = _mux_audio(video_only_path, audio_path, out_path)
        try:
            _os.remove(audio_path)
        except OSError:
            pass
    if muxed:
        try:
            _os.remove(video_only_path)
        except OSError:
            pass
        print(f"Wrote preview video (with MIDI audio) to {out_path}")
    else:
        # No ffmpeg / audio: fall back to the silent video as preview.mp4.
        try:
            _os.replace(video_only_path, out_path)
        except OSError:
            out_path = video_only_path
        print(f"Wrote preview video (silent) to {out_path}")


def _pick_file_dialog(title: str, initialdir: str, patterns):
    """Open a native file-open dialog (Tkinter, already used for the key
    picker) rooted at ``initialdir``. Returns the chosen path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(title=title, initialdir=initialdir, filetypes=patterns)
        root.destroy()
        return path or None
    except Exception:
        return None


def _resolve_media_path(arg, kind: str, patterns):
    """Resolve a MIDI/video path, defaulting to ~/Downloads. If not given,
    open a file dialog there; if given as a bare name that isn't found, try
    ~/Downloads/<name>."""
    if arg:
        if os.path.exists(arg):
            return arg
        cand = os.path.join(DOWNLOADS_DIR, arg)
        if os.path.exists(cand):
            return cand
        return arg  # let the later open fail with a clear error
    initial = DOWNLOADS_DIR if os.path.isdir(DOWNLOADS_DIR) else os.getcwd()
    picked = _pick_file_dialog(f"Select {kind} file", initial, patterns)
    if not picked:
        raise SystemExit(f"No {kind} file selected.")
    return picked


def interactive_align(video_path: str, calib, midi_data, offset_sec: float) -> float:
    """Play the video with the keyboard overlay + MIDI beeps so the user can
    fine-tune the video<->MIDI offset BEFORE the slow per-frame hand tracking.
    Notes flash their key (red ring) at ``note.start_sec + offset``; nudge the
    offset until the flashes/beeps land on the visible key presses.

    Controls: SPACE play/pause, <-/-> nudge 10ms, Up/Down 50ms, a/d seek 2s,
    R restart, ENTER accept, ESC/q keep current. Returns the chosen offset."""
    import cv2  # lazy import
    import time
    import shutil
    import subprocess
    import tempfile
    from keyboard import draw_keyboard_overlay

    onsets = sorted((n.start_sec, n.duration_sec, n.pitch) for n in midi_data.notes)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  (could not open video for alignment; skipping)")
        return offset_sec
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    disp_scale = min(1.0, 1280.0 / max(1, max(w, h)))

    # Pre-render the beep track in MIDI time (offset 0) so ffplay can be seeked
    # to (video_time - offset) regardless of the current offset.
    beep_path = None
    if shutil.which("ffplay"):
        dur = (total / fps) if (total and fps) else ((onsets[-1][0] + 2.0) if onsets else 5.0)
        tmp = os.path.join(tempfile.gettempdir(), "align_beep.wav")
        if _render_midi_wav(midi_data.notes, 0.0, dur, tmp):
            beep_path = tmp

    audio = {"proc": None}

    def stop_audio():
        if audio["proc"] is not None:
            try:
                audio["proc"].kill()
            except Exception:
                pass
            audio["proc"] = None

    def start_audio(t):
        stop_audio()
        if not beep_path:
            return
        # The beep wav is rendered at raw MIDI time (offset 0), so the point
        # in it that should sound "now" is midi_time = t - offset_sec. When
        # that's negative (video is still in the lead-in silence before the
        # offset), seeking to 0 would play the first beep immediately --
        # silently ignoring a positive offset on every restart. Instead delay
        # ffplay's audible output by the missing amount (ffmpeg `adelay`
        # filter) so it starts exactly when the video reaches midi_time 0.
        midi_time = t - offset_sec
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
        if midi_time >= 0:
            cmd += ["-ss", f"{midi_time:.3f}"]
        else:
            delay_ms = int(round(-midi_time * 1000))
            cmd += ["-af", f"adelay={delay_ms}:all=1"]
        cmd.append(beep_path)
        try:
            audio["proc"] = subprocess.Popen(cmd)
        except Exception:
            audio["proc"] = None

    win = "Align  |  SPACE play/pause   <-/-> 10ms   Up/Dn 50ms   a/d seek   R restart   ENTER ok   ESC keep"
    cv2.namedWindow(win)

    ARROW_L = {65361, 81, 2424832}
    ARROW_U = {65362, 82, 2490368}
    ARROW_R = {65363, 83, 2555904}
    ARROW_D = {65364, 84, 2621440}

    def seek(newt):
        newt = max(0.0, newt)
        cap.set(cv2.CAP_PROP_POS_MSEC, newt * 1000.0)
        ok, fr = cap.read()
        return (newt, fr if ok else None)

    playing = False
    t, last_frame = seek(0.0)
    wall0 = 0.0
    t0 = 0.0

    while True:
        if playing:
            target = t0 + (time.perf_counter() - wall0)
            ok = True
            while cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 < target:
                ok, fr = cap.read()
                if not ok:
                    break
                last_frame = fr
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not ok:
                playing = False
                stop_audio()

        if last_frame is None:
            ok, last_frame = cap.read()
            if not ok:
                break
        frame = last_frame.copy()

        # Static keyboard overlay + a ring on every currently-sounding note:
        # thick red right at the onset (the moment that must line up with the
        # visible key press), thinner orange while the note is held -- so the
        # marker stays visible long enough to actually judge the alignment.
        draw_keyboard_overlay(frame, calib)
        for st, dur, pitch in onsets:
            vt = st + offset_sec
            if not (vt - 0.03 <= t <= vt + max(dur, 0.35)):
                continue
            try:
                kx, ky = calib.pitch_to_screen(pitch)
                if t <= vt + 0.12:
                    cv2.circle(frame, (int(kx), int(ky)), 16, (0, 0, 255), 4, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (int(kx), int(ky)), 12, (0, 140, 255), 2, cv2.LINE_AA)
            except Exception:
                pass

        disp = cv2.resize(frame, (int(w * disp_scale), int(h * disp_scale))) if disp_scale < 1.0 else frame
        hud = f"offset {offset_sec:+.3f}s   t {t:6.2f}s   {'PLAY' if playing else 'PAUSE'}"
        hint = "R restart+play | SPACE play/pause | </> 10ms  up/dn 50ms | a/d seek | ENTER accept"
        cv2.putText(disp, hud, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(disp, hud, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp, hint, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, hint, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.imshow(win, disp)

        key = cv2.waitKeyEx(1 if playing else 20)
        if key == -1:
            continue
        ka = key & 0xFF

        def resync():
            nonlocal wall0, t0
            if playing:
                wall0 = time.perf_counter()
                t0 = t
                start_audio(t)

        if ka in (13, 10):  # ENTER accept
            break
        if ka in (27, ord("q")):  # ESC / q keep current
            break
        if ka == ord(" "):
            playing = not playing
            if playing:
                wall0 = time.perf_counter()
                t0 = t
                start_audio(t)
            else:
                stop_audio()
        elif key in ARROW_L:
            offset_sec -= 0.01
            resync()
        elif key in ARROW_R:
            offset_sec += 0.01
            resync()
        elif key in ARROW_U:
            offset_sec += 0.05
            resync()
        elif key in ARROW_D:
            offset_sec -= 0.05
            resync()
        elif ka == ord("a"):
            t, last_frame = seek(t - 2.0)
            resync()
        elif ka == ord("d"):
            t, last_frame = seek(t + 2.0)
            resync()
        elif ka == ord("r"):
            # Restart video + MIDI together from the beginning and PLAY, so a
            # freshly nudged offset can be re-checked from the top (works from
            # pause too -- restart always starts playback).
            t, last_frame = seek(0.0)
            playing = True
            wall0 = time.perf_counter()
            t0 = t
            start_audio(t)

    stop_audio()
    cap.release()
    cv2.destroyWindow(win)
    return offset_sec


def _derive_pitch_range(args: argparse.Namespace, midi_data, quiet: bool = False) -> tuple:
    """Keyboard overlay range: auto-derived from the MIDI (floor/ceil to a C)
    unless the user explicitly overrode it via --low-pitch/--high-pitch.
    With C/G-key calibration the range no longer affects the mapping -- it
    only sets how far the on-screen overlay is drawn.

    Factored out of `analyze()` because it's needed twice: once early (from
    the preview clip's MIDI-derived range, before the full video download
    finishes) when the early-calibration path runs, and once at its
    original spot otherwise. ``quiet`` suppresses the auto-set print (used
    when re-deriving the already-announced range after early calibration,
    so it isn't logged twice).
    """
    low_pitch, high_pitch = args.low_pitch, args.high_pitch
    if low_pitch is None or high_pitch is None:
        pitches = [n.pitch for n in midi_data.notes] or [21, 108]
        if low_pitch is None:
            low_pitch = max(0, (min(pitches) // 12) * 12)
        if high_pitch is None:
            high_pitch = min(127, ((max(pitches) // 12) + 1) * 12)
        if not quiet:
            print(f"  keyboard overlay range auto-set from MIDI: {low_pitch}-{high_pitch}")
    return low_pitch, high_pitch


def analyze(args: argparse.Namespace) -> dict:
    from keyboard import Calibration, pitch_to_white_index
    from midi_io import read_midi_notes, read_pedal_segments
    from hands import extract_fingertip_frames, interpolate_fingertips, Fingertip
    from sync import estimate_offset_from_audio, estimate_offset_from_press_moments, extract_audio_wav
    from match import Candidate, NoteToMatch, resolve_chord_conflicts, group_simultaneous
    from reconcile import (
        reconcile_note, confidence_by_window, _frame_times,
        analyze_hand_spread, HAND_SPREAD_WARN_THRESHOLD_KEYS, CLOSE_HANDS_THRESHOLD_KEYS,
    )
    if args.render:
        from render_hands import (
            assign_hands_for_notes, assign_hands_from_reference_colors,
            interactive_pick_hand_colors, sample_key_baselines, sample_notes,
            lit_end_time, trim_note_durations,
        )

    # Resolve media paths (default to ~/Downloads; file dialog if omitted).
    # --transcribe supplies its own MIDI (from the video's audio), so --midi
    # is neither required nor prompted for in that mode.
    if not args.transcribe:
        args.midi = _resolve_media_path(args.midi, "MIDI", [("MIDI files", "*.mid *.midi"), ("All files", "*.*")])
    if not args.video:
        args.video = _resolve_media_path(
            args.video, "video", [("Video files", "*.mp4 *.mkv *.mov *.webm *.avi"), ("All files", "*.*")]
        )

    # download_dir needs a directory before --out is known when --out is
    # omitted (there may be no --midi path to derive one from yet, e.g. when
    # transcribing). Default to ~/Downloads -- matching every other default
    # in this tool (MIDI/video file dialogs, GUI browse buttons) -- rather
    # than the directory the script happened to be launched from.
    provisional_out_dir = os.path.dirname(os.path.abspath(args.out)) if args.out else DOWNLOADS_DIR
    download_dir = os.path.join(provisional_out_dir, "downloads")

    is_url = args.video.startswith("http://") or args.video.startswith("https://")

    # `calib`/`midi_data` are set here (instead of at their "natural" spot
    # further down) only when the early-preview path below runs and
    # succeeds; the normal path further down fills them in otherwise. This
    # lets `_get_or_create_calibration` stay a single call site either way.
    video_path = None
    source_audio_path = None
    channel_id = None
    calib = None
    midi_data = None
    picked_hand_colors = None

    # --- Early (pre-download) calibration for URL video sources ----------
    # The slow part of a URL --video is `_download_if_url`: two full-
    # resolution HTTP streams plus a local ffmpeg mux, which can take
    # minutes. Historically nothing interactive happened until that
    # finished, so pasting a URL meant staring at a terminal before the
    # calibration window (`_get_or_create_calibration`) ever appeared.
    #
    # Instead: fetch a small ~`--preview-seconds` clip (fast -- seconds, not
    # minutes), open the calibration window on THAT clip immediately, and
    # run the real, full `_download_if_url` in a background thread the
    # whole time the user is scrubbing/clicking in the calibration UI. By
    # the time they confirm calibration, the full download has often
    # already finished (or finishes shortly after `join()` below) -- the
    # download's dead time is hidden behind calibration instead of coming
    # before it.
    #
    # This only applies when:
    #   - the input is a URL (a local file has no slow download to hide
    #     calibration behind -- see the `else` branch: completely
    #     unchanged, no preview, no thread);
    #   - `--preview-seconds` > 0 (0 explicitly disables this).
    # `_get_or_create_calibration` is otherwise invoked unconditionally
    # further down regardless of --render/--no-cv-calibration/etc, so no
    # extra "does this run even need calibration" check is needed here.
    if is_url and args.preview_seconds > 0 and not args.midi_only:
        print(f"Fetching a {args.preview_seconds:.0f}s preview clip for early calibration...")
        preview_path, preview_channel_id = _download_preview(args.video, download_dir, seconds=args.preview_seconds)
        if preview_path is not None:
            # Kick off the real download now, in the background, so its
            # minutes of dead time overlap with the calibration step below
            # instead of preceding it. Exceptions are captured here and
            # re-raised on the main thread after join() -- a real download
            # failure must never be silently swallowed.
            download_result = {}

            def _bg_download():
                try:
                    download_result["value"] = _download_if_url(args.video, download_dir)
                except Exception as exc:
                    download_result["error"] = exc

            download_thread = threading.Thread(target=_bg_download, daemon=True)
            download_thread.start()

            if args.transcribe:
                # No MIDI exists yet in --transcribe mode (it comes from the
                # FULL video's audio, downloading right now in the background),
                # so the pitch range can't be derived from notes here.
                # Calibrate against the full 88-key range instead (or the
                # explicit --low-pitch/--high-pitch overrides if given).
                # That's safe and consistent:
                #   - the CV auto-detector already tries the full 88-key
                #     range FIRST regardless of the requested range (see
                #     detect_keyboard_calibration's Phase A notes in
                #     keyboard_cv.py), so it behaves identically;
                #   - for manual clicks, Calibration's low/high_pitch only
                #     define the normalized-u span and the drawn overlay
                #     extent -- clicked C/G points carry their own pitches,
                #     so a full-range calibration maps any subset keyboard
                #     self-consistently.
                # midi_data intentionally stays None: the transcription and
                # MIDI read below run exactly as before, and the narrower
                # note-derived low/high_pitch is still recomputed afterwards
                # for everything downstream -- only the CALIBRATION uses
                # this wide range.
                low_pitch = args.low_pitch if args.low_pitch is not None else 21
                high_pitch = args.high_pitch if args.high_pitch is not None else 108
            else:
                print(f"Reading MIDI: {args.midi}")
                midi_data = read_midi_notes(args.midi)
                print(f"  {len(midi_data.notes)} notes, ppq={midi_data.ppq}")
                low_pitch, high_pitch = _derive_pitch_range(args, midi_data)

            from calibration_store import calibration_key

            print("\n" + "="*70)
            print(">>> WAITING FOR CALIBRATION <<<")
            print("="*70)
            print("An OpenCV window should appear with the video preview.")
            print("Click on white key centers to calibrate the keyboard.")
            print("C and G keys are easiest to identify.")
            print("Need at least 4 points. Press ESC when done.")
            print("(Full video downloads in the background...)")
            print("="*70 + "\n")
            calib = _get_or_create_calibration(
                preview_path, args.calibration, low_pitch, high_pitch,
                use_cv_calibration=not args.no_cv_calibration,
                calib_key=calibration_key(args.video, preview_channel_id),
            )

            # Also do hand-color picking from the (already fully downloaded)
            # preview clip here, rather than waiting for the full video --
            # the picker just needs *some* representative footage to scrub/
            # click, not the complete video, so there's no reason to block
            # on the (often much larger, slower) full download for this.
            # Falls through to picking from the full video further below
            # only if this never ran (preview download failed/unsupported,
            # or --preview-seconds 0). If the default --preview-seconds
            # (20s) doesn't show both hands' lit keys, pass a larger value.
            if args.render and args.pick_hand_colors:
                from render_hands import interactive_pick_hand_colors

                print("Opening hand-color picker on the preview clip "
                      "(scrub with arrow keys, click a lit key, ESC when done)...")
                picked_hand_colors = interactive_pick_hand_colors(preview_path)
                picked_hand_colors = _apply_hand_color_fallback_and_log(args.video, picked_hand_colors)
                print(f"  picked colors: {picked_hand_colors}")

            print("Waiting for the full video download to finish...")
            download_thread.join()
            if "error" in download_result:
                raise download_result["error"]
            video_path, source_audio_path, channel_id = download_result["value"]
            print("✓ Download complete. Starting analysis...")
        else:
            print("Preview download isn't available for this source (unsupported by this extractor) -- "
                  "falling back to the normal sequential download.")

    if video_path is None:
        # Either a local file, --preview-seconds 0, or the preview download
        # above failed -- exactly today's behavior.
        video_path, source_audio_path, channel_id = _download_if_url(args.video, download_dir)

    if source_audio_path is None:
        # Local file (or a rare progressive-stream fallback in
        # _download_if_url that already has audio embedded): fall back to
        # treating video_path as also being the audio source, exactly as
        # every ffmpeg call here already handles a file that mixes both.
        source_audio_path = video_path

    if args.out is None:
        args.out = default_out_path(args.midi) if args.midi else default_out_path_from_video(video_path)

    # Apply output directory override if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        basename = os.path.basename(args.out)
        args.out = os.path.join(args.output_dir, basename)
        print(f"Output directory set to: {args.output_dir}")

    # Manual hand-color picking only needs SOME representative video to
    # scrub/click, not any MIDI/transcription/calibration output -- normally
    # already done above from the fast preview clip (see the early-preview
    # block). This is the fallback for when that never ran (local file,
    # preview download unsupported/failed, or --preview-seconds 0): only
    # now, using the full video, do we have anything to pick from.
    if args.render and args.pick_hand_colors and picked_hand_colors is None:
        # interactive_pick_hand_colors already imported above (~line 896-899,
        # gated on the same args.render).
        print("Opening hand-color picker (scrub with arrow keys, click a lit key, ESC when done)...")
        picked_hand_colors = interactive_pick_hand_colors(video_path)
        picked_hand_colors = _apply_hand_color_fallback_and_log(args.video, picked_hand_colors)
        print(f"  picked colors: {picked_hand_colors}")

    if args.transcribe:
        transcribed_midi_path = _strip_fingering_json_suffix(args.out) + ".transcribed.mid"
        print("Transcribing MIDI from the video's audio (Kong high-resolution model)...")
        from transcribe import transcribe_audio

        wav_path = extract_audio_wav(source_audio_path)
        try:
            transcribe_audio(
                wav_path,
                transcribed_midi_path,
                onset_threshold=args.onset_threshold,
                min_velocity=args.min_velocity,
                min_duration_sec=args.min_duration,
            )
        finally:
            try:
                os.remove(wav_path)
                os.rmdir(os.path.dirname(wav_path))
            except OSError:
                pass
        print(f"  Wrote transcribed MIDI to {transcribed_midi_path}")
        args.midi = transcribed_midi_path

    if args.midi_only:
        print("MIDI-only mode: skipping calibration and hand/finger analysis.")
        return {
            "version": 2,
            "source": args.video,
            "midi": args.midi,
            "midiOnly": True,
            "notes": [],
        }

    if midi_data is None:
        # Not already read by the early-preview-calibration path above.
        print(f"Reading MIDI: {args.midi}")
        midi_data = read_midi_notes(args.midi)
        print(f"  {len(midi_data.notes)} notes, ppq={midi_data.ppq}")

    # Sustain-pedal segments (seconds), read generically from whatever MIDI
    # ends up in args.midi -- transcribed (the Kong model captures CC64) or
    # user-supplied (harmless no-op if it has none). Feeds the de-pedal
    # reconciliation step below.
    pedal_segments = read_pedal_segments(args.midi) if not args.no_reconcile else []
    if pedal_segments:
        print(f"  {len(pedal_segments)} sustain-pedal segment(s) found")

    if calib is None:
        # Not already calibrated (from the preview) above.
        low_pitch, high_pitch = _derive_pitch_range(args, midi_data)

        from calibration_store import calibration_key
        calib = _get_or_create_calibration(
            video_path, args.calibration, low_pitch, high_pitch,
            use_cv_calibration=not args.no_cv_calibration,
            calib_key=calibration_key(args.video, channel_id),
        )
    else:
        # Calibrated early from the preview clip -- low_pitch/high_pitch
        # still need to be (re)derived from the actual notes for the rest
        # of analyze(): in --transcribe mode calibration deliberately used
        # the full 88-key range (no MIDI existed yet), and everything
        # downstream should use the note-derived range as before. Cheap:
        # no I/O, just re-deriving from midi_data/args already in hand.
        # quiet unless --transcribe: the non-transcribe early path already
        # printed the auto-set line when it read the MIDI; the transcribe
        # path is announcing this range for the first time here.
        low_pitch, high_pitch = _derive_pitch_range(args, midi_data, quiet=not args.transcribe)

    # Seed the offset BEFORE the slow hand tracking so it can be tuned first.
    # Transcribed MIDI is derived FROM this video's own audio, so its timeline
    # already IS the video's timeline -- offset is 0 by construction, and
    # running audio-onset (or press-moments) sync on top would be redundant.
    # --offset still overrides, and the interactive aligner (below) still runs
    # so a residual model-latency nudge can be applied if ever needed.
    use_press_moments = args.offset is None and args.sync_method == "press-moments" and not args.transcribe
    if args.offset is not None:
        offset_sec = args.offset
        print(f"Using manual offset: {offset_sec:.3f}s")
    elif args.transcribe:
        offset_sec = 0.0
        print("Offset: 0.000s (transcribed MIDI shares the video's own timeline)")
    elif not use_press_moments:
        print("Estimating offset (audio onset detection)...")
        offset_sec = estimate_offset_from_audio(source_audio_path, midi_data)
        print(f"Auto-estimated offset: {offset_sec:.3f}s")
    else:
        offset_sec = 0.0

    # --render mode: use lit-key color clustering for hand assignment instead of hand tracking
    if args.render:
        print("Render mode: sampling lit-key colors for hand assignment...")
        pitches = [n.pitch for n in midi_data.notes]

        print("  sampling key baselines...")
        baselines = sample_key_baselines(video_path, calib, pitches, n_samples=8)

        print("  sampling note colors...")
        note_samples = sample_notes(video_path, calib, midi_data.notes)

        colors = [note_samples[i].get("onset_rgb") for i in range(len(midi_data.notes))]
        if args.pick_hand_colors and picked_hand_colors:
            # picked_hand_colors was already gathered earlier (right after
            # the full video download, before --transcribe) -- see the
            # comment above that block for why.
            hands, hand_confidences = assign_hands_from_reference_colors(colors, picked_hand_colors, flip=args.flip_render_hands)
        elif args.pick_hand_colors:
            # Hand color picking wasn't done earlier (e.g., preview failed) -- do it now
            print("Opening hand-color picker on the full video "
                  "(scrub with arrow keys, click a lit key, ESC when done)...")
            picked_hand_colors = interactive_pick_hand_colors(video_path)
            picked_hand_colors = _apply_hand_color_fallback_and_log(args.video, picked_hand_colors)
            print(f"  picked colors: {picked_hand_colors}")
            hands, hand_confidences = assign_hands_from_reference_colors(colors, picked_hand_colors, flip=args.flip_render_hands)
        else:
            print("  assigning hands from color clustering...")
            hands, confidence, _degenerate = assign_hands_for_notes(colors, pitches, flip=args.flip_render_hands)
            hand_confidences = [confidence] * len(hands)

        print("  detecting key release times...")
        # Must be a list PARALLEL to midi_data.notes (None where undetermined)
        # -- trim_note_durations does `zip(notes, lit_end_times)`, so a dict
        # here (previously only populated for notes with a determined
        # lit_end) silently truncated to the dict's length: zip() stops at
        # the shorter iterable, so every note beyond that count -- and any
        # note whose lit_end came out None -- was dropped from the output
        # entirely, not just left untrimmed.
        lit_end_times = []
        for idx, note in enumerate(midi_data.notes):
            samples = note_samples[idx].get("samples", [])
            baseline = baselines.get(note.pitch, (200, 200, 200))
            onset_sec = note.start_sec
            lit_end_times.append(lit_end_time(samples, baseline, onset_sec, threshold=40.0, gap_samples=2))

        print("  trimming note durations to lit-key release...")
        trimmed_notes = trim_note_durations(midi_data.notes, lit_end_times, min_duration_sec=0.05)

        out_notes = []
        confidences = []
        notes_with_time = []
        trims = {}  # (onset_tick, pitch) -> new_duration_sec, for the bundled MIDI rewrite

        for idx, note in enumerate(trimmed_notes):
            note_entry = {
                "onsetTick": note.onset_tick,
                "pitch": note.pitch,
                "hand": hands[idx],
                "confidence": round(hand_confidences[idx], 4),
            }

            if note.duration_sec != midi_data.notes[idx].duration_sec:
                note_entry["durationSec"] = round(note.duration_sec, 4)
                trims[(note.onset_tick, note.pitch)] = note.duration_sec

            out_notes.append(note_entry)
            confidences.append(hand_confidences[idx])
            notes_with_time.append((note.start_sec, hand_confidences[idx]))

        dropped_count = 0
        flagged_count = 0
        depedaled_count = 0

        # Rewrite the bundled MIDI so the app actually receives the
        # lit-key-trimmed note offsets (not just JSON metadata it never
        # consumes for timing -- see the module docstring in render_hands.py
        # for why this exists). CC64 pedal events pass through untouched.
        if trims:
            from midi_io import write_trimmed_midi

            trimmed_midi_path = _strip_fingering_json_suffix(args.out) + ".render-trimmed.mid"
            write_trimmed_midi(args.midi, trimmed_midi_path, trims)
            print(f"  wrote lit-trimmed MIDI to {trimmed_midi_path} ({len(trims)} note(s) trimmed)")
            args.midi = trimmed_midi_path

    else:
        # Interactive pre-alignment: watch/hear the MIDI against the video and tune
        # the offset before committing to per-frame tracking (unless --no-align, or
        # the legacy press-moments method which needs tracking first).
        if not args.no_align and not use_press_moments:
            offset_sec = interactive_align(video_path, calib, midi_data, offset_sec)
            print(f"Offset after alignment: {offset_sec:.3f}s")

        print(f"Tracking hands in video at {args.fps} fps (this may take a while)...")
        fingertip_frames = extract_fingertip_frames(
            video_path, fps=args.fps, flip_handedness=args.flip_handedness, k1=calib.k1,
            min_hand_confidence=args.min_hand_confidence,
        )
        print(f"  {len(fingertip_frames)} sampled frames")

        # Diagnostic: predicts no-finger-support risk before the (much slower)
        # matching pass runs -- pieces where both hands stay in the same
        # narrow register confuse the nearest-fingertip check far more than
        # register-separated (bass LH / melody RH) playing. See reconcile.py.
        spread = analyze_hand_spread(fingertip_frames, calib)
        if spread is not None:
            print(
                f"  hand spread: mean {spread.mean_distance_keys:.1f} keys apart "
                f"({spread.frames_with_both_hands} dual-hand frames, "
                f"{spread.frac_close * 100:.0f}% closer than {CLOSE_HANDS_THRESHOLD_KEYS:.0f} keys)"
            )
            if spread.mean_distance_keys < HAND_SPREAD_WARN_THRESHOLD_KEYS:
                print(
                    "  ⚠ hands stay unusually close together in this video -- expect a "
                    "higher no-finger-support rate than typical (register-separated) pieces"
                )

        if use_press_moments:
            print("Estimating offset using sync method: press-moments (legacy fallback)")
            offset_sec = estimate_offset_from_press_moments(fingertip_frames, midi_data)
            print(f"Auto-estimated offset: {offset_sec:.3f}s (override with --offset)")

        if args.preview:
            run_preview(video_path, calib, midi_data, offset_sec, fingertip_frames, args.fps)

        # Group simultaneous MIDI notes (chords) so we can resolve finger
        # conflicts within each chord.
        note_times = [n.start_sec for n in midi_data.notes]
        groups = group_simultaneous(note_times, tolerance_sec=0.03)

        # Match in depth-invariant keyboard-x space (continuous white-key index):
        # a finger identifies a key by its position ALONG the keyboard, not by how
        # far down the key it presses. Matching in raw 2-D pixels penalizes the
        # natural depth spread of fingers and collapses confidence for every note.
        # KEY_MATCH_SCALE is the characteristic mismatch (in white keys) at which
        # confidence drops to ~1/e -- ~0.8 of a white key, so a finger on the right
        # key scores high and one a whole key off scores low.
        KEY_MATCH_SCALE = 0.8

        results: dict = {}
        for group in groups:
            video_time = midi_data.notes[group[0]].start_sec + offset_sec
            tips = interpolate_fingertips(fingertip_frames, video_time)
            candidates = [
                Candidate(hand=h, finger=f, x=calib.screen_to_white_index((x, y)), y=0.0)
                for h, f, x, y in tips
            ]

            notes_to_match = []
            for idx in group:
                note = midi_data.notes[idx]
                key_x = pitch_to_white_index(note.pitch)
                notes_to_match.append(NoteToMatch(note_id=idx, key_xy=(float(key_x), 0.0)))

            group_results = resolve_chord_conflicts(notes_to_match, candidates, scale=KEY_MATCH_SCALE)
            results.update(group_results)

        # Video-based reconciliation (Phase D): cross-checks each matched note
        # against the video's own hand tracking to flag/drop ghost notes and
        # trim pedal-inflated durations. Conservative: flags more than it drops,
        # and never judges a note whose onset falls in an occluded stretch of
        # video (no fingertip tracked nearby at all) -- audio wins there. See
        # reconcile.py's module docstring for why this exists: a real comparison
        # against another AMT tool's output showed the disputed notes were loud
        # and sustained, not quiet/short -- no audio-only threshold could ever
        # separate them, so this uses the video as independent evidence instead.
        frame_times = _frame_times(fingertip_frames)

        out_notes = []
        confidences = []
        notes_with_time = []  # (start_sec, confidence) for the per-window table
        dropped_count = 0
        flagged_count = 0
        depedaled_count = 0

        for idx, note in enumerate(midi_data.notes):
            m = results.get(idx)
            if m is None:
                continue
            if m.confidence < args.confidence_threshold:
                continue

            note_entry = {
                "onsetTick": note.onset_tick,
                "pitch": note.pitch,
                "hand": m.hand,
                "finger": m.finger,
                "confidence": round(m.confidence, 4),
            }

            if not args.no_reconcile:
                video_onset_sec = note.start_sec + offset_sec
                rec = reconcile_note(
                    velocity=note.velocity,
                    duration_sec=note.duration_sec,
                    video_onset_sec=video_onset_sec,
                    pitch=note.pitch,
                    match_distance_keys=m.distance,
                    frame_times=frame_times,
                    frames=fingertip_frames,
                    calib=calib,
                    pedal_segments=pedal_segments,
                )
                if "dropped" in rec.flags:
                    dropped_count += 1
                    continue
                if rec.flags:
                    note_entry["flags"] = rec.flags
                    if "no-finger-support" in rec.flags:
                        flagged_count += 1
                    if "depedaled" in rec.flags:
                        depedaled_count += 1
                        # Seconds, not ticks: the reconciliation math above is
                        # entirely in seconds (video/audio time), and nothing
                        # downstream currently consumes a tick-based duration --
                        # keeping these in seconds avoids an unused tick
                        # conversion (see midi_io.py for the tick<->sec map if
                        # a future consumer needs it).
                        note_entry["durationSec"] = round(rec.trimmed_duration_sec, 4)
                        note_entry["audioDurationSec"] = round(rec.audio_duration_sec, 4)

            out_notes.append(note_entry)
            confidences.append(m.confidence)
            notes_with_time.append((note.start_sec, m.confidence))

    out = {
        "version": 2,
        "source": args.video,
        "midi": args.midi,
        "ppq": midi_data.ppq,
        "offsetSec": offset_sec,
        "notes": out_notes,
    }
    if pedal_segments:
        out["pedal"] = [{"startSec": round(s, 4), "endSec": round(e, 4)} for s, e in pedal_segments]

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    print()
    print("Summary:")
    print(f"  notes matched: {len(out_notes)}/{len(midi_data.notes)}")
    print(f"  mean confidence: {mean_conf:.3f}")
    print(f"  offset used: {offset_sec:.3f}s")
    if not args.no_reconcile:
        print(f"  reconciliation: {dropped_count} dropped, {flagged_count} flagged (no-finger-support), "
              f"{depedaled_count} de-pedal trimmed")

    # Per-10s-window mean confidence: a declining trend over a long song is
    # the signature of video/MIDI sync DRIFT (a single constant offset can't
    # fix it), not a uniformly-bad calibration -- see README "Sync".
    windows = confidence_by_window(notes_with_time)
    if len(windows) > 1:
        print("  confidence by 10s window:")
        for start_sec, mean_window_conf, count in windows:
            print(f"    {start_sec:6.0f}s: {mean_window_conf:.3f}  ({count} notes)")

    return out


def _apply_hand_color_fallback_and_log(video: str, picked: dict) -> dict:
    """Fills in a missing black-key reference color from that hand's picked
    white-key color (assign_hands_from_reference_colors deliberately
    doesn't guess this itself -- see its docstring), and appends a record
    of what was actually clicked by the user vs. what the tool assumed to
    hand_colors_log.jsonl (append-only, one JSON object per line).

    This is a growing ground-truth dataset -- real user-picked colors per
    real render -- for later comparing/training the automatic hue-
    clustering approach (render_hands.assign_hands_for_notes) against,
    since that's what motivated manual picking to begin with (see its
    docstring/history: automatic clustering has produced implausibly
    skewed hand splits on some renders despite reporting high confidence).
    """
    user_picked = dict(picked)
    assumed = {}
    if "LH_black" not in picked and "LH_white" in picked:
        picked["LH_black"] = picked["LH_white"]
        assumed["LH_black"] = picked["LH_white"]
    if "RH_black" not in picked and "RH_white" in picked:
        picked["RH_black"] = picked["RH_white"]
        assumed["RH_black"] = picked["RH_white"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "hand_colors_log.jsonl")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "video": video,
        "user_picked": user_picked,
        "assumed": assumed,
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        print(f"  (could not write hand_colors_log.jsonl: {exc})")

    return picked


def _log_youtube_run(video: str, out_path: str, transcribed_midi_path: str | None, bundle_path: str | None) -> None:
    """Log a YouTube video URL and its output files to log.txt (append-only).
    No-op if video is a local file path (not http:// or https://)."""
    if not (video.startswith("http://") or video.startswith("https://")):
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "log.txt")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n")  # blank line separator
        f.write(f"{video}\n")
        if bundle_path:
            f.write(f"{os.path.basename(bundle_path)}\n")
        if transcribed_midi_path:
            f.write(f"{os.path.basename(transcribed_midi_path)}\n")
        f.write(f"{os.path.basename(out_path)}\n")


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.midi_only:
        args.transcribe = True
    if args.pick_hand_colors:
        args.render = True

    if args.selftest:
        import selftest

        return 0 if selftest.run() else 1

    result = analyze(args)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out}")

    bundle_path = None
    if not args.no_bundle:
        from bundle import write_symple_bundle, default_bundle_path

        bundle_path = default_bundle_path(args.out)
        try:
            metadata = {"artist": args.artist, "title": args.title, "genre": args.genre, "difficulty": args.difficulty} if (args.artist or args.title or args.genre or args.difficulty) else None
            write_symple_bundle(bundle_path, args.midi, args.out, source_video=args.video, metadata=metadata)
            print(f"Wrote {bundle_path} (MIDI + fingering, for one-step loading in Symplethesia)")
        except Exception as exc:
            print(f"  (could not write .symple bundle: {exc})")
            bundle_path = None

    # Log YouTube runs to log.txt for traceability (which video produced which files).
    transcribed_midi_path = None
    if args.transcribe:
        transcribed_midi_path = _strip_fingering_json_suffix(args.out) + ".transcribed.mid"
    _log_youtube_run(args.video, args.out, transcribed_midi_path, bundle_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
