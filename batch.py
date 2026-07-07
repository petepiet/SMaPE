"""Batch processing: turn a playlist URL, a list of loose video URLs, or a
.txt file of URLs into a flat list of videos, then run this tool's usual
--transcribe pipeline over all of them in two phases -- front-loaded
interactive setup (calibration, and for --render, hand-colour picking), then
a fully unattended per-video transcribe+analyze pass. See
ai/tasks/004-batch-processing/PLAN.md for the overall design.

Split the same way as the rest of this codebase: pure list/dict logic
(`normalize_entries`, `plan_phase1`) is fully covered by selftest.py;
yt-dlp/cv2-touching orchestration (`expand_sources`, `batch_process`) is not
-- verified against real playlists/videos instead.
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Callable, Optional


# --------------------------------------------------------------------------
# Pure logic -- unit-tested in selftest.py
# --------------------------------------------------------------------------

def normalize_entries(info_entries: list) -> list:
    """Normalizes yt-dlp-style info dicts into a flat
    ``[{"url":..., "channel_id":..., "title":...}, ...]`` list.

    Accepts either:
      - a list of "flat" playlist entry dicts (what
        ``yt_dlp.YoutubeDL(...).extract_info(playlist_url)['entries']``
        yields with ``extract_flat: True`` -- each entry commonly has
        'id'/'url'/'channel_id'/'title', but not consistently all of them
        depending on the extractor), or
      - a single video's own info dict passed directly (no 'entries' key --
        the caller is expected to have already unwrapped
        ``info.get('entries') or [info]`` before calling this, so every
        item here is already "one video's dict").

    Missing 'channel_id'/'title' become None/"" rather than raising --
    yt-dlp's flat-playlist extraction doesn't guarantee either field is
    present for every extractor. A watchable url is derived from 'url' if
    present, else built from 'id' as
    "https://www.youtube.com/watch?v=<id>" (flat-playlist entries commonly
    carry only the bare video id, not a full url). An entry with neither
    'url' nor 'id' is skipped (nothing to watch).
    """
    normalized = []
    for entry in info_entries:
        if not entry:
            continue
        url = entry.get("url")
        if not url:
            video_id = entry.get("id")
            if not video_id:
                continue
            url = f"https://www.youtube.com/watch?v={video_id}"
        normalized.append({
            "url": url,
            "channel_id": entry.get("channel_id") or entry.get("uploader_id"),
            "title": entry.get("title") or "",
        })
    return normalized


def expand_sources(sources: list) -> list:
    """Expands ``sources`` (playlist URLs, single-video URLs, and/or paths to
    a .txt file of one-URL-per-line) into a flat, de-duplicated
    ``[{"url", "channel_id", "title"}, ...]`` list, in the order first seen.

    For a .txt path, each non-blank line is treated as its own URL (itself
    possibly a playlist -- expanded the same as any other URL). Each URL is
    resolved via a flat (fast, no per-video metadata fetch)
    ``yt_dlp.YoutubeDL`` extraction: a playlist URL yields multiple entries,
    a single-video URL yields just itself (no 'entries' key at all).
    """
    import yt_dlp  # lazy import -- keeps this module importable from selftest.py

    all_entries = []
    for source in sources:
        if source.lower().endswith(".txt") and os.path.exists(source):
            with open(source, "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]
        else:
            urls = [source]

        for url in urls:
            ydl_opts = {"extract_flat": True, "quiet": True, "skip_download": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            raw_entries = info.get("entries") or [info]
            all_entries.extend(normalize_entries(raw_entries))

    # De-dup by url, preserving first-seen order.
    seen = set()
    deduped = []
    for entry in all_entries:
        if entry["url"] in seen:
            continue
        seen.add(entry["url"])
        deduped.append(entry)
    return deduped


def plan_phase1(
    sources: list,
    has_saved_calib: Callable[[str], bool],
    has_saved_colors: Callable[[str], bool],
    needs_colors: bool,
) -> dict:
    """Decides, per source, whether Phase 1 (interactive setup) needs to
    touch it at all -- the batch-processing homogeneity check.

    ``sources`` is a list of entries as returned by `expand_sources`/
    `normalize_entries`, each requiring a "channel_key" the caller has
    already resolved onto the entry (see `batch_process`, which stores it
    under entry["channel_key"] before calling this) -- kept as a plain
    string key here (not `calibration_key`/`color_key` themselves) so this
    function stays pure/injectable: ``has_saved_calib``/``has_saved_colors``
    are callables (``channel_key -> bool``) the caller backs with real
    `load_saved_calibration`/`load_saved_colors` disk reads; a test can pass
    fakes instead, with no disk or network access.

    Grouped by channel: within one run of `plan_phase1`, only the FIRST
    source of a given channel that still lacks saved data is "manual" --
    once that first one is (about to be, or already) saved, every later
    source from the SAME channel in this same batch is "reuse", since
    Phase 1 will have populated the store for that channel by the time
    Phase 2 needs it. A channel that already has saved data before this
    batch even started (``has_saved_calib``/``has_saved_colors`` already
    True) needs no manual entry at all.

    ``needs_colors`` gates whether the colour-store check participates at
    all (only True for --render/synthesia-mode batches; --transcribe-only
    batches never need a colour pick).

    Returns ``{"reuse": [...], "manual": [...], "report": "<summary>"}``.
    """
    reuse = []
    manual = []
    # Channels this planning pass has already decided will be handled (either
    # already saved, or the first manual entry queued to populate it) --
    # lets every SUBSEQUENT source from that channel default to reuse.
    resolved_channels = set()

    for entry in sources:
        channel_key = entry.get("channel_key") or entry["url"]

        calib_ok = has_saved_calib(channel_key)
        colors_ok = (not needs_colors) or has_saved_colors(channel_key)
        already_good = calib_ok and colors_ok

        if already_good or channel_key in resolved_channels:
            reuse.append(entry)
            resolved_channels.add(channel_key)
        else:
            manual.append(entry)
            resolved_channels.add(channel_key)

    total = len(sources)
    n_reuse = len(reuse)
    n_manual = len(manual)
    if n_manual == 0:
        report = f"{n_reuse} of {total} video(s) reuse the channel calibration/colours; none need manual setup."
    else:
        manual_titles = ", ".join(e.get("title") or e["url"] for e in manual)
        report = (
            f"{n_reuse} of {total} video(s) reuse the channel calibration/colours; "
            f"{n_manual} need manual setup: {manual_titles}."
        )
    return {"reuse": reuse, "manual": manual, "report": report}


# --------------------------------------------------------------------------
# Orchestration -- uses yt-dlp/cv2 (via extract_fingering), not unit-tested
# directly; verified against real playlists/videos.
# --------------------------------------------------------------------------

def batch_process(sources: list, args: argparse.Namespace) -> int:
    """Runs the full three-phase batch pipeline over ``sources``.

    Phase 1 (fast preview download): fetch only the first ~45 s of every
      video so the calibration UI can open immediately without waiting for
      a full multi-GB download.  Each video gets its own preview subfolder
      so previews don't clobber each other.
    Phase 2 (interactive, front-loaded): for EVERY video in order, show
      the key-overlay calibration UI (and the colour picker in --render
      mode).  ``allow_headless_reuse=False`` forces the UI even for channels
      that already have a saved calibration -- the user sees and confirms
      every video before the long unattended phase starts.
    Phase 3 (unattended): full download + KONG transcription + analysis for
      each video, reusing the calibration saved in Phase 2.
    """
    from calibration_store import calibration_key
    from color_store import color_key
    import extract_fingering

    needs_colors = bool(args.render)
    low_pitch = args.low_pitch if args.low_pitch is not None else 21
    high_pitch = args.high_pitch if args.high_pitch is not None else 108

    entries = expand_sources(sources)
    print(f"Batch: {len(entries)} video(s) found across {len(sources)} source(s).")
    if not entries:
        print("Nothing to process.")
        return 0

    for entry in entries:
        entry["channel_key"] = calibration_key(entry["url"], entry.get("channel_id"))

    download_dir = os.path.join(extract_fingering.DOWNLOADS_DIR, "downloads")

    # ------------------------------------------------------------------
    # Phase 1: fast preview download (~45 s) for every video
    # ------------------------------------------------------------------
    print(f"\nPhase 1 of 3: fast preview download for {len(entries)} video(s)...")
    for i, entry in enumerate(entries):
        label = entry.get("title") or entry["url"]
        print(f"  [{i + 1}/{len(entries)}] {label}")
        url = entry["url"]
        if url.startswith(("http://", "https://")):
            # Each video gets its own subfolder so previews don't overwrite
            # each other -- preview.mp4 inside batch_preview_000/, _001/, etc.
            preview_dir = os.path.join(download_dir, f"batch_preview_{i:03d}")
            preview_path, channel_id = extract_fingering._download_preview(url, preview_dir)
            if channel_id:
                entry["channel_id"] = channel_id
                entry["channel_key"] = calibration_key(url, channel_id)
            if preview_path is None:
                print(f"    preview download failed; will try full download in Phase 3")
        else:
            # Local file: use it directly as the calibration source
            preview_path = url if os.path.exists(url) else None
            if preview_path is None:
                print(f"    local file not found: {url}")
        entry["_preview_path"] = preview_path

    # ------------------------------------------------------------------
    # Phase 2: interactive key overlay + colour check for every video
    # ------------------------------------------------------------------
    print(f"\nPhase 2 of 3: key overlay check for {len(entries)} video(s) (one by one)...")
    for i, entry in enumerate(entries):
        label = entry.get("title") or entry["url"]
        preview = entry.get("_preview_path")
        if preview is None:
            print(f"\n  [{i + 1}/{len(entries)}] SKIP (no preview): {label}")
            continue
        print(f"\n--- Calibration [{i + 1}/{len(entries)}]: {label} ---")
        try:
            calib_key_ = calibration_key(entry["url"], entry.get("channel_id"))
            # allow_headless_reuse=False: always open the UI so the user can
            # verify every video before the long unattended phase begins.
            extract_fingering._get_or_create_calibration(
                preview, args.calibration, low_pitch, high_pitch,
                use_cv_calibration=not args.no_cv_calibration,
                calib_key=calib_key_,
                allow_headless_reuse=False,
            )
            if needs_colors:
                from render_hands import interactive_pick_hand_colors
                from color_store import save_colors

                print("Opening hand-color picker (scrub with arrow keys, click a lit key, ESC when done)...")
                picked = interactive_pick_hand_colors(preview)
                picked = extract_fingering._apply_hand_color_fallback_and_log(entry["url"], picked)
                save_colors(color_key(entry["url"], entry.get("channel_id")), picked)
        except Exception as exc:  # noqa: BLE001
            print(f"  Calibration failed: {exc} -- this video will likely fail Phase 3 too.")

    # ------------------------------------------------------------------
    # Phase 3: unattended full download + transcribe + analyze
    # ------------------------------------------------------------------
    print(f"\nPhase 3 of 3: unattended transcription + analysis for {len(entries)} video(s).")
    results = []  # (url, ok, err)
    for entry in entries:
        print(f"\n--- Phase 2: {entry.get('title') or entry['url']} ---")  # "Phase 2" kept for GUI parser
        video_args = copy.deepcopy(args)
        video_args.video = entry["url"]
        video_args.transcribe = True
        video_args.no_align = True
        video_args.out = None
        video_args.output_dir = args.output_dir
        video_args.headless_reuse = True

        try:
            video_args._analysis_result = extract_fingering.analyze(video_args)
            if video_args.out is None:
                video_args.out = extract_fingering.default_out_path_from_video(video_args.video)
            extract_fingering._write_analysis_outputs(video_args)
            results.append((entry["url"], True, None))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}")
            results.append((entry["url"], False, str(exc)))

    ok_count = sum(1 for _, ok, _ in results if ok)
    fail_count = len(results) - ok_count
    print(f"\nbatch done: {ok_count} ok, {fail_count} failed")
    if fail_count:
        print("Failures:")
        for url, ok, err in results:
            if not ok:
                print(f"  {url}: {err}")
    return 0 if fail_count == 0 else 1
