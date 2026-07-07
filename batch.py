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
    """Runs the full two-phase batch pipeline over ``sources`` and returns a
    process exit code (0 if every video succeeded, 1 if any failed).

    Phase 0: expand playlists/files into a flat video list.
    Phase 1 (interactive, front-loaded): for each video, resolve its
      calibration/colour storage keys, print a homogeneity report, and run
      interactive calibration (+ colour picking, in --render mode) ONLY for
      videos `plan_phase1` marks "manual" -- the first video of each channel
      still missing saved data. Persists via the same stores
      `_get_or_create_calibration`/`interactive_pick_hand_colors` already
      write to, so Phase 2 (and later batches) find them saved.
    Phase 2 (unattended): for each video, build a per-video copy of ``args``
      (--transcribe, --no-align, headless reuse of saved calibration/colours)
      and run `extract_fingering.analyze()` inside try/except so one bad
      video never aborts the batch.
    """
    from calibration_store import calibration_key, load_saved_calibration
    from color_store import color_key, load_saved_colors
    import extract_fingering

    needs_colors = bool(args.render)

    entries = expand_sources(sources)
    print(f"Batch: {len(entries)} video(s) found across {len(sources)} source(s).")
    if not entries:
        print("Nothing to process.")
        return 0

    # Resolve each entry's storage key up front (channel_id when known, else
    # a per-path hash -- same convention calibration_key/color_key already
    # use elsewhere) and stash it on the entry for plan_phase1/Phase 2 reuse.
    for entry in entries:
        entry["channel_key"] = calibration_key(entry["url"], entry.get("channel_id"))
        # color_key uses the identical (channel_id or path-hash) scheme, so
        # for a given entry calibration_key/color_key always agree on
        # channel-vs-path -- one channel_key string is enough for both
        # has_saved_calib/has_saved_colors checks below.

    def has_saved_calib(channel_key: str) -> bool:
        return load_saved_calibration(channel_key) is not None

    def has_saved_colors(channel_key: str) -> bool:
        return load_saved_colors(channel_key) is not None

    plan = plan_phase1(entries, has_saved_calib, has_saved_colors, needs_colors)
    print(f"Phase 1: {plan['report']}")

    download_dir = os.path.join(extract_fingering.DOWNLOADS_DIR, "downloads")

    for entry in plan["manual"]:
        print(f"\n--- Phase 1 (manual): {entry.get('title') or entry['url']} ---")
        try:
            video_path, _audio_path, resolved_channel_id = extract_fingering._download_if_url(
                entry["url"], download_dir,
            )
            calib_key = calibration_key(entry["url"], resolved_channel_id or entry.get("channel_id"))
            color_key_ = color_key(entry["url"], resolved_channel_id or entry.get("channel_id"))

            # Full 88-key range: batch's --transcribe mode has no MIDI yet at
            # calibration time, same reasoning as analyze()'s early-preview path.
            low_pitch = args.low_pitch if args.low_pitch is not None else 21
            high_pitch = args.high_pitch if args.high_pitch is not None else 108

            extract_fingering._get_or_create_calibration(
                video_path, args.calibration, low_pitch, high_pitch,
                use_cv_calibration=not args.no_cv_calibration,
                calib_key=calib_key,
            )

            if needs_colors:
                from render_hands import interactive_pick_hand_colors
                from color_store import save_colors

                print("Opening hand-color picker (scrub with arrow keys, click a lit key, ESC when done)...")
                picked = interactive_pick_hand_colors(video_path)
                picked = extract_fingering._apply_hand_color_fallback_and_log(entry["url"], picked)
                save_colors(color_key_, picked)
        except Exception as exc:  # noqa: BLE001
            print(f"  Phase 1 setup failed for {entry['url']}: {exc} -- interactive calibration/colours is not possible headlessly; this video will likely fail Phase 2 too.")

    print("\nPhase 2: unattended transcribe + analyze for all videos.")
    results = []  # (url, ok, err)
    for entry in entries:
        print(f"\n--- Phase 2: {entry.get('title') or entry['url']} ---")
        video_args = copy.deepcopy(args)
        video_args.video = entry["url"]
        video_args.transcribe = True
        video_args.no_align = True
        video_args.out = None  # re-derive per video from its own filename
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
    print(f"\nbatch done: {ok_count} ok, {fail_count} failed, 0 skipped")
    if fail_count:
        print("Failures:")
        for url, ok, err in results:
            if not ok:
                print(f"  {url}: {err}")
    return 0 if fail_count == 0 else 1
