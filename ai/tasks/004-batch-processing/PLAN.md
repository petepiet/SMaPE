# 004 — Batch processing (YouTube playlist → multiple MIDI/.symple)

**Status:** CLI batch processing (steps A-F) COMPLETE. GUI queue (step G) still pending.
**Owner model split:** Opus (planning/verification) · Sonnet (interlocking core) · Haiku (mechanical mirror)
**Started:** 2026-07-06

Convert a YouTube playlist (or several video URLs) into MIDI + `.symple`
bundles in one run, by front-loading all human-in-the-loop steps
(calibration; for Synthesia videos also LH/RH colour picking) and then
running the slow, unattended work (transcribe → track/colour → match →
reconcile → release → export) as a batch.

## Scope decisions (confirmed)

1. **`--transcribe` mode first.** Offset = 0 by construction, so alignment
   is skippable (`--no-align`) and phase 2 is fully unattended. Batch with
   user-supplied MIDI per video = later extension.
2. **One `--batch` command** runs phase 1 (interactive) for all videos, then
   phase 2 (unattended) for all videos.
3. **CLI-first.** GUI queue is a separate later phase.
4. **Consistency checks** so "calibrate/pick once per channel" is safe:
   - camera angle: reuse existing `calibrations_agree` headless;
   - colours: new `colors_agree()` (match against known LH/RH reference —
     easier/more robust than from-scratch clustering);
   - phase 1 prompts the human ONLY when a check fails (bumped camera,
     different framing, or a different colour theme);
   - plus an up-front homogeneity report (how many videos share the
     channel calibration/colours; which need manual setup).

## Architecture

```
Phase 0  playlist/URL list → flat video list        (batch.expand_sources)
Phase 1  INTERACTIVE, batched — only what a check flags as unknown/changed:
         per video: preview clip → calibration (persist) → [synthesia: colour pick (persist)]
         trusted saved calib/colours for this channel → reuse headless (no window)
Phase 2  UNATTENDED, batch — per video:
         transcribe → track/colour → match → reconcile → release → MIDI + .symple
         (--no-align + saved calibration + saved colours)
```

## Steps

Legend: [ ] todo · [~] in progress · [x] done · owner in ()

- [x] **A. `color_store.py`** (Haiku) — per-channel LH/RH colour persistence,
  exact mirror of `calibration_store.py` (key/load/save only, NO
  `colors_agree`). + round-trip selftest. **DONE** — verified by Opus,
  clean mirror; selftest 82/82 green (`test_color_store_round_trip` added).
- [x] **B. `colors_agree()`** (Sonnet) — in `render_hands.py`; sample a few
  lit-key colours from a video and compare (hue distance) against saved
  LH/RH reference; within tolerance → reuse, else → re-pick. + tests.
  **DONE** — keys on LH_white/RH_white, 25° tolerance converted to hue-circle
  chord distance; 4 new tests, selftest 86/86 green.
- [x] **C. Headless calibration reuse** (Sonnet) — `_get_or_create_calibration`
  gets `allow_headless_reuse`: when a trusted saved calib exists, return it
  WITHOUT opening `interactive_calibrate`. Serves phase 1 (skip already-
  calibrated channels) and phase 2 (never a window).
  **DONE** — new `get_trusted_calibration()` helper (never opens a window;
  CV-corroborates when possible, else returns the saved calib uncorroborated,
  returns None only when nothing saved or CV actively disagrees);
  `allow_headless_reuse=False` default keeps single-video behaviour
  unchanged. py_compile clean.
- [x] **D. Playlist expansion** (Sonnet) — `batch.expand_sources(urls)` via
  yt-dlp `--flat-playlist`; accepts playlist URL, loose URLs, or a text file
  of URLs; returns `[{url, channel_id, title}]`. + parsing test on fake JSON.
  **DONE** — pure `normalize_entries()` (id-only entries get a built watch
  url, missing channel_id/title default sanely) + `expand_sources()`
  (lazy yt-dlp import, .txt file support, de-dup by url). 3 new tests.
- [x] **E. Batch orchestrator** (Sonnet) — `batch.batch_process(sources, args)`:
  phase-1 loop (interactive only on check-failure) + homogeneity report,
  phase-2 loop (unattended `analyze()` per video, per-video error isolation,
  final summary). + planning test (which videos need phase-1 interaction
  given what's already saved).
  **DONE** — pure `plan_phase1()` (channel-grouped: first not-yet-saved
  source per channel is "manual", rest "reuse") + `batch_process()`
  orchestrator wiring `_download_if_url`/`_get_or_create_calibration`/
  `interactive_pick_hand_colors`/`save_colors` for Phase 1 and
  `analyze()` + `args.headless_reuse=True` for Phase 2. 2 new tests.
  selftest 91/91 green.
- [x] **F. CLI + README + tests green** (Sonnet) — `--batch <url|file>` in
  `build_arg_parser`, routing in `main()`, README batch section, full
  selftest green.
  **DONE** — `--batch` (nargs='+') added, `main()` routes to
  `batch.batch_process()` before the single-video path when set, README
  "Batch processing" section added before "Matching". py_compile clean on
  all changed files; `selftest.py` and `extract_fingering.py --selftest`
  both report "OK - all 91 tests passed".
- [ ] **G. GUI queue** (later) — paste playlist/URLs, per-video progress.

## Decisions log

- 2026-07-06: field/model decisions inherited from task 003 note-release
  (camelCase + L/R). Batch reuses `analyze()` unchanged.
- 2026-07-06: `colors_agree` lives in `render_hands.py` (where hue helpers
  are), NOT in `color_store.py` (persistence only) — clean split for the
  Haiku/Sonnet division.
- 2026-07-07: `analyze()`'s headless-reuse plumbing uses `getattr(args,
  "headless_reuse", False)` rather than a `build_arg_parser` default, so a
  plain single-video run's `args` namespace need not carry the attribute at
  all -- `batch.batch_process()` sets `video_args.headless_reuse = True` on
  its per-video copy before calling `analyze()` directly (bypassing
  `main()`/argparse entirely for Phase 2). Colour reuse in `--render` mode
  is plumbed by loading saved colours into `picked_hand_colors` and forcing
  `args.pick_hand_colors = True` for that run, so it flows through the
  existing `assign_hands_from_reference_colors` branch unchanged rather than
  adding a third hand-assignment code path.

## Test plan (pure-logic, no cv/video, in selftest.py)

- color_store round-trip (A).
- colors_agree: same theme agrees, different theme disagrees (B).
- playlist parsing from fake yt-dlp JSON (D).
- batch planning: given saved calib/colours, which videos still need a
  manual pick (E).

## Related completed work

- Task 003 note-release correction — DONE, committed `056c927`
  (key vs sound duration, `note_release.py`).
