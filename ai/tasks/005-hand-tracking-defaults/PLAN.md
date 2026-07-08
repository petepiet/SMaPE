# 005 — Hand-tracking defaults + dual-hand health check

**Status:** DONE (not yet committed)
**Owner:** Fable (diagnosis, implementation, verification — small targeted edits, no agents)
**Date:** 2026-07-08

## Trigger

Real-world failure on "A Groovy Kind Of Love" (real-hands overhead cover):
390 notes came back 381 L / 9 R, 343 flagged `no-finger-support`, median
confidence 0.000 — a useless hand split in Symplethesia despite palm/wrist
tracking being present.

## Measured diagnosis (30s clip of the failing video, fps 10)

| min_hand_confidence | frames with BOTH hands |
|---|---|
| 0.5 (old default) | **2%** |
| 0.3 | 90% |
| 0.15 | 93% |

- MediaPipe's detector, at the library-default 0.5, silently drops the
  SECOND hand on the typical overhead piano shot (keys at the bottom of the
  frame, hands only partly visible) — the single surviving hand absorbs
  every matchable note; the other hand's notes go no-finger-support at ~0
  confidence.
- Palm/wrist landmarks could not help: they are outputs of a successful
  detection, not a substitute for it.
- Handedness labels at 0.3 verified correct (236 frames L-left-of-R,
  0 swapped) — no flip needed.
- With `fix_handedness_continuity` + the calib crop, the same clip reaches
  **79% dual-hand at the new 0.3 default** — the failing video is fixed by
  the default alone, no retry needed.

## Changes (all done)

- [x] `extract_fingering.py` — `--min-hand-confidence` default 0.5 → 0.3
  (help text explains the measured why).
- [x] `hands.py` — `extract_fingertip_frames` param default 0.5 → 0.3
  (docstring updated with measurements).
- [x] `gui.py` — Min hand confidence default "0.5" → "0.3"; FPS default
  "20" → "30" (CLI already defaulted to 30); tooltip updated.
- [x] `reconcile.py` — new pure `dual_hand_rate(frames)` +
  `DUAL_HAND_RATE_MIN = 0.2` + `RETRY_HAND_CONFIDENCE = 0.15`.
- [x] `extract_fingering.py` analyze() — post-tracking health check: prints
  the dual-hand rate; if < 20% with headroom, auto-retries tracking ONCE at
  0.15 and keeps whichever pass saw both hands more often; final warning if
  still low (points to --preview + README section).
- [x] `selftest.py` — `test_dual_hand_rate` added and registered.
- [x] `README.md` — CLI table + "a hand frequently missing" section updated
  (new default, measured numbers, built-in mitigation described).

## Verification

- py_compile clean on all touched files.
- selftest: **OK — all 92 tests passed** (was 91).
- End-to-end on the failing clip with the new default: 79% dual-hand rate,
  no retry triggered — the original video's failure mode is gone.

## Notes / follow-ups

- The affected song should be re-run (old .symple was produced at 0.5).
- Related: task 004 batch processing (unattended Phase 2 benefits directly
  from the auto-retry — no human watching the tracking there).
