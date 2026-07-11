# 006 — Hand-assignment benchmark harness

**Status:** Harness complete — awaiting human labeling of `benchmark/groovy-kind-of-love/truth.json`
**Owner:** Opus (design) · Sonnet (implementation)
**Date:** 2026-07-09

## Why

Before adding any more tracking layers (Kalman, optical flow, background
subtraction, etc. — see the critique of the 10-idea proposal), we need a
**measuring stick**. Otherwise every new layer is faith-based: we can't tell
if it helps or hurts. The one metric that matters most: **% of notes with the
correct hand** (finger accuracy secondary).

## Core idea: labeling = "correct a copy of the output"

Authoring ground truth by hand per note is too tedious. Instead: run SMaPE,
take its `fingering.json`, and FIX the wrong hand labels (start from the
prediction, flip the mistakes). That corrected file is the ground truth. A
`benchmark.py init` helper bootstraps the template pre-filled with SMaPE's
current guesses so the user only flips what's wrong.

## Layout

```
benchmark/
  <song-name>/
    truth.json       # hand-corrected reference (ground truth)
    predicted.json   # latest SMaPE fingering.json output (regenerate to re-measure)
    notes.md         # optional: video url + how truth was made
```

## Schema

`truth.json`:
```json
{ "video": "https://...",
  "notes": [ {"pitch": 60, "onsetSec": 12.43, "hand": "R", "finger": 2}, ... ] }
```
`finger` optional per note. `onsetSec` matches the pred's `startSec` (the
init helper copies it), so exact-onset matching is trivial; a small tolerance
handles re-transcribed predictions whose onsets shifted slightly.

## API (benchmark.py)

Pure, unit-tested in selftest.py:
- `join_notes(truth, pred, sec_tol=0.1) -> list[(truth_note, pred_note|None)]`
  — match by pitch AND nearest onset within tol; each pred note used at most once.
- `score(pairs) -> dict` — {labeled, matched, coverage, hand_correct,
  hand_accuracy, finger_labeled, finger_correct, finger_accuracy,
  confusion:{L_as_R, R_as_L}}. Accuracy denominators = matched notes.

I/O orchestration (not unit-tested):
- `load_truth(path)`, `load_pred(path)` (normalize fingering.json notes:
  use `startSec` for onset, `hand`, `finger`).
- `init_truth(fingering_path, out_path)` — write a truth.json template from a
  prediction (current guesses) for the user to correct.
- `run_dir(bench_dir)` — scan subdirs, score each, print a per-song table +
  an aggregate row.

CLI:
- `python benchmark.py`            → scan ./benchmark/, print the table.
- `python benchmark.py init <fingering.json> [out]` → bootstrap a truth template.

## Steps

Legend: [ ] todo · [~] in progress · [x] done

- [x] **A. `benchmark.py`** — pure `join_notes`/`score` + I/O
  (`load_truth`/`load_pred`/`init_truth`/`run_dir`) + CLI. Done.
- [x] **B. selftest tests** — join (exact, within-tol, miss, nearest-of-two),
  score (all-correct=1.0, one-wrong-hand, coverage with a miss, finger acc over
  finger-labeled only, confusion counts). Registered in TESTS. 8 new tests,
  all 100 tests pass (`.venv/bin/python3 selftest.py`).
- [x] **C. `benchmark/README.md` + main README section** — the authoring
  workflow (run → init → correct → re-run). Added `benchmark/README.md` and
  a "Benchmarking hand-assignment accuracy" section in the main README.
- [x] **D. Seed the first case** — bootstrap `benchmark/groovy-kind-of-love/`
  from the existing output (video watch?v=WTxmmqbHe_M); the USER still needs
  to correct `truth.json` by hand (labeling is a human task — we only
  scaffolded it, current numbers are a trivial 100% since truth==predicted).

## Non-goals

- Not part of CI's video pipeline (no MediaPipe in selftest) — only the pure
  scoring is unit-tested; predictions are produced offline.
- Not auto-labeling: ground truth is human-corrected. The harness only
  measures and scaffolds.
