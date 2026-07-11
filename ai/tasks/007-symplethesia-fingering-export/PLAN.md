# 007 — Symplethesia: "Export fingering analysis (dev)"

**Status:** DONE
**Owner:** Opus (design) · Sonnet (implementation)
**Date:** 2026-07-09
**Repo:** /home/pjotter/Documenten/VS Code/symplethesia (NOT the SMaPE repo)

## Why

To validate SMaPE's video-based hand assignment we need human-verified ground
truth (SMaPE benchmark, task 006). The user prefers to correct hands in the
Symplethesia app UI (nicer than editing JSON), then export. The app has
`applyFingeringJson` (import) + `setFingering` edits + note.hand/fingerOverride,
but NO export. This adds the mirror.

Bonus: the exported, human-corrected file also upgrades Symplethesia's OWN
`eval:inference` fixtures from "SMaPE output (may be wrong)" to "human-verified"
— today that eval trusts raw SMaPE output as ground truth (a real blind spot,
since SMaPE can drop a hand entirely).

## Building blocks (already located)

- Schema: `FingeringJson` / `FingeringJsonNote` in `src/app/fingeringImport.ts`.
- Note model: `note.hand?: 'L'|'R'|'N'`, `note.start` (ticks), `note.pitch`;
  finger via `note.fingerOverride` merged with a computed suggestion in
  `App.recomputeFingering()` (App.ts:3560, builds a merged finger map).
- Time: `ticksToSeconds(ticks, tempoMap, ppq)` in `src/core/utils/time.ts`.
- Download: `downloadBlob(blob, filename)` in `src/app/platform/saveFile.ts`.
- Import wiring to mirror: `applyFingeringJson` calls at App.ts ~4014 (.symple)
  and ~4094 (standalone fingering import / dev menu).

## Design

New `src/app/fingeringExport.ts`:
- Pure `buildFingeringJson(notes, tempoMap, ppq, fingerByNoteId, meta?) -> FingeringJson`.
  Per note (sorted by start): onsetTick=note.start, pitch, hand=note.hand ?? 'N',
  finger=fingerByNoteId.get(id) ?? null, confidence=1.0 (human-reviewed),
  startSec=ticksToSeconds(note.start,…). version=2, offsetSec=0, ppq, source/midi
  from meta or "". Pure + unit-tested.

App.ts: dev-only "Export fingering analysis (dev)" action next to the import
one: resolve the effective finger map (reuse recomputeFingering's merged-map
logic), call buildFingeringJson, JSON.stringify(indent 2), downloadBlob as
`<project>.fingering.json`. Gated behind the same dev flag as import.

## Notes / decisions

- `finger` = effective (override ?? suggestion). Caveat for truth use: only
  human-set hands are verified; fingers are "best available". Hand is the
  primary metric anyway.
- Export includes BOTH onsetTick and startSec so SMaPE `benchmark.py` (matches
  on startSec) works; onsetTicks stay identical to predicted.json (same .symple).
- Whole-project export; trimming to the first ~90s for labeling is a tiny
  SMaPE-side `benchmark.py` helper (separate, Python) — NOT this task.

## Steps
- [x] A. `src/app/fingeringExport.ts` (pure `buildFingeringJson`, reuses `FingeringJson`/`FingeringJsonNote` from fingeringImport.ts) + `tests/app/fingeringExport.test.ts` (4 tests: hand/finger mapping, null finger + 'N' hand defaults, startSec via tempo map, sort by start, meta passthrough).
- [x] B. `index.html`: added `#hmExportFingering` hamburger item ("Export fingering analysis (dev)", hidden by default) right after the import item. `App.ts`: dev-gated visibility toggle (same `isDevMode()` check as import), `export-fingering` menu action wired to new `handleExportFingering()`, which reuses `this.fingering` (the merged suggestion+override map `recomputeFingering` already maintains — no new finger-resolution logic added) and the existing `saveTextFile` helper (same pattern as `handleExport`/`handleExportMidi`), downloading `<exportBaseName>.fingering.json`.
- [x] C. `npx tsc --noEmit` clean; `npx vitest run` — 98 files / 1081 tests passed (no regressions).

## Follow-up (SMaPE repo, separate, not here)
- `benchmark.py trim <truth.json> <max_sec>` to keep only the first-N-seconds
  labeled window (user labels the first ~90s).
