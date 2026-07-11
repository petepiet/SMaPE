# Hand-assignment benchmark

Measures how accurately SMaPE assigns notes to the left/right hand (and,
secondarily, fingers) against a human-corrected ground truth. This is the
one metric that matters most for judging whether a new tracking layer
(Kalman filter, optical flow, background subtraction, etc.) actually helps.

## Layout

```
benchmark/
  <song-name>/
    truth.json       # hand-corrected reference (ground truth)
    predicted.json   # latest SMaPE fingering.json output (regenerate to re-measure)
    notes.md          # optional: video url + how truth was made
```

## Authoring workflow

Ground truth is authored by **correcting a copy of SMaPE's own output**,
not by labeling notes from scratch — that would be too tedious. The idea:
SMaPE's hand guesses are already right most of the time, so start from its
prediction and only flip what's wrong.

1. Run SMaPE on a video to produce a `fingering.json`.
2. Bootstrap a truth template pre-filled with SMaPE's current guesses:

   ```
   .venv/bin/python3 benchmark.py init fingering.json benchmark/<song-name>/truth.json
   ```

3. Open `benchmark/<song-name>/truth.json` and **correct the wrong `hand`
   values by hand** (and `finger` values too, if you want finger-accuracy
   to mean something). Leave everything that's already correct alone.
4. Copy the same `fingering.json` to `benchmark/<song-name>/predicted.json`.
5. Run the benchmark:

   ```
   .venv/bin/python3 benchmark.py
   ```

   This scans every subdirectory of `benchmark/` that has both `truth.json`
   and `predicted.json`, scores each song, and prints a per-song table plus
   a micro-averaged aggregate row.

## Re-measuring after improving SMaPE

`truth.json` is the fixed reference — leave it alone once corrected.
To measure whether a change to SMaPE's hand-assignment logic helped:

1. Re-run SMaPE on the same video to get a fresh `fingering.json`.
2. Overwrite `benchmark/<song-name>/predicted.json` with it.
3. Run `.venv/bin/python3 benchmark.py` again and compare the hand/finger
   accuracy numbers to the previous run.

## Metrics

- **coverage** — fraction of truth notes that could be matched to a
  predicted note (same pitch, nearest onset within a small tolerance).
  Low coverage usually means the prediction's transcription/timing
  diverged from truth, not a hand-assignment problem.
- **hand accuracy** — of the *matched* notes, the fraction whose predicted
  hand equals the truth hand. This is the primary metric.
- **finger accuracy** — of the matched notes where truth also has a
  finger label, the fraction whose predicted finger equals the truth
  finger. Secondary metric; notes with no finger label in truth are
  excluded from the denominator.
- **confusion (L_as_R / R_as_L)** — directional breakdown of hand
  mistakes, useful for spotting a systematic left/right bias.

## CLI reference

```
.venv/bin/python3 benchmark.py                     # scan ./benchmark, print table
.venv/bin/python3 benchmark.py <dir>                # scan a different directory
.venv/bin/python3 benchmark.py init <fingering.json> [out.json]
                                                     # bootstrap a truth.json template
```
