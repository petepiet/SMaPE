# SMaPE — Symple Midi and Playstyle Extractor

**Turn a piano performance video into a playable, hand-split score.** Point SMaPE at a video of someone playing piano and it works out **which hand plays each note** (and often which finger), then packages everything into a `.symple` bundle you open in one click in [Symplethesia](https://app.symplethesia.com) — ready to practise hands separately.

No MIDI file required: SMaPE can transcribe the notes straight from the video's own audio.

<div style="background: #0a0d14; padding: 20px; border-radius: 8px; display: inline-block;">
  <img src="smape.png" alt="SMaPE Tool" width="500">
</div>

## Get started — just run the app

**You don't need Python or the command line.** Grab the ready-to-run app from the [latest release](https://github.com/petepiet/SMaPE/releases/latest):

- **Windows** — download **`SMaPE-windows-x64.zip`**, unzip it anywhere, and double-click **`SMaPE.exe`**. Nothing to install.
- **Linux** — download **`SMaPE-linux-amd64.deb`** and install it (double-click, or `sudo apt install ./SMaPE-linux-amd64.deb`). Launch **SMaPE** from your apps menu.

Then it's a short wizard:

1. **Add a video** — paste a YouTube (or any yt-dlp) link, or drop in a local file.
2. **Say what it is** — a real piano player, a Synthesia-style training video, or "just extract the MIDI".
3. **Run** — SMaPE downloads the video, transcribes the MIDI if you don't have one, opens a quick calibration window (click a few C/G keys so it knows where the keyboard is), and analyses the hands.

While it runs, a **live view** shows the hand tracking frame-by-frame and a **yellow box marks exactly the area being analysed**, so you can see at a glance that it's working. When it finishes you get a `.symple` bundle — open it in Symplethesia and start playing.

> 💡 **Trying it out?** Use **public-domain / royalty-free music** so you can freely share the results — e.g. a Chopin prelude, Debussy's *Arabesque No. 1*, or a Bach invention. Classical piano is public domain and plentiful on YouTube.

## What's new in v1.0.0

v1.0.0 is a big step up in **hand-assignment accuracy** — getting the left/right split right, which is what makes hands-separate practice actually work.

- **Fixed the #1 tracking failure.** Overhead camera shots (where the hands sit *below* the keys) used to lose the hands entirely because the analysed area clipped the palms. It now includes the whole hand — on a real test video, both-hands detection jumped from **0% → 88%**.
- **Much smarter left/right assignment.** Several cooperating signals now decide which hand played each note — musical register, a bass/melody "voice" model learned from the clearly-tracked passages, and targeted re-detection where a hand was missed — so far fewer notes bleed onto the wrong hand.
- **Skin-blob hand recovery** (optional checkbox). For ballads where both hands play close together and get merged into one, an independent skin-colour tracker finds the two hands and splits them apart.
- **Contrast boost** for dark or low-contrast footage.
- **Clearer previews.** Every sounding note is drawn on the keyboard **coloured by its assigned hand**, and the live/preview view shows the analysed region — so you can eyeball the result instead of guessing.

Everything above is **on by default** (skin-blob recovery is a single optional checkbox) — you don't have to configure anything.

## What it does

- **Which hand plays what:** the core output — a reliable left/right split for hands-separate practice — via video hand tracking (MediaPipe) plus the multi-signal assignment above.
- **Finger numbers too:** where the video is clear, each note also gets a finger (1–5).
- **No MIDI? No problem:** transcribes the notes from the video's audio with the Kong high-resolution piano model.
- **Synthesia videos:** also handles rendered keyboard videos with colour-coded keys instead of real hands.
- **One-click sharing:** exports a `.symple` bundle (MIDI + hand/finger data + song metadata) that opens directly in Symplethesia.

Prefer the command line, or want the full flag reference? Everything the app does is also available via `extract_fingering.py` — see [Command-line use](#command-line-use-optional) further down.

## Windows: portable app (no install)

Each push to `master` builds **SMaPE-windows-x64.zip** via GitHub Actions
(tagged `v*` pushes also publish it as a GitHub Release). Download it,
unzip anywhere, run `SMaPE.exe` — no Python, ffmpeg, or pip needed. The zip
contains:

- `SMaPE.exe` — the GUI
- `app/` — the analysis engine (plain Python scripts)
- `runtime/` — a private embedded Python with the core dependencies
  (mediapipe, opencv, librosa, ...) preinstalled
- `ffmpeg/` — static ffmpeg/ffprobe binaries

The heavy transcription stack (CPU PyTorch + the Kong model package, ~1 GB)
is **not** in the zip. The first time you run the *video2mid* (`--transcribe`)
mode, the GUI offers to download and install it into the bundle's own
`runtime/` — a one-time step that touches nothing outside the SMaPE folder.

## Install

Full install (on the machine that will actually run the analysis over
video). On Debian/Ubuntu the system Python is "externally managed" (PEP 668),
so a **virtual environment is required** — a plain `pip install` fails with
`error: externally-managed-environment`:

```bash
cd SMaPE
python3 -m venv .venv          # if this fails: sudo apt install python3-venv
source .venv/bin/activate      # prompt now shows (.venv)
pip install -r requirements.txt
```

Then always `source .venv/bin/activate` before running the tool
(`deactivate` to leave it).

Requirements: `mediapipe`, `opencv-python`, `numpy`, `mido`, `yt-dlp`, `librosa`.
(mediapipe supports Python ≤3.12 — your 3.12 is fine.)

`librosa` (used for audio-based sync, the default -- see "Sync" below) also
requires the system `ffmpeg` binary to be installed, since it's used to
extract the video's audio track:

```bash
sudo apt install ffmpeg   # Debian/Ubuntu
brew install ffmpeg       # Mac
```

(`ffmpeg` is available on most systems already.) If `ffmpeg`/`librosa` are
unavailable, pass `--sync-method press-moments` to use the older hand-motion
sync method instead, which doesn't need them.

Hand tracking uses MediaPipe's **Tasks API** (`HandLandmarker`), not the
legacy `mp.solutions.hands` API (which recent mediapipe pip releases
0.10.31+ broke/removed -- see mediapipe GitHub issue #6204). Because of
this, `requirements.txt` does not need to pin a specific mediapipe version;
any current pip release works.

The hand-landmark model file (`hand_landmarker.task`, ~10MB) is downloaded
automatically the first time you run the tool (needs network access on
that first run) and cached at `models/hand_landmarker.task`.
If the automatic download fails (e.g. no network access on the analysis
machine), download it manually from:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

and place it at `models/hand_landmarker.task` (relative to the repo root).

### Install (transcription, `--transcribe`, optional)

Only needed if you want the tool to generate the MIDI itself from the
video's audio instead of supplying `--midi`. This is a separate, heavier
install -- skip it if you always have a MIDI file already.

Uses [`piano_transcription_inference`](https://github.com/qiuqiangkong/piano_transcription_inference)
(MIT license; checkpoint hosted on Zenodo, also unencumbered), the "Kong"
high-resolution piano transcription model. It depends on **PyTorch**, which
must be installed with the **CPU-only** wheel first -- a plain
`pip install torch` pulls a multi-GB CUDA build you don't need for this:

```bash
# inside the activated .venv, in addition to the main Install above:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install piano_transcription_inference
```

The first `--transcribe` run downloads a **~165MB checkpoint** to
`~/piano_transcription_inference_data/` (needs network access -- one-time;
the checkpoint is cached after that). CPU inference is fine: a few minutes of solo piano audio typically
takes on the order of a minute or two on a modern CPU, no GPU required.

### Self-test only (no video libs needed)

The core geometry/sync/matching logic is pure numpy and has its own test
suite that runs with **only numpy installed** (no mediapipe/opencv/mido):

```bash
python3 -m pip install --user numpy
python3 selftest.py
# or:
python3 extract_fingering.py --selftest
```

## Command-line use (optional)

> The app (above) is the recommended way to use SMaPE and needs none of this.
> The CLI is here for automation, batch jobs, and people who prefer a terminal.
> It first needs the Python dependencies installed — see [Install](#install).

```bash
python3 extract_fingering.py \
  --video chopin-prelude.mp4 --midi chopin-prelude.mid \
  --out fingering.json
```

`--video`/`--midi` may be omitted entirely — a file dialog opens (rooted at
`~/Downloads`) to pick them. A bare filename is also resolved against
`~/Downloads`. `--video` can be a local file path or a YouTube (or any
yt-dlp-supported) URL — URLs are downloaded first via `yt-dlp` into a
`downloads/` folder next to the output JSON.

If `--out` is omitted, the output JSON defaults to the **same directory and
basename as `--midi`**, with a `.fingering.json` suffix -- e.g.
`--midi "/home/user/songs/piece.mid"` (no `--out`) writes
`/home/user/songs/piece.fingering.json`.

Full CLI:

```
--video PATH_OR_URL       local file or yt-dlp URL (file dialog if omitted)
--midi PATH               exact same performance as --video (file dialog if omitted; not used with --transcribe)
--transcribe              generate the MIDI from --video's own audio instead of requiring --midi (see "Transcription")
--midi-only               transcribe MIDI from --video's audio and stop there -- no calibration, no hand tracking,
                          no fingering/hand analysis. Implies --transcribe. Still writes a fingering JSON/.symple
                          bundle (empty notes list, "midiOnly": true) so the MIDI stays one-step-importable.
--onset-threshold F       --transcribe only: raise the model's onset-confidence cutoff (default 0.3) to reduce
                          ghost notes (see "Reducing ghost notes")
--min-velocity N          --transcribe only: drop transcribed notes quieter than this, 0-127 (default 0 = off)
--min-duration SECONDS    --transcribe only: drop transcribed notes shorter than this (default 0.0 = off)
--out PATH                output JSON path (default: <midi-directory>/<midi-basename>.fingering.json, or
                          ~/Downloads/<video-title>.fingering.json when --transcribe has no --midi to key off)
--offset SECONDS          manual video-vs-MIDI offset (video_time = midi_time + offset); seeds the aligner
--sync-method {audio,press-moments}  how to seed the offset (default: audio)
--no-align                skip the interactive video/MIDI alignment step (before tracking)
--no-bundle               don't write the .symple bundle (MIDI + fingering JSON) alongside --out
--preview                 also render preview.mp4 (keyboard overlay + fingertips + MIDI audio)
--fps N                   frame sampling rate for hand tracking (default 30)
--min-hand-confidence F   MediaPipe hand-detection confidence threshold, 0..1 (default 0.3; lower, e.g. 0.15,
                          if a hand frequently goes undetected) -- see "Hand tracking" note below
--confidence-threshold F  drop matches below this confidence, 0..1 (default 0.0 = keep all)
--no-clahe                skip CLAHE contrast enhancement of the tracker's input (helps on dark video); on by default
--no-midi-recover         skip the MIDI-anchored recovery pass (re-runs MediaPipe cropped around keys where
                          a note sounds but no hand was tracked); on by default
--no-midi-hand-prior      skip the MIDI register-clustering hand prior that fixes low-confidence L/R where
                          hands are close together; on by default
--no-merge-split          skip recovering a second hand when two register-separated hands merged into one
                          (uses the MIDI cluster split point); on by default
--no-voice-separation     skip the video-seeded bass/melody voice model that reassigns ambiguous notes
                          (register + pitch recurrence + lowest-voice); on by default
--blob-recover            enable skin-blob (YCrCb) hand recovery for hands that touch/overlap in the same
                          register; OFF by default (may need per-video skin tuning)
--no-reconstruct          skip the kinematic outlier-removal pass that drops teleporting fingertip glitches
                          before matching; on by default
--flip-handedness         swap detected L/R hand labels (see "Hand tracking" note below)
--low-pitch N             (optional) leftmost overlay key; default auto from MIDI
--high-pitch N            (optional) rightmost overlay key; default auto from MIDI
--selftest                run the pure-numpy self-tests and exit
```

> **Note on `--low-pitch`/`--high-pitch`:** with the C/G-key calibration
> below these no longer affect the mapping or the results — they're
> auto-derived from the MIDI and only size the on-screen keyboard overlay.
> They remain as optional overrides but you normally never set them.

### Hand tracking: left/right swapped?

If `--preview` shows the detected hands swapped (left tagged as right or
vice versa), pass `--flip-handedness`. This is a known MediaPipe quirk: its
handedness label assumes a mirror-selfie camera view, which doesn't always
match an overhead shot of the performer's own hands depending on camera
placement.

### Hand tracking: a hand frequently missing entirely?

If `--preview` shows one hand's dots vanishing for long stretches even
though it's clearly visible and playing (very low mean confidence, and a
`reconciliation:` summary line flagging a large fraction of notes as
`no-finger-support`, are the symptoms -- caught this way on a real run:
confidence dropped to ~0.02-0.08 and 86% of notes were flagged, while the
keyboard calibration overlay itself was pixel-accurate throughout), this is
MediaPipe's hand *detector* dropping a hand, not a calibration problem --
check by extracting a few frames from `preview.mp4` at the low-confidence
timestamps (`ffmpeg -ss T -i preview.mp4 -frames:v 1 out.jpg`) and looking
for whether both hands have colored fingertip dots.

**Likely causes:** the typical overhead piano shot itself (keys at the
bottom of the frame, hands only partly visible -- measured on a real cover
video: both hands seen in just 2% of frames at threshold 0.5, vs 90% at 0.3
and 93% at 0.15), and monochrome/black-and-white source video (MediaPipe's
hand landmark model is trained mostly on color imagery and partly relies on
skin-tone cues that desaturated footage lacks).

**Built-in mitigation:** the default `--min-hand-confidence` is 0.3
(deliberately below the library's own 0.5, for exactly the overhead-shot
reason above). After tracking, the tool also measures the fraction of frames
where BOTH hands were seen, prints it, and -- when it comes back under 20%
with headroom left -- automatically retries the tracking once at 0.15,
keeping whichever pass saw both hands more often.

**Manual fix if it still misses a hand:** lower `--min-hand-confidence`
further (e.g. `--min-hand-confidence 0.15`). This relaxes both
`min_hand_detection_confidence` and `min_hand_presence_confidence`, trading
a higher false-detection rate for fewer missed real hands. Available in the
GUI too. Raise it back toward 0.5 if you see ghost hand detections instead.

The tool prints a summary when done:

```
Summary:
  notes matched: 142/150
  mean confidence: 0.81
  offset used: 1.35s
```

## Calibration (multi-point, "click the C and G keys")

Every run opens a calibration window on the video's first frame. You click
the **centre of several white keys spread across the keyboard** — clicking
**C and G keys** is easiest because they're quick to identify by eye — and
for each click a small picker asks which key it was (C1…C8, G1…G8). Place at
least **4** points (more, spread wider, is better); the tool fits a **1-D
projective row map** from your clicks to screen pixels and draws the inferred
keyboard live so you can verify it lines up before finishing.

Controls:

- **Left-click** an empty spot → place a point (then pick which key).
- **Left-drag** a point → reposition it (generous 20 px grab radius).
- **Right-click**, or **Del / Backspace / x** on the selected point → remove
  a misplaced point.
- **Tab** cycles the selected point; **arrow keys** nudge it by 1 px.
- **U** undoes the last-placed point.
- **Esc** finishes (needs 4+ points).

**Why C/G clicks instead of the old 4 corners:** the corner method required
you to also state which pitch sat at each end (`--low-pitch`/`--high-pitch`),
and a single row of corner clicks can't constrain camera perspective in the
near↔far direction. Clicking real, labelled keys anchors actual pitches
directly to pixels, so the keyboard range no longer matters (it's auto-set
from the MIDI just to size the overlay), and the fit is well-posed from one
row. See the docstrings in `keyboard.py` (`_fit_row_projective`,
`interactive_calibrate`) for the geometry.

The fitted map is stored on the `Calibration` as a `row` field. The
`--calibration` path is currently a compatibility placeholder — calibration
runs interactively each time (it is not loaded from disk). Older 4-corner
`calibration.json` files still load and work via the corner-homography path;
only the interactive flow uses the row map.

**Stretch goal (not implemented):** automatic black/white key edge
detection from the image, to skip manual clicking entirely.

## Sync: video <-> MIDI offset

Because the video and MIDI are the **same performance**, the alignment
between them is a single constant time offset -- no time-warping/tempo
matching is needed. `video_time = midi_time + offsetSec`.

- **Interactive alignment (default, before tracking):** the offset is first
  seeded (manual or audio, below), then an alignment window opens where the
  video plays with the keyboard overlay and each note flashes its key at
  `start + offset`, while the **MIDI is played as beeps** (via `ffplay`) so
  you can *see and hear* whether the flashes/beeps land on the visible key
  presses. Nudge with ←/→ (10 ms) or ↑/↓ (50 ms), SPACE play/pause, `a`/`d`
  seek, `R` restart, ENTER to accept. This lets you fix the offset *before*
  committing to the slow per-frame hand tracking. Pass `--no-align` to skip
  it (e.g. headless runs).
- **Manual:** pass `--offset SECONDS` directly (e.g. if you already know
  how many seconds into the video the MIDI's t=0 falls). Overrides
  `--sync-method` entirely.
- **`--sync-method audio` (default):** the tool extracts the video's audio
  track (via `ffmpeg`) to a temporary `.wav` file, then uses `librosa`'s
  onset detection to find the first significant piano-sound onset. That
  onset time is compared against the MIDI's first note (`notes[0].start_sec`)
  to compute `offsetSec = audio_first_onset_time - midi_first_note_time`.
  This listens to the actual performance audio, so it's far more reliable
  than inferring key-presses from hand motion. Requires `ffmpeg` + `librosa`
  (see Install above). The temp `.wav` is deleted after use.
- **`--sync-method press-moments` (legacy fallback):** the tool detects
  "press moments" in the video -- moments where a tracked fingertip's
  downward motion stops (a local minimum of a descent), a proxy for the
  finger reaching a key -- and cross-correlates the density of those
  moments against the density of MIDI note onsets over a range of candidate
  offsets (±5s by default, 50ms bins), picking the offset that maximizes
  alignment. This is noisier than audio-based sync (it depends on accurate
  hand tracking); use it only if audio extraction fails or the video has no
  usable audio track. The estimate is printed either way; pass `--offset` to
  override it if it looks wrong.
- **`--preview`:** renders `preview.mp4` with the inferred **keyboard
  overlay** (white/black key ticks + note labels), detected fingertips
  (colored dots `L1`..`R5`), a red ring over the expected key of the note
  nearest each frame's synced time, **and a MIDI beep audio track** muxed in
  (via `ffmpeg`) so you can verify sync by ear as well as by eye.

## Transcription (`--transcribe`)

Don't have (or don't want to separately find) an exact-performance MIDI for
the video? `--transcribe` generates one straight from the video's own audio:

```bash
python3 extract_fingering.py --video performance.mp4 --transcribe
```

`--midi` is not needed (and is ignored if given) in this mode. This uses the
Kong high-resolution piano transcription model (see "Install (transcription)"
above for the one-time setup) to detect note pitch, onset/offset timing,
**velocity**, and **sustain-pedal** events directly from the audio -- a
solo-piano-specific model, meaningfully more accurate for this purpose than a
general-purpose audio-to-MIDI tool.

The transcribed MIDI is written to `<out-basename>.transcribed.mid` (next to
the output JSON) and then re-read through this tool's own `midi_io.py`
exactly like a user-supplied MIDI, so the rest of the pipeline (calibration,
alignment, hand tracking, matching) is completely unchanged. The written
`.transcribed.mid` also carries the model's sustain-pedal events as CC64 --
if you later import that MIDI into Symplethesia, the pedal comes along
automatically (see the app's pedal support).

**Offset is 0 by construction:** because the MIDI is derived from this exact
video's own audio, its timeline already IS the video's timeline -- there's no
separate offset to estimate, so the audio-onset (or press-moments) auto-sync
step is skipped. `--offset` still works as a manual override, and the
interactive alignment step (see "Sync" above) still runs by default in case
the model's own timing benefits from a small manual nudge; pass `--no-align`
to skip it.

**Honest expectations:** this is AI transcription, not a perfect transcript --
expect occasional missed/extra notes, quantization wobble, and pedal-inflated
note durations, especially on complex or fast passages. It's a strong draft to
clean up in Symplethesia's editor (or a future automated reconciliation pass
against the video's hand-tracking, which can help de-pedal and reject
ghost notes -- see `ai/tasks/003-amt-pedal-transcription/PLAN.md` Phase D),
not a finished score.

### Reducing ghost notes

Three knobs, all no-ops at their defaults (identical output to not passing
them at all), for when the transcription has more false-positive ("ghost")
notes than you'd like:

```bash
--onset-threshold 0.4    # model default is 0.3 -- higher = pickier about declaring a note at all
--min-velocity 15        # drop notes quieter than this (0-127) -- ghosts tend to be quiet (pedal resonance/harmonics)
--min-duration 0.05      # drop notes shorter than this (seconds) -- ghosts tend to be very short blips
```

`--onset-threshold` changes what the model itself decides is a note (fewer
false positives, but risk of losing quiet/ambiguous real notes too);
`--min-velocity`/`--min-duration` are a cheap independent post-filter applied
after transcription, on the assumption that ghost notes tend to be quiet
and/or short-lived. All three are exposed in the GUI too (enabled once
Transcribe is ticked).

**There's no universal "right" value** -- what counts as a ghost note depends
on the recording and the model's behavior on it. Tune by comparing against a
reference: if you have an export from another transcription tool (e.g.
Ivory) for the same video, a quick note-level diff (match by pitch + onset
time, after finding the constant offset between the two files' timelines --
they often don't start at the same silence-trimmed point) will show you
concretely how many notes disagree and whether a given threshold change
actually helped, rather than guessing.

## Batch processing (`--batch`)

Process a whole YouTube playlist (or several loose video URLs) in one
command, targeting `--transcribe` mode:

```bash
python3 extract_fingering.py --batch "https://youtube.com/playlist?list=..." --transcribe
python3 extract_fingering.py --batch video1_url video2_url urls.txt --transcribe --render
```

`--batch` accepts one or more SOURCEs: playlist URLs, single video URLs, or
a path to a `.txt` file with one URL per line -- any mix, in any order. Runs
in two phases:

- **Phase 1 (interactive, front-loaded):** for each video, resolves its
  channel's calibration/colour storage key (same per-channel cache as a
  normal single-video run -- see "Calibration" above and `--render`'s
  colour picking below) and prints a **homogeneity report**: how many
  videos already have saved calibration/colours for their channel and can
  be reused, versus how many need a manual pick. Only the *first* video of
  each not-yet-saved channel opens an interactive calibration window (and,
  in `--render` mode, the hand-colour picker); every later video from that
  same channel reuses what was just saved, headlessly.
- **Phase 2 (unattended):** transcribes and analyzes every video in the
  batch back-to-back, with `--no-align` and the saved per-channel
  calibration/colours applied automatically -- no windows, no prompts. One
  video failing (bad download, transcription error, etc.) is caught and
  logged rather than aborting the rest of the batch.

The batch always ends with a summary (`batch done: X ok, Y failed, Z
skipped`) and a list of any failures.

**Why `--transcribe` only, for now:** batch mode assumes `--no-align` is
safe, which is only true when the offset is 0 by construction (see
"Transcription" above) -- a user-supplied `--midi` per video would need its
own offset estimated or supplied per video, which is a later extension.

**Consistency checks keep "set up once per channel" safe**, so a batch
never silently reuses the wrong camera angle or colour theme:
- camera angle: the existing headless `calibrations_agree` check (a fresh
  CV read of the new video must still line up with the saved calibration);
- hand colours (`--render` only): `colors_agree` (in `render_hands.py`)
  matches a few sampled lit-key colours from the new video against the
  saved LH/RH white-key references by hue distance (25° tolerance) --
  deliberately an easier, more robust check than clustering from scratch,
  since it's matching against two already-known target colours rather than
  discovering new ones.

## Matching

For each MIDI note onset:

1. Compute the synced video time (`onset_time_sec + offsetSec`).
2. Interpolate every tracked fingertip's screen position to that exact time
   (linear interpolation between the two nearest sampled video frames).
3. Convert both the note's pitch and every fingertip to a **continuous
   white-key index** (`screen_to_white_index` / `pitch_to_white_index`) —
   i.e. match in **depth-invariant keyboard-x space**, not raw 2-D pixels.
   A finger identifies a key by its position *along* the keyboard, not by how
   far down the key's depth it presses; matching in 2-D pixels penalises the
   natural depth spread of fingers and collapses confidence for every note.
4. Pick the fingertip nearest in key-index; emit a confidence in `[0, 1]`
   that decays with the mismatch in **white keys** (`confidence_from_distance`
   in `match.py`, characteristic scale `KEY_MATCH_SCALE ≈ 0.8` of a key). So
   confidence is now meaningful in key units: **>0.5 ≈ finger within half a
   key**; ~0.3 ≈ about one key off on average.

**Chord handling:** MIDI notes within 30ms of each other are grouped and
matched together (`group_simultaneous` in `match.py`) using greedy
nearest-first assignment so that no `(hand, finger)` pair is reused within
one chord (`resolve_chord_conflicts`).

**Held notes:** because matching is done independently per onset against
the fingertip positions *at that onset's synced time*, a sustained note
naturally keeps whichever finger/hand was nearest at the moment it was
struck (fingertip tracking after the onset doesn't change the recorded
assignment).

## Output JSON schema

```json
{
  "version": 2,
  "source": "<video path or url>",
  "midi": "<midi path>",
  "ppq": 480,
  "offsetSec": 0.0,
  "notes": [
    {
      "onsetTick": 0,
      "pitch": 60,
      "hand": "L",
      "finger": 1,
      "confidence": 0.87,
      "startSec": 0.0,
      "correctedEndSec": 0.485,
      "soundEndSec": 1.9,
      "keyDurationSec": 0.485,
      "soundDurationSec": 1.9,
      "releaseReason": "same_finger_new_note"
    }
  ]
}
```

(This is the hand-tracking path's per-note shape; `--render` mode still uses
a separate `durationSec`-based shape -- see "Note release correction" below
for why the two differ.)

- `onsetTick` -- tick position of the note-on, in the MIDI's own `ppq`
  (matches Symplethesia's tick units when `ppq` agrees; the importer joins
  on pitch + nearest tick within a tolerance regardless).
- `pitch` -- MIDI note number (0-127).
- `hand` -- `"L"` or `"R"`.
- `finger` -- `1`-`5` (1 = thumb, 5 = pinky).
- `confidence` -- `0.0`-`1.0`, derived from fingertip-to-key pixel distance.
- `flags` -- optional list of reconciliation flags (e.g.
  `"no-finger-support"`); present only when non-empty.
- `startSec` -- the note's onset, in the MIDI/note timeline (`note.start_sec`).
- `correctedEndSec` -- the estimated PHYSICAL key-release time (when the
  finger actually left the key), in the same timeline.
- `soundEndSec` -- the original, unmodified audio note-off (`startSec +`
  the MIDI/AMT duration) -- when sustain pedal is down this is later than
  the key was actually released, and that's intentional: it's a correct
  description of the *sound*, not the key.
- `keyDurationSec` / `soundDurationSec` -- `correctedEndSec`/`soundEndSec`
  minus `startSec`, provided pre-computed for convenience.
- `releaseReason` -- which rule produced `correctedEndSec`; see below.

### Note release correction

Audio transcription reports when a note stops being *audible*, not when the
key was physically released -- sustain pedal (or room resonance) can make a
short staccato note look like it lasted a second or more. This tool
therefore tracks **two distinct durations** per note:

- **sound duration** (`soundDurationSec`) -- unchanged from the audio/MIDI
  transcription; this is what should drive audio-accuracy comparisons.
- **key duration** (`keyDurationSec`) -- a corrected, always-<=-sound-duration
  estimate of how long the finger actually held the key down; this is what
  should drive on-screen key-release animation/visuals.

`correctedEndSec` is chosen as the **smallest** of several candidate caps
(ties broken toward the higher-priority rule):

1. **same-finger rule** -- a real hand can't still be holding this key down
   once the SAME (hand, finger) is already pressing its next note, so that
   next note's onset (minus a small safety margin, `--release-margin-ms`,
   default 15ms) caps this note's release.
2. **same-pitch rule** -- likewise, a key must come back up before it can be
   struck again, so the next note at the same pitch (restrike) caps it too.
3. **visual key-release** -- if the video's own hand tracking observed the
   finger leaving the key before the audio-reported offset (the existing
   sustain-pedal de-pedal check in `reconcile.py`), that trusted observation
   caps it.
4. **original MIDI/audio offset** -- if nothing above applies, the original
   sound end is used (key duration == sound duration).
5. **fallback estimate** -- only when no MIDI duration exists at all
   (start + 0.3s).

Sustain pedal never enters this computation directly -- it only ever
lengthens `soundDurationSec`, never `keyDurationSec`.

## Output bundle (`.symple`)

Alongside `--out`'s fingering JSON, every run also writes a **`.symple`
bundle** next to it (same basename, e.g. `piece.fingering.json` ->
`piece.symple`) -- pass `--no-bundle` (or untick the GUI checkbox) to skip it.

A `.symple` file is a plain **ZIP** (written with Python's stdlib `zipfile`,
no new dependency) containing:

```
manifest.json    -- format version, generator, source video/midi, contents list
song.mid         -- the exact MIDI used for this analysis
fingering.json   -- the fingering analysis output (same as --out)
```

It's a plain zip specifically so both sides get a mature reader for free:
`zipfile` here, and Symplethesia's existing 7z-wasm-based archive reader in
the browser (`src/core/library/archive.ts`) on the app side -- no new
dependency there either. The manifest's `contents` list lets future versions
add more files (e.g. pedal data) without breaking older readers, which only
require `song.mid` to be present and treat everything else as optional.

**Loading a bundle in Symplethesia:** File -> Open -> pick the `.symple` file
(or drag it in). The app imports the bundled MIDI through the normal import
wizard, then automatically applies the bundled fingering -- equivalent to
importing the MIDI and using "Import fingering analysis (dev)" separately,
but one step. If the wizard is cancelled or the import fails, the fingering
is not applied (and nothing in the previously-open song is touched).

## How the hand-assignment works (v1.0.0)

Getting left/right right is the whole point, so SMaPE stacks several
independent signals — each a step that only steps in where the ones before it
were unsure. All are on by default except skin-blob recovery.

| Step | What it does |
|------|--------------|
| **Wider analysis area** | Includes the whole hand (palm + wrist), so overhead shots don't clip the hands and lose detection |
| **Contrast boost (CLAHE)** | Lifts detail in dark/low-contrast footage before tracking |
| **MediaPipe tracking** | Finds hands and fingertips per frame |
| **Targeted re-detection** | Where a note sounds but no hand was tracked there, re-runs the detector zoomed in on that key |
| **Register prior** | Bass → left, treble → right, with a boundary that follows the music |
| **Merge-split** | When two register-separated hands merge into one detection, rebuilds the missing one |
| **Voice model** | Learns each hand's bass/melody pattern from the clear passages and applies it to the ambiguous ones |
| **Skin-blob recovery** *(optional)* | An independent skin-colour tracker that splits two hands playing close together |
| **Kinematic cleanup** | Drops physically-impossible "teleporting" fingertip glitches |

## Earlier improvements

- **Metadata support:** Song metadata (Artist, Title, Genre, Difficulty) is stored in `.symple` bundles and imported directly into Symplethesia
- **Auto-fill metadata:** Extracts song info from video titles intelligently (supports "Artist - Song" patterns)
- **Better feedback:** Progress messages show what the tool is doing at each phase (download, calibration, transcription)
- **Enhanced UI:** Dialogs auto-fit to screen size; frame navigation with Page Up/Page Down (±150 frames)
- **Octave shifting:** Shift the entire keyboard calibration overlay by octave (< / > keys) when auto-detection is off by one

## GUI

A small desktop GUI (`gui.py`) wraps the CLI above so you don't have to
remember the flags. It's built with **Tkinter** (Python stdlib), so it has
no required dependencies beyond Python itself.

### Launching

```bash
python3 gui.py
```

`tkinter` itself doesn't need mediapipe/opencv/etc., so this launch command
works with **plain system Python** -- you do not need to be inside the
venv just to open the window. However, when you click **Run**, the GUI
needs the actual analysis dependencies to do anything useful. It handles
this automatically: at startup it looks for `.venv/bin/python` next to
`gui.py` and, if found, uses *that* interpreter to run
`extract_fingering.py` as a subprocess. If no `.venv` is found, it falls
back to the system Python and shows a yellow warning banner at the top of
the window telling you to install dependencies first (see "Install"
above) -- it will not silently fail or hide the problem.

Because the tool's calibration and alignment steps open blocking OpenCV
windows (click the C/G keys; watch/hear the alignment), the GUI always runs
`extract_fingering.py` as a **subprocess**, never by importing it, so those
windows can pop up and behave normally.

### Flow

The GUI is a multi-step wizard, styled to match the main Symplethesia app's dark theme:

1. **Video** -- paste a YouTube (or yt-dlp-supported) URL, **Open file...**
   to browse (defaults to `~/Downloads`), or drag a local file onto the
   field. Click **Next →**.
2. **What kind of video is this?** -- three buttons, each hoverable for a
   tooltip explaining what it needs/produces:
   - **Piano player** -- real hands on a real piano; you supply the
     exact-performance MIDI file on the next screen.
   - **Synthesia training video** -- a rendered keyboard with lit,
     colour-coded keys (no real hands); MIDI is transcribed from audio,
     hand (not finger) comes from key colour. See "Synthesia-render
     support" below.
   - **Extract MIDI only** -- transcribes MIDI from audio and stops there;
     no calibration, no hand/finger analysis (`--midi-only`).
3. **Metadata** -- fill in song information (Artist, Title, Genre, Difficulty). The **Auto-fill** button parses the video title to populate these fields automatically. Click **Next →**.
4. **Run** -- the MIDI-file field only appears here for "Piano player" mode.
   Click **Run**; stdout/stderr streams live into the scrollable log box.
   Click **Stop** to terminate a running analysis. When the process exits,
   the status line turns green with `Done -- wrote <path>` on success, or
   red with the failure/exit code on error (full details remain in the log
   above). "Open output folder" opens the output JSON's containing folder
   in your file manager. **← Back** returns to previous screens at
   any point. **↻ Restart** returns to the video selection screen.

A tooltip's **Don't show this again** checkbox persists across launches (in
`.gui_prefs.json`, next to `gui.py`, gitignored).

### Settings (gear icon)

Everything not covered by the 3-step flow lives behind the **⚙** icon,
pinned bottom-right and clickable at any time -- including mid-run, since it
opens as a separate, non-blocking window rather than a modal dialog. It's
grouped into labelled sections; the defaults are tuned so most people never
need to touch it.

Under **Hand tracking** you'll find the accuracy toggles from
["How the hand-assignment works"](#how-the-hand-assignment-works-v100) as
plain checkboxes — all on by default except one:

- **MIDI-anchored hand recovery** — re-find a hand the tracker missed (on).
- **Skin-blob hand recovery** — for hands that touch/overlap in the same
  register (**off** by default; turn it on for close-hands ballads where one
  hand is under-tracked).
- **Enhance contrast for detection (CLAHE)** — for dark video (on).
- **Live tracking view** — show the hand detection in the run screen (on).

The rest covers Transcribe/Render overrides, MIDI file, FPS, Offset, Sync
method, Min hand confidence, Confidence threshold, alignment, preview video,
`.symple` bundle, flip render hand colours, and the three ghost-note fields
(Onset threshold, Min velocity, Min duration -- see "Reducing ghost notes").

> The Low/High pitch fields and the on-screen 88-key picker were **removed** —
> with C/G-key calibration the range is auto-derived from the MIDI and no
> longer affects results.

### Drag-and-drop (optional)

By default the Video/MIDI drop zones work as simple "click to Browse"
buttons. For real OS-level drag-and-drop, install the optional
[`tkinterdnd2`](https://pypi.org/project/tkinterdnd2/) package:

```bash
pip install tkinterdnd2
```

(Install this into whichever Python environment you'll use to *launch*
`gui.py` -- it's unrelated to the `.venv` used for the actual video
analysis subprocess.) If `tkinterdnd2` isn't installed, `gui.py` detects
this at startup and falls back gracefully -- no crash, no required
dependency.

## Importing into Symplethesia

See `src/app/fingeringImport.ts` in the main app. In short: open the
Symplethesia app with the dev flag on (`?dev=1` in the URL, or
`localStorage.setItem('sympl.dev', '1')`), open the hamburger menu, and use
"Import fingering analysis (dev)" to pick the `fingering.json` produced by
this tool. It joins each JSON note to a project note by exact pitch match +
nearest `note.start` within `ppq/8` ticks, then applies hand + finger
assignments in two undo steps.

Alternatively, open the `.symple` bundle directly in Symplethesia (File → Open)
for one-step import with metadata and bundled MIDI.

## Support & Feedback

- **GitHub:** [github.com/petepiet/SMaPE](https://github.com/petepiet/SMaPE) — report issues, request features
- **Symplethesia:** [app.symplethesia.com](https://app.symplethesia.com) — the companion piano-learning app
- **Support the project:** [ko-fi.com/pieterg](https://ko-fi.com/pieterg)

---

**Created by Pieter Geljon**
