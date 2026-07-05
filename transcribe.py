"""Audio -> MIDI transcription via the Kong high-resolution piano
transcription model (`piano_transcription_inference`, MIT license --
https://github.com/qiuqiangkong/piano_transcription_inference, checkpoint
hosted on Zenodo). Lets the tool run directly off a video's own audio track
instead of requiring the user to separately supply the exact-performance
MIDI (`--transcribe` in extract_fingering.py).

Heavy (torch) dependency -- imported lazily inside `transcribe_audio` only,
so the rest of the tool (and selftest.py, which must stay pure-numpy) never
needs torch installed. See README "Install (transcription)" for the CPU-only
torch install command and the one-time ~165MB checkpoint download (needs
`wget` on PATH and network access on first use only -- it's cached under
~/piano_transcription_inference_data/ after that).
"""
from __future__ import annotations

# The model's own default (RegressionPostProcessor.onset_threshold in
# PianoTranscription.__init__, verified against the library's source): the
# minimum onset-confidence score for the model to declare a note. None here
# means "leave the library's own default alone" -- see transcribe_audio.
DEFAULT_ONSET_THRESHOLD = 0.3


def filter_note_events(note_events: list, min_velocity: int = 0, min_duration_sec: float = 0.0) -> list:
    """Drops transcribed notes that look like ghost/artifact notes: very
    quiet ones (often pedal resonance or sympathetic-string harmonics rather
    than an actual key press) and implausibly short ones (tracking blips).
    `note_events` is the `est_note_events` list from
    `PianoTranscription.transcribe()`'s return dict (each a dict with
    onset_time/offset_time/midi_note/velocity -- see that function's
    docstring). Pure list filtering, no torch needed -- safe to unit test
    without the heavy transcription dependencies installed.

    Defaults (min_velocity=0, min_duration_sec=0.0) filter nothing, so this
    is a no-op unless the caller explicitly opts in.
    """
    return [
        e for e in note_events
        if e["velocity"] >= min_velocity and (e["offset_time"] - e["onset_time"]) >= min_duration_sec
    ]


def transcribe_audio(
    audio_path: str,
    out_midi_path: str,
    onset_threshold: float | None = None,
    min_velocity: int = 0,
    min_duration_sec: float = 0.0,
) -> None:
    """Transcribes a solo-piano audio file (anything librosa can load: wav,
    mp3, ...) to MIDI, writing the result to `out_midi_path`.

    Uses `PianoTranscription.transcribe()`, which -- verified against the
    library's own source (`utilities.py` `write_events_to_midi`) -- writes a
    clean, constant-tempo (120bpm, 384 ticks/beat) MIDI file containing both
    note events (with real velocities) AND sustain-pedal CC64 events. Both
    come along "for free": the caller re-parses the written file with this
    tool's own `midi_io.read_midi_notes` (so ticks/ppq stay consistent with
    the rest of the pipeline, exactly as with a user-supplied MIDI), and the
    embedded CC64 pedal data is preserved if the file is later imported into
    Symplethesia (see the app's pedal-import support).

    CPU-only inference (no GPU required) -- a few minutes of solo piano
    typically takes on the order of a minute or two on a modern CPU.

    Ghost-note controls (all optional, all no-ops at their defaults so
    behavior is unchanged unless you opt in -- see README "Reducing ghost
    notes" for how to pick values and how to check whether a change actually
    helped, using the Ivory-comparison workflow):
      onset_threshold  -- raises the model's own onset-confidence cutoff
                           (library default 0.3) so it's pickier about
                           declaring a note at all. Higher = fewer false
                           positives, but risks losing quiet/ambiguous real
                           notes. None = leave the library default alone.
      min_velocity      -- drop notes quieter than this (0-127 scale) after
                           transcription -- ghost notes (resonance/harmonics)
                           tend to be quiet.
      min_duration_sec  -- drop notes shorter than this (seconds) after
                           transcription -- tracking blips tend to be very
                           short.
    """
    import os
    import librosa  # already a dependency (used for audio-onset sync too)
    import torch
    from piano_transcription_inference import PianoTranscription, sample_rate
    from piano_transcription_inference.utilities import write_events_to_midi
    import time

    # Kong's model is GRU-heavy, and GRUs on CPU are hurt (not helped) by
    # many threads: on hybrid P/E-core CPUs (e.g. i5-1235U) PyTorch's default
    # of one thread per logical core makes every parallel section wait for
    # the slowest E-core at each sync point. Empirically observed making a
    # ~4-min job take hours. Cap at 4 threads unless the user overrides via
    # TORCH_NUM_THREADS.
    num_threads = int(os.environ.get("TORCH_NUM_THREADS", "0")) or min(4, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    print(f"  Using {num_threads} CPU threads for inference (override with TORCH_NUM_THREADS)", flush=True)

    print("  Loading audio...", flush=True)
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio_duration_sec = len(audio) / sr
    print(f"  Audio loaded: {audio_duration_sec:.1f} seconds", flush=True)

    print("  Loading transcription model (first run downloads a ~165MB checkpoint)...", flush=True)
    transcriptor = PianoTranscription(device="cpu")
    if onset_threshold is not None:
        transcriptor.onset_threshold = onset_threshold
        print(f"  Using onset_threshold={onset_threshold} (library default is {DEFAULT_ONSET_THRESHOLD})", flush=True)

    print(f"  Transcribing {audio_duration_sec:.1f}s of audio (this can take {audio_duration_sec/60:.0f}-{audio_duration_sec/30:.0f} min on CPU)...", flush=True)
    start_time = time.time()
    # Pass midi_path=None so the library doesn't write the file itself --
    # we write it ourselves below, AFTER filtering, so the filtered notes
    # (and not the raw model output) are what ends up in the MIDI file and
    # everything downstream of it.
    result = transcriptor.transcribe(audio, None)
    elapsed_sec = time.time() - start_time
    note_events = result["est_note_events"]
    pedal_events = result["est_pedal_events"]

    print(f"✓ Transcription complete: {len(note_events)} notes detected in {elapsed_sec:.1f}s", flush=True)

    if min_velocity > 0 or min_duration_sec > 0.0:
        before = len(note_events)
        note_events = filter_note_events(note_events, min_velocity, min_duration_sec)
        dropped = before - len(note_events)
        print(f"  Filtered {dropped}/{before} notes below min_velocity={min_velocity}, "
              f"min_duration_sec={min_duration_sec} (likely ghost notes)", flush=True)

    print(f"  Writing {len(note_events)} notes to {out_midi_path}...", flush=True)
    write_events_to_midi(start_time=0, note_events=note_events, pedal_events=pedal_events, midi_path=out_midi_path)
    print(f"✓ MIDI written successfully", flush=True)
