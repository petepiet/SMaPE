"""MIDI parsing: turn a MIDI file into a flat list of note-on events with
absolute tick and second timestamps, using `mido`.

`mido` is imported lazily so this module can be *referenced* (e.g. for type
hints or by selftest.py using synthetic data) without mido being installed.
Only `read_midi_notes` actually needs it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MidiNote:
    onset_tick: int
    pitch: int
    start_sec: float
    duration_sec: float
    velocity: int
    channel: int = 0


@dataclass
class MidiData:
    ppq: int
    notes: list  # list[MidiNote], sorted by start_sec then pitch


def _build_tick_to_sec(mid) -> "callable":
    """Builds a tick -> seconds converter for `mid` (a mido.MidiFile),
    honoring mid-file tempo changes. Shared by `read_midi_notes` and
    `read_pedal_segments` so the tricky tempo-map/accumulation logic exists
    in exactly one place.
    """
    ppq = mid.ticks_per_beat

    # First pass: get absolute tick for every event, per-track, then merge.
    # We build one merged, time-sorted event stream with both tick and
    # second timestamps by combining mido's tempo map ourselves.
    tempo_map: list = [(0, 500000)]  # (tick, microseconds_per_beat), default 120bpm
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo_map.append((abs_tick, msg.tempo))
    tempo_map.sort(key=lambda t: t[0])
    # De-dup / keep first tempo at tick 0
    dedup: list = []
    for tick, tempo in tempo_map:
        if dedup and dedup[-1][0] == tick:
            dedup[-1] = (tick, tempo)
        else:
            dedup.append((tick, tempo))
    tempo_map = dedup

    def tick_to_sec(tick: int) -> float:
        sec = 0.0
        last_tick, last_tempo = 0, tempo_map[0][1]
        for t, tempo in tempo_map[1:]:
            if t >= tick:
                break
            sec += (t - last_tick) * (last_tempo / 1_000_000.0) / ppq
            last_tick, last_tempo = t, tempo
        sec += (tick - last_tick) * (last_tempo / 1_000_000.0) / ppq
        return sec

    return tick_to_sec


def read_midi_notes(path: str) -> MidiData:
    """Parse a MIDI file into absolute-time note-on/off pairs.

    Uses mido's merged-track view (`mido.MidiFile.tracks` iterated with
    running delta times) and mido's own tempo-aware `msg.time` (seconds)
    accumulation when iterating the file directly, which mido provides via
    `MidiFile.__iter__` (yields messages with `.time` = seconds since the
    previous message, already accounting for tempo changes). We also track
    ticks by accumulating raw delta ticks per track and merging by time.
    """
    import mido  # lazy import

    mid = mido.MidiFile(path)
    ppq = mid.ticks_per_beat
    tick_to_sec = _build_tick_to_sec(mid)

    events: list = []  # (abs_tick, msg, channel)
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type in ("note_on", "note_off"):
                events.append((abs_tick, msg))

    # Match note_on -> next note_off (or note_on velocity 0) per (channel, pitch).
    open_notes: dict = {}
    notes: list = []
    for abs_tick, msg in sorted(events, key=lambda e: e[0]):
        key = (getattr(msg, "channel", 0), msg.note)
        if msg.type == "note_on" and msg.velocity > 0:
            open_notes.setdefault(key, []).append((abs_tick, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            stack = open_notes.get(key)
            if stack:
                onset_tick, vel = stack.pop(0)
                notes.append(
                    MidiNote(
                        onset_tick=onset_tick,
                        pitch=msg.note,
                        start_sec=tick_to_sec(onset_tick),
                        duration_sec=max(0.0, tick_to_sec(abs_tick) - tick_to_sec(onset_tick)),
                        velocity=vel,
                        channel=getattr(msg, "channel", 0),
                    )
                )
    notes.sort(key=lambda n: (n.start_sec, n.pitch))
    return MidiData(ppq=ppq, notes=notes)


def write_trimmed_midi(in_path: str, out_path: str, trims: dict) -> None:
    """Rewrites ``in_path`` -> ``out_path``, shortening each note whose
    ``(onset_tick, pitch)`` key appears in ``trims`` (mapping to a new
    duration in seconds).

    The new note_off tick is computed by scaling the note's own onset->offset
    TICK span by the ratio new_duration_sec / original_duration_sec (using
    tick_to_sec only to get the original duration for that ratio) -- this
    avoids needing a global seconds->ticks inverse across tempo changes, and
    is exact under constant tempo (the common case for these renders). Never
    extends a note (ratio is clamped to <= 1) and never trims to zero-length
    (floors at 1 tick). All other events -- including CC64 sustain-pedal --
    keep their exact original absolute tick; only matched note_off events
    move earlier, and each track's delta-times are recomputed from the
    resulting absolute-tick set (a stable sort by tick, so ties keep their
    original relative order and end-of-track markers -- already the largest
    tick in their track -- stay last).
    """
    import mido  # lazy import

    mid = mido.MidiFile(in_path)
    tick_to_sec = _build_tick_to_sec(mid)

    for track in mid.tracks:
        abs_tick = 0
        events = []  # [abs_tick, msg]
        for msg in track:
            abs_tick += msg.time
            events.append([abs_tick, msg])

        open_notes: dict = {}
        for entry in events:
            tick, msg = entry
            if msg.type not in ("note_on", "note_off"):
                continue
            key = (getattr(msg, "channel", 0), msg.note)
            if msg.type == "note_on" and msg.velocity > 0:
                open_notes.setdefault(key, []).append((tick, entry))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                stack = open_notes.get(key)
                if stack:
                    onset_tick, onset_entry = stack.pop(0)
                    trim_key = (onset_tick, msg.note)
                    if trim_key in trims:
                        new_duration_sec = trims[trim_key]
                        orig_duration_sec = tick_to_sec(tick) - tick_to_sec(onset_tick)
                        if orig_duration_sec > 0 and new_duration_sec < orig_duration_sec:
                            ratio = max(0.0, new_duration_sec / orig_duration_sec)
                            new_off_tick = onset_tick + round(ratio * (tick - onset_tick))
                            entry[0] = max(onset_tick + 1, min(tick, new_off_tick))

        events.sort(key=lambda e: e[0])
        prev_tick = 0
        for tick, msg in events:
            msg.time = tick - prev_tick
            prev_tick = tick

    mid.save(out_path)


def read_pedal_segments(path: str) -> list:
    """Collapses a MIDI's CC64 (sustain pedal) events into down/up
    (start_sec, end_sec) intervals, merged across all tracks and sorted.
    Returns [] if the file has no CC64 events at all.

    Note: mido exposes the RAW 0-127 MIDI byte for control-change values
    (unlike @tonejs/midi on the TypeScript side, which normalizes to 0..1 --
    verified separately for that library). The down threshold here is
    therefore 64, the standard MIDI on/off cutoff for a two-state pedal
    model -- not 0.5.
    """
    import mido  # lazy import

    mid = mido.MidiFile(path)
    tick_to_sec = _build_tick_to_sec(mid)

    events: list = []  # (abs_tick, value)
    last_tick = 0
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "control_change" and msg.control == 64:
                events.append((abs_tick, msg.value))
            last_tick = max(last_tick, abs_tick)
    events.sort(key=lambda e: e[0])

    segments: list = []
    down_since = None
    for tick, value in events:
        down = value >= 64
        if down and down_since is None:
            down_since = tick
        elif not down and down_since is not None:
            if tick > down_since:
                segments.append((tick_to_sec(down_since), tick_to_sec(tick)))
            down_since = None
    if down_since is not None and last_tick > down_since:
        segments.append((tick_to_sec(down_since), tick_to_sec(last_tick)))

    return segments
