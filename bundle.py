"""Writes a `.symple` bundle: a plain ZIP containing the MIDI + fingering
analysis (and, in future phases, other derived files like pedal data) for a
single song, so Symplethesia can load the whole result in one step instead of
importing the MIDI and separately picking a fingering JSON.

ZIP (not a bespoke format) so both sides get a mature, well-tested reader for
free: Python's stdlib `zipfile` here, and the app's existing 7z-wasm-based
archive reader (`src/core/library/archive.ts`) in the browser -- no new
dependency on either side. `manifest.json` records the format version and the
contents list so the reader can evolve (extra files, renamed entries) without
breaking old bundles, and old readers can ignore files they don't understand.
"""
from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime, timezone

BUNDLE_VERSION = 1


def write_symple_bundle(
    out_path: str,
    midi_path: str,
    fingering_json_path: str,
    source_video: str | None = None,
    extra_files: dict | None = None,
    metadata: dict | None = None,
) -> str:
    """Writes ``out_path`` (conventionally ending in `.symple`) containing:
      - manifest.json  -- format version + contents list + provenance + metadata
      - song.mid        -- copy of the MIDI actually used for this analysis
      - fingering.json  -- the fingering analysis output

    ``metadata`` (optional): ``{"artist": str, "title": str, "genre": str}``
    for song metadata (Symplethesia uses this for library entries and PDF export).

    ``extra_files`` (optional): ``{archive_name: source_path}`` for any
    additional derived files a later phase wants to include (e.g. pedal
    data) -- written as-is and listed in the manifest, so older bundles
    without them still load fine in readers that check for presence.

    Returns ``out_path``.
    """
    contents = ["song.mid", "fingering.json", "DISCLAIMER.txt"]
    if extra_files:
        contents.extend(sorted(extra_files.keys()))

    manifest = {
        "version": BUNDLE_VERSION,
        "generator": "piano-fingering-tool",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "source": {"video": source_video, "midi": os.path.basename(midi_path)},
        "contents": contents,
    }

    # Add metadata if provided
    if metadata:
        manifest["metadata"] = {k: v for k, v in metadata.items() if v}  # exclude empty fields

    # Find DISCLAIMER.txt in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    disclaimer_path = os.path.join(script_dir, "DISCLAIMER.txt")

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.write(midi_path, "song.mid")
        zf.write(fingering_json_path, "fingering.json")
        if os.path.exists(disclaimer_path):
            zf.write(disclaimer_path, "DISCLAIMER.txt")
        if extra_files:
            for arcname, path in extra_files.items():
                zf.write(path, arcname)

    return out_path


def default_bundle_path(fingering_json_path: str) -> str:
    """`<dir>/<basename>.fingering.json` -> `<dir>/<basename>.symple`."""
    base = fingering_json_path
    for suffix in (".fingering.json", ".json"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base + ".symple"
