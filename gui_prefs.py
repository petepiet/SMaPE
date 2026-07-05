"""Small persisted-preferences store for gui.py.

Currently stores only which tooltips the user has dismissed ("Don't show
this again"). Deliberately minimal -- no window geometry, no last-used
mode, etc. -- to avoid the store growing into a second source of truth for
things the widgets themselves already track for the session.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFS_PATH = HERE / ".gui_prefs.json"


class Prefs:
    def __init__(self, path: Path = PREFS_PATH):
        self.path = Path(path)
        self._data = {"dismissed_tooltips": []}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._data.update(loaded)
            except (json.JSONDecodeError, OSError):
                pass  # corrupt/unreadable prefs -> fall back to defaults

    def is_tooltip_dismissed(self, key: str) -> bool:
        return key in self._data.get("dismissed_tooltips", [])

    def dismiss_tooltip(self, key: str) -> None:
        dismissed = set(self._data.setdefault("dismissed_tooltips", []))
        dismissed.add(key)
        self._data["dismissed_tooltips"] = sorted(dismissed)
        self._save()

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except OSError:
            pass  # best-effort; a prefs write failure must never crash the GUI
