"""Hover-tooltip helper for gui.py.

Tkinter has no built-in tooltip widget. This binds a small delayed popup
to a widget's <Enter>/<Leave> events, keyed by a stable string so
dismissal ("Don't show this again") can persist across launches via a
gui_prefs.Prefs instance.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from gui_prefs import Prefs


class Tooltip:
    # Every live instance, so a page switch (pack_forget on the hovered
    # widget's ancestor) can force-close any open popup even though hiding
    # a widget this way does not fire a real <Leave> event -- without this,
    # a tooltip shown right before navigating away is orphaned: its Toplevel
    # is a separate top-level window, not a child of the hidden page, so it
    # stays stuck on screen indefinitely.
    _instances: list["Tooltip"] = []

    def __init__(self, widget: tk.Widget, key: str, text: str, prefs: Prefs, delay_ms: int = 500):
        self.widget = widget
        self.key = key
        self.text = text
        self.prefs = prefs
        self.delay_ms = delay_ms
        self._after_id = None
        self._tipwin = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._cancel, add="+")
        widget.bind("<ButtonPress>", self._cancel, add="+")
        Tooltip._instances.append(self)

    @classmethod
    def hide_all(cls):
        for instance in cls._instances:
            instance._cancel()

    def _schedule(self, _event=None):
        if self.prefs.is_tooltip_dismissed(self.key):
            return
        self._cancel_after()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _show(self):
        if self._tipwin is not None or self.prefs.is_tooltip_dismissed(self.key):
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6

        win = tk.Toplevel(self.widget)
        win.wm_overrideredirect(True)
        win.wm_geometry(f"+{x}+{y}")
        try:
            win.attributes("-topmost", True)
        except tk.TclError:
            pass
        self._tipwin = win

        frame = ttk.Frame(win, style="Tooltip.TFrame", padding=10)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame, text=self.text, wraplength=280, justify="left", style="Tooltip.TLabel"
        ).pack(anchor="w")

        dont_show = tk.BooleanVar(value=False)

        def _on_toggle():
            if dont_show.get():
                self.prefs.dismiss_tooltip(self.key)
                self._cancel()

        ttk.Checkbutton(
            frame,
            text="Don't show this again",
            variable=dont_show,
            command=_on_toggle,
            style="Tooltip.TCheckbutton",
        ).pack(anchor="w", pady=(6, 0))

    def _cancel_after(self):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _cancel(self, _event=None):
        self._cancel_after()
        if self._tipwin is not None:
            self._tipwin.destroy()
            self._tipwin = None
