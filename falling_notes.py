"""Ambient falling-notes background, ported from the main Symplethesia app's
Library-view decoration (src/ui/libraryBg.ts + the `.falling-note-bg` CSS in
src/style.css) to a Tkinter Canvas.

The original is 28 DOM bars, each animated by its own infinite CSS
@keyframes loop, with a simulated depth-of-field: a random per-note `depth`
in [0, 1] jointly lerps opacity, fall speed, size, blur and paint order.
Tkinter canvas items have no native alpha, blur, or gradient fill, so this
reimplements the same per-note parameters (palette, depth lerps,
spawn/respawn/loop logic, soft upward glow trail) driving plain rectangles
via a `Canvas.after()` tick loop instead of CSS: "opacity" is approximated
by pre-blending each note's color toward the canvas background color, and
the CSS gradient trail by a few stacked bands each blended at a different
alpha. (CSS border-radius on the body was tried via small corner ovals and
dropped -- at these small sizes it read as an odd bulge, not a subtle
rounding; not worth it for a background decoration.)
"""

from __future__ import annotations

import random
import tkinter as tk

# Same 5-color palette as libraryBg.ts's PALETTE -- a muted, dark set
# unrelated to the app's hand/accent colors, chosen for a low-key ambient
# effect rather than for meaning.
PALETTE = ["#2f6da8", "#6a52b0", "#a64d79", "#a85f2c", "#268a68"]

NOTE_COUNT = 28
FRAME_MS = 40  # ~25fps -- plenty smooth for slow-falling background bars
TRAIL_BANDS = 3  # stacked rectangles approximating the CSS gradient trail


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _blend(fg_hex: str, bg_hex: str, alpha: float) -> str:
    """Opaque blend of fg_hex over bg_hex at `alpha` (0..1) -- stands in for
    the CSS `opacity` the original uses, since canvas fills have no native
    alpha."""
    fr, fg, fb = _hex_to_rgb(fg_hex)
    br, bgc, bb = _hex_to_rgb(bg_hex)
    r = round(_lerp(br, fr, alpha))
    g = round(_lerp(bgc, fg, alpha))
    b = round(_lerp(bb, fb, alpha))
    return f"#{r:02x}{g:02x}{b:02x}"


class FallingNotesBackground:
    """Owns a set of note-dicts and draws/animates them on `canvas`.

    Each note's canvas items (main body + N trail bands) are created once
    and repositioned/recolored in place on every respawn -- never recreated
    -- so a respawn can never leave an orphaned item behind
    at its last (bottom-of-screen) position."""

    def __init__(self, canvas: tk.Canvas, bg_color: str):
        self.canvas = canvas
        self.bg_color = bg_color
        self._running = False
        self._after_id = None
        self.notes = []
        for _ in range(NOTE_COUNT):
            note = self._make_note(randomize_y=True)
            note.update(item=None, trail_items=[None] * TRAIL_BANDS)
            self.notes.append(note)

    def _make_note(self, randomize_y: bool = False) -> dict:
        depth = random.random()  # 0 = far, 1 = near -- drives everything below
        scale = _lerp(0.5, 1.15, depth)
        # Wider base range than the original CSS's 6-20px: at this scale
        # (flat canvas fill, no glow) anything much narrower reads as a
        # needle-thin line rather than a "block".
        w = random.uniform(9, 24) * scale
        h = random.uniform(28, 90) * scale
        # Duration in the original is lerp(18, 6, depth) seconds to fall one
        # screen height; expressed here directly as px/tick instead.
        speed = _lerp(1.0, 3.0, depth) * random.uniform(0.85, 1.15)
        # Noticeably higher than the original CSS's 0.03-0.085: that value
        # relies on a glowing DOM blend mode we don't have here, and read as
        # nearly invisible once reproduced as a flat canvas-rectangle blend.
        opacity = _lerp(0.14, 0.38, depth) * random.uniform(0.85, 1.1)
        # Fall back to a plausible size if the canvas hasn't been realized/
        # laid out yet (winfo_width/height return 1 before the first draw) --
        # otherwise every note would spawn clustered at x~0 and only spread
        # out once each one individually falls off-screen and respawns.
        canvas_w = max(self.canvas.winfo_width(), 400)
        canvas_h = max(self.canvas.winfo_height(), 300)
        x = random.uniform(0.01, 0.99) * canvas_w
        y = random.uniform(-h, canvas_h) if randomize_y else -h - 20
        return {
            "x": x, "y": y, "w": w, "h": h,
            "speed": speed, "opacity": opacity,
            "color": random.choice(PALETTE), "depth": depth,
        }

    def start(self):
        if self._running:
            return
        self._running = True
        self._tick()

    def stop(self):
        self._running = False
        if self._after_id is not None:
            self.canvas.after_cancel(self._after_id)
            self._after_id = None

    def _respawn(self, note: dict, canvas_w: float):
        fresh = self._make_note()
        fresh["x"] = random.uniform(0.01, 0.99) * canvas_w
        # Only the motion/appearance fields -- item/trail_items are
        # deliberately left untouched so the existing canvas items get
        # reused (repositioned next _draw_note call) instead of abandoned
        # in place.
        for key in ("x", "y", "w", "h", "speed", "opacity", "color", "depth"):
            note[key] = fresh[key]

    def _draw_note(self, note: dict):
        x, y, w, h = note["x"], note["y"], note["w"], note["h"]
        color, opacity = note["color"], note["opacity"]
        fill = _blend(color, self.bg_color, opacity)

        # Soft glow trail above the block (direction it fell from) -- a
        # short, gentle 3-step gradient rather than many thin bands, which
        # read as a "stepped"/disconnected sliver at this size instead of a
        # smooth fade. Approximates the CSS ::after
        # linear-gradient(to top, color 50% -> 18% -> transparent).
        trail_h = h * 0.9
        band_h = trail_h / TRAIL_BANDS
        for i in range(TRAIL_BANDS):
            band_top = y - trail_h + i * band_h
            t = i / (TRAIL_BANDS - 1)  # 0 = topmost band (faintest), 1 = nearest the block
            band_fill = _blend(color, self.bg_color, opacity * _lerp(0.1, 0.55, t))
            item = note["trail_items"][i]
            if item is None:
                item = self.canvas.create_rectangle(0, 0, 0, 0, fill=band_fill, outline="")
                note["trail_items"][i] = item
            self.canvas.coords(item, x, band_top, x + w, band_top + band_h)
            self.canvas.itemconfigure(item, fill=band_fill)
            self.canvas.tag_raise(item)

        # Main body -- a plain rectangle. (An earlier version patched small
        # ovals onto the bottom corners to fake CSS border-radius, but at
        # these small sizes that read as an odd bulge rather than a subtle
        # rounding -- not worth the visual risk for a background decoration.)
        if note["item"] is None:
            note["item"] = self.canvas.create_rectangle(0, 0, 0, 0, fill=fill, outline="")
        self.canvas.coords(note["item"], x, y, x + w, y + h)
        self.canvas.itemconfigure(note["item"], fill=fill)
        self.canvas.tag_raise(note["item"])

    def _tick(self):
        if not self._running:
            return
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        # Nearer (higher-depth) notes are brighter/larger and must paint on
        # top -- process/raise in ascending depth order each frame.
        for note in sorted(self.notes, key=lambda n: n["depth"]):
            note["y"] += note["speed"]
            if note["y"] > canvas_h + 40:
                self._respawn(note, canvas_w)
            self._draw_note(note)
        self._after_id = self.canvas.after(FRAME_MS, self._tick)
