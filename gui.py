#!/usr/bin/env python3
"""Desktop GUI for extract_fingering.py -- "Symple Midi and Playstyle Extractor" (SMaPE).

A thin Tkinter front-end that builds the argv for extract_fingering.py and
runs it as a subprocess (never imports it), because the tool's own
calibration step opens a blocking OpenCV window that must be able to appear
normally in its own process.

Flow: paste/pick a video (page 1) -> pick what kind of video it is (page 2,
three buttons that preset the right combination of flags) -> run screen
(page 3). Everything else lives behind the gear icon (bottom-right,
reachable at any time, including mid-run).

No new *required* dependency: this uses only the Python stdlib (tkinter).
If the optional `tkinterdnd2` package is installed, real OS-level drag-and-
drop is enabled for the Video and MIDI fields; otherwise those fields fall
back gracefully to Browse-button-only behavior.
"""
import io
import os
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter.font as tkfont
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gui_prefs import Prefs  # noqa: E402
from tooltip import Tooltip  # noqa: E402
from falling_notes import FallingNotesBackground  # noqa: E402

# --- Optional drag-and-drop support -----------------------------------
# tkinterdnd2 is NOT a required dependency. If it's missing, we catch the
# ImportError and fall back to a plain tk.Tk() root; the drop-zone widgets
# then behave as plain clickable "Browse..." buttons instead of accepting
# OS drag-and-drop.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_DND = True
except ImportError:
    _HAS_DND = False

# --- Optional Pillow support (for a nicely-scaled, transparency-keyed
# header logo) -- not a required dependency: without it, the header logo
# falls back to tkinter's own (integer-ratio, lower quality, no white-
# keying) PhotoImage.subsample. Deliberately only imports PIL.Image, not
# PIL.ImageTk: the latter is packaged separately on some distros (e.g.
# Debian/Ubuntu's `python3-pil.imagetk`) and is easy to have missing even
# when core Pillow is installed. Avoided entirely by handing Tk a PNG byte
# buffer instead (tk.PhotoImage decodes PNG -- including alpha -- natively
# since Tk 8.6, no ImageTk required).
try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# Frozen (PyInstaller) Windows bundle layout: SMaPE.exe sits next to app/
# (the engine scripts, run as plain files) and runtime/ (an embeddable
# CPython that runs them), so the GUI keeps its subprocess model unchanged
# -- and `runtime/python.exe -m pip` can install the optional transcription
# stack on demand (see _ensure_transcription_deps).
if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(sys.executable).resolve().parent
    HERE = BUNDLE_DIR / "app"
else:
    BUNDLE_DIR = Path(__file__).resolve().parent
    HERE = BUNDLE_DIR
RUNTIME_PYTHON = BUNDLE_DIR / "runtime" / ("python.exe" if os.name == "nt" else "bin/python3")
EXTRACT_SCRIPT = HERE / "extract_fingering.py"
VENV_PYTHON = HERE / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

# Bundled ffmpeg (used by librosa's audio loading and yt-dlp): prepend to
# PATH so child processes find it without a system-wide install.
_FFMPEG_DIR = BUNDLE_DIR / "ffmpeg"
if _FFMPEG_DIR.is_dir():
    os.environ["PATH"] = str(_FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")

# Without this, a child process's stdout isn't a real console (piped, and on
# Windows the GUI itself is a no-console exe), so Python falls back to the
# legacy ANSI codepage (cp1252) for sys.stdout -- which can't encode the
# unicode symbols (e.g. checkmarks) some scripts print, raising
# UnicodeEncodeError. Force UTF-8 for every subprocess we launch below.
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# On Windows the GUI is a windowed (no-console) exe; without this flag every
# child process would flash open its own console window.
POPEN_KWARGS = (
    {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
)
DOWNLOADS_DIR = os.path.expanduser("~/Downloads")
APP_TITLE = "Symple Midi and Playstyle Extractor (SMaPE)"
ICON_PATH = HERE / "icon.png"
LOGO_PATH = HERE / "smape.png"

# --- Symplethesia palette (ported from src/style.css :root) --------------
COLOR_BG = "#171b24"
COLOR_BG2 = "#0f1119"
COLOR_CARD = "#1e2230"
COLOR_INK = "#cdd3e8"
COLOR_MUT = "#6b7491"
COLOR_ACC = "#5477e8"
COLOR_OK = "#3dcf82"
COLOR_WARN = "#f5c842"
COLOR_DEL = "#e85454"
# Flat approximation of the app's rgba(255,255,255,0.09) hairline border --
# Tk widget backgrounds don't support alpha blending.
COLOR_BORDER = "#3a3f52"


def _strip_dnd_braces(path: str) -> str:
    """tkinterdnd2 wraps paths containing spaces in {curly braces}."""
    path = path.strip()
    if path.startswith("{") and path.endswith("}"):
        path = path[1:-1]
    return path


def _pick_font(preferred, fallbacks):
    available = set(tkfont.families())
    for name in (preferred, *fallbacks):
        if name in available:
            return name
    return "TkDefaultFont"


def _key_out_white(img, low=225, high=250):
    """Return an RGBA copy of `img` with near-white pixels made transparent
    (a linear falloff between `low` and `high` brightness, rather than a
    hard cutoff, so anti-aliased edges fade out smoothly instead of leaving
    a jagged/haloed outline). Only meaningful for Pillow images; requires
    _HAS_PIL. Cheap because this is called after the logo is already
    downscaled to header size, not on the full-resolution source."""
    img = img.convert("RGBA")
    pixels = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            whiteness = min(r, g, b)
            if whiteness >= high:
                new_a = 0
            elif whiteness <= low:
                new_a = a
            else:
                new_a = round(a * (high - whiteness) / (high - low))
            if new_a != a:
                pixels[x, y] = (r, g, b, new_a)
    return img


MODES = {
    "hands": {
        "title": "Piano player",
        "caption": "Overhead video of real hands on a real piano.",
        "tooltip": (
            "For a video showing real hands playing a real piano. You supply the "
            "exact-performance MIDI file; the tool tracks fingertips and matches them "
            "to notes, giving you hand + finger (1-5) for every note."
        ),
    },
    "render": {
        "title": "Synthesia training video",
        "caption": "Rendered keyboard, lit colour-coded keys, no real hands.",
        "tooltip": (
            "For Synthesia-style renders: a computer-generated keyboard where keys "
            "light up per hand, falling bars, no real hands visible. MIDI is "
            "transcribed from the video's own audio; hand (not finger) is read from "
            "the lit-key colour at each note."
        ),
    },
    "midi_only": {
        "title": "Extract MIDI only",
        "caption": "Just the notes -- no fingering or hand assignments.",
        "tooltip": (
            "Transcribes MIDI (pitch, timing, velocity, sustain pedal) from the "
            "video's own audio and stops there -- no calibration, no hand tracking, "
            "no fingering/hand analysis. Fastest option if you only want the notes."
        ),
    },
}


class FingeringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1000x700")
        self.root.minsize(820, 560)

        self.prefs = Prefs()
        self.mode = None
        self.proc = None
        self.log_queue = queue.Queue()
        self.reader_thread = None
        self._last_auto_out = None  # tracks the last value we auto-filled into Output JSON
        self._settings_win = None

        self._resolve_python()
        self._setup_style()
        self._set_window_icon()
        self._init_vars()
        self._build_widgets()
        self._show_page(1)
        self._poll_log_queue()

    # -- interpreter selection -------------------------------------------------
    def _resolve_python(self):
        if RUNTIME_PYTHON.exists():
            # Bundled runtime (Windows zip build) -- ships the lite deps.
            self.python_exe = str(RUNTIME_PYTHON)
            self.venv_warning = None
        elif VENV_PYTHON.exists():
            self.python_exe = str(VENV_PYTHON)
            self.venv_warning = None
        else:
            self.python_exe = sys.executable
            self.venv_warning = (
                "No .venv found — install dependencies first: see README.md; "
                "using system Python which likely lacks mediapipe/opencv"
            )

    def _transcription_deps_present(self):
        """True if the Python that will run the job can import the optional
        transcription stack (torch + piano_transcription_inference).

        Checked via find_spec in a subprocess (fast -- does not import torch)
        rather than in-process: the GUI's interpreter is not the one that
        runs the job (frozen bundle / .venv)."""
        probe = ("import importlib.util as u, sys; "
                 "sys.exit(0 if u.find_spec('torch') "
                 "and u.find_spec('piano_transcription_inference') else 1)")
        try:
            return subprocess.run(
                [self.python_exe, "-c", probe],
                capture_output=True, timeout=30, **POPEN_KWARGS,
            ).returncode == 0
        except Exception:
            return False

    # -- visual style, matching the main Symplethesia app ----------------------
    def _setup_style(self):
        self.ui_font_family = _pick_font("Inter", ["Inter Tight", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"])
        self.mono_font_family = _pick_font("JetBrains Mono", ["Consolas", "Menlo", "Courier New", "DejaVu Sans Mono"])
        self.default_font = (self.ui_font_family, 11)
        self.bold_font = (self.ui_font_family, 13, "bold")
        self.mono_font = (self.mono_font_family, 11)

        self.root.configure(bg=COLOR_BG)
        self.root.option_add("*Font", self.default_font)

        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("TFrame", background=COLOR_BG)
        style.configure("Card.TFrame", background=COLOR_CARD)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_INK, font=self.default_font)
        style.configure("Muted.TLabel", background=COLOR_BG, foreground=COLOR_MUT, font=self.default_font)
        style.configure("Heading.TLabel", background=COLOR_BG, foreground=COLOR_INK, font=self.bold_font)
        # Card-background variants -- for labels sitting inside a
        # Card.TFrame panel (pages 1/2, floated over the animated
        # background) rather than directly on the plain window background.
        style.configure("CardMuted.TLabel", background=COLOR_CARD, foreground=COLOR_MUT, font=self.default_font)
        style.configure("CardHeading.TLabel", background=COLOR_CARD, foreground=COLOR_INK, font=self.bold_font)
        style.configure("Tooltip.TFrame", background=COLOR_CARD)
        style.configure("Tooltip.TLabel", background=COLOR_CARD, foreground=COLOR_INK, font=self.default_font)
        style.configure("Tooltip.TCheckbutton", background=COLOR_CARD, foreground=COLOR_MUT)

        style.configure(
            "TButton", background=COLOR_BG, foreground=COLOR_MUT,
            bordercolor=COLOR_MUT, borderwidth=1, focusthickness=0,
            padding=(10, 6), relief="flat", font=self.default_font,
        )
        style.map(
            "TButton",
            foreground=[("active", COLOR_INK), ("disabled", COLOR_MUT)],
            bordercolor=[("active", COLOR_ACC), ("disabled", COLOR_BORDER)],
        )

        style.configure(
            "Primary.TButton", background=COLOR_ACC, foreground=COLOR_BG,
            bordercolor=COLOR_ACC, borderwidth=1, padding=(14, 8), relief="flat",
            font=(self.ui_font_family, 12, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("disabled", COLOR_CARD)],
            foreground=[("disabled", COLOR_MUT)],
        )

        style.configure(
            "Mode.TButton", background=COLOR_CARD, foreground=COLOR_INK,
            bordercolor=COLOR_MUT, borderwidth=1, padding=(20, 34), relief="flat",
            font=(self.ui_font_family, 13, "bold"),
        )
        style.map("Mode.TButton", bordercolor=[("active", COLOR_ACC)])

        style.configure(
            "TEntry", fieldbackground=COLOR_CARD, foreground=COLOR_INK,
            bordercolor=COLOR_MUT, insertcolor=COLOR_INK, borderwidth=1,
        )
        style.map(
            "TEntry",
            bordercolor=[("focus", COLOR_ACC)],
            fieldbackground=[("disabled", COLOR_BG)],
            foreground=[("disabled", COLOR_MUT)],
        )

        style.configure(
            "TCheckbutton", background=COLOR_BG, foreground=COLOR_INK,
            focuscolor=COLOR_BG, font=self.default_font,
        )
        style.map(
            "TCheckbutton",
            foreground=[("active", COLOR_INK), ("disabled", COLOR_MUT)],
        )

        style.configure(
            "TCombobox", fieldbackground=COLOR_CARD, background=COLOR_CARD,
            foreground=COLOR_INK, arrowcolor=COLOR_INK, bordercolor=COLOR_MUT,
        )
        style.map("TCombobox", fieldbackground=[("readonly", COLOR_CARD)])

        style.configure("Vertical.TScrollbar", background=COLOR_CARD, troughcolor=COLOR_BG, bordercolor=COLOR_BG)

        style.configure("Treeview", background=COLOR_CARD, foreground=COLOR_INK,
                        fieldbackground=COLOR_CARD, borderwidth=0, font=self.default_font)
        style.configure("Treeview.Heading", background=COLOR_BG2, foreground=COLOR_MUT,
                        borderwidth=0, font=self.default_font)
        style.map("Treeview", background=[("selected", COLOR_ACC)], foreground=[("selected", COLOR_BG)])

    def _set_window_icon(self):
        try:
            if ICON_PATH.exists():
                self._icon_img = tk.PhotoImage(file=str(ICON_PATH))
                self.root.iconphoto(True, self._icon_img)
        except tk.TclError:
            pass

    # -- variable initialization (independent of which widgets exist) ---------
    def _init_vars(self):
        self.video_var = tk.StringVar()
        self.midi_var = tk.StringVar()
        self.transcribe_var = tk.BooleanVar(value=True)
        self.render_var = tk.BooleanVar(value=False)
        self.midi_only_var = tk.BooleanVar(value=False)
        self.out_var = tk.StringVar(value="")
        self.output_dir_var = tk.StringVar(value="")
        self.fps_var = tk.StringVar(value="30")
        self.align_var = tk.BooleanVar(value=True)
        self.preview_var = tk.BooleanVar(value=False)
        self.bundle_var = tk.BooleanVar(value=True)
        self.offset_var = tk.StringVar(value="")
        self.calibration_var = tk.StringVar(value=str(HERE / "calibration.json"))
        self.sync_method_var = tk.StringVar(value="audio")
        self.min_hand_conf_var = tk.StringVar(value="0.3")
        self.conf_var = tk.StringVar(value="0.0")
        self.flip_render_hands_var = tk.BooleanVar(value=False)
        # Default ON for now: automatic hue clustering has been observed to
        # produce implausibly skewed hand splits on some renders (see
        # render_hands.py's assign_hands_for_notes docstring/history) --
        # manual picking is the more reliable path until that's improved.
        self.pick_hand_colors_var = tk.BooleanVar(value=True)
        self.midi_recover_var = tk.BooleanVar(value=True)
        self.blob_recover_var = tk.BooleanVar(value=False)
        self.clahe_var = tk.BooleanVar(value=True)
        self.live_view_var = tk.BooleanVar(value=True)
        self._live_frame_path = None
        self._live_photo = None
        self._live_polling_active = False
        self.onset_threshold_var = tk.StringVar(value="")
        self.min_velocity_var = tk.StringVar(value="0")
        self.min_duration_var = tk.StringVar(value="0")
        self.no_gpu_var = tk.BooleanVar(value=False)
        # Windows only: plain PyPI torch is CPU-only there (Linux's PyPI
        # wheel already bundles CUDA -- see the install-command comment in
        # _on_run), so getting GPU accel on Windows needs an extra opt-in
        # download of the CUDA build.
        self.gpu_torch_var = tk.BooleanVar(value=False)
        self.artist_var = tk.StringVar(value="")
        self.title_var = tk.StringVar(value="")
        self.genre_var = tk.StringVar(value="")
        self.difficulty_var = tk.StringVar(value="")
        self.extracted_video_title = None  # Store title extracted from log
        self.batch_mode_var = tk.BooleanVar(value=False)
        self.batch_urls: list = []
        self._batch_tree_ids: list = []
        self._batch_current_idx: int = -1
        self._last_run_json: str = None
        self._last_run_bundle: str = None

    def _build_header_logo(self, parent):
        """Persistent header banner (smape.png), scaled to a fixed height.
        Uses Pillow for a smooth resize if available; otherwise falls back
        to tkinter's own (integer-ratio, lower quality) PhotoImage.subsample
        -- either way this is optional decoration, never required to run.

        The source PNG has no alpha channel (it's a flat white background,
        not real transparency), so on the Pillow path the near-white
        background is chroma-keyed to transparent after resizing (cheap at
        this point since the image is already shrunk to header size) so it
        blends into the dark window instead of showing a white box."""
        if not LOGO_PATH.exists():
            return None
        target_height = 77  # 64px base, +20%
        try:
            if _HAS_PIL:
                img = Image.open(LOGO_PATH).convert("RGBA")
                w, h = img.size
                scale = target_height / h
                img = img.resize((max(1, round(w * scale)), target_height), Image.LANCZOS)
                img = _key_out_white(img)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                photo = tk.PhotoImage(data=buf.getvalue())
            else:
                photo = tk.PhotoImage(file=str(LOGO_PATH))
                factor = max(1, round(photo.height() / target_height))
                photo = photo.subsample(factor, factor)
        except (tk.TclError, OSError):
            return None
        self._logo_img = photo  # keep a reference; Tk drops it if GC'd
        return tk.Label(parent, image=photo, bg=COLOR_BG, bd=0)

    # -- top-level widget construction ------------------------------------------
    def _build_widgets(self):
        # Ambient falling-notes background (ported from the main app's
        # Library view, see falling_notes.py) fills the whole window as the
        # backmost layer. Deliberately no full-window "content" or
        # "page_host" wrapper Frame on top of it -- a ttk.Frame always
        # paints its ENTIRE allocated area opaque even where its own
        # children don't fill it, so any such wrapper would silently block
        # the animation across its whole footprint, not just where widgets
        # actually sit. Instead every real widget (logo, warning banner,
        # each page, the gear button) is placed directly on `self.root`,
        # sized to its own content -- only those specific footprints are
        # opaque, and the canvas shows through everywhere else.
        self.bg_canvas = tk.Canvas(self.root, bg=COLOR_BG, highlightthickness=0)
        self.bg_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        # Force geometry to resolve before seeding initial note positions --
        # otherwise winfo_width()/height() read back 1 (unrealized widget)
        # and every note spawns clustered at x~0 until it individually
        # falls off-screen and respawns with a correctly-randomized x.
        self.root.update_idletasks()
        self.falling_notes = FallingNotesBackground(self.bg_canvas, COLOR_BG)
        self.root.after(50, self.falling_notes.start)

        # Logo + optional venv warning float at a fixed spot near the top,
        # placed (not packed) so they don't reserve pack-managed space --
        # page 3 (the only page that needs to know where they end) gets the
        # measured total height below.
        top_offset = 14
        logo_label = self._build_header_logo(self.root)
        if logo_label is not None:
            logo_label.place(relx=0.5, y=10, anchor="n")
            self.root.update_idletasks()
            top_offset = 10 + logo_label.winfo_reqheight() + 10

        if self.venv_warning:
            warn = tk.Label(
                self.root, text=self.venv_warning, bg=COLOR_WARN, fg="#3a2c00",
                anchor="w", justify="left", wraplength=960, font=self.default_font,
            )
            warn.place(relx=0, y=top_offset, relwidth=1, anchor="nw")
            self.root.update_idletasks()
            top_offset += warn.winfo_reqheight() + 10
        self._page3_top_offset = top_offset

        self.page1 = self._build_page1(self.root)
        self.page2 = self._build_page2(self.root)
        self.page3 = self._build_page3(self.root)
        self.page4 = self._build_page4(self.root)
        self.pages = {1: self.page1, 2: self.page2, 3: self.page3, 4: self.page4}

        # Footer: links and donate button at the bottom
        self._build_footer(self.root)

        # Gear icon: placed directly on root (not inside any page), so it
        # stays pinned bottom-right and reachable regardless of which page
        # is showing, including mid-run.
        self.gear_button = ttk.Button(self.root, text="⚙", width=3, command=self._open_settings)
        self.gear_button.place(relx=1.0, rely=1.0, anchor="se", x=-14, y=-14)

    def _show_page(self, n):
        # A tooltip left open when navigating away would otherwise be
        # orphaned: hiding its widget's page via pack_forget doesn't fire a
        # real mouse-<Leave> event, so the popup (a separate Toplevel) would
        # stay stuck on screen. See Tooltip.hide_all's docstring/comment.
        Tooltip.hide_all()
        # Pages 1/2 are compact and mostly-empty by design -- shrink-wrapped
        # and floated centered (via place, not fill) rather than stretched
        # to fill the whole window, so the falling-notes canvas behind them
        # stays visible in the surrounding space instead of being covered by
        # an opaque full-window frame. Page 3 (the run screen) is genuinely
        # content-dense -- Back/mode label, MIDI field, buttons, status, a
        # log box that should use the available space -- so it stays full-
        # bleed as before.
        for page in self.pages.values():
            page.place_forget()
        margin = 40
        page = self.pages[n]
        if n == 3:
            page.place(
                relx=0, rely=0, relwidth=1, relheight=1,
                x=margin, y=self._page3_top_offset,
                width=-2 * margin, height=-(self._page3_top_offset + margin),
            )
        else:
            page.place(relx=0.5, rely=0.5, anchor="center")

    def _build_footer(self, parent):
        """Footer with links and donate button at the bottom of the window."""
        footer = tk.Frame(parent, bg=COLOR_BG, height=60)
        footer.place(relx=0, rely=1.0, relwidth=1, anchor="sw")

        # Container for footer content (centered)
        content = tk.Frame(footer, bg=COLOR_BG)
        content.pack(side="bottom", pady=12, padx=20)

        # Link styling
        link_style = {"fg": COLOR_ACC, "bg": COLOR_BG, "font": self.default_font, "cursor": "hand2", "relief": "flat", "bd": 0}

        # Symplethesia link
        symple_btn = tk.Label(content, text="Made by the team of Symplethesia", **link_style)
        symple_btn.pack(side="left", padx=8)
        symple_btn.bind("<Button-1>", lambda e: self._open_url("https://app.symplethesia.com"))
        symple_btn.bind("<Enter>", lambda e: symple_btn.config(fg=COLOR_OK, font=(self.ui_font_family, 11, "underline")))
        symple_btn.bind("<Leave>", lambda e: symple_btn.config(fg=COLOR_ACC, font=self.default_font))

        # Separator
        sep = tk.Label(content, text="•", fg=COLOR_MUT, bg=COLOR_BG, font=self.default_font)
        sep.pack(side="left", padx=4)

        # Ko-Fi donate button
        kofi_btn = tk.Label(content, text="☕ Ko-Fi Donate", **link_style)
        kofi_btn.pack(side="left", padx=8)
        kofi_btn.bind("<Button-1>", lambda e: self._open_url("https://ko-fi.com/pieterg"))
        kofi_btn.bind("<Enter>", lambda e: kofi_btn.config(fg=COLOR_OK, font=(self.ui_font_family, 11, "underline")))
        kofi_btn.bind("<Leave>", lambda e: kofi_btn.config(fg=COLOR_ACC, font=self.default_font))

        # Separator
        sep2 = tk.Label(content, text="•", fg=COLOR_MUT, bg=COLOR_BG, font=self.default_font)
        sep2.pack(side="left", padx=4)

        # GitHub link
        github_btn = tk.Label(content, text="GitHub", **link_style)
        github_btn.pack(side="left", padx=8)
        github_btn.bind("<Button-1>", lambda e: self._open_url("https://github.com/petepiet/SMaPE"))
        github_btn.bind("<Enter>", lambda e: github_btn.config(fg=COLOR_OK, font=(self.ui_font_family, 11, "underline")))
        github_btn.bind("<Leave>", lambda e: github_btn.config(fg=COLOR_ACC, font=self.default_font))

    def _open_url(self, url):
        """Open a URL in the default browser."""
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Could not open {url}: {e}")

    # -- page 1: video input -----------------------------------------------------
    def _build_page1(self, parent):
        page = ttk.Frame(parent, style="Card.TFrame", padding=28)

        ttk.Label(
            page, text="Paste a YouTube link, or open a local video file.",
            style="CardMuted.TLabel",
        ).pack(pady=(0, 18))

        entry_row = ttk.Frame(page, style="Card.TFrame")
        entry_row.pack(fill="x")
        self.video_entry = ttk.Entry(entry_row, textvariable=self.video_var, width=52)
        self.video_entry.pack(side="left", fill="x", expand=True, ipady=4)
        ttk.Button(entry_row, text="Paste link", command=self._paste_video).pack(side="left", padx=(8, 0))
        ttk.Button(entry_row, text="Open file...", command=self._browse_video).pack(side="left", padx=(8, 0))

        self.page1_status_var = tk.StringVar(value="")
        self.page1_status_label = ttk.Label(page, textvariable=self.page1_status_var, style="CardMuted.TLabel")
        self.page1_status_label.pack(anchor="w", pady=(8, 0))

        ttk.Checkbutton(
            page, text="Process multiple videos (batch mode)",
            variable=self.batch_mode_var, command=self._on_batch_toggled,
        ).pack(anchor="w", pady=(10, 0))

        # Batch queue panel (hidden by default, revealed by _on_batch_toggled)
        self.batch_queue_frame = ttk.Frame(page, style="Card.TFrame")

        queue_inner = ttk.Frame(self.batch_queue_frame, style="Card.TFrame")
        queue_inner.pack(fill="x")
        self.batch_listbox = tk.Listbox(
            queue_inner, height=4, bg=COLOR_BG, fg=COLOR_INK,
            selectbackground=COLOR_ACC, selectforeground=COLOR_BG,
            font=self.mono_font, relief="flat", highlightthickness=1,
            highlightbackground=COLOR_BORDER, width=52,
        )
        lb_scroll = ttk.Scrollbar(queue_inner, orient="vertical", command=self.batch_listbox.yview)
        self.batch_listbox.configure(yscrollcommand=lb_scroll.set)
        self.batch_listbox.pack(side="left", fill="x", expand=True)
        lb_scroll.pack(side="right", fill="y")

        batch_btn_row = ttk.Frame(self.batch_queue_frame, style="Card.TFrame")
        batch_btn_row.pack(fill="x", pady=(6, 0))
        ttk.Button(batch_btn_row, text="Add URL ↑", command=self._add_to_batch).pack(side="left")
        ttk.Button(batch_btn_row, text="Remove", command=self._remove_from_batch).pack(side="left", padx=(6, 0))
        ttk.Button(batch_btn_row, text="Clear", command=self._clear_batch).pack(side="left", padx=(6, 0))
        self.batch_count_label = ttk.Label(batch_btn_row, text="", style="CardMuted.TLabel")
        self.batch_count_label.pack(side="right")

        ttk.Button(
            page, text="Next →", style="Primary.TButton", command=self._on_next_from_page1,
        ).pack(anchor="e", pady=(18, 0))

        if _HAS_DND:
            self.video_entry.drop_target_register(DND_FILES)
            self.video_entry.dnd_bind("<<Drop>>", self._on_drop_video)

        return page

    def _on_next_from_page1(self):
        if self.batch_mode_var.get():
            if not self.batch_urls:
                self.page1_status_var.set("Add at least one URL to the batch queue first.")
                self.page1_status_label.configure(foreground=COLOR_DEL)
                return
        else:
            video = self.video_var.get().strip()
            if not video:
                self.page1_status_var.set("Paste a YouTube link or open a local file first.")
                self.page1_status_label.configure(foreground=COLOR_DEL)
                return
        self.page1_status_var.set("")
        self._show_page(2)

    # -- page 2: mode selection ---------------------------------------------------
    def _build_page2(self, parent):
        page = ttk.Frame(parent, style="Card.TFrame", padding=28)

        ttk.Label(page, text="What kind of video is this?", style="CardHeading.TLabel").pack(pady=(0, 18))

        # One column per mode: bold title on top, clickable image button under
        # it, muted caption underneath. Grid with uniform columns so the three
        # spread evenly. The source PNGs are large (~1254px); Tk's PhotoImage
        # can't smooth-scale, so use PIL to resize and hand Tk a PNG byte
        # buffer (same ImageTk-free pattern as the header logo) -- without
        # PIL the title falls back to being the clickable button.
        image_map = {
            "hands": "buttons/pianist.png",
            "render": "buttons/synthesia.png",
            "midi_only": "buttons/video2mid.png",
        }
        self._page2_images = {}  # keep references so Tk doesn't garbage-collect them
        buttons_row = ttk.Frame(page, style="Card.TFrame")
        buttons_row.pack(fill="x")
        for col, mode_key in enumerate(("hands", "render", "midi_only")):
            buttons_row.columnconfigure(col, weight=1, uniform="mode")
            info = MODES[mode_key]
            cell = ttk.Frame(buttons_row, style="Card.TFrame")
            cell.grid(row=0, column=col, padx=12, sticky="n")

            img = None
            img_path = HERE / image_map[mode_key]
            if _HAS_PIL and img_path.exists():
                try:
                    pil_img = Image.open(img_path)
                    pil_img.thumbnail((170, 170), Image.LANCZOS)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    img = tk.PhotoImage(data=buf.getvalue())
                    self._page2_images[mode_key] = img
                except Exception as e:
                    print(f"Warning: couldn't load {img_path}: {e}", file=sys.stderr)

            if img is not None:
                ttk.Label(cell, text=info["title"], style="CardHeading.TLabel").pack()
                btn = tk.Button(
                    cell, image=img, bg=COLOR_CARD, bd=0, cursor="hand2",
                    command=lambda m=mode_key: self._select_mode(m),
                    activebackground=COLOR_ACC, highlightthickness=0,
                )
                btn.pack(pady=(14, 0))
            else:
                btn = ttk.Button(
                    cell, text=info["title"], style="Mode.TButton",
                    command=lambda m=mode_key: self._select_mode(m),
                )
                btn.pack()
            ttk.Label(
                cell, text=info["caption"], style="CardMuted.TLabel",
                wraplength=190, justify="center",
            ).pack(pady=(14, 0))
            Tooltip(btn, key=f"mode_{mode_key}", text=info["tooltip"], prefs=self.prefs)

        ttk.Button(page, text="← Back", command=lambda: self._show_page(1)).pack(anchor="w", pady=(22, 0))

        return page

    def _select_mode(self, mode):
        self.mode = mode
        if mode == "hands":
            self.midi_only_var.set(False)
            self.render_var.set(False)
            self._on_render_toggled()
            # Default: transcribe from audio ("Supply a MIDI file" starts
            # unchecked); the MIDI file field only appears once the user
            # opts in via that checkbox (see _refresh_midi_visibility).
            self.transcribe_var.set(True)
            self._on_transcribe_toggled()
        elif mode == "render":
            self.midi_only_var.set(False)
            self.render_var.set(True)
            self._on_render_toggled()
        elif mode == "midi_only":
            self.render_var.set(False)
            self._on_render_toggled()
            self.transcribe_var.set(True)
            self.midi_only_var.set(True)
            self._on_transcribe_toggled()
        self._refresh_page3_for_mode()
        self._refresh_batch_progress_visibility()
        if self.transcribe_var.get() and not self.video_var.get().strip().startswith(("http://", "https://")):
            self._maybe_default_out(self.video_var.get().strip())
        self._show_page(3)

    # -- page 3: run screen -------------------------------------------------------
    def _build_page3(self, parent):
        page = ttk.Frame(parent)
        pad = {"padx": 0, "pady": 6}

        top = ttk.Frame(page)
        top.pack(fill="x", padx=16, pady=(16, 0))
        ttk.Button(top, text="← Back", command=self._on_back_from_page3).pack(side="left")
        self.mode_label_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.mode_label_var, style="Muted.TLabel").pack(side="left", padx=(12, 0))
        ttk.Button(top, text="↻ Restart", command=self._on_restart).pack(side="right")

        form = ttk.Frame(page)
        form.pack(fill="x", padx=16, pady=(10, 0))
        form.columnconfigure(1, weight=1)

        # MIDI file row -- only relevant/shown in "hands" mode, where a real
        # performance MIDI must be supplied. Shares self.midi_var with the
        # secondary copy in Settings, so either one edits the same value.
        self.page3_midi_row = ttk.Frame(form)
        self.page3_midi_row.grid(row=0, column=0, columnspan=3, sticky="ew", **pad)
        self.page3_midi_row.columnconfigure(1, weight=1)
        ttk.Label(self.page3_midi_row, text="MIDI file (exact same performance):").grid(row=0, column=0, sticky="w")
        self.midi_entry = ttk.Entry(self.page3_midi_row, textvariable=self.midi_var)
        self.midi_entry.grid(row=0, column=1, sticky="ew", padx=8)
        self.midi_browse_button = ttk.Button(self.page3_midi_row, text="Browse...", command=self._browse_midi)
        self.midi_browse_button.grid(row=0, column=2)
        if _HAS_DND:
            self.midi_entry.drop_target_register(DND_FILES)
            self.midi_entry.dnd_bind("<<Drop>>", self._on_drop_midi)

        # Middle section: always-packed container; swaps between metadata
        # (single-video) and batch-progress Treeview (batch mode).
        middle_container = ttk.Frame(page)
        middle_container.pack(fill="x")

        # Batch progress panel (hidden by default)
        self.batch_progress_frame = ttk.Frame(middle_container)
        bp_label = ttk.Label(self.batch_progress_frame, text="Batch queue", style="Muted.TLabel")
        bp_label.pack(fill="x", padx=16, pady=(10, 4))
        tree_frame = ttk.Frame(self.batch_progress_frame)
        tree_frame.pack(fill="x", padx=16, pady=(0, 6))
        self.queue_tree = ttk.Treeview(
            tree_frame, columns=("status",), show="headings", height=5, selectmode="none",
        )
        self.queue_tree.heading("status", text="Video / Status", anchor="w")
        self.queue_tree.column("status", width=560, stretch=True)
        qt_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.queue_tree.yview)
        self.queue_tree.configure(yscrollcommand=qt_scroll.set)
        self.queue_tree.pack(side="left", fill="x", expand=True)
        qt_scroll.pack(side="right", fill="y")

        btn_frame = ttk.Frame(page)
        btn_frame.pack(fill="x", padx=16, pady=(14, 6))
        self.run_button = ttk.Button(btn_frame, text="Run", style="Primary.TButton", command=self._on_run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(btn_frame, text="Stop", command=self._on_stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        ttk.Button(btn_frame, text="Open output folder", command=self._open_output_folder).pack(side="left", padx=8)
        self.continue_button = ttk.Button(btn_frame, text="Continue to metadata →",
                                          command=lambda: self._show_page(4))
        # Shown only after a successful run (hidden until then)

        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = tk.Label(
            page, textvariable=self.status_var, anchor="w", bg=COLOR_BG, fg=COLOR_OK, font=self.default_font,
        )
        self.status_label.pack(fill="x", padx=16, pady=(0, 6))

        # Live tracking frame (shown when live view is enabled during a run)
        self.live_view_container = ttk.Frame(page)
        self.live_view_container.pack(fill="x", padx=16, pady=(0, 4))
        self.live_view_label = tk.Label(
            self.live_view_container, bg=COLOR_CARD, bd=0, anchor="center", height=0,
        )
        self.live_view_label.pack(fill="x")

        log_frame = ttk.Frame(page)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.log_text = tk.Text(
            log_frame, wrap="word", font=self.mono_font, state="disabled",
            bg=COLOR_CARD, fg=COLOR_INK, insertbackground=COLOR_INK,
            relief="flat", highlightthickness=1, highlightbackground=COLOR_BORDER,
        )
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        return page

    # -- page 4: metadata editor (shown after a successful single-video run) ----
    def _build_page4(self, parent):
        page = ttk.Frame(parent, style="Card.TFrame", padding=28)

        ttk.Label(page, text="Metadata", style="CardHeading.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(
            page,
            text="Auto-filled from the video title — edit if needed, then save to the .symple bundle.",
            style="CardMuted.TLabel",
        ).pack(anchor="w", pady=(0, 18))

        form = ttk.Frame(page, style="Card.TFrame")
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Artist:", style="CardMuted.TLabel").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.artist_var, width=36).grid(row=0, column=1, sticky="ew", padx=8, pady=5)

        swap_btn = ttk.Button(form, text="⇄", width=3, command=self._swap_artist_title)
        swap_btn.grid(row=0, column=2, rowspan=2, padx=(0, 0), pady=5)
        Tooltip(swap_btn, key="swap_artist_title_p4", text="Swap Artist and Title", prefs=self.prefs)

        ttk.Label(form, text="Title:", style="CardMuted.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.title_var, width=36).grid(row=1, column=1, sticky="ew", padx=8, pady=5)

        ttk.Label(form, text="Genre:", style="CardMuted.TLabel").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.genre_var, width=36).grid(row=2, column=1, sticky="ew", padx=8, pady=5)

        ttk.Label(form, text="Difficulty:", style="CardMuted.TLabel").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Combobox(
            form, textvariable=self.difficulty_var, state="readonly",
            values=["", "easy", "intermediate", "advanced", "expert"], width=20,
        ).grid(row=3, column=1, sticky="w", padx=8, pady=5)

        self.p4_status_var = tk.StringVar(value="")
        self.p4_status_label = tk.Label(
            page, textvariable=self.p4_status_var, anchor="w",
            bg=COLOR_CARD, fg=COLOR_OK, font=self.default_font,
        )
        self.p4_status_label.pack(fill="x", pady=(16, 0))

        btn_row = ttk.Frame(page, style="Card.TFrame")
        btn_row.pack(fill="x", pady=(12, 0))
        ttk.Button(btn_row, text="← Run again", command=lambda: self._show_page(3)).pack(side="left")
        ttk.Button(
            btn_row, text="Save to .symple bundle", style="Primary.TButton",
            command=self._save_metadata,
        ).pack(side="left", padx=(10, 0))
        ttk.Button(btn_row, text="Done →", command=self._on_done_metadata).pack(side="right")

        return page

    def _on_done_metadata(self):
        self.video_var.set("")
        self.extracted_video_title = None
        self._last_run_json = None
        self._last_run_bundle = None
        self._show_page(1)

    def _save_metadata(self):
        import zipfile, json as _json
        bundle_path = self._last_run_bundle
        if not bundle_path or not os.path.exists(bundle_path):
            # Derive from known JSON path
            out_json = self._last_run_json or self.out_var.get().strip()
            if out_json:
                base = out_json
                for sfx in (".fingering.json", ".json"):
                    if base.endswith(sfx):
                        base = base[: -len(sfx)]
                        break
                bundle_path = base + ".symple"
        if not bundle_path or not os.path.exists(bundle_path):
            self.p4_status_var.set("No .symple bundle found — run the analysis first.")
            self.p4_status_label.configure(fg=COLOR_DEL)
            return
        metadata = {k: v for k, v in [
            ("artist", self.artist_var.get().strip()),
            ("title", self.title_var.get().strip()),
            ("genre", self.genre_var.get().strip()),
            ("difficulty", self.difficulty_var.get().strip()),
        ] if v}
        try:
            with zipfile.ZipFile(bundle_path, "r") as zf:
                manifest = _json.loads(zf.read("manifest.json"))
                contents = {name: zf.read(name) for name in zf.namelist() if name != "manifest.json"}
            if metadata:
                manifest["metadata"] = metadata
            elif "metadata" in manifest:
                del manifest["metadata"]
            with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("manifest.json", _json.dumps(manifest, indent=2))
                for name, data in contents.items():
                    zf.writestr(name, data)
            self.p4_status_var.set(f"Saved to {Path(bundle_path).name}")
            self.p4_status_label.configure(fg=COLOR_OK)
        except Exception as exc:
            self.p4_status_var.set(f"Save failed: {exc}")
            self.p4_status_label.configure(fg=COLOR_DEL)

    def _on_back_from_page3(self):
        self._show_page(2)

    def _refresh_page3_for_mode(self):
        if self.mode in MODES:
            self.mode_label_var.set(f"Mode: {MODES[self.mode]['title']}")
        self._refresh_midi_visibility()

    def _refresh_midi_visibility(self):
        """The MIDI file field (page 3's primary copy, and Settings' secondary
        copy) is only relevant once the user opts out of transcription via
        "Supply a MIDI file" -- hidden entirely otherwise, not just greyed
        out. Only "hands" mode ever allows that opt-out (Render/MIDI-only
        force transcription and disable the checkbox), so gate on both."""
        show = self.mode == "hands" and not self.transcribe_var.get()
        for row_attr in ("page3_midi_row", "midi_settings_row"):
            row = getattr(self, row_attr, None)
            if row is None or not row.winfo_exists():
                continue
            if show:
                row.grid()
            else:
                row.grid_remove()

    def _on_restart(self):
        """Go back to page 1 (video selection) after stopping any running process."""
        self._on_stop()
        self.video_var.set("")
        self.extracted_video_title = None
        self._show_page(1)

    def _autofill_from_extracted_title(self):
        """Parse the video title extracted from download log and populate metadata."""
        if not self.extracted_video_title:
            return
        title = self.extracted_video_title
        self._parse_and_populate_metadata(title)

    def _parse_and_populate_metadata(self, title: str):
        """Parse a title string and populate metadata fields."""
        # YouTube titles often use the fullwidth bar (U+FF5C) because "|" is
        # not allowed in filenames -- normalize it. Then drop everything after
        # the first bar: it's almost always channel/branding junk
        # ("Song - Artist | SomeChannel piano tutorial"), not metadata.
        title = title.replace("｜", "|")
        if "|" in title:
            title = title.split("|", 1)[0].strip()

        # Try to split what's left on common delimiters: " - ", " by "
        artist = ""
        song_title = ""

        if " - " in title:
            parts = title.split(" - ", 1)
            artist = parts[0].strip()
            song_title = parts[1].strip()
        elif " by " in title:
            parts = title.split(" by ", 1)
            song_title = parts[0].strip()
            artist = parts[1].strip()
        elif ": " in title:
            # "Artist: Song Title" pattern common on YouTube
            parts = title.split(": ", 1)
            artist = parts[0].strip()
            song_title = parts[1].strip()
        else:
            song_title = title

        # Detect difficulty from title
        detected_difficulty = ""
        difficulty_keywords = {
            "beginner": "easy",
            "easy": "easy",
            "elementary": "easy",
            "intermediate": "intermediate",
            "advanced": "advanced",
            "expert": "expert",
            "hard": "advanced",
            "difficult": "advanced",
        }
        title_lower = song_title.lower()
        for keyword, difficulty in difficulty_keywords.items():
            if keyword in title_lower:
                detected_difficulty = difficulty
                break  # Use the first match

        # Clean up common keywords from song title
        for keyword in [" EASY Piano", " HARD Piano", " (Piano Cover)", " Piano Cover", " Piano Tutorial", " Easy Piano", " Piano", " Tutorial", " (Cover)", " Cover", " (Easy)", " (Hard)", " (Advanced)", " (Expert)", " - Easy", " - Hard", " - Advanced", " - Expert"]:
            song_title = song_title.replace(keyword, "").replace(keyword.lower(), "").strip()

        # Set genre as "Piano" by default
        genre = "Piano"

        # Set the fields
        self.artist_var.set(artist)
        self.title_var.set(song_title)
        self.genre_var.set(genre)
        if detected_difficulty:
            self.difficulty_var.set(detected_difficulty)

    def _swap_artist_title(self):
        artist, title = self.artist_var.get(), self.title_var.get()
        self.artist_var.set(title)
        self.title_var.set(artist)

    def _autofill_metadata(self):
        """Auto-fill artist, title, and genre from available sources."""
        # First, try using extracted title from log
        if self.extracted_video_title:
            self._parse_and_populate_metadata(self.extracted_video_title)
            return

        out_path = self.out_var.get().strip()

        # Only use output path if it looks like a real filename (not a URL)
        if out_path and not ("http://" in out_path or "https://" in out_path or "watch?v=" in out_path):
            # Extract filename and remove the .fingering.json suffix
            title = Path(out_path).stem
            if title.endswith(".fingering"):
                title = title[:-10]  # Remove ".fingering"
            self._parse_and_populate_metadata(title)
            return

        # Try to get title from video input
        video_input = self.video_var.get().strip()
        if not video_input or "http://" in video_input or "https://" in video_input or "watch?v=" in video_input:
            # Can't extract from URL; suggest running first
            import tkinter.messagebox as msgbox
            msgbox.showinfo(
                "Auto-fill",
                "To auto-fill metadata, please run the tool first.\n\n"
                "The video title will be extracted from the download and\n"
                "metadata will be populated automatically.\n\n"
                "Or, if you have a local file with a descriptive name,\n"
                "it will be used for auto-fill."
            )
            return

        # Extract filename from local file path
        title = Path(video_input).stem
        self._parse_and_populate_metadata(title)

    # -- settings popover (gear icon) ---------------------------------------------
    def _open_settings(self):
        if self._settings_win is not None and self._settings_win.winfo_exists():
            self._settings_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.configure(bg=COLOR_BG)
        win.transient(self.root)  # stays above the main window; no grab_set, so it's non-modal
        self._settings_win = win

        self._build_settings_contents(win)

        self.root.update_idletasks()
        w, h = 700, 620
        x = self.root.winfo_x() + self.root.winfo_width() - w - 30
        y = self.root.winfo_y() + self.root.winfo_height() - h - 30
        win.geometry(f"{w}x{h}+{max(x, 0)}+{max(y, 0)}")

        # Re-sync enabled/disabled states onto the freshly created widgets.
        self._on_transcribe_toggled()
        self._on_render_toggled()

    def _build_settings_contents(self, win):
        canvas = tk.Canvas(win, bg=COLOR_BG, highlightthickness=0)
        vscroll = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        adv = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=adv, anchor="nw")
        adv.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        pad = {"padx": 12, "pady": 5}
        adv.columnconfigure(1, weight=1)
        arow = 0

        def _sec(title):
            nonlocal arow
            ttk.Separator(adv, orient="horizontal").grid(
                row=arow, column=0, columnspan=3, sticky="ew", padx=12, pady=(12, 0)
            )
            arow += 1
            ttk.Label(adv, text=title, style="Muted.TLabel").grid(
                row=arow, column=0, columnspan=3, sticky="w", padx=14, pady=(4, 2)
            )
            arow += 1

        # ── Input / Mode ─────────────────────────────────────────────────────
        ttk.Label(adv, text="Settings", style="Heading.TLabel").grid(
            row=arow, column=0, columnspan=3, sticky="w", **pad
        )
        arow += 1

        # Transcribing is the default; this checkbox is phrased as the opt-out
        # so that transcribe_var keeps its True-means-transcribing meaning.
        supply_midi_check = ttk.Checkbutton(
            adv, text="Supply a MIDI file (exact performance of this video, extracted elsewhere)",
            variable=self.transcribe_var, onvalue=False, offvalue=True,
            command=self._on_transcribe_toggled,
        )
        supply_midi_check.grid(row=arow, column=0, columnspan=3, sticky="w", **pad)
        self.transcribe_check = supply_midi_check
        Tooltip(
            supply_midi_check, key="supply_midi",
            text="Off by default: the tool transcribes MIDI from the video's own audio. Check this if "
            "you already have the exact-performance MIDI file for this video (e.g. extracted elsewhere) "
            "and want to supply it directly instead.",
            prefs=self.prefs,
        )
        arow += 1

        # Only shown when "Supply a MIDI file" is checked.
        self.midi_settings_row = ttk.Frame(adv)
        self.midi_settings_row.grid(row=arow, column=0, columnspan=3, sticky="ew", padx=0, pady=0)
        self.midi_settings_row.columnconfigure(1, weight=1)
        ttk.Label(self.midi_settings_row, text="MIDI file:").grid(row=0, column=0, sticky="w", **pad)
        self.midi_entry_settings = ttk.Entry(self.midi_settings_row, textvariable=self.midi_var)
        self.midi_entry_settings.grid(row=0, column=1, sticky="ew", **pad)
        self.midi_browse_button_settings = ttk.Button(self.midi_settings_row, text="Browse...", command=self._browse_midi)
        self.midi_browse_button_settings.grid(row=0, column=2, **pad)
        arow += 1

        self.render_check = ttk.Checkbutton(
            adv, text="Synthesia-style render (lit-key colour → hand assignment, no hand tracking)",
            variable=self.render_var, command=self._on_render_toggled,
        )
        self.render_check.grid(row=arow, column=0, columnspan=3, sticky="w", **pad)
        arow += 1

        # ── Sync & Alignment ─────────────────────────────────────────────────
        _sec("SYNC & ALIGNMENT")

        ttk.Label(adv, text="FPS:").grid(row=arow, column=0, sticky="w", **pad)
        ttk.Entry(adv, textvariable=self.fps_var, width=12).grid(row=arow, column=1, sticky="w", **pad)
        arow += 1

        offset_label = ttk.Label(adv, text="Offset (sec, blank = auto):")
        offset_label.grid(row=arow, column=0, sticky="w", **pad)
        ttk.Entry(adv, textvariable=self.offset_var, width=12).grid(row=arow, column=1, sticky="w", **pad)
        Tooltip(
            offset_label, key="offset",
            text="Manual video/MIDI offset in seconds (video_time = midi_time + offset). "
            "Leave blank to auto-estimate from the audio.",
            prefs=self.prefs,
        )
        arow += 1

        sync_label = ttk.Label(adv, text="Sync method:")
        sync_label.grid(row=arow, column=0, sticky="w", **pad)
        sync_method_combo = ttk.Combobox(
            adv, textvariable=self.sync_method_var, values=["audio", "press-moments"],
            state="readonly", width=18,
        )
        sync_method_combo.grid(row=arow, column=1, sticky="w", **pad)
        Tooltip(
            sync_label, key="sync_method",
            text="audio (default): detect the first piano onset in the video's own audio track -- "
            "reliable, works from a static camera. press-moments: legacy fallback using hand motion "
            "instead of audio, only useful if audio extraction fails.",
            prefs=self.prefs,
        )
        arow += 1

        self.align_check = ttk.Checkbutton(
            adv, text="Align video/MIDI before analysis (watch + hear, tune the offset)",
            variable=self.align_var,
        )
        self.align_check.grid(row=arow, column=0, columnspan=3, sticky="w", **pad)
        arow += 1

        # ── Hand Tracking ─────────────────────────────────────────────────────
        _sec("HAND TRACKING")

        hand_conf_label = ttk.Label(adv, text="Min hand confidence (0–1):")
        hand_conf_label.grid(row=arow, column=0, sticky="w", **pad)
        ttk.Entry(adv, textvariable=self.min_hand_conf_var, width=12).grid(row=arow, column=1, sticky="w", **pad)
        Tooltip(
            hand_conf_label, key="min_hand_confidence",
            text="MediaPipe hand-detection confidence threshold (default 0.3 -- deliberately below the "
            "library's 0.5, which drops the second hand on typical overhead piano shots). Lower further "
            "(e.g. 0.15) if a clearly visible hand still goes undetected; raise toward 0.5 if ghost "
            "hands appear.",
            prefs=self.prefs,
        )
        arow += 1

        ttk.Label(adv, text="Confidence threshold:").grid(row=arow, column=0, sticky="w", **pad)
        ttk.Entry(adv, textvariable=self.conf_var, width=12).grid(row=arow, column=1, sticky="w", **pad)
        arow += 1

        midi_recover_check = ttk.Checkbutton(
            adv, text="MIDI-anchored hand recovery (re-run MediaPipe where a note sounds but no hand was tracked)",
            variable=self.midi_recover_var,
        )
        midi_recover_check.grid(row=arow, column=0, columnspan=3, sticky="w", **pad)
        Tooltip(
            midi_recover_check, key="midi_recover",
            text="After tracking, for any frame where a hand is missing but a MIDI note is sounding in a "
            "register with no tracked hand, re-run MediaPipe cropped tightly around that key to recover the "
            "missing hand. Uses the played note as ground truth for where a hand must be -- improves the L/R "
            "hand split on videos where one hand is frequently dropped. On by default; uncheck to skip it.",
            prefs=self.prefs,
        )
        arow += 1

        blob_recover_check = ttk.Checkbutton(
            adv, text="Skin-blob hand recovery (for hands that touch/overlap — may need tuning)",
            variable=self.blob_recover_var,
        )
        blob_recover_check.grid(row=arow, column=0, columnspan=3, sticky="w", **pad)
        Tooltip(
            blob_recover_check, key="blob_recover",
            text="Independent skin-colour (YCrCb) segmentation that finds the two hands as blobs, splitting a "
            "merged blob at its seam. Recovers a hand MediaPipe merged/dropped when the hands play close "
            "together in the same register (where the MIDI register split and voice separation can't help). "
            "Off by default: skin segmentation can need per-video tuning; enable it for close-hands pieces "
            "where one hand is under-tracked.",
            prefs=self.prefs,
        )
        arow += 1

        clahe_check = ttk.Checkbutton(
            adv, text="Enhance contrast for detection (CLAHE — helps on dark/low-contrast video)",
            variable=self.clahe_var,
        )
        clahe_check.grid(row=arow, column=0, columnspan=3, sticky="w", **pad)
        Tooltip(
            clahe_check, key="clahe",
            text="Applies CLAHE local-contrast enhancement to the lightness channel of MediaPipe's input "
            "before hand detection. Lifts detail in dark or low-contrast footage (a common cause of missed "
            "hands) without changing hue/skin-tone cues. Affects detection only, not the exported video. "
            "On by default; uncheck if it makes detection worse on already well-lit video.",
            prefs=self.prefs,
        )
        arow += 1

        ttk.Checkbutton(
            adv, text="Live tracking view (show hand detection in the run screen while processing)",
            variable=self.live_view_var,
        ).grid(row=arow, column=0, columnspan=3, sticky="w", **pad)
        arow += 1

        # ── Render Mode ───────────────────────────────────────────────────────
        _sec("RENDER MODE  (Synthesia-style only)")

        self.flip_render_hands_check = ttk.Checkbutton(
            adv, text="Flip render hand colours (use if L/R look swapped)",
            variable=self.flip_render_hands_var,
        )
        self.flip_render_hands_check.grid(row=arow, column=0, columnspan=3, sticky="w", **pad)
        arow += 1

        pick_colors_check = ttk.Checkbutton(
            adv, text="Pick hand colours manually (click-sample instead of auto-clustering)",
            variable=self.pick_hand_colors_var,
        )
        pick_colors_check.grid(row=arow, column=0, columnspan=3, sticky="w", **pad)
        self.pick_hand_colors_check = pick_colors_check
        Tooltip(
            pick_colors_check, key="pick_hand_colors",
            text="Opens a window on the video where you scrub with arrow keys and click directly on a lit "
            "key to sample its color, labeling it LH/RH white/black key. Use this when automatic color "
            "clustering produces an implausible hand split (e.g. one hand gets nearly the whole keyboard) "
            "despite reporting high confidence.",
            prefs=self.prefs,
        )
        arow += 1

        # ── Transcription ─────────────────────────────────────────────────────
        _sec("TRANSCRIPTION  (auto-MIDI from audio)")

        self.transcribe_opts_frame = ttk.Frame(adv)
        self.transcribe_opts_frame.grid(row=arow, column=0, columnspan=3, sticky="ew", padx=12)
        arow += 1
        ghost_label = ttk.Label(self.transcribe_opts_frame, text="Ghost-note filtering", style="Muted.TLabel")
        ghost_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(4, 4))
        Tooltip(
            ghost_label, key="ghost_notes",
            text="Three independent knobs for reducing false-positive notes in transcription, all "
            "no-ops at their defaults. Onset threshold: pickier about declaring a note at all. "
            "Min velocity / Min duration: drop notes that are quiet / very short after the fact.",
            prefs=self.prefs,
        )
        ttk.Label(self.transcribe_opts_frame, text="Onset threshold (blank = default 0.3):").grid(
            row=1, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Entry(self.transcribe_opts_frame, textvariable=self.onset_threshold_var, width=10).grid(
            row=1, column=1, sticky="w", padx=(7, 0), pady=(0, 5)
        )
        ttk.Label(self.transcribe_opts_frame, text="Min velocity (0–127, 0 = off):").grid(
            row=2, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Entry(self.transcribe_opts_frame, textvariable=self.min_velocity_var, width=10).grid(
            row=2, column=1, sticky="w", padx=(7, 0), pady=(0, 5)
        )
        ttk.Label(self.transcribe_opts_frame, text="Min duration sec (0 = off):").grid(
            row=3, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Entry(self.transcribe_opts_frame, textvariable=self.min_duration_var, width=10).grid(
            row=3, column=1, sticky="w", padx=(7, 0), pady=(0, 5)
        )
        no_gpu_check = ttk.Checkbutton(
            self.transcribe_opts_frame, text="Force CPU (disable GPU acceleration)",
            variable=self.no_gpu_var,
        )
        no_gpu_check.grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 5))
        Tooltip(
            no_gpu_check, key="no_gpu",
            text="By default transcription uses a CUDA GPU automatically if one is available -- this "
            "forces CPU-only inference instead. Only useful for troubleshooting a suspected GPU-specific "
            "issue; leave unchecked to get the (usually much faster) GPU path when you have one.",
            prefs=self.prefs,
        )
        if os.name == "nt":
            gpu_torch_check = ttk.Checkbutton(
                self.transcribe_opts_frame,
                text="Use NVIDIA GPU (installs CUDA PyTorch, ~2.5 GB extra download)",
                variable=self.gpu_torch_var,
            )
            gpu_torch_check.grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 5))
            Tooltip(
                gpu_torch_check, key="gpu_torch_windows",
                text="Only relevant the first time transcription support installs itself. Plain PyPI "
                "PyTorch on Windows is CPU-only, so GPU acceleration needs a separate ~2.5 GB CUDA build "
                "from download.pytorch.org. If that download fails with a TLS/handshake error (a known "
                "issue with some AV/network setups against that host), the installer automatically falls "
                "back to the CPU-only build and prints manual download instructions in the log, so the "
                "run is never left broken -- you can retry the GPU install later.",
                prefs=self.prefs,
            )

        # ── Output ────────────────────────────────────────────────────────────
        _sec("OUTPUT")

        ttk.Checkbutton(
            adv, text="Render preview video (with MIDI audio) after analysis", variable=self.preview_var,
        ).grid(row=arow, column=0, columnspan=3, sticky="w", **pad)
        arow += 1

        ttk.Checkbutton(
            adv, text="Write a .symple bundle (MIDI + fingering, one-step load in Symplethesia)",
            variable=self.bundle_var,
        ).grid(row=arow, column=0, columnspan=3, sticky="w", **pad)
        arow += 1

        ttk.Label(adv, text="Output JSON:").grid(row=arow, column=0, sticky="w", **pad)
        ttk.Entry(adv, textvariable=self.out_var).grid(row=arow, column=1, sticky="ew", **pad)
        ttk.Button(adv, text="Browse...", command=self._browse_out).grid(row=arow, column=2, **pad)
        arow += 1

        ttk.Label(adv, text="Output directory:").grid(row=arow, column=0, sticky="w", **pad)
        ttk.Entry(adv, textvariable=self.output_dir_var).grid(row=arow, column=1, sticky="ew", **pad)
        ttk.Button(adv, text="Browse...", command=self._browse_output_dir).grid(row=arow, column=2, **pad)
        arow += 1

        # ── Advanced ──────────────────────────────────────────────────────────
        _sec("ADVANCED")

        calib_label = ttk.Label(adv, text="Calibration file:")
        calib_label.grid(row=arow, column=0, sticky="w", **pad)
        ttk.Entry(adv, textvariable=self.calibration_var).grid(row=arow, column=1, sticky="ew", **pad)
        ttk.Button(adv, text="Browse...", command=self._browse_calibration).grid(row=arow, column=2, **pad)
        Tooltip(
            calib_label, key="calibration_file",
            text="Currently a compatibility placeholder -- calibration is never loaded from or saved "
            "to disk; every run recalibrates the keyboard position interactively.",
            prefs=self.prefs,
        )
        arow += 1

    # -- browse / drop handlers ---------------------------------------------
    def _browse_video(self):
        path = filedialog.askopenfilename(title="Select video file", initialdir=DOWNLOADS_DIR)
        if path:
            self.video_var.set(path)
            if self.transcribe_var.get():
                self._maybe_default_out(path)

    def _paste_video(self):
        """Paste the clipboard (a YouTube URL) into the Video field, replacing
        whatever's currently there."""
        try:
            text = self.root.clipboard_get().strip()
        except Exception:
            text = ""
        if text:
            self.video_var.set(text)
            self.page1_status_var.set("Pasted video link from clipboard.")
            self.page1_status_label.configure(foreground=COLOR_OK)
            # A URL's title isn't known until it's downloaded, so don't guess
            # an Output JSON path from the raw URL string (that produced a
            # garbled path, e.g. treating "https://" as a filesystem path).
            # Leave Output JSON blank in that case -- extract_fingering.py
            # derives a sensible ~/Downloads/<video-title>.fingering.json
            # default itself once the video is resolved.
            if self.transcribe_var.get() and not text.startswith(("http://", "https://")):
                self._maybe_default_out(text)
        else:
            self.page1_status_var.set("Clipboard is empty — nothing to paste.")
            self.page1_status_label.configure(foreground=COLOR_DEL)

    def _on_transcribe_toggled(self):
        """Grey out the MIDI field(s) when transcribing from the video's own
        audio instead (no MIDI file needed), enable the ghost-note controls
        (only meaningful in this mode), re-derive the Output JSON default
        from the video path instead of a MIDI path, and default Align off:
        a transcribed MIDI is derived FROM this video's own audio, so its
        timeline already IS the video's timeline (offset is 0 by
        construction -- same reasoning extract_fingering.py itself uses to
        skip the auto-offset estimate), so there's nothing to tune by
        default. Left enabled rather than force-disabled, since a manual
        nudge is still occasionally useful for a residual transcription-
        model latency -- the user can re-check it if actually needed.

        Widgets referenced here may not exist yet (the Settings popover is
        built lazily, and page 3 is built once at startup) -- every access
        is guarded so this is safe to call from _select_mode before any
        widgets exist, and again once Settings is opened to sync state."""
        transcribing = self.transcribe_var.get()
        state = "disabled" if transcribing else "normal"
        self._set_midi_field_state(state)
        self._set_transcribe_opts_state("normal" if transcribing else "disabled")
        self._refresh_midi_visibility()
        if transcribing:
            self._align_var_before_transcribe = self.align_var.get()
            self.align_var.set(False)
            video = self.video_var.get().strip()
            if video and not video.startswith(("http://", "https://")):
                self._maybe_default_out(video)
        else:
            self.align_var.set(getattr(self, "_align_var_before_transcribe", True))

    def _on_render_toggled(self):
        """Render mode (Synthesia-style: lit-key colour drives hand assignment,
        no hand tracking) needs the notes to come from somewhere -- since a
        render has no real hands to track, Transcribe is forced on (and its
        checkbox disabled while Render is checked, so the two can't drift out
        of sync). Toggling Render off restores whatever Transcribe was set to
        before. See _on_transcribe_toggled's docstring re: guarded widget access."""
        rendering = self.render_var.get()
        if rendering:
            self._transcribe_var_before_render = self.transcribe_var.get()
            self.transcribe_var.set(True)
            self._on_transcribe_toggled()
        elif hasattr(self, "_transcribe_var_before_render"):
            # Only restore if Render was actually toggled on at some point;
            # otherwise this is just a state-resync call (e.g. opening
            # Settings) and transcribe_var must be left exactly as-is.
            self.transcribe_var.set(self._transcribe_var_before_render)
            self._on_transcribe_toggled()
        self._configure_if_exists("transcribe_check", state="disabled" if rendering else "normal")
        self._configure_if_exists("flip_render_hands_check", state="normal" if rendering else "disabled")
        self._configure_if_exists("pick_hand_colors_check", state="normal" if rendering else "disabled")

    def _configure_if_exists(self, attr_name, **kwargs):
        widget = getattr(self, attr_name, None)
        if widget is None:
            return
        try:
            if widget.winfo_exists():
                widget.configure(**kwargs)
        except tk.TclError:
            pass

    def _set_midi_field_state(self, state):
        for entry_attr, btn_attr in (
            ("midi_entry", "midi_browse_button"),
            ("midi_entry_settings", "midi_browse_button_settings"),
        ):
            self._configure_if_exists(entry_attr, state=state)
            self._configure_if_exists(btn_attr, state=state)

    def _set_transcribe_opts_state(self, state):
        frame = getattr(self, "transcribe_opts_frame", None)
        if frame is None or not frame.winfo_exists():
            return
        for child in frame.winfo_children():
            if isinstance(child, ttk.Entry):
                child.configure(state=state)

    def _browse_midi(self):
        path = filedialog.askopenfilename(
            title="Select MIDI file", initialdir=DOWNLOADS_DIR,
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")]
        )
        if path:
            self.midi_var.set(path)
            self._maybe_default_out(path)

    def _browse_calibration(self):
        path = filedialog.asksaveasfilename(
            title="Calibration JSON path",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.calibration_var.set(path)

    def _browse_out(self):
        path = filedialog.asksaveasfilename(
            title="Output JSON path",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.out_var.set(path)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Output directory for fingering.json and .symple bundle")
        if path:
            self.output_dir_var.set(path)

    def _on_drop_video(self, event):
        self.video_var.set(_strip_dnd_braces(event.data))

    def _on_drop_midi(self, event):
        path = _strip_dnd_braces(event.data)
        self.midi_var.set(path)
        self._maybe_default_out(path)

    def _maybe_default_out(self, midi_path):
        """Auto-fill the Output JSON field to sit next to the MIDI file
        (same directory + basename, `.fingering.json` suffix), matching
        extract_fingering.py's own --out default -- but only if the user
        hasn't already customized Output JSON away from a prior default."""
        if not midi_path:
            return
        current = self.out_var.get().strip()
        if current and current != self._last_auto_out:
            return  # user edited it manually; leave it alone
        default_out = str(Path(midi_path).with_suffix("")) + ".fingering.json"
        self.out_var.set(default_out)
        self._last_auto_out = default_out

    def _open_output_folder(self):
        out_path = Path(self.out_var.get()).expanduser()
        folder = out_path.parent if out_path.parent.exists() else HERE
        self._log(f"Output folder: {folder.resolve()}\n")
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(folder)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            elif sys.platform.startswith("win"):
                os.startfile(str(folder))  # type: ignore[attr-defined]
        except Exception as exc:
            self._log(f"(could not open folder automatically: {exc})\n")

    # -- validation -----------------------------------------------------------
    def _validate(self):
        if self.batch_mode_var.get():
            if not self.batch_urls:
                return "Add at least one URL to the batch queue."
        else:
            video = self.video_var.get().strip()
            midi = self.midi_var.get().strip()
            if not video:
                return "Video field is required (paste a YouTube URL, or Browse/drop a local file)."
            if not midi and not self.transcribe_var.get():
                return "MIDI field is required (Browse/drop a .mid file), or choose a Transcribe-based mode."

        offset = self.offset_var.get().strip()
        if offset:
            try:
                float(offset)
            except ValueError:
                return "Offset must be a number (or blank for auto-estimate)."

        try:
            float(self.fps_var.get())
        except ValueError:
            return "FPS must be a number."

        try:
            float(self.min_hand_conf_var.get())
        except ValueError:
            return "Min hand confidence must be a number (0-1)."

        try:
            float(self.conf_var.get())
        except ValueError:
            return "Confidence threshold must be a number."

        if not self.calibration_var.get().strip():
            return "Calibration path must not be empty."
        # Output JSON is intentionally allowed to be blank: extract_fingering.py
        # derives a sensible default itself (next to --midi, or in ~/Downloads
        # named after the video's title when transcribing) when --out is omitted.

        onset_threshold = self.onset_threshold_var.get().strip()
        if onset_threshold:
            try:
                float(onset_threshold)
            except ValueError:
                return "Onset threshold must be a number (or blank for the model default)."

        try:
            int(self.min_velocity_var.get().strip() or "0")
        except ValueError:
            return "Min velocity must be a whole number (0-127)."

        try:
            float(self.min_duration_var.get().strip() or "0")
        except ValueError:
            return "Min duration must be a number of seconds."

        return None

    # -- run / stop -------------------------------------------------------------
    def _build_argv(self):
        self._live_frame_path = None  # reset each run
        if self.batch_mode_var.get():
            return self._build_batch_argv()
        argv = [
            self.python_exe,
            str(EXTRACT_SCRIPT),
            "--video", self.video_var.get().strip(),
            "--calibration", self.calibration_var.get().strip(),
            "--fps", str(float(self.fps_var.get())),
            "--min-hand-confidence", str(float(self.min_hand_conf_var.get())),
            "--confidence-threshold", str(float(self.conf_var.get())),
        ]

        out = self.out_var.get().strip()
        if out:
            argv += ["--out", out]

        output_dir = self.output_dir_var.get().strip()
        if output_dir:
            argv += ["--output-dir", output_dir]

        if self.transcribe_var.get():
            argv.append("--transcribe")
            if self.midi_only_var.get():
                argv.append("--midi-only")
            onset_threshold = self.onset_threshold_var.get().strip()
            if onset_threshold:
                argv += ["--onset-threshold", onset_threshold]
            min_velocity = self.min_velocity_var.get().strip()
            if min_velocity and min_velocity != "0":
                argv += ["--min-velocity", min_velocity]
            min_duration = self.min_duration_var.get().strip()
            if min_duration and float(min_duration) != 0.0:
                argv += ["--min-duration", min_duration]
            if self.no_gpu_var.get():
                argv.append("--no-gpu")
        else:
            argv += ["--midi", self.midi_var.get().strip()]

        offset = self.offset_var.get().strip()
        if offset:
            argv += ["--offset", offset]

        if self.preview_var.get():
            argv.append("--preview")

        if not self.align_var.get():
            argv.append("--no-align")

        if not self.bundle_var.get():
            argv.append("--no-bundle")
        else:
            # Include metadata in the bundle
            artist = self.artist_var.get().strip()
            title = self.title_var.get().strip()
            genre = self.genre_var.get().strip()
            difficulty = self.difficulty_var.get().strip()
            if artist:
                argv += ["--artist", artist]
            if title:
                argv += ["--title", title]
            if genre:
                argv += ["--genre", genre]
            if difficulty:
                argv += ["--difficulty", difficulty]

        if self.render_var.get():
            argv.append("--render")
            if self.flip_render_hands_var.get():
                argv.append("--flip-render-hands")
            if self.pick_hand_colors_var.get():
                argv.append("--pick-hand-colors")

        if not self.midi_recover_var.get():
            argv.append("--no-midi-recover")
        if self.blob_recover_var.get():
            argv.append("--blob-recover")
        if not self.clahe_var.get():
            argv.append("--no-clahe")

        if self.live_view_var.get() and _HAS_PIL:
            self._live_frame_path = tempfile.mktemp(suffix=".jpg", prefix="smape_live_")
            argv += ["--live-frame-path", self._live_frame_path]

        return argv

    def _on_run(self):
        if self.proc is not None:
            return

        error = self._validate()
        if error:
            self._set_status(error, error=True)
            return

        if self.batch_mode_var.get():
            self._init_batch_progress()

        argv = self._build_argv()

        # The transcription stack (CPU torch + the Kong model package) is not
        # part of the lite bundle/venv -- offer a one-time in-place install
        # into whatever Python will run the job.
        commands = []
        if "--transcribe" in argv and not self._transcription_deps_present():
            import tkinter.messagebox as msgbox
            gpu_requested = os.name == "nt" and self.gpu_torch_var.get()
            size_note = "~2.5 GB: CUDA-enabled PyTorch" if gpu_requested else "~1 GB: CPU-only PyTorch"
            if not msgbox.askyesno(
                "Install transcription support?",
                f"Audio-to-MIDI transcription needs a one-time extra download "
                f"({size_note} + the piano transcription model "
                "package).\n\nDownload and install it now into SMaPE's own "
                "Python runtime? Nothing outside SMaPE is modified.",
            ):
                self._set_status("Transcription support not installed — run cancelled.", error=True)
                return
            # A system-installed bundle (e.g. the .deb's /opt/smape) is not
            # user-writable -- install into the user site-packages instead,
            # which the runtime interpreter picks up automatically.
            prefix = Path(self.python_exe).resolve().parent
            if prefix.name in ("bin", "Scripts"):
                prefix = prefix.parent
            user_flag = [] if os.access(prefix, os.W_OK) else ["--user"]
            pip = [self.python_exe, "-m", "pip", "install",
                   "--no-warn-script-location"] + user_flag
            # Windows: plain PyPI torch there is CPU-only (the CUDA builds
            # only exist on download.pytorch.org). Linux: PyPI torch already
            # bundles CUDA, so no separate GPU install step is ever needed
            # there -- the cpu index stays the default to avoid the
            # multi-GB CUDA build when the user hasn't asked for it.
            if os.name == "nt":
                cpu_torch_cmd = pip + ["torch"]
            else:
                cpu_torch_cmd = pip + ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"]
            if os.name == "nt" and self.gpu_torch_var.get():
                # download.pytorch.org's CloudFront TLS is broken by some
                # AV/network filters (observed in the wild:
                # SSLV3_ALERT_HANDSHAKE_FAILURE) even with AV/VPN disabled --
                # most likely the bundled runtime's own OpenSSL build failing
                # to negotiate with CloudFront, not a network block. If the
                # automated install fails, fall back to the CPU-only build
                # (so the run isn't left broken) and print manual steps.
                cuda_torch_cmd = pip + ["torch", "--index-url", "https://download.pytorch.org/whl/cu121"]
                fallback_msg = (
                    "\n[GUI] CUDA PyTorch download failed (commonly a TLS/handshake issue "
                    "against download.pytorch.org, not necessarily your network) -- falling "
                    "back to the CPU-only build so this run can continue.\n"
                    "[GUI] To get GPU acceleration manually instead: download the matching "
                    "wheel via a browser (often succeeds where this automated download "
                    "didn't) from https://download.pytorch.org/whl/cu121/ , then run:\n"
                    f'[GUI]   "{self.python_exe}" -m pip install --no-warn-script-location '
                    "<path-to-downloaded-torch-whl>\n\n"
                )
                commands.append(("try_then_fallback", cuda_torch_cmd, cpu_torch_cmd, fallback_msg))
            else:
                commands.append(cpu_torch_cmd)
            commands.append(pip + ["piano_transcription_inference"])
        commands.append(argv)

        self._clear_log()
        self._set_status("Running...", error=False)
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.continue_button.pack_forget()
        if self._live_frame_path:
            self._live_polling_active = True
            self._poll_live_frame()

        def run_one(cmd):
            self.log_queue.put(("line", f"$ {' '.join(cmd)}\n\n"))
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    # Decode the child's output as UTF-8 (it emits UTF-8 --
                    # we set PYTHONIOENCODING/PYTHONUTF8 for it above). Without
                    # this, text mode uses the parent's locale codec (cp1252 on
                    # Windows), which chokes on the first non-cp1252 byte (e.g.
                    # 0x90 from pip/yt-dlp/ffmpeg output, or any UTF-8 multibyte
                    # char) and killed the whole run with a "charmap codec can't
                    # decode" error surfaced as a bogus "Failed to launch". Any
                    # stray undecodable byte from a non-Python child becomes the
                    # replacement char instead of crashing the reader.
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    **POPEN_KWARGS,
                )
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    self.log_queue.put(("line", line))
                return self.proc.wait()
            except Exception as exc:
                self.log_queue.put(("line", f"\n[GUI] Failed to launch subprocess: {exc}\n"))
                return -1

        def worker():
            returncode = -1
            for cmd in commands:
                if isinstance(cmd, tuple) and cmd[0] == "try_then_fallback":
                    _, primary, fallback, fallback_msg = cmd
                    returncode = run_one(primary)
                    if returncode != 0:
                        self.log_queue.put(("line", fallback_msg))
                        returncode = run_one(fallback)
                else:
                    returncode = run_one(cmd)
                if returncode != 0:
                    break
            self.log_queue.put(("done", returncode))

        self.reader_thread = threading.Thread(target=worker, daemon=True)
        self.reader_thread.start()

    def _on_stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self._log("\n[GUI] Stopping subprocess...\n")
            try:
                self.proc.terminate()
            except Exception as exc:
                self._log(f"[GUI] Error terminating: {exc}\n")

    # -- thread-safe log queue polling -------------------------------------------
    def _poll_log_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "line":
                    self._log(payload)
                    # Parse batch progress lines
                    if self.batch_mode_var.get():
                        stripped = payload.strip()
                        if stripped.startswith("--- Phase 2:") and stripped.endswith("---"):
                            title = stripped[len("--- Phase 2:"):].rstrip("-").strip()
                            # Mark previous as done if still running
                            if self._batch_current_idx >= 0:
                                if self._get_batch_item_status(self._batch_current_idx) == "▶ running":
                                    self._update_batch_item(self._batch_current_idx, status="✓ done")
                            self._batch_current_idx += 1
                            self._update_batch_item(self._batch_current_idx, label=title, status="▶ running")
                        elif stripped.startswith("FAILED:") and self._batch_current_idx >= 0:
                            self._update_batch_item(self._batch_current_idx, status="✗ failed")
                        elif stripped.startswith("batch done:"):
                            for i in range(len(self._batch_tree_ids)):
                                if self._get_batch_item_status(i) == "▶ running":
                                    self._update_batch_item(i, status="✓ done")
                    # Track output paths for the metadata page (single-video only)
                    if not self.batch_mode_var.get():
                        s = payload.strip()
                        if s.startswith("Wrote ") and s.endswith(".fingering.json"):
                            p = s[len("Wrote "):].strip()
                            if os.path.exists(p):
                                self._last_run_json = p
                        elif s.startswith("Wrote ") and ".symple" in s:
                            tail = s[len("Wrote "):].strip()
                            idx = tail.find(".symple")
                            p = tail[:idx + len(".symple")] if idx >= 0 else ""
                            if p.endswith(".symple") and os.path.exists(p):
                                self._last_run_bundle = p
                    # Extract video title from a yt-dlp download log line. Two
                    # patterns, depending on whether the file is fresh or cached:
                    #   "[download] Destination: /path/Title Here.mp4"
                    #   "[download] /path/Title Here.mp4 has already been downloaded"
                    if (
                        not self.extracted_video_title
                        and "[download]" in payload
                        and ".mp4" in payload
                        and ("Destination:" in payload or "has already been downloaded" in payload)
                    ):
                        try:
                            start = payload.rfind("/") + 1
                            end = payload.find(".mp4")
                            if start > 0 and end > start:
                                title = payload[start:end]
                                # The 20s calibration clip downloads as
                                # "preview.mp4" -- not the video's title.
                                if title and title.lower() != "preview":
                                    self.extracted_video_title = title
                                    self._autofill_from_extracted_title()
                        except Exception:
                            pass
                elif kind == "done":
                    self._on_process_done(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _on_process_done(self, returncode):
        self._live_polling_active = False
        # Clean up the temp frame file (keep last frame in the label until next run)
        if self._live_frame_path and os.path.exists(self._live_frame_path):
            try:
                os.remove(self._live_frame_path)
            except OSError:
                pass
        self._live_frame_path = None

        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.proc = None
        if self.batch_mode_var.get():
            if returncode == 0:
                self._set_status(f"Batch done — {len(self.batch_urls)} video(s) processed. See log for paths.", error=False)
            else:
                self._set_status("Batch finished with failures — see log above for details.", error=True)
        elif returncode == 0:
            self._autofill_metadata()
            self.p4_status_var.set(
                f"Bundle ready: {Path(self._last_run_bundle).name}"
                if self._last_run_bundle and os.path.exists(self._last_run_bundle)
                else "Run completed — fill in metadata and save to bundle."
            )
            self.p4_status_label.configure(fg=COLOR_OK)
            # Stay on the run page so the user can review the last tracking frame
            self._set_status("Done — review tracking above, then continue to metadata.", error=False)
            self.continue_button.pack(side="right")
        else:
            self._set_status(f"Failed (exit code {returncode}) — see log above for details.", error=True)

    # -- batch helpers ----------------------------------------------------------
    def _on_batch_toggled(self):
        if self.batch_mode_var.get():
            self.batch_queue_frame.pack(fill="x", pady=(8, 0))
        else:
            self.batch_queue_frame.pack_forget()

    def _add_to_batch(self):
        url = self.video_var.get().strip()
        if not url:
            self.page1_status_var.set("Enter a URL in the field above first.")
            self.page1_status_label.configure(foreground=COLOR_DEL)
            return
        if url in self.batch_urls:
            self.page1_status_var.set("That URL is already in the queue.")
            self.page1_status_label.configure(foreground=COLOR_WARN)
            return
        self.batch_urls.append(url)
        self.batch_listbox.insert("end", url)
        self.batch_count_label.configure(text=f"{len(self.batch_urls)} URL(s)")
        self.page1_status_var.set(f"Added to queue. {len(self.batch_urls)} URL(s) queued.")
        self.page1_status_label.configure(foreground=COLOR_OK)

    def _remove_from_batch(self):
        sel = self.batch_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.batch_listbox.delete(idx)
        del self.batch_urls[idx]
        self.batch_count_label.configure(text=f"{len(self.batch_urls)} URL(s)" if self.batch_urls else "")

    def _clear_batch(self):
        self.batch_listbox.delete(0, "end")
        self.batch_urls.clear()
        self.batch_count_label.configure(text="")

    def _refresh_batch_progress_visibility(self):
        if self.batch_mode_var.get():
            self.batch_progress_frame.pack(fill="x")
        else:
            self.batch_progress_frame.pack_forget()

    def _init_batch_progress(self):
        for item in self.queue_tree.get_children():
            self.queue_tree.delete(item)
        self._batch_tree_ids = []
        self._batch_current_idx = -1
        for url in self.batch_urls:
            item_id = self.queue_tree.insert("", "end", values=(f"⏳  {url}",))
            self._batch_tree_ids.append(item_id)

    def _update_batch_item(self, idx: int, label: str = None, status: str = None):
        if 0 <= idx < len(self._batch_tree_ids):
            item_id = self._batch_tree_ids[idx]
            if label is not None or status is not None:
                cur = self.queue_tree.item(item_id, "values")
                cur_text = cur[0] if cur else ""
                # Separate stored label (after first emoji+spaces) from prefix
                if status is None:
                    # Keep current status emoji, update label
                    prefix = cur_text.split("  ", 1)[0] + "  " if "  " in cur_text else "  "
                    new_text = prefix + (label or "")
                elif label is None:
                    # Keep current label, update status emoji
                    existing_label = cur_text.split("  ", 1)[1] if "  " in cur_text else cur_text
                    emoji = {"⏳ pending": "⏳", "▶ running": "▶", "✓ done": "✓", "✗ failed": "✗"}.get(status, "")
                    new_text = f"{emoji}  {existing_label}"
                else:
                    emoji = {"⏳ pending": "⏳", "▶ running": "▶", "✓ done": "✓", "✗ failed": "✗"}.get(status, "")
                    new_text = f"{emoji}  {label}"
                self.queue_tree.item(item_id, values=(new_text,))
            self.queue_tree.see(item_id)

    def _get_batch_item_status(self, idx: int) -> str:
        if 0 <= idx < len(self._batch_tree_ids):
            vals = self.queue_tree.item(self._batch_tree_ids[idx], "values")
            if vals:
                text = vals[0]
                if text.startswith("▶"):
                    return "▶ running"
                if text.startswith("✓"):
                    return "✓ done"
                if text.startswith("✗"):
                    return "✗ failed"
        return "⏳ pending"

    def _build_batch_argv(self) -> list:
        argv = [self.python_exe, str(EXTRACT_SCRIPT), "--batch"] + list(self.batch_urls) + [
            "--transcribe",
            "--calibration", self.calibration_var.get().strip(),
            "--fps", str(float(self.fps_var.get())),
            "--min-hand-confidence", str(float(self.min_hand_conf_var.get())),
            "--confidence-threshold", str(float(self.conf_var.get())),
        ]
        output_dir = self.output_dir_var.get().strip()
        if output_dir:
            argv += ["--output-dir", output_dir]
        if not self.bundle_var.get():
            argv.append("--no-bundle")
        if self.render_var.get():
            argv.append("--render")
            if self.flip_render_hands_var.get():
                argv.append("--flip-render-hands")
            if self.pick_hand_colors_var.get():
                argv.append("--pick-hand-colors")
        elif self.midi_only_var.get():
            argv.append("--midi-only")
        if self.no_gpu_var.get():
            argv.append("--no-gpu")
        if not self.midi_recover_var.get():
            argv.append("--no-midi-recover")
        if self.blob_recover_var.get():
            argv.append("--blob-recover")
        if not self.clahe_var.get():
            argv.append("--no-clahe")
        onset_threshold = self.onset_threshold_var.get().strip()
        if onset_threshold:
            argv += ["--onset-threshold", onset_threshold]
        return argv

    # -- log / status helpers -----------------------------------------------
    def _log(self, text):
        # Only auto-scroll when the user is already at (or near) the bottom,
        # so manually scrolling up to read earlier output isn't interrupted.
        at_bottom = self.log_text.yview()[1] >= 0.98
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        if at_bottom:
            self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.live_view_label.configure(image="", height=0)
        self._live_photo = None

    # -- live tracking frame polling ----------------------------------------
    def _poll_live_frame(self):
        if not self._live_polling_active:
            return
        path = self._live_frame_path
        if path and os.path.exists(path) and _HAS_PIL:
            try:
                img = Image.open(path)
                img.load()  # force read before file may be overwritten
                avail_w = max(200, self.live_view_label.winfo_width())
                max_h = 220
                scale = min(avail_w / img.width, max_h / img.height, 1.0)
                new_w = max(1, int(img.width * scale))
                new_h = max(1, int(img.height * scale))
                img = img.resize((new_w, new_h), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                self._live_photo = tk.PhotoImage(data=buf.getvalue())
                self.live_view_label.configure(image=self._live_photo, height=new_h)
            except Exception:
                pass
        self.root.after(150, self._poll_live_frame)

    def _set_status(self, text, error):
        self.status_var.set(text)
        self.status_label.configure(fg=COLOR_DEL if error else COLOR_OK)


def main():
    if _HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    FingeringGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
