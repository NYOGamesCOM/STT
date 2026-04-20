#!/usr/bin/env python3
"""
STT (Speech to text) — voice-to-text desktop app
Hold hotkey → speak → release → transcribed text is typed into the focused window.

Usage:
    python stt.py

Packaging (Windows):
    pyinstaller --onefile --noconsole --name STT stt.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import logging
import queue
import sys
import threading
import time
from enum import Enum, auto
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import sounddevice as sd
import keyboard
import pyperclip
import pystray
from PIL import Image, ImageDraw
from faster_whisper import WhisperModel


# ═════════════════════════════════════════════════════════════════════════════
# Paths — PyInstaller-safe
# ═════════════════════════════════════════════════════════════════════════════

def _app_dir() -> Path:
    """Directory containing the executable (or script), works in frozen builds."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


APP_DIR = _app_dir()
CONFIG_PATH = APP_DIR / "config.json"
LOG_PATH = APP_DIR / "stt.log"
HISTORY_PATH = APP_DIR / "history.jsonl"
MARKSOFT_URL = "https://marksoft.ro"


# ═════════════════════════════════════════════════════════════════════════════
# Logging
# ═════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("stt")


# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG: dict = {
    "model": "base",          # "tiny" | "base" | "small" — Whisper model size
    "hotkey": "right alt",    # Key name as recognised by the keyboard library
    "language": "en",         # None = auto-detect (slower); or e.g. "en", "ro"
    "sample_rate": 16000,     # Hz — Whisper expects 16 kHz
    "device": "cpu",          # "cpu" | "cuda"
    "compute_type": "int8",   # "int8" | "float16" | "float32"
    "paste_method": "clipboard",  # "clipboard" | "type"
    "min_audio_seconds": 0.5,     # Ignore recordings shorter than this
    "beam_size": 1,           # 1 = greedy (fastest); 5 = max accuracy
    "cpu_threads": 0,         # 0 = auto (use all available cores)
    "persistent_mic": False,  # True = keep mic stream open (lower latency, but
                              #        some Windows PyInstaller bundles capture silence)
    "vad_min_seconds": 3.0,   # Skip VAD for clips shorter than this (faster)
    "show_overlay": True,     # Bottom-center on-screen indicator
    "show_main_window": True, # Open the main window on launch
    "start_minimized": False, # Start in tray instead of showing the window
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Merge: new keys from DEFAULT_CONFIG survive upgrades
            return {**DEFAULT_CONFIG, **data}
        except Exception as exc:
            log.warning("Could not load config (%s); using defaults.", exc)
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        log.debug("Config saved → %s", CONFIG_PATH)
    except Exception as exc:
        log.error("Could not save config: %s", exc)


# ═════════════════════════════════════════════════════════════════════════════
# App state
# ═════════════════════════════════════════════════════════════════════════════

class AppState(Enum):
    LOADING_MODEL = auto()
    IDLE = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()


# ═════════════════════════════════════════════════════════════════════════════
# Tray icon images
# ═════════════════════════════════════════════════════════════════════════════

_ICON_SIZE = 64
_ICON_BG: dict[AppState, tuple] = {
    AppState.LOADING_MODEL: (60,  110, 220, 255),   # Blue
    AppState.IDLE:          (45,  185,  78, 255),   # Green
    AppState.RECORDING:     (210,  50,  50, 255),   # Red
    AppState.TRANSCRIBING:  (230, 160,  20, 255),   # Amber
}
_ICON_TOOLTIPS: dict[AppState, str] = {
    AppState.LOADING_MODEL: "STT (Speech to text) — Loading model…",
    AppState.IDLE:          "STT (Speech to text) — Idle  (hold hotkey to record)",
    AppState.RECORDING:     "STT (Speech to text) — Recording…",
    AppState.TRANSCRIBING:  "STT (Speech to text) — Transcribing…",
}

_ICON_CACHE: dict[AppState, Image.Image] = {}


def _make_icon(state: AppState) -> Image.Image:
    """Draw a simple coloured-circle icon with a symbolic mark."""
    if state in _ICON_CACHE:
        return _ICON_CACHE[state]

    sz = _ICON_SIZE
    img = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bg = _ICON_BG[state]

    # Outer coloured disc
    d.ellipse([2, 2, sz - 2, sz - 2], fill=bg, outline=(255, 255, 255, 160), width=2)

    w = (255, 255, 255, 220)  # white

    if state == AppState.IDLE:
        # Microphone body
        d.rounded_rectangle([25, 12, 39, 34], radius=7, fill=w)
        # Stand arc
        d.arc([16, 26, 48, 48], start=0, end=180, fill=w, width=3)
        # Stem + base
        d.line([32, 48, 32, 54], fill=w, width=3)
        d.line([24, 54, 40, 54], fill=w, width=3)

    elif state == AppState.RECORDING:
        # Filled circle = recording indicator
        d.ellipse([18, 18, 46, 46], fill=w)

    elif state == AppState.TRANSCRIBING:
        # Three dots = processing
        for cx in [18, 32, 46]:
            d.ellipse([cx - 5, 27, cx + 5, 37], fill=w)

    elif state == AppState.LOADING_MODEL:
        # Hourglass silhouette
        pts_top = [(18, 16), (46, 16), (32, 32)]
        pts_bot = [(18, 48), (46, 48), (32, 32)]
        d.polygon(pts_top, fill=w)
        d.polygon(pts_bot, fill=w)

    _ICON_CACHE[state] = img
    return img


# ═════════════════════════════════════════════════════════════════════════════
# Audio recorder
# ═════════════════════════════════════════════════════════════════════════════

class AudioRecorder:
    """Captures audio from the default microphone.

    Two modes:
      - persistent=True  → mic stream stays open continuously; start()/stop()
        just toggle a capture flag. Eliminates device-open latency (~100–300 ms).
      - persistent=False → stream is opened on start() and closed on stop().

    Exposes `level` (0.0–1.0) from every callback so the overlay can draw a
    live waveform without re-sampling audio itself.
    """

    def __init__(self, sample_rate: int = 16000, persistent: bool = True):
        self.sample_rate = sample_rate
        self.persistent = persistent
        self._frames: list[np.ndarray] = []
        self._active = False
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self.level: float = 0.0  # Live RMS, updated every callback

    # ── Internal ──────────────────────────────────────────────────────────────

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.debug("sounddevice status: %s", status)
        # Cheap RMS for overlay animation (always updated, not just when active)
        try:
            rms = float(np.sqrt(np.mean(indata * indata)))
            # Scale roughly into 0..1 for normal speech
            self.level = min(1.0, rms * 6.0)
        except Exception:
            pass
        if self._active:
            with self._lock:
                self._frames.append(indata.copy())

    def _open_stream(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._callback,
        )
        self._stream.start()
        log.info("Audio stream opened (%.0f Hz, persistent=%s)",
                 self.sample_rate, self.persistent)

    def _close_stream(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as exc:
            log.warning("Error closing audio stream: %s", exc)
        self._stream = None

    # ── Public ────────────────────────────────────────────────────────────────

    def prime(self) -> None:
        """Pre-open the mic stream (persistent mode)."""
        if self.persistent:
            try:
                self._open_stream()
            except Exception as exc:
                log.error("Priming audio stream failed: %s", exc)

    def start(self) -> None:
        with self._lock:
            self._frames = []
        self._active = True
        if not self.persistent:
            try:
                self._open_stream()
            except Exception as exc:
                self._active = False
                log.error("Failed to open audio stream: %s", exc)
                raise
        log.info("Audio recording started.")

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a 1-D float32 array."""
        self._active = False
        if not self.persistent:
            self._close_stream()

        with self._lock:
            frames = self._frames[:]
            self._frames = []

        if not frames:
            log.info("No audio captured.")
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames, axis=0).flatten()
        duration = len(audio) / self.sample_rate
        log.info("Recording stopped — %.2f s captured.", duration)
        return audio

    def shutdown(self) -> None:
        self._active = False
        self._close_stream()


# ═════════════════════════════════════════════════════════════════════════════
# Transcriber (wraps faster-whisper)
# ═════════════════════════════════════════════════════════════════════════════

class Transcriber:
    """Lazy-loading wrapper around WhisperModel."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        cpu_threads: int = 0,
        vad_min_seconds: float = 3.0,
        sample_rate: int = 16000,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = max(1, int(beam_size))
        self.cpu_threads = max(0, int(cpu_threads))
        self.vad_min_seconds = float(vad_min_seconds)
        self.sample_rate = int(sample_rate)
        self._model: WhisperModel | None = None
        self._ready = threading.Event()
        self._loading = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_async(
        self,
        on_start=None,
        on_done=None,
        on_error=None,
    ) -> threading.Thread:
        """Download / load the model in a daemon thread."""

        def _run():
            self._loading = True
            if on_start:
                on_start()
            try:
                log.info("Loading WhisperModel '%s' (device=%s, compute=%s, threads=%s)…",
                         self.model_name, self.device, self.compute_type, self.cpu_threads or "auto")
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    num_workers=1,
                )
                self._warm_up()
                self._ready.set()
                log.info("Model '%s' ready.", self.model_name)
                if on_done:
                    on_done()
            except Exception as exc:
                log.error("Model load failed: %s", exc)
                if on_error:
                    on_error(exc)
            finally:
                self._loading = False

        t = threading.Thread(target=_run, name="model-loader", daemon=True)
        t.start()
        return t

    def _warm_up(self) -> None:
        """Run one tiny silent clip so CTranslate2 compiles kernels up front —
        otherwise the very first real transcription pays that cost."""
        if self._model is None:
            return
        try:
            t0 = time.perf_counter()
            dummy = np.zeros(16000, dtype=np.float32)  # 1 s of silence
            segs, _ = self._model.transcribe(
                dummy,
                beam_size=1,
                vad_filter=False,
                without_timestamps=True,
                condition_on_previous_text=False,
            )
            for _ in segs:
                pass
            log.info("Model warm-up done in %.2fs.", time.perf_counter() - t0)
        except Exception as exc:
            log.debug("Warm-up skipped: %s", exc)

    # ── Transcription ─────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        """Return transcribed text. Raises if model not loaded."""
        if self._model is None:
            raise RuntimeError("Model not loaded yet.")
        if len(audio) == 0:
            return ""

        t0 = time.perf_counter()
        # Skip VAD on short clips — the VAD pass costs more than it saves.
        duration = len(audio) / max(1, self.sample_rate)
        use_vad = duration >= self.vad_min_seconds
        try:
            kwargs = dict(
                language=language,
                beam_size=self.beam_size,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            if use_vad:
                kwargs["vad_filter"] = True
                kwargs["vad_parameters"] = {"min_silence_duration_ms": 500}
            segments, _info = self._model.transcribe(audio, **kwargs)
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:
            # VAD can fail in rare edge cases — retry without it
            log.warning("Transcription failed (%s); retrying without VAD.", exc)
            segments, _info = self._model.transcribe(
                audio,
                language=language,
                beam_size=self.beam_size,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()

        log.info("Transcribed in %.2fs: %r", time.perf_counter() - t0, text)
        return text


# ═════════════════════════════════════════════════════════════════════════════
# Text injection
# ═════════════════════════════════════════════════════════════════════════════

def inject_text(text: str, method: str = "clipboard") -> None:
    """Type/paste *text* into whatever window currently has focus."""
    if not text:
        return

    # Tiny settle so the hotkey-release event is flushed before we send input
    time.sleep(0.02)

    if method == "clipboard":
        _inject_via_clipboard(text)
    else:
        _inject_via_type(text)


def _restore_clipboard_async(old: str, delay: float = 0.35) -> None:
    """Restore the previous clipboard contents after the paste has landed.
    Runs on a daemon thread so the worker can return to IDLE immediately."""
    def _worker():
        try:
            time.sleep(delay)
            pyperclip.copy(old)
        except Exception:
            pass
    threading.Thread(target=_worker, name="clipboard-restore", daemon=True).start()


def _inject_via_clipboard(text: str) -> None:
    old: str = ""
    try:
        old = pyperclip.paste() or ""
    except Exception:
        pass

    try:
        pyperclip.copy(text)
        # Shorter settle than before — clipboard writes are near-instant on Win10+
        time.sleep(0.015)
        keyboard.send("ctrl+v")
    except Exception as exc:
        log.error("Clipboard injection failed: %s — falling back to type.", exc)
        _inject_via_type(text)
        return

    # Restore asynchronously so transcription worker isn't blocked.
    _restore_clipboard_async(old)


def _inject_via_type(text: str) -> None:
    try:
        keyboard.write(text, delay=0.008)
    except Exception as exc:
        log.error("Type injection failed: %s", exc)


# ═════════════════════════════════════════════════════════════════════════════
# Hotkey dialog (tkinter, runs in a daemon thread)
# ═════════════════════════════════════════════════════════════════════════════

# Map Tk keysym names → keyboard-library names
_TK_KEY_MAP: dict[str, str] = {
    "alt_r":       "right alt",
    "alt_l":       "left alt",
    "control_r":   "right ctrl",
    "control_l":   "left ctrl",
    "shift_r":     "right shift",
    "shift_l":     "left shift",
    "super_r":     "right windows",
    "super_l":     "left windows",
    "caps_lock":   "caps lock",
    "scroll_lock": "scroll lock",
    "num_lock":    "num lock",
    "f1":  "f1",  "f2":  "f2",  "f3":  "f3",  "f4":  "f4",
    "f5":  "f5",  "f6":  "f6",  "f7":  "f7",  "f8":  "f8",
    "f9":  "f9",  "f10": "f10", "f11": "f11", "f12": "f12",
}


def show_hotkey_dialog(current: str, on_set) -> None:
    """Display a small window that listens for a keypress and reports it."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        log.error("tkinter is not available — cannot show hotkey dialog.")
        return

    root = tk.Tk()
    root.title("Set Hotkey — STT (Speech to text)")
    root.geometry("340x170")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    # Centre on screen
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw - 340) // 2}+{(sh - 170) // 2}")

    captured = [current]

    tk.Label(
        root,
        text="Press the key you want to use as the\nhold-to-record hotkey, then click Set.",
        pady=12, font=("Segoe UI", 10),
    ).pack()

    var = tk.StringVar(value=current)
    entry = ttk.Entry(root, textvariable=var, state="readonly",
                      justify="center", font=("Segoe UI", 13, "bold"))
    entry.pack(pady=4, padx=24, fill="x")

    def on_key(event: tk.Event) -> None:
        sym = event.keysym.lower()
        name = _TK_KEY_MAP.get(sym, sym.replace("_", " "))
        captured[0] = name
        var.set(name)

    root.bind("<KeyPress>", on_key)
    root.focus_force()

    btn_row = tk.Frame(root)
    btn_row.pack(pady=10)

    def do_set():
        on_set(captured[0])
        root.destroy()

    ttk.Button(btn_row, text="Set",    command=do_set,        width=10).pack(side="left", padx=6)
    ttk.Button(btn_row, text="Cancel", command=root.destroy,  width=10).pack(side="left", padx=6)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


# ═════════════════════════════════════════════════════════════════════════════
# Transcription history  (local JSONL store + dark-themed browser window)
# ═════════════════════════════════════════════════════════════════════════════

_HISTORY_LOCK = threading.Lock()
_HISTORY_MAX_ENTRIES = 1000


def append_history(text: str, *, model: str, language: str | None,
                   duration: float) -> None:
    """Append one transcription to history.jsonl (fire-and-forget)."""
    if not text:
        return

    entry = {
        "ts":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "text":     text,
        "model":    model,
        "language": language or "",
        "duration": round(duration, 2),
    }

    def _worker():
        try:
            with _HISTORY_LOCK:
                with open(HISTORY_PATH, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            log.debug("History append failed: %s", exc)

    threading.Thread(target=_worker, name="history-append", daemon=True).start()


def load_history() -> list[dict]:
    """Return history entries, newest first. Caps to _HISTORY_MAX_ENTRIES."""
    if not HISTORY_PATH.exists():
        return []
    out: list[dict] = []
    try:
        with _HISTORY_LOCK:
            with open(HISTORY_PATH, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
    except Exception as exc:
        log.warning("Could not read history: %s", exc)
        return []

    # Trim on read if the file has grown too large
    if len(out) > _HISTORY_MAX_ENTRIES:
        out = out[-_HISTORY_MAX_ENTRIES:]
        try:
            with _HISTORY_LOCK:
                with open(HISTORY_PATH, "w", encoding="utf-8") as fh:
                    for e in out:
                        fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        except Exception:
            pass

    out.reverse()  # newest first
    return out


def _rewrite_history(entries_newest_first: list[dict]) -> None:
    """Persist the given list back to disk (oldest-first on disk)."""
    try:
        with _HISTORY_LOCK:
            with open(HISTORY_PATH, "w", encoding="utf-8") as fh:
                for e in reversed(entries_newest_first):
                    fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.warning("Could not rewrite history: %s", exc)


def clear_history() -> None:
    try:
        with _HISTORY_LOCK:
            if HISTORY_PATH.exists():
                HISTORY_PATH.unlink()
    except Exception as exc:
        log.warning("Could not clear history: %s", exc)


# ── History window ───────────────────────────────────────────────────────────

# Dark palette — shared with the overlay
_UI_BG      = "#14161a"
_UI_BG2     = "#1c1f25"
_UI_BG3     = "#262a32"
_UI_FG      = "#e7e9ee"
_UI_FG_DIM  = "#8a93a2"
_UI_ACCENT  = "#3b82f6"
_UI_DANGER  = "#ef4444"
_UI_BORDER  = "#2f343d"


def show_history_window() -> None:
    """Open the dark-themed transcription history browser."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        import webbrowser
    except ImportError:
        log.error("tkinter unavailable — history window disabled.")
        return

    root = tk.Tk()
    root.title("History — STT (Speech to text)")
    root.geometry("720x520")
    root.minsize(560, 420)
    root.configure(bg=_UI_BG)

    # Centre on screen
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    x = (sw - 720) // 2
    y = (sh - 520) // 2
    root.geometry(f"+{x}+{y}")

    # ── ttk dark styling ──────────────────────────────────────────────────
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure(".", background=_UI_BG, foreground=_UI_FG,
                    fieldbackground=_UI_BG2, bordercolor=_UI_BORDER,
                    lightcolor=_UI_BG, darkcolor=_UI_BG)
    style.configure("TFrame", background=_UI_BG)
    style.configure("Card.TFrame", background=_UI_BG2)
    style.configure("TLabel", background=_UI_BG, foreground=_UI_FG,
                    font=("Segoe UI", 10))
    style.configure("Dim.TLabel", background=_UI_BG, foreground=_UI_FG_DIM,
                    font=("Segoe UI", 9))
    style.configure("Title.TLabel", background=_UI_BG, foreground=_UI_FG,
                    font=("Segoe UI Semibold", 13))
    style.configure("TEntry", fieldbackground=_UI_BG2, foreground=_UI_FG,
                    bordercolor=_UI_BORDER, insertcolor=_UI_FG)
    style.configure("TButton", background=_UI_BG3, foreground=_UI_FG,
                    bordercolor=_UI_BORDER, focusthickness=0,
                    padding=(12, 6), font=("Segoe UI", 10))
    style.map("TButton",
              background=[("active", _UI_BORDER), ("pressed", _UI_BORDER)])
    style.configure("Accent.TButton", background=_UI_ACCENT, foreground="#ffffff")
    style.map("Accent.TButton",
              background=[("active", "#2563eb"), ("pressed", "#1d4ed8")])
    style.configure("Danger.TButton", background=_UI_BG3, foreground=_UI_DANGER)
    style.map("Danger.TButton",
              background=[("active", "#3a1f22"), ("pressed", "#3a1f22")])

    # ── Layout: header / search / list / preview / footer ─────────────────
    outer = ttk.Frame(root, style="TFrame")
    outer.pack(fill="both", expand=True, padx=16, pady=14)

    header = ttk.Frame(outer, style="TFrame")
    header.pack(fill="x")
    ttk.Label(header, text="Transcription history",
              style="Title.TLabel").pack(side="left")

    search_var = tk.StringVar()
    search_row = ttk.Frame(outer, style="TFrame")
    search_row.pack(fill="x", pady=(10, 8))
    ttk.Label(search_row, text="Search", style="Dim.TLabel").pack(side="left", padx=(0, 8))
    search_entry = ttk.Entry(search_row, textvariable=search_var)
    search_entry.pack(side="left", fill="x", expand=True)

    # Listbox (Tk — ttk has no Listbox)
    body = ttk.Frame(outer, style="TFrame")
    body.pack(fill="both", expand=True)

    list_frame = tk.Frame(body, bg=_UI_BG2, highlightthickness=1,
                          highlightbackground=_UI_BORDER)
    list_frame.pack(side="top", fill="both", expand=True)

    scrollbar = tk.Scrollbar(list_frame, bg=_UI_BG2, troughcolor=_UI_BG,
                             activebackground=_UI_BG3)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(
        list_frame,
        bg=_UI_BG2, fg=_UI_FG,
        selectbackground=_UI_ACCENT, selectforeground="#ffffff",
        highlightthickness=0, bd=0,
        activestyle="none",
        font=("Segoe UI", 10),
        yscrollcommand=scrollbar.set,
    )
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    # Full-text preview
    preview_frame = tk.Frame(body, bg=_UI_BG2, highlightthickness=1,
                             highlightbackground=_UI_BORDER)
    preview_frame.pack(side="top", fill="x", pady=(10, 0))
    preview = tk.Text(preview_frame, height=6, bg=_UI_BG2, fg=_UI_FG,
                      insertbackground=_UI_FG, bd=0, highlightthickness=0,
                      wrap="word", padx=10, pady=8,
                      font=("Segoe UI", 10))
    preview.pack(fill="both", expand=True)
    preview.configure(state="disabled")

    # Action buttons
    btn_row = ttk.Frame(outer, style="TFrame")
    btn_row.pack(fill="x", pady=(12, 0))

    status_var = tk.StringVar(value="")
    ttk.Label(btn_row, textvariable=status_var, style="Dim.TLabel").pack(side="left")

    # ── State ─────────────────────────────────────────────────────────────
    all_entries: list[dict] = load_history()
    visible_entries: list[dict] = list(all_entries)

    def _fmt_row(e: dict) -> str:
        ts = e.get("ts", "")[:19].replace("T", " ")
        txt = (e.get("text", "") or "").replace("\n", " ")
        if len(txt) > 80:
            txt = txt[:77] + "…"
        return f"{ts}   {txt}"

    def refresh_list() -> None:
        q = search_var.get().strip().lower()
        visible_entries.clear()
        for e in all_entries:
            if not q or q in (e.get("text", "") or "").lower():
                visible_entries.append(e)
        listbox.delete(0, "end")
        for e in visible_entries:
            listbox.insert("end", _fmt_row(e))
        status_var.set(f"{len(visible_entries)} of {len(all_entries)} entries")
        _set_preview("")

    def _set_preview(text: str) -> None:
        preview.configure(state="normal")
        preview.delete("1.0", "end")
        preview.insert("1.0", text)
        preview.configure(state="disabled")

    def _selected_entry() -> dict | None:
        sel = listbox.curselection()
        if not sel:
            return None
        i = sel[0]
        if 0 <= i < len(visible_entries):
            return visible_entries[i]
        return None

    def on_select(_evt=None) -> None:
        e = _selected_entry()
        _set_preview((e or {}).get("text", ""))

    def do_copy(_evt=None) -> None:
        e = _selected_entry()
        if not e:
            return
        try:
            pyperclip.copy(e.get("text", ""))
            status_var.set("Copied to clipboard.")
        except Exception as exc:
            status_var.set(f"Copy failed: {exc}")

    def do_delete() -> None:
        e = _selected_entry()
        if not e:
            return
        try:
            all_entries.remove(e)
        except ValueError:
            return
        _rewrite_history(all_entries)
        refresh_list()
        status_var.set("Entry deleted.")

    def do_clear() -> None:
        if not all_entries:
            return
        if not messagebox.askyesno(
            "Clear history",
            "Remove all transcription history? This cannot be undone.",
            parent=root,
        ):
            return
        clear_history()
        all_entries.clear()
        refresh_list()
        status_var.set("History cleared.")

    ttk.Button(btn_row, text="Copy", style="Accent.TButton",
               command=do_copy).pack(side="right", padx=(6, 0))
    ttk.Button(btn_row, text="Delete",
               command=do_delete).pack(side="right", padx=(6, 0))
    ttk.Button(btn_row, text="Clear all", style="Danger.TButton",
               command=do_clear).pack(side="right", padx=(6, 0))

    # ── Footer (powered by MarkSoft) ──────────────────────────────────────
    footer = ttk.Frame(outer, style="TFrame")
    footer.pack(fill="x", pady=(14, 0))

    ttk.Label(footer, text="Powered by ", style="Dim.TLabel").pack(side="left")
    link = tk.Label(footer, text="MarkSoft", bg=_UI_BG, fg=_UI_ACCENT,
                    cursor="hand2", font=("Segoe UI", 9, "underline"))
    link.pack(side="left")
    link.bind("<Button-1>", lambda _e: webbrowser.open(MARKSOFT_URL))
    ttk.Label(footer, text=f"  ·  {MARKSOFT_URL}",
              style="Dim.TLabel").pack(side="left")

    # ── Events ────────────────────────────────────────────────────────────
    search_var.trace_add("write", lambda *_: refresh_list())
    listbox.bind("<<ListboxSelect>>", on_select)
    listbox.bind("<Double-Button-1>", do_copy)
    listbox.bind("<Return>", do_copy)
    root.bind("<Escape>", lambda _e: root.destroy())

    refresh_list()
    search_entry.focus_set()
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


# ═════════════════════════════════════════════════════════════════════════════
# On-screen overlay indicator  (WisperFlow-style pill, bottom-centre)
# ═════════════════════════════════════════════════════════════════════════════

class Overlay:
    """
    Small, borderless, always-on-top, click-through pill shown at the bottom
    centre of the primary monitor while recording/transcribing.

    Owns its own Tk root and mainloop on a dedicated daemon thread — all
    state changes from other threads are marshalled via `root.after(0, ...)`.
    """

    W, H = 150, 42                       # Pill size
    MARGIN_BOTTOM = 60                   # Gap above taskbar
    BG       = "#14161a"
    FG_IDLE  = "#7a8290"
    FG_REC   = "#ef4444"
    FG_TRANS = "#f59e0b"

    def __init__(self, get_level):
        self._get_level = get_level      # callable → float in [0, 1]
        self._state = "hidden"           # "hidden" | "recording" | "transcribing"
        self._root = None
        self._canvas = None
        self._phase = 0
        self._ready = threading.Event()
        self._t = threading.Thread(target=self._run, name="overlay", daemon=True)
        self._t.start()
        self._ready.wait(timeout=3.0)

    # ── Tk thread ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            import tkinter as tk
        except ImportError:
            log.error("tkinter unavailable — overlay disabled.")
            self._ready.set()
            return

        root = tk.Tk()
        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.0)
        root.configure(bg=self.BG)

        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = (sw - self.W) // 2
        y = sh - self.H - self.MARGIN_BOTTOM
        root.geometry(f"{self.W}x{self.H}+{x}+{y}")

        canvas = tk.Canvas(root, width=self.W, height=self.H, bg=self.BG,
                           highlightthickness=0, bd=0)
        canvas.pack(fill="both", expand=True)

        self._root = root
        self._canvas = canvas

        # Windows: make the window click-through and hide from Alt-Tab
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            GWL_EXSTYLE = -20
            WS_EX_LAYERED     = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_TOOLWINDOW  = 0x00000080
            WS_EX_NOACTIVATE  = 0x08000000
            ex = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE,
                ex | WS_EX_LAYERED | WS_EX_TRANSPARENT
                   | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            )
        except Exception as exc:
            log.debug("Click-through setup failed: %s", exc)

        self._ready.set()
        self._tick()
        try:
            root.mainloop()
        except Exception as exc:
            log.debug("Overlay mainloop ended: %s", exc)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_pill(self) -> None:
        c = self._canvas
        W, H = self.W, self.H
        r = H // 2
        c.create_oval(0, 0, 2 * r, H, fill=self.BG, outline="")
        c.create_oval(W - 2 * r, 0, W, H, fill=self.BG, outline="")
        c.create_rectangle(r, 0, W - r, H, fill=self.BG, outline="")

    def _tick(self) -> None:
        import math
        if self._canvas is None or self._root is None:
            return

        self._phase += 1
        st = self._state

        if st != "hidden":
            c = self._canvas
            c.delete("all")
            self._draw_pill()
            W, H = self.W, self.H

            if st == "recording":
                level = max(0.08, min(1.0, self._get_level()))
                bars, bw, gap = 9, 3, 4
                total = bars * bw + (bars - 1) * gap
                x0 = (W - total) // 2
                cy = H // 2
                phase = self._phase / 3.0
                for i in range(bars):
                    amp = (0.25 + 0.75 * abs(math.sin(phase + i * 0.8))) * level
                    bh = max(4, int(amp * (H - 16)))
                    x = x0 + i * (bw + gap)
                    c.create_rectangle(
                        x, cy - bh // 2, x + bw, cy + bh // 2,
                        fill=self.FG_REC, outline="",
                    )

            elif st == "transcribing":
                cy = H // 2
                phase = self._phase / 5.0
                for i in range(3):
                    amp = 0.5 + 0.5 * math.sin(phase + i * 0.9)
                    size = 3 + int(amp * 4)
                    x = W // 2 + (i - 1) * 16
                    c.create_oval(
                        x - size, cy - size, x + size, cy + size,
                        fill=self.FG_TRANS, outline="",
                    )

        # 33 ms ≈ 30 FPS
        self._canvas.after(33, self._tick)

    # ── Fade helpers (run on Tk thread) ───────────────────────────────────────

    def _fade(self, target: float, step: float) -> None:
        if self._root is None:
            return
        try:
            cur = float(self._root.attributes("-alpha"))
        except Exception:
            return
        done = (step > 0 and cur + step >= target) or \
               (step < 0 and cur + step <= target)
        if done:
            self._root.attributes("-alpha", target)
            if target == 0.0:
                self._root.withdraw()
            return
        self._root.attributes("-alpha", cur + step)
        self._root.after(16, lambda: self._fade(target, step))

    # ── Public (thread-safe) ──────────────────────────────────────────────────

    def set_state(self, state: str) -> None:
        """state: 'hidden' | 'recording' | 'transcribing'"""
        if self._root is None:
            return

        def apply():
            prev = self._state
            self._state = state
            if state == "hidden":
                self._fade(0.0, -0.14)
            else:
                if prev == "hidden":
                    self._root.deiconify()
                self._fade(0.94, 0.18)

        try:
            self._root.after(0, apply)
        except Exception:
            pass

    def shutdown(self) -> None:
        if self._root is None:
            return
        try:
            self._root.after(0, self._root.destroy)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# Main window  (optional desktop UI — close to tray, big record button,
#                hotkey/model/language controls, live waveform, recent clips)
# ═════════════════════════════════════════════════════════════════════════════

class MainWindow:
    """
    Dark-themed desktop window. Runs its own Tk root on a daemon thread.
    All state updates from other threads go through `root.after(0, ...)`.

    Callbacks (all optional):
      on_record_toggle()  — click the big record button
      on_change_hotkey()  — "Change" button next to hotkey label
      on_model_change(m)  — model dropdown
      on_language_change(l) — language dropdown
      on_open_history()   — "Open full history" button
      on_quit()           — File menu Quit
    """

    W, H = 480, 640

    def __init__(
        self,
        *,
        get_level,
        get_state,
        get_config,
        get_recent_history,
        on_record_toggle=None,
        on_change_hotkey=None,
        on_model_change=None,
        on_language_change=None,
        on_open_history=None,
        on_quit=None,
    ):
        self._get_level = get_level
        self._get_state = get_state
        self._get_config = get_config
        self._get_recent = get_recent_history
        self._cb_record = on_record_toggle
        self._cb_hotkey = on_change_hotkey
        self._cb_model  = on_model_change
        self._cb_lang   = on_language_change
        self._cb_history = on_open_history
        self._cb_quit   = on_quit

        self._root = None
        self._widgets: dict = {}
        self._wave_phase = 0
        self._last_state = None
        self._last_hotkey = None
        self._last_model = None
        self._last_language = None
        self._ready = threading.Event()

        threading.Thread(target=self._run, name="main-window", daemon=True).start()
        self._ready.wait(timeout=4.0)

    # ── Tk thread ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError:
            log.error("tkinter unavailable — main window disabled.")
            self._ready.set()
            return

        root = tk.Tk()
        root.title("STT (Speech to text)")
        root.configure(bg=_UI_BG)
        root.geometry(f"{self.W}x{self.H}")
        root.minsize(420, 540)

        # Centre
        root.update_idletasks()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"+{(sw - self.W) // 2}+{(sh - self.H) // 2}")

        # ── ttk dark style ───────────────────────────────────────────────
        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background=_UI_BG, foreground=_UI_FG,
                        fieldbackground=_UI_BG2, bordercolor=_UI_BORDER,
                        lightcolor=_UI_BG, darkcolor=_UI_BG)
        style.configure("TFrame", background=_UI_BG)
        style.configure("Card.TFrame", background=_UI_BG2)
        style.configure("TLabel", background=_UI_BG, foreground=_UI_FG,
                        font=("Segoe UI", 10))
        style.configure("Dim.TLabel", background=_UI_BG, foreground=_UI_FG_DIM,
                        font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=_UI_BG, foreground=_UI_FG,
                        font=("Segoe UI Semibold", 16))
        style.configure("TButton", background=_UI_BG3, foreground=_UI_FG,
                        bordercolor=_UI_BORDER, focusthickness=0,
                        padding=(10, 6), font=("Segoe UI", 10))
        style.map("TButton", background=[("active", _UI_BORDER)])
        style.configure("Accent.TButton", background=_UI_ACCENT, foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#2563eb")])
        style.configure("TCombobox", fieldbackground=_UI_BG2, background=_UI_BG3,
                        foreground=_UI_FG, arrowcolor=_UI_FG, bordercolor=_UI_BORDER)
        root.option_add("*TCombobox*Listbox.background", _UI_BG2)
        root.option_add("*TCombobox*Listbox.foreground", _UI_FG)
        root.option_add("*TCombobox*Listbox.selectBackground", _UI_ACCENT)
        root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

        outer = ttk.Frame(root, style="TFrame")
        outer.pack(fill="both", expand=True, padx=18, pady=16)

        # ── Header ───────────────────────────────────────────────────────
        header = ttk.Frame(outer, style="TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="STT", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text=" · Speech to text", style="Dim.TLabel"
                  ).pack(side="left", pady=(4, 0))

        # ── Status card ──────────────────────────────────────────────────
        card = tk.Frame(outer, bg=_UI_BG2, highlightthickness=1,
                        highlightbackground=_UI_BORDER)
        card.pack(fill="x", pady=(14, 12))

        status_row = tk.Frame(card, bg=_UI_BG2)
        status_row.pack(fill="x", padx=16, pady=(14, 4))
        dot = tk.Canvas(status_row, width=16, height=16, bg=_UI_BG2,
                        highlightthickness=0, bd=0)
        dot.pack(side="left", padx=(0, 10))
        status_lbl = tk.Label(status_row, text="LOADING", bg=_UI_BG2, fg=_UI_FG,
                              font=("Segoe UI Semibold", 13))
        status_lbl.pack(side="left")

        hint_lbl = tk.Label(
            card, text="Getting ready…",
            bg=_UI_BG2, fg=_UI_FG_DIM, font=("Segoe UI", 10),
            anchor="w",
        )
        hint_lbl.pack(fill="x", padx=16, pady=(0, 4))

        # Live waveform canvas
        wave = tk.Canvas(card, height=42, bg=_UI_BG2,
                         highlightthickness=0, bd=0)
        wave.pack(fill="x", padx=12, pady=(4, 14))

        # ── Big record button ────────────────────────────────────────────
        rec_btn = tk.Button(
            outer, text="●  Click or hold Right Alt to record",
            bg=_UI_ACCENT, fg="#ffffff", activebackground="#2563eb",
            activeforeground="#ffffff", bd=0, relief="flat",
            font=("Segoe UI Semibold", 11),
            cursor="hand2",
            command=lambda: self._cb_record and self._cb_record(),
        )
        rec_btn.pack(fill="x", ipady=12, pady=(0, 14))

        # ── Settings rows ────────────────────────────────────────────────
        def row(parent, label_text):
            r = ttk.Frame(parent, style="TFrame")
            r.pack(fill="x", pady=(0, 8))
            ttk.Label(r, text=label_text, style="Dim.TLabel", width=10
                      ).pack(side="left")
            return r

        # Hotkey
        hot_row = row(outer, "Hotkey")
        hk_val = tk.Label(hot_row, text="—", bg=_UI_BG, fg=_UI_FG,
                          font=("Segoe UI Semibold", 10))
        hk_val.pack(side="left")
        ttk.Button(hot_row, text="Change",
                   command=lambda: self._cb_hotkey and self._cb_hotkey()
                   ).pack(side="right")

        # Model
        mod_row = row(outer, "Model")
        mod_var = tk.StringVar(value="base")
        mod_cb = ttk.Combobox(mod_row, textvariable=mod_var, state="readonly",
                              values=["tiny", "base", "small"], width=12)
        mod_cb.pack(side="left")
        mod_cb.bind("<<ComboboxSelected>>",
                    lambda _e: self._cb_model and self._cb_model(mod_var.get()))

        # Language
        lang_row = row(outer, "Language")
        lang_var = tk.StringVar(value="en")
        lang_cb = ttk.Combobox(
            lang_row, textvariable=lang_var, state="readonly",
            values=["auto", "en", "ro", "fr", "de", "es", "it", "pt", "nl", "pl", "ja", "zh"],
            width=12,
        )
        lang_cb.pack(side="left")
        lang_cb.bind("<<ComboboxSelected>>",
                     lambda _e: self._cb_lang and self._cb_lang(lang_var.get()))

        # ── Recent history ───────────────────────────────────────────────
        ttk.Label(outer, text="Recent", style="Dim.TLabel"
                  ).pack(anchor="w", pady=(10, 6))

        hist_frame = tk.Frame(outer, bg=_UI_BG2, highlightthickness=1,
                              highlightbackground=_UI_BORDER)
        hist_frame.pack(fill="both", expand=True)
        hist_sb = tk.Scrollbar(hist_frame, bg=_UI_BG2, troughcolor=_UI_BG,
                               activebackground=_UI_BG3)
        hist_sb.pack(side="right", fill="y")
        hist = tk.Listbox(
            hist_frame, bg=_UI_BG2, fg=_UI_FG,
            selectbackground=_UI_ACCENT, selectforeground="#ffffff",
            highlightthickness=0, bd=0, activestyle="none",
            font=("Segoe UI", 9), yscrollcommand=hist_sb.set,
        )
        hist.pack(side="left", fill="both", expand=True)
        hist_sb.config(command=hist.yview)

        def _copy_selected(_e=None):
            sel = hist.curselection()
            if not sel:
                return
            # Entries are stored as tuples in self._widgets["hist_data"]
            data = self._widgets.get("hist_data") or []
            i = sel[0]
            if 0 <= i < len(data):
                try:
                    pyperclip.copy(data[i].get("text", ""))
                except Exception:
                    pass
        hist.bind("<Double-Button-1>", _copy_selected)
        hist.bind("<Return>", _copy_selected)

        bottom_row = ttk.Frame(outer, style="TFrame")
        bottom_row.pack(fill="x", pady=(8, 0))
        ttk.Button(bottom_row, text="Open full history…",
                   command=lambda: self._cb_history and self._cb_history()
                   ).pack(side="left")
        ttk.Button(bottom_row, text="Minimize to tray",
                   command=self.hide).pack(side="right")

        # ── Footer ───────────────────────────────────────────────────────
        footer = ttk.Frame(outer, style="TFrame")
        footer.pack(fill="x", pady=(12, 0))
        ttk.Label(footer, text="Powered by ", style="Dim.TLabel").pack(side="left")
        link = tk.Label(footer, text="MarkSoft", bg=_UI_BG, fg=_UI_ACCENT,
                        cursor="hand2", font=("Segoe UI", 9, "underline"))
        link.pack(side="left")
        import webbrowser
        link.bind("<Button-1>", lambda _e: webbrowser.open(MARKSOFT_URL))
        ttk.Label(footer, text=f"  ·  {MARKSOFT_URL}",
                  style="Dim.TLabel").pack(side="left")

        # ── State stash + bindings ───────────────────────────────────────
        self._root = root
        self._widgets = {
            "dot": dot, "status": status_lbl, "hint": hint_lbl,
            "wave": wave, "rec_btn": rec_btn,
            "hk_val": hk_val, "mod_var": mod_var, "lang_var": lang_var,
            "hist": hist, "hist_data": [],
        }

        # Close (X) = hide to tray, not quit
        root.protocol("WM_DELETE_WINDOW", self.hide)
        root.bind("<Escape>", lambda _e: self.hide())

        self._ready.set()
        self._tick()
        try:
            root.mainloop()
        except Exception as exc:
            log.debug("Main window mainloop ended: %s", exc)

    # ── Drawing / polling ─────────────────────────────────────────────────────

    _STATE_COLORS = {
        "LOADING_MODEL": ("#6b80ff", "Loading model…"),
        "IDLE":          ("#5ad48e", "Ready — hold Right Alt or click record."),
        "RECORDING":     ("#ef4444", "Recording — release to transcribe."),
        "TRANSCRIBING":  ("#f59e0b", "Transcribing…"),
    }

    def _tick(self) -> None:
        import math
        if self._root is None:
            return
        w = self._widgets

        # State
        st = self._get_state()
        st_name = getattr(st, "name", str(st))
        color, hint = self._STATE_COLORS.get(st_name, ("#8a93a2", ""))
        if st_name != self._last_state:
            self._last_state = st_name
            dot = w["dot"]
            dot.delete("all")
            dot.create_oval(2, 2, 14, 14, fill=color, outline="")
            w["status"].config(text=st_name.replace("_", " "), fg=color)
            w["hint"].config(text=hint)
            if st_name == "RECORDING":
                w["rec_btn"].config(
                    text="■  Recording…  (release or click to stop)",
                    bg="#ef4444", activebackground="#dc2626",
                )
            else:
                w["rec_btn"].config(
                    text="●  Click or hold Right Alt to record",
                    bg=_UI_ACCENT, activebackground="#2563eb",
                )

        # Config-driven labels
        cfg = self._get_config() or {}
        hk = cfg.get("hotkey", "—")
        if hk != self._last_hotkey:
            self._last_hotkey = hk
            w["hk_val"].config(text=hk)
        model = cfg.get("model", "base")
        if model != self._last_model:
            self._last_model = model
            w["mod_var"].set(model)
        lang = cfg.get("language") or "auto"
        if lang != self._last_language:
            self._last_language = lang
            w["lang_var"].set(lang)

        # Waveform
        self._wave_phase += 1
        wave = w["wave"]
        wave.delete("all")
        cw = int(wave.winfo_width()) or (self.W - 80)
        ch = int(wave.winfo_height()) or 42
        bars = 28
        gap = 3
        bw = max(2, (cw - gap * (bars - 1)) // bars)
        if st_name == "RECORDING":
            level = max(0.08, min(1.0, self._get_level()))
            col = "#ef4444"
        else:
            level = 0.12
            col = _UI_BG3
        for i in range(bars):
            amp = (0.25 + 0.75 * abs(math.sin(self._wave_phase / 3.0 + i * 0.7))) * level
            bh = max(3, int(amp * (ch - 8)))
            x = i * (bw + gap)
            y0 = (ch - bh) // 2
            wave.create_rectangle(x, y0, x + bw, y0 + bh, fill=col, outline="")

        # History (update every ~1 s to keep it cheap)
        if self._wave_phase % 30 == 0:
            try:
                entries = self._get_recent() or []
            except Exception:
                entries = []
            hist = w["hist"]
            hist.delete(0, "end")
            for e in entries[:10]:
                ts = (e.get("ts") or "")[11:16]  # HH:MM
                txt = (e.get("text") or "").replace("\n", " ")
                if len(txt) > 60:
                    txt = txt[:57] + "…"
                hist.insert("end", f"{ts}   {txt}")
            w["hist_data"] = entries[:10]

        wave.after(33, self._tick)

    # ── Public (thread-safe) ──────────────────────────────────────────────────

    def show(self) -> None:
        if self._root is None:
            return
        def _apply():
            self._root.deiconify()
            self._root.lift()
            self._root.focus_force()
        try:
            self._root.after(0, _apply)
        except Exception:
            pass

    def hide(self) -> None:
        if self._root is None:
            return
        try:
            self._root.after(0, self._root.withdraw)
        except Exception:
            pass

    def shutdown(self) -> None:
        if self._root is None:
            return
        try:
            self._root.after(0, self._root.destroy)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# Main application
# ═════════════════════════════════════════════════════════════════════════════

class STTApp:
    """
    Orchestrates audio capture, transcription, text injection and the
    system-tray icon.  pystray runs on the main thread; everything else
    runs on daemon threads.
    """

    def __init__(self) -> None:
        self.config = load_config()

        self._state = AppState.IDLE
        self._state_lock = threading.Lock()

        self.recorder = AudioRecorder(
            sample_rate=self.config["sample_rate"],
            persistent=bool(self.config.get("persistent_mic", True)),
        )
        self.transcriber = self._make_transcriber()

        # On-screen overlay (created lazily in run() so Tk starts cleanly)
        self.overlay: Overlay | None = None
        self.main_window: MainWindow | None = None

        # Keyboard state
        self._recording_active = False     # True while hotkey is held down
        self._kb_hook = None               # keyboard library hook handle

        # Audio ready for transcription
        self._audio_q: queue.Queue[np.ndarray | None] = queue.Queue()

        # pystray Icon — assigned in run()
        self.icon: pystray.Icon | None = None

        # Start the transcription worker
        threading.Thread(target=self._transcription_worker,
                         name="transcription-worker", daemon=True).start()

    # ── Factory helpers ───────────────────────────────────────────────────────

    def _make_transcriber(self) -> Transcriber:
        return Transcriber(
            model_name=self.config["model"],
            device=self.config["device"],
            compute_type=self.config["compute_type"],
            beam_size=self.config.get("beam_size", 1),
            cpu_threads=self.config.get("cpu_threads", 0),
            vad_min_seconds=self.config.get("vad_min_seconds", 3.0),
            sample_rate=self.config.get("sample_rate", 16000),
        )

    # ── State management ──────────────────────────────────────────────────────

    @property
    def state(self) -> AppState:
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, new: AppState) -> None:
        with self._state_lock:
            self._state = new
        self._refresh_icon()
        self._refresh_overlay(new)

    def _refresh_overlay(self, st: AppState) -> None:
        if self.overlay is None:
            return
        if st == AppState.RECORDING:
            self.overlay.set_state("recording")
        elif st == AppState.TRANSCRIBING:
            self.overlay.set_state("transcribing")
        else:
            self.overlay.set_state("hidden")

    def _refresh_icon(self) -> None:
        if self.icon is None:
            return
        st = self.state
        try:
            self.icon.icon = _make_icon(st)
            self.icon.title = _ICON_TOOLTIPS[st]
        except Exception as exc:
            log.debug("Icon refresh error: %s", exc)

    # ── Notifications ─────────────────────────────────────────────────────────

    def _notify(self, message: str, title: str = "STT (Speech to text)") -> None:
        if self.icon:
            try:
                self.icon.notify(message, title)
            except Exception:
                pass
        log.info("[notify] %s: %s", title, message)

    # ── Model management ──────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Kick off background model loading with UI feedback."""
        self.state = AppState.LOADING_MODEL

        def on_start():
            self._notify(
                f"Downloading / loading the '{self.config['model']}' model.\n"
                "This may take a moment on first run…"
            )

        def on_done():
            self.state = AppState.IDLE
            self._notify(f"Model '{self.config['model']}' is ready. Start speaking!")

        def on_error(exc):
            self.state = AppState.IDLE
            self._notify(f"Failed to load model: {exc}", "STT (Speech to text) — Error")

        self.transcriber.load_async(on_start=on_start, on_done=on_done, on_error=on_error)

    # ── Keyboard hook ─────────────────────────────────────────────────────────

    def _register_hotkey(self) -> None:
        self._unregister_hotkey()
        hotkey_name = self.config["hotkey"].lower()

        def _hook(event: keyboard.KeyboardEvent) -> None:
            # Normalise the event key name for comparison
            name = (event.name or "").lower().strip()
            if name != hotkey_name:
                return
            if event.event_type == keyboard.KEY_DOWN:
                self._on_press()
            elif event.event_type == keyboard.KEY_UP:
                self._on_release()

        try:
            self._kb_hook = keyboard.hook(_hook, suppress=False)
            log.info("Hotkey registered: '%s'", hotkey_name)
        except Exception as exc:
            log.error("Could not register hotkey: %s", exc)

    def _unregister_hotkey(self) -> None:
        if self._kb_hook is not None:
            try:
                keyboard.unhook(self._kb_hook)
            except Exception:
                pass
            self._kb_hook = None

    # ── Hotkey callbacks ──────────────────────────────────────────────────────

    # ── UI-initiated config changes ───────────────────────────────────────────

    def _ui_change_hotkey(self) -> None:
        def on_set(new_key: str):
            self.config["hotkey"] = new_key
            save_config(self.config)
            self._register_hotkey()
            self._notify(f"Hotkey updated → '{new_key}'")
        threading.Thread(
            target=show_hotkey_dialog,
            args=(self.config["hotkey"], on_set),
            daemon=True,
        ).start()

    def _ui_change_model(self, name: str) -> None:
        if self.config.get("model") == name:
            return
        if self.state != AppState.IDLE:
            self._notify("Finish the current operation first.")
            return
        self.config["model"] = name
        save_config(self.config)
        self.transcriber = self._make_transcriber()
        self._load_model()
        if self.icon:
            try:
                self.icon.update_menu()
            except Exception:
                pass

    def _ui_change_language(self, lang: str) -> None:
        new = None if lang == "auto" else lang
        if self.config.get("language") == new:
            return
        self.config["language"] = new
        save_config(self.config)
        self._notify(f"Language → {lang}")

    def toggle_record(self) -> None:
        """UI entry point: click-to-toggle record (alternative to hotkey)."""
        if self._recording_active:
            self._on_release()
        else:
            self._on_press()

    def _on_press(self) -> None:
        if self._recording_active:
            return  # Hold repeat — ignore

        st = self.state
        if st == AppState.LOADING_MODEL:
            self._notify("Model is still loading — please wait a moment.")
            return
        if st != AppState.IDLE:
            return  # Busy transcribing

        self._recording_active = True
        self.state = AppState.RECORDING
        try:
            self.recorder.start()
        except Exception as exc:
            log.error("Could not start recording: %s", exc)
            self._recording_active = False
            self.state = AppState.IDLE

    def _on_release(self) -> None:
        if not self._recording_active:
            return

        self._recording_active = False

        if self.state != AppState.RECORDING:
            return

        audio = self.recorder.stop()

        # Too short? Silently discard.
        min_dur = self.config.get("min_audio_seconds", 0.5)
        duration = len(audio) / self.config["sample_rate"]
        if duration < min_dur:
            log.info("Recording too short (%.2f s < %.2f s); skipped.", duration, min_dur)
            self.state = AppState.IDLE
            return

        self.state = AppState.TRANSCRIBING
        self._audio_q.put(audio)

    # ── Transcription worker ──────────────────────────────────────────────────

    def _transcription_worker(self) -> None:
        """Runs forever on a daemon thread; processes queued audio clips."""
        while True:
            audio = self._audio_q.get()
            if audio is None:          # Shutdown sentinel
                break
            try:
                language = self.config.get("language") or None
                duration = len(audio) / max(1, self.config["sample_rate"])
                text = self.transcriber.transcribe(audio, language=language)
                if text:
                    inject_text(text, method=self.config.get("paste_method", "clipboard"))
                    append_history(
                        text,
                        model=self.config.get("model", ""),
                        language=language,
                        duration=duration,
                    )
                else:
                    log.info("Empty transcription — nothing to inject.")
            except Exception as exc:
                log.error("Transcription/injection error: %s", exc)
            finally:
                self.state = AppState.IDLE
                self._audio_q.task_done()

    # ── Tray menu ─────────────────────────────────────────────────────────────

    def _build_menu(self) -> pystray.Menu:

        def _is_model(name: str):
            return lambda item: self.config["model"] == name

        def _switch_model(name: str):
            def action(icon, item):
                if self.config["model"] == name:
                    return
                if self.state != AppState.IDLE:
                    self._notify("Finish the current operation first.")
                    return
                self.config["model"] = name
                save_config(self.config)
                self.transcriber = self._make_transcriber()
                self._load_model()
                if self.icon:
                    self.icon.update_menu()
            return action

        def _show_window(icon, item):
            if self.main_window is not None:
                self.main_window.show()

        def _open_history(icon, item):
            threading.Thread(target=show_history_window,
                             name="history-window", daemon=True).start()

        def _set_hotkey(icon, item):
            def on_set(new_key: str):
                self.config["hotkey"] = new_key
                save_config(self.config)
                self._register_hotkey()
                self._notify(f"Hotkey updated → '{new_key}'")

            threading.Thread(
                target=show_hotkey_dialog,
                args=(self.config["hotkey"], on_set),
                daemon=True,
            ).start()

        def _quit(icon, item):
            log.info("Quit requested.")
            self._unregister_hotkey()
            self._audio_q.put(None)   # stop transcription worker
            icon.stop()

        return pystray.Menu(
            pystray.MenuItem(
                lambda item: f"Status: {self.state.name.replace('_', ' ').title()}",
                action=None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Show window", _show_window, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Model",
                pystray.Menu(
                    pystray.MenuItem(
                        "tiny (fastest)",
                        _switch_model("tiny"),
                        checked=_is_model("tiny"),
                        radio=True,
                    ),
                    pystray.MenuItem(
                        "base",
                        _switch_model("base"),
                        checked=_is_model("base"),
                        radio=True,
                    ),
                    pystray.MenuItem(
                        "small",
                        _switch_model("small"),
                        checked=_is_model("small"),
                        radio=True,
                    ),
                ),
            ),
            pystray.MenuItem("Set Hotkey…", _set_hotkey),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("History…", _open_history),
            pystray.MenuItem("Quit", _quit),
        )

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("STT (Speech to text) starting… (config: %s)", CONFIG_PATH)

        # Pre-open the mic so the first press records instantly
        self.recorder.prime()

        # On-screen indicator (optional)
        if self.config.get("show_overlay", True):
            try:
                self.overlay = Overlay(get_level=lambda: self.recorder.level)
            except Exception as exc:
                log.warning("Overlay init failed: %s", exc)
                self.overlay = None

        # Main desktop window (optional)
        if self.config.get("show_main_window", True):
            try:
                self.main_window = MainWindow(
                    get_level=lambda: self.recorder.level,
                    get_state=lambda: self.state,
                    get_config=lambda: self.config,
                    get_recent_history=load_history,
                    on_record_toggle=self.toggle_record,
                    on_change_hotkey=self._ui_change_hotkey,
                    on_model_change=self._ui_change_model,
                    on_language_change=self._ui_change_language,
                    on_open_history=lambda: threading.Thread(
                        target=show_history_window, daemon=True
                    ).start(),
                    on_quit=lambda: self.icon and self.icon.stop(),
                )
                if self.config.get("start_minimized", False):
                    self.main_window.hide()
            except Exception as exc:
                log.warning("Main window init failed: %s", exc)
                self.main_window = None

        self._register_hotkey()
        self._load_model()

        self.icon = pystray.Icon(
            name="STT",
            icon=_make_icon(AppState.LOADING_MODEL),
            title=_ICON_TOOLTIPS[AppState.LOADING_MODEL],
            menu=self._build_menu(),
        )

        log.info("System tray icon running.")
        try:
            self.icon.run()      # blocks until icon.stop() is called
        finally:
            try:
                self.recorder.shutdown()
            except Exception:
                pass
            if self.overlay is not None:
                self.overlay.shutdown()
            if self.main_window is not None:
                self.main_window.shutdown()
        log.info("STT (Speech to text) shut down.")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = STTApp()
    app.run()