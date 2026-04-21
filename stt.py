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


APP_VERSION = "0.2.2"
APP_DIR = _app_dir()
CONFIG_PATH = APP_DIR / "config.json"
LOG_PATH = APP_DIR / "stt.log"
HISTORY_PATH = APP_DIR / "history.jsonl"
MARKSOFT_URL = "https://marksoft.ro"
GITHUB_URL = "https://github.com/NYOGamesCOM/STT"


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
    "last_seen_version": "", # Used to trigger the "What's new" card once per update
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
# Status badge colours — matched to the overlay palette for visual consistency
_ICON_BADGE: dict[AppState, tuple] = {
    AppState.LOADING_MODEL: (124, 196, 255, 255),   # Cyan-blue
    AppState.IDLE:          ( 90, 212, 142, 255),   # Mint
    AppState.RECORDING:     (255,  77,  96, 255),   # Red
    AppState.TRANSCRIBING:  (247, 169,  64, 255),   # Amber
}
_ICON_TOOLTIPS: dict[AppState, str] = {
    AppState.LOADING_MODEL: "STT — Loading model…",
    AppState.IDLE:          "STT — Ready (hold hotkey to record)",
    AppState.RECORDING:     "STT — Recording…",
    AppState.TRANSCRIBING:  "STT — Transcribing…",
}

_ICON_CACHE: dict[AppState, Image.Image] = {}

# Icon base colour — aligns with overlay BG for a cohesive brand
_ICON_BG_FILL    = (14, 16, 20, 255)
_ICON_BG_BORDER  = (255, 255, 255, 28)
_ICON_BG_HL_TOP  = (255, 255, 255, 18)
_ICON_S_FILL     = (240, 244, 252, 240)


def _load_bold_font(size: int):
    """Try to load a bold system font; fall back to PIL's default."""
    from PIL import ImageFont
    candidates = [
        "segoeuib.ttf",        # Windows Segoe UI Bold
        "segoeuisb.ttf",       # Windows Segoe UI Semibold
        "arialbd.ttf",         # Windows Arial Bold
        "Arial Bold.ttf",
        "Helvetica.ttc",       # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _make_icon(state: AppState) -> Image.Image:
    """
    Draw a branded STT tray icon: a rounded-square dark tile with a large
    white 'S' always visible and a small state-coloured badge in the
    top-right corner. Same mark at every state → instant brand recognition.
    """
    if state in _ICON_CACHE:
        return _ICON_CACHE[state]

    sz = _ICON_SIZE
    img = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # ── Rounded square base ────────────────────────────────────────────
    pad = 2
    radius = 14
    d.rounded_rectangle(
        [pad, pad, sz - pad, sz - pad],
        radius=radius, fill=_ICON_BG_FILL,
        outline=_ICON_BG_BORDER, width=1,
    )
    # Faint top inner highlight — that "glass" feel
    d.rounded_rectangle(
        [pad + 1, pad + 1, sz - pad - 1, sz - pad - 1],
        radius=radius - 1, fill=None,
        outline=_ICON_BG_HL_TOP, width=1,
    )

    # ── Centered 'S' wordmark ──────────────────────────────────────────
    font = _load_bold_font(42)
    text = "S"
    if font is not None:
        try:
            bbox = d.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = (sz - tw) // 2 - bbox[0]
            ty = (sz - th) // 2 - bbox[1] - 1
            d.text((tx, ty), text, font=font, fill=_ICON_S_FILL)
        except Exception:
            # Fallback: stylised S from two arcs if text rendering fails
            d.arc([14, 10, 50, 38], start=0, end=180, fill=_ICON_S_FILL, width=6)
            d.arc([14, 26, 50, 54], start=180, end=360, fill=_ICON_S_FILL, width=6)
    else:
        d.arc([14, 10, 50, 38], start=0, end=180, fill=_ICON_S_FILL, width=6)
        d.arc([14, 26, 50, 54], start=180, end=360, fill=_ICON_S_FILL, width=6)

    # ── Status badge dot (top-right) ───────────────────────────────────
    badge_color = _ICON_BADGE[state]
    br = 9                          # badge radius
    bx, by = sz - br - 6, br + 6
    # Dark ring for contrast against any wallpaper
    d.ellipse([bx - br - 2, by - br - 2, bx + br + 2, by + br + 2],
              fill=_ICON_BG_FILL)
    d.ellipse([bx - br, by - br, bx + br, by + br], fill=badge_color)
    # Small specular highlight on the badge for depth
    d.ellipse([bx - br + 2, by - br + 2, bx - 2, by - 2],
              fill=(255, 255, 255, 60))

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
# Update checker  (polls GitHub Releases, shows a pop-up if a newer tag exists)
# ═════════════════════════════════════════════════════════════════════════════

_UPDATE_API = "https://api.github.com/repos/NYOGamesCOM/STT/releases/latest"
_UPDATE_CACHE_SECONDS = 3600   # don't hit GitHub more than once an hour


def _parse_version(tag: str) -> tuple[int, ...]:
    """Parse 'v0.1.5' / '0.1.5' → (0, 1, 5). Non-numeric parts become 0."""
    s = (tag or "").lstrip("v").strip()
    parts: list[int] = []
    for p in s.split("."):
        num = ""
        for ch in p:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return tuple(parts) or (0,)


def _fetch_latest_release(timeout: float = 6.0) -> dict | None:
    """GET /releases/latest via stdlib urllib (no extra deps)."""
    return _fetch_github(
        "https://api.github.com/repos/NYOGamesCOM/STT/releases/latest",
        timeout=timeout,
    )


def _fetch_release_for_tag(tag: str, timeout: float = 6.0) -> dict | None:
    """GET /releases/tags/<tag> — used for the 'What's new' card."""
    tag = (tag or "").strip()
    if not tag:
        return None
    if not tag.startswith("v"):
        tag = "v" + tag
    return _fetch_github(
        f"https://api.github.com/repos/NYOGamesCOM/STT/releases/tags/{tag}",
        timeout=timeout,
    )


def _fetch_github(url: str, timeout: float = 6.0) -> dict | None:
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={"User-Agent": f"STT/{APP_VERSION}",
                     "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        log.debug("GitHub fetch failed (%s): %s", url, exc)
        return None


def check_for_update(force: bool = False) -> dict | None:
    """
    Return release dict when a *newer* version is available, else None.
    Caches the last hit for an hour to avoid hammering the API.
    """
    now = time.time()
    cache = getattr(check_for_update, "_cache", (0.0, None))
    if not force and now - cache[0] < _UPDATE_CACHE_SECONDS:
        return cache[1]

    rel = _fetch_latest_release()
    result: dict | None = None
    if rel and rel.get("tag_name"):
        remote = _parse_version(rel["tag_name"])
        local  = _parse_version(APP_VERSION)
        if remote > local:
            result = {
                "tag":     rel["tag_name"],
                "name":    rel.get("name") or rel["tag_name"],
                "body":    rel.get("body") or "",
                "html_url": rel.get("html_url"),
                "published_at": rel.get("published_at", ""),
            }
    check_for_update._cache = (now, result)  # type: ignore[attr-defined]
    return result


# ── Helpers for rendering release notes ──────────────────────────────────────

def _relative_time(iso_str: str) -> str:
    """'2026-04-21T14:00:00Z' → 'just now' | '3h ago' | '2d ago' | '4mo ago'."""
    if not iso_str:
        return ""
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        diff = (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return ""
    if diff < 45:                return "just now"
    if diff < 60 * 60:           return f"{int(diff/60)}m ago"
    if diff < 60 * 60 * 24:      return f"{int(diff/3600)}h ago"
    if diff < 60 * 60 * 24 * 30: return f"{int(diff/86400)}d ago"
    if diff < 60 * 60 * 24 * 365:return f"{int(diff/(86400*30))}mo ago"
    return f"{int(diff/(86400*365))}y ago"


# Palette used by the update dialog (matches the main window).
_DLG_BG        = "#0b0d11"
_DLG_BG_CARD   = "#13161c"
_DLG_BG_ELEV   = "#1a1e26"
_DLG_BG_HOVER  = "#222832"
_DLG_BORDER    = "#242a33"
_DLG_BORDER_S  = "#1c2028"
_DLG_FG        = "#e9ecf2"
_DLG_FG_DIM    = "#8289a0"
_DLG_FG_MUTE   = "#5a6071"
_DLG_ACCENT    = "#0a84ff"
_DLG_ACCENT_H  = "#3398ff"
_DLG_OK        = "#32d74b"
_DLG_AMBER     = "#ff9f0a"
_DLG_REC       = "#ff453a"


def _render_release_notes(text_widget, md: str) -> None:
    """Render our CHANGELOG-flavoured markdown into a styled tk.Text.

    Understands:
      '### ✨ New'     → mint-coloured header
      '### 🎨 Design'  → amber-coloured header
      '### 🐛 Fixed'   → red-coloured header
      '### <other>'    → neutral header
      '- xxx'          → bullet
      '## <version>'   → skipped (shown in the dialog header already)
      '## Downloads'   → stops rendering (we don't show the download table)
    """
    text_widget.configure(state="normal")
    text_widget.delete("1.0", "end")

    text_widget.tag_configure("h_new",    foreground=_DLG_OK,
                              font=("Segoe UI Semibold", 10), spacing1=10, spacing3=4)
    text_widget.tag_configure("h_design", foreground=_DLG_AMBER,
                              font=("Segoe UI Semibold", 10), spacing1=10, spacing3=4)
    text_widget.tag_configure("h_fix",    foreground=_DLG_REC,
                              font=("Segoe UI Semibold", 10), spacing1=10, spacing3=4)
    text_widget.tag_configure("h_other",  foreground=_DLG_FG,
                              font=("Segoe UI Semibold", 10), spacing1=10, spacing3=4)
    text_widget.tag_configure("bullet",   foreground=_DLG_FG_MUTE,
                              font=("Segoe UI", 10))
    text_widget.tag_configure("body",     foreground=_DLG_FG,
                              font=("Segoe UI", 10), lmargin1=6, lmargin2=24,
                              spacing3=2)
    text_widget.tag_configure("plain",    foreground=_DLG_FG,
                              font=("Segoe UI", 10), spacing3=2)

    def header_tag(header: str) -> str:
        low = header.lower()
        if any(k in low for k in ("new", "added", "✨")):
            return "h_new"
        if any(k in low for k in ("design", "changed", "🎨")):
            return "h_design"
        if any(k in low for k in ("fix", "bug", "🐛")):
            return "h_fix"
        return "h_other"

    first_line = True
    for line in (md or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            # 'Downloads' marks the boilerplate footer the workflow appends
            if stripped[3:].strip().lower().startswith("downloads"):
                break
            # Skip the '## vX.Y.Z — title' header itself
            continue
        if stripped.startswith("### "):
            header = stripped[4:]
            text_widget.insert("end", header + "\n", header_tag(header))
        elif stripped.startswith("- "):
            text_widget.insert("end", "   •  ", "bullet")
            text_widget.insert("end", stripped[2:] + "\n", "body")
        elif stripped == "":
            if not first_line:
                text_widget.insert("end", "\n")
        else:
            text_widget.insert("end", line + "\n", "plain")
        first_line = False

    text_widget.configure(state="disabled")


def show_update_dialog(update: dict, *, mode: str = "update",
                       on_dismiss=None) -> None:
    """Dark-themed update / what's-new pop-up with a custom title bar.

    `update` keys:  tag, name, body, html_url, published_at
    `mode`:         'update'   — a newer version is available
                    'whatsnew' — user just launched an already-installed
                                 newer version; this is the welcome card.
    `on_dismiss`:   optional callable fired when the dialog closes.
    """
    try:
        import tkinter as tk
        from tkinter import ttk
        import webbrowser
    except ImportError:
        return

    W, H = 500, 460
    TITLE_H = 38

    root = tk.Tk()
    root.overrideredirect(True)
    root.configure(bg=_DLG_BG)
    root.geometry(f"{W}x{H}")
    root.attributes("-topmost", True)

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw - W) // 2}+{(sh - H) // 2}")

    # Icon so the taskbar / Alt-Tab gets our brand
    try:
        from PIL import ImageTk
        root._icon = ImageTk.PhotoImage(_make_icon(AppState.IDLE))       # type: ignore
        root.iconphoto(True, root._icon)                                 # type: ignore
    except Exception:
        pass
    try:
        import tempfile, os
        ico_path = os.path.join(tempfile.gettempdir(), f"stt_{APP_VERSION}.ico")
        _make_icon(AppState.IDLE).save(
            ico_path, format="ICO",
            sizes=[(16, 16), (32, 32), (48, 48), (64, 64)],
        )
        root.iconbitmap(default=ico_path)
    except Exception:
        pass

    # Make an overrideredirect window show up in the taskbar on Windows
    def _taskbar_fix():
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            GWL_EXSTYLE = -20
            WS_EX_APPWINDOW  = 0x00040000
            WS_EX_TOOLWINDOW = 0x00000080
            ex = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ex = (ex & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW
            ctypes.windll.user32.ShowWindow(hwnd, 0)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex)
            ctypes.windll.user32.ShowWindow(hwnd, 5)
        except Exception:
            pass
    root.after(10, _taskbar_fix)

    # ── Custom title bar (drag + close) ────────────────────────────────
    bar = tk.Frame(root, bg=_DLG_BG, height=TITLE_H)
    bar.pack(fill="x", side="top")
    bar.pack_propagate(False)

    # Tiny brand icon
    try:
        from PIL import ImageTk
        ti = _make_icon(AppState.IDLE).resize((20, 20))
        bar._ti = ImageTk.PhotoImage(ti)                 # type: ignore
        tk.Label(bar, image=bar._ti, bg=_DLG_BG          # type: ignore
                 ).pack(side="left", padx=(11, 8), pady=9)
    except Exception:
        pass

    tk.Label(bar, text=("What's new" if mode == "whatsnew" else "Update available"),
             bg=_DLG_BG, fg=_DLG_FG,
             font=("Segoe UI Semibold", 10)).pack(side="left")

    def _close():
        if on_dismiss:
            try: on_dismiss()
            except Exception: pass
        try: root.destroy()
        except Exception: pass

    close_btn = tk.Label(bar, text="\u2715", bg=_DLG_BG, fg=_DLG_FG_DIM,
                         cursor="hand2", font=("Segoe UI", 11),
                         padx=14, pady=0)
    close_btn.pack(side="right")
    close_btn.bind("<Enter>", lambda _e: close_btn.configure(bg=_DLG_REC, fg="#fff"))
    close_btn.bind("<Leave>", lambda _e: close_btn.configure(bg=_DLG_BG, fg=_DLG_FG_DIM))
    close_btn.bind("<Button-1>", lambda _e: _close())

    drag_off = {"x": 0, "y": 0}
    def _drag_start(e):
        drag_off["x"] = e.x_root - root.winfo_x()
        drag_off["y"] = e.y_root - root.winfo_y()
    def _drag_motion(e):
        root.geometry(f"+{e.x_root - drag_off['x']}+{e.y_root - drag_off['y']}")
    for w in (bar, *bar.winfo_children()):
        if isinstance(w, (tk.Frame, tk.Label)) and w is not close_btn:
            w.bind("<Button-1>",  _drag_start)
            w.bind("<B1-Motion>", _drag_motion)

    # Animated accent strip under the title bar
    strip = tk.Canvas(root, height=2, bg=_DLG_BG, bd=0, highlightthickness=0)
    strip.pack(fill="x", side="top")
    strip_state = {"x": 0}
    def _animate_strip():
        try:
            strip.delete("all")
            cw = strip.winfo_width() or W
            strip_state["x"] = (strip_state["x"] + 3) % (cw + 120)
            # Two overlapping gradients give a smooth drifting sheen
            for i, (col, off, w_) in enumerate((
                (_DLG_ACCENT,   strip_state["x"] - 120, 120),
                (_DLG_ACCENT_H, strip_state["x"] - 260, 140),
            )):
                strip.create_rectangle(off, 0, off + w_, 2, fill=col, outline="")
            strip.after(40, _animate_strip)
        except Exception:
            pass
    _animate_strip()

    # ── Body ────────────────────────────────────────────────────────────
    body = tk.Frame(root, bg=_DLG_BG)
    body.pack(fill="both", expand=True, side="top", padx=22, pady=(16, 14))

    if mode == "whatsnew":
        title = f"Welcome to STT {update['tag']}"
    else:
        title = "A new version of STT is available"
    tk.Label(body, text=title, bg=_DLG_BG, fg=_DLG_FG,
             font=("Segoe UI Semibold", 15)).pack(anchor="w")

    # Version pill + relative time
    pill = tk.Frame(body, bg=_DLG_BG)
    pill.pack(anchor="w", pady=(6, 10))
    if mode == "whatsnew":
        tk.Label(pill, text=f" {update['tag']}  ·  latest ",
                 bg=_DLG_BG_ELEV, fg=_DLG_OK,
                 font=("Consolas", 9, "bold")).pack(side="left")
    else:
        tk.Label(pill, text=f"  v{APP_VERSION}  ",
                 bg=_DLG_BG_ELEV, fg=_DLG_FG_DIM,
                 font=("Consolas", 9, "bold")).pack(side="left")
        tk.Label(pill, text="  \u2192  ", bg=_DLG_BG, fg=_DLG_FG_DIM,
                 font=("Segoe UI", 10)).pack(side="left")
        tk.Label(pill, text=f"  {update['tag']}  ",
                 bg=_DLG_BG_ELEV, fg=_DLG_ACCENT,
                 font=("Consolas", 9, "bold")).pack(side="left")
    rel_t = _relative_time(update.get("published_at") or "")
    if rel_t:
        tk.Label(pill, text=f"   released {rel_t}", bg=_DLG_BG, fg=_DLG_FG_MUTE,
                 font=("Segoe UI", 9)).pack(side="left")

    # Notes panel
    notes_wrap = tk.Frame(body, bg=_DLG_BG_CARD,
                          highlightthickness=1, highlightbackground=_DLG_BORDER_S)
    notes_wrap.pack(fill="both", expand=True)

    notes = tk.Text(notes_wrap, bg=_DLG_BG_CARD, fg=_DLG_FG,
                    bd=0, highlightthickness=0, wrap="word",
                    padx=14, pady=12, font=("Segoe UI", 10))
    sb = ttk.Scrollbar(notes_wrap, orient="vertical", command=notes.yview)
    notes.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    notes.pack(side="left", fill="both", expand=True)

    _render_release_notes(notes, update.get("body") or "")

    # ── Actions ─────────────────────────────────────────────────────────
    actions = tk.Frame(root, bg=_DLG_BG)
    actions.pack(fill="x", side="top", padx=22, pady=(0, 18))

    def _gh_link():
        try:
            webbrowser.open(update.get("html_url") or GITHUB_URL + "/releases")
        except Exception:
            pass

    # Tertiary: See on GitHub (left)
    gh = tk.Label(actions, text="See on GitHub \u2197",
                  bg=_DLG_BG, fg=_DLG_FG_DIM,
                  cursor="hand2", font=("Segoe UI", 10))
    gh.pack(side="left")
    gh.bind("<Enter>", lambda _e: gh.configure(fg=_DLG_FG))
    gh.bind("<Leave>", lambda _e: gh.configure(fg=_DLG_FG_DIM))
    gh.bind("<Button-1>", lambda _e: _gh_link())

    def _primary_btn(parent, label: str, cmd, *, accent: bool):
        bg = _DLG_ACCENT if accent else _DLG_BG_ELEV
        fg = "#ffffff"  if accent else _DLG_FG
        hover = _DLG_ACCENT_H if accent else _DLG_BG_HOVER
        b = tk.Label(parent, text=label, bg=bg, fg=fg,
                     font=("Segoe UI Semibold", 10),
                     padx=16, pady=8, cursor="hand2")
        b.bind("<Enter>", lambda _e: b.configure(bg=hover))
        b.bind("<Leave>", lambda _e: b.configure(bg=bg))
        b.bind("<Button-1>", lambda _e: cmd())
        return b

    if mode == "whatsnew":
        _primary_btn(actions, "Got it", _close, accent=True).pack(side="right")
    else:
        def _download():
            _gh_link()
            _close()
        _primary_btn(actions, "Download update", _download,
                     accent=True).pack(side="right", padx=(10, 0))
        _primary_btn(actions, "Remind me later", _close,
                     accent=False).pack(side="right")

    root.bind("<Escape>", lambda _e: _close())
    root.protocol("WM_DELETE_WINDOW", _close)
    root.mainloop()


# ═════════════════════════════════════════════════════════════════════════════
# On-screen overlay indicator  (WisperFlow-style pill, bottom-centre)
# ═════════════════════════════════════════════════════════════════════════════

class Overlay:
    """
    Liquid-glass pill shown at the bottom centre while recording/transcribing.

    - True transparent rounded corners on Windows (via -transparentcolor).
    - Click-through (WS_EX_TRANSPARENT) and hidden from Alt-Tab.
    - Smooth scrolling bezier waveform mirrored top/bottom while recording.
    - Pulsing status dot on the left, monospace elapsed timer on the right.

    Owns its own Tk root + mainloop on a dedicated daemon thread; every
    state change from other threads is marshalled via `root.after(0, ...)`.
    """

    W, H = 200, 32                       # Thinner, more elegant proportions
    MARGIN_BOTTOM = 52                   # Gap above taskbar
    LEVELS_N = 64                        # Rolling waveform sample buffer

    # iOS-inspired palette
    BG        = "#0a0c10"                # near-black, slightly warmer
    HL        = "#1b2029"                # 1-px inner top highlight
    FG_DIM    = "#6e6e73"                # iOS secondary label
    FG_REC    = "#ff453a"                # iOS systemRed (dark)
    FG_REC_GL = "#ff6b62"                # halo
    FG_TRANS  = "#ff9f0a"                # iOS systemOrange (dark)
    FG_LOAD   = "#0a84ff"                # iOS systemBlue (dark)
    # Windows transparency key — any pixel this exact color becomes see-through
    KEY       = "#ff00ff"

    def __init__(self, get_level):
        self._get_level = get_level
        self._state = "hidden"           # "hidden" | "recording" | "transcribing"
        self._root = None
        self._canvas = None
        self._phase = 0
        self._rec_start = 0.0
        self._levels: list[float] = [0.0] * self.LEVELS_N
        self._transparent_ok = False
        self._ready = threading.Event()
        threading.Thread(target=self._run, name="overlay", daemon=True).start()
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

        # Try true transparent corners (Windows). Fallback: opaque pill.
        try:
            root.configure(bg=self.KEY)
            root.attributes("-transparentcolor", self.KEY)
            self._transparent_ok = True
        except Exception:
            root.configure(bg=self.BG)
            self._transparent_ok = False

        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        x = (sw - self.W) // 2
        y = sh - self.H - self.MARGIN_BOTTOM
        root.geometry(f"{self.W}x{self.H}+{x}+{y}")

        canvas_bg = self.KEY if self._transparent_ok else self.BG
        canvas = tk.Canvas(
            root, width=self.W, height=self.H, bg=canvas_bg,
            highlightthickness=0, bd=0,
        )
        canvas.pack(fill="both", expand=True)

        self._root = root
        self._canvas = canvas

        # Click-through + hide from Alt-Tab (Windows)
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

    # ── Color helper (blend hex → bg for faux-alpha glows) ────────────────────

    @staticmethod
    def _mix(hex_a: str, hex_b: str, t: float) -> str:
        ar, ag, ab = int(hex_a[1:3], 16), int(hex_a[3:5], 16), int(hex_a[5:7], 16)
        br, bg_, bb = int(hex_b[1:3], 16), int(hex_b[3:5], 16), int(hex_b[5:7], 16)
        r = int(ar + (br - ar) * t)
        g = int(ag + (bg_ - ag) * t)
        b = int(ab + (bb - ab) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_pill(self) -> None:
        c = self._canvas
        W, H = self.W, self.H
        r = H // 2
        # Pill body
        c.create_oval(0, 0, 2 * r, H, fill=self.BG, outline="")
        c.create_oval(W - 2 * r, 0, W, H, fill=self.BG, outline="")
        c.create_rectangle(r, 0, W - r, H, fill=self.BG, outline="")
        # 1-px inner top highlight — the "glass" shine
        c.create_arc(1, 1, 2 * r - 1, H - 1, start=45, extent=90,
                     style="arc", outline=self.HL, width=1)
        c.create_arc(W - 2 * r + 1, 1, W - 1, H - 1, start=45, extent=90,
                     style="arc", outline=self.HL, width=1)
        c.create_line(r, 1, W - r, 1, fill=self.HL)

    def _tick(self) -> None:
        import math
        if self._canvas is None or self._root is None:
            return

        self._phase += 1
        st = self._state

        if st == "hidden":
            # Keep the loop alive so re-show renders immediately
            self._canvas.after(33, self._tick)
            return

        c = self._canvas
        c.delete("all")
        W, H = self.W, self.H
        cy = H // 2

        self._draw_pill()

        # Roll waveform buffer (smooth even when idle state flips to recording)
        live = 0.10
        if st == "recording":
            live = max(0.10, min(1.0, self._get_level()))
        # Ease toward target so we don't see visible jitter from raw RMS
        self._levels.pop(0)
        prev = self._levels[-1]
        eased = prev + (live - prev) * 0.35
        self._levels.append(eased)

        # ── Left: pulsing state dot (single soft halo, iOS-ish) ──────────
        dot_x = 14
        if st == "recording":
            color = self.FG_REC
            pulse = 0.5 + 0.5 * math.sin(self._phase / 7.0)
            core = 2.6 + pulse * 0.8
            # One soft halo ring instead of three
            rr = core + 3.2
            halo = self._mix(self.BG, self.FG_REC_GL, 0.42 * (0.4 + pulse * 0.6))
            c.create_oval(dot_x - rr, cy - rr, dot_x + rr, cy + rr,
                          outline=halo, width=1)
            c.create_oval(dot_x - core, cy - core, dot_x + core, cy + core,
                          fill=color, outline="")
        elif st == "transcribing":
            color = self.FG_TRANS
            pulse = 0.5 + 0.5 * math.sin(self._phase / 5.5)
            rr = 2.6 + pulse * 0.9
            c.create_oval(dot_x - rr, cy - rr, dot_x + rr, cy + rr,
                          fill=color, outline="")

        # ── Right: elapsed timer (recording only) ─────────────────────────
        right_edge = W - 13
        if st == "recording":
            elapsed = int(max(0, time.time() - self._rec_start))
            mm, ss = elapsed // 60, elapsed % 60
            c.create_text(
                right_edge, cy, text=f"{mm}:{ss:02d}",
                fill=self.FG_DIM, font=("Consolas", 9, "bold"),
                anchor="e",
            )
            right_edge -= 36

        # ── Center: waveform / dots ───────────────────────────────────────
        wave_x0 = dot_x + 11
        wave_x1 = right_edge - 6
        wave_w = max(20, wave_x1 - wave_x0)

        if st == "recording":
            # Smooth mirrored curve, thinner line for a sleeker look
            n = len(self._levels)
            step = wave_w / (n - 1)
            max_amp = (H - 12) / 2
            top_pts: list[float] = []
            bot_pts: list[float] = []
            phase = self._phase / 4.0
            for i, v in enumerate(self._levels):
                x = wave_x0 + i * step
                wiggle = 0.85 + 0.15 * math.sin(phase + i * 0.35)
                a = v * max_amp * wiggle
                top_pts += [x, cy - a]
                bot_pts += [x, cy + a]
            c.create_line(*top_pts, smooth=True, width=1.5,
                          fill=self.FG_REC,
                          capstyle="round", joinstyle="round")
            c.create_line(*bot_pts, smooth=True, width=1.5,
                          fill=self.FG_REC,
                          capstyle="round", joinstyle="round")

        elif st == "transcribing":
            # Three breathing dots, tighter spacing
            gap = 11
            mid_x = (wave_x0 + wave_x1) // 2
            for i in range(3):
                px = mid_x + (i - 1) * gap
                ph = self._phase / 6.0 + i * 0.7
                amp = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(ph))
                sz = 1.6 + amp * 1.8
                c.create_oval(px - sz, cy - sz, px + sz, cy + sz,
                              fill=self.FG_TRANS, outline="")

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
            if state == "recording":
                self._rec_start = time.time()
                self._levels = [0.0] * self.LEVELS_N
            if state == "hidden":
                self._fade(0.0, -0.14)
            else:
                if prev == "hidden":
                    self._root.deiconify()
                self._fade(0.96, 0.20)

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
    Fully dark-themed desktop window with a custom title bar, compact
    status strip, history-first layout, click-to-preview pane, and a
    slide-in Settings drawer from the right.

    Runs its own Tk root on a daemon thread; every state update from
    other threads is marshalled via `root.after(0, ...)`.
    """

    # Proportions — calmer, tighter than before
    W, H = 460, 600
    TITLE_H = 40
    STATUS_H = 44
    WAVE_H = 28
    FOOTER_H = 34
    DRAWER_W = 300

    # Palette (iOS-inspired, reused by the drawer / preview / hover states)
    BG          = "#0b0d11"
    BG_CARD     = "#13161c"
    BG_ELEV     = "#1a1e26"
    BG_HOVER    = "#222832"
    BORDER      = "#242a33"
    BORDER_SOFT = "#1c2028"
    FG          = "#e9ecf2"
    FG_DIM      = "#8289a0"
    FG_MUTE     = "#5a6071"
    ACCENT      = "#0a84ff"        # iOS blue
    ACCENT_H    = "#3398ff"
    REC         = "#ff453a"        # iOS red
    OK          = "#32d74b"        # iOS green
    AMBER       = "#ff9f0a"        # iOS orange
    CYAN        = "#7cc4ff"

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
        on_show_whats_new=None,
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
        self._cb_whats_new = on_show_whats_new

        self._root = None
        self._widgets: dict = {}
        self._wave_phase = 0
        self._levels: list[float] = [0.0] * 80
        self._last_state = None
        self._last_hotkey = None
        self._last_model = None
        self._last_language = None
        self._drawer_open = False
        self._is_maximized = False
        self._pre_max_geom = ""
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
        root.title(f"STT by MarkSoft")
        root.configure(bg=self.BG)
        root.geometry(f"{self.W}x{self.H}")
        root.overrideredirect(True)
        root.minsize(420, 480)

        # Position: centred on the primary monitor
        root.update_idletasks()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"+{(sw - self.W) // 2}+{(sh - self.H) // 2}")

        # Branded icon — held on self so it isn't garbage-collected
        self._set_window_icon(root)

        # Make an override-redirect window appear in the taskbar + Alt-Tab
        root.after(10, lambda: self._make_appear_in_taskbar(root))

        # ── ttk base style — proper dark theme, including Combobox ───────
        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background=self.BG, foreground=self.FG,
                        fieldbackground=self.BG_ELEV, bordercolor=self.BORDER,
                        lightcolor=self.BG, darkcolor=self.BG,
                        insertcolor=self.FG, selectbackground=self.ACCENT,
                        selectforeground="#ffffff")
        style.configure("TFrame", background=self.BG)
        style.configure("Card.TFrame", background=self.BG_CARD)
        style.configure("Elev.TFrame", background=self.BG_ELEV)

        # Combobox — the default often leaks native white on Windows
        style.configure("TCombobox",
                        fieldbackground=self.BG_ELEV,
                        background=self.BG_ELEV,
                        foreground=self.FG,
                        arrowcolor=self.FG_DIM,
                        bordercolor=self.BORDER,
                        selectbackground=self.BG_ELEV,
                        selectforeground=self.FG)
        style.map("TCombobox",
                  fieldbackground=[("readonly", self.BG_ELEV)],
                  selectbackground=[("readonly", self.BG_ELEV)],
                  selectforeground=[("readonly", self.FG)],
                  foreground=[("readonly", self.FG)],
                  bordercolor=[("active", self.ACCENT)])
        root.option_add("*TCombobox*Listbox.background",       self.BG_ELEV)
        root.option_add("*TCombobox*Listbox.foreground",       self.FG)
        root.option_add("*TCombobox*Listbox.selectBackground", self.ACCENT)
        root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        root.option_add("*TCombobox*Listbox.borderWidth",      0)
        root.option_add("*TCombobox*Listbox.font",             ("Segoe UI", 10))

        # Scrollbar — minimal dark
        style.configure("Vertical.TScrollbar",
                        background=self.BG_ELEV, troughcolor=self.BG,
                        bordercolor=self.BG, arrowcolor=self.FG_DIM,
                        gripcount=0)
        style.map("Vertical.TScrollbar",
                  background=[("active", self.BG_HOVER)])

        # ── Build the UI ─────────────────────────────────────────────────
        self._root = root
        self._build_title_bar(root)
        tk.Frame(root, bg=self.BORDER_SOFT, height=1).pack(fill="x", side="top")

        body = tk.Frame(root, bg=self.BG)
        body.pack(fill="both", expand=True, side="top")

        self._build_status_strip(body)
        self._build_waveform(body)
        self._build_history(body)
        self._build_actions(body)
        self._build_footer(body)

        # Drawer sits on top of body — placed absolute
        self._build_drawer(root)

        root.bind("<Escape>", lambda _e: self._escape_key())
        root.bind("<Configure>", self._on_configure)

        self._ready.set()
        self._tick()
        try:
            root.mainloop()
        except Exception as exc:
            log.debug("Main window mainloop ended: %s", exc)

    # ── Icon + Windows taskbar integration ────────────────────────────────────

    def _set_window_icon(self, root) -> None:
        try:
            from PIL import ImageTk
            self._wm_icon = ImageTk.PhotoImage(_make_icon(AppState.IDLE))
            root.iconphoto(True, self._wm_icon)
        except Exception as exc:
            log.debug("iconphoto failed: %s", exc)
        # .ico fallback so the Windows taskbar / Alt-Tab get a clean icon
        try:
            import tempfile, os
            path = os.path.join(tempfile.gettempdir(), f"stt_{APP_VERSION}.ico")
            _make_icon(AppState.IDLE).save(
                path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)],
            )
            root.iconbitmap(default=path)
        except Exception as exc:
            log.debug("iconbitmap fallback failed: %s", exc)

    def _make_appear_in_taskbar(self, root) -> None:
        """Override-redirect windows hide from the taskbar by default on
        Windows. Toggle WS_EX_APPWINDOW so the OS treats us as a top-level app."""
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            GWL_EXSTYLE = -20
            WS_EX_APPWINDOW  = 0x00040000
            WS_EX_TOOLWINDOW = 0x00000080
            ex = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ex = (ex & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW
            # Hide → set → show so the shell picks up the style change
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex)
            ctypes.windll.user32.ShowWindow(hwnd, 5)  # SW_SHOW
        except Exception as exc:
            log.debug("Taskbar style fix skipped: %s", exc)

    # ── Title bar (drag + min/max/close) ──────────────────────────────────────

    def _build_title_bar(self, root) -> None:
        bar = tk.Frame(root, bg=self.BG, height=self.TITLE_H)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        # Left: branded icon + wordmark
        try:
            from PIL import ImageTk
            icon_img = _make_icon(AppState.IDLE).resize((22, 22))
            self._tb_icon = ImageTk.PhotoImage(icon_img)
            tk.Label(bar, image=self._tb_icon, bg=self.BG
                     ).pack(side="left", padx=(12, 8), pady=9)
        except Exception:
            pass
        tk.Label(bar, text="STT", bg=self.BG, fg=self.FG,
                 font=("Segoe UI Semibold", 10)).pack(side="left")
        tk.Label(bar, text=" by MarkSoft", bg=self.BG, fg=self.FG_DIM,
                 font=("Segoe UI", 9)).pack(side="left")

        # Right: window buttons  (— □ ×)
        close_btn = self._window_btn(bar, "\u2715", self.hide,
                                     hover_bg=self.REC, hover_fg="#ffffff")
        close_btn.pack(side="right")
        max_btn = self._window_btn(bar, "\u25A1", self._toggle_max)
        max_btn.pack(side="right")
        min_btn = self._window_btn(bar, "\u2012", self._minimize)
        min_btn.pack(side="right")

        # Drag on any empty part of the bar (not on icon/label/buttons)
        for w in (bar,):
            w.bind("<Button-1>",  self._drag_start)
            w.bind("<B1-Motion>", self._drag_motion)
            w.bind("<Double-Button-1>", lambda _e: self._toggle_max())
        # Let clicks on labels still drag
        for child in bar.winfo_children():
            if isinstance(child, tk.Label):
                child.bind("<Button-1>",  self._drag_start)
                child.bind("<B1-Motion>", self._drag_motion)

    def _window_btn(self, parent, symbol: str, command,
                    hover_bg: str | None = None,
                    hover_fg: str | None = None):
        hb = hover_bg or self.BG_HOVER
        hf = hover_fg or self.FG
        b = tk.Label(parent, text=symbol, bg=self.BG, fg=self.FG_DIM,
                     font=("Segoe UI", 11), cursor="hand2",
                     padx=14, pady=0)
        def on_enter(_e): b.configure(bg=hb, fg=hf)
        def on_leave(_e): b.configure(bg=self.BG, fg=self.FG_DIM)
        b.bind("<Enter>", on_enter)
        b.bind("<Leave>", on_leave)
        b.bind("<Button-1>", lambda _e: command())
        return b

    def _drag_start(self, event) -> None:
        self._drag_off = (event.x_root - self._root.winfo_x(),
                          event.y_root - self._root.winfo_y())

    def _drag_motion(self, event) -> None:
        if getattr(self, "_drag_off", None) is None:
            return
        ox, oy = self._drag_off
        self._root.geometry(f"+{event.x_root - ox}+{event.y_root - oy}")

    def _minimize(self) -> None:
        # For overrideredirect windows on Windows, Tk's iconify() is flaky —
        # go straight through the Win32 API so 'Show window' / SW_RESTORE
        # can reliably bring it back.
        try:
            import ctypes
            hwnd = self._root.winfo_id()
            parent = ctypes.windll.user32.GetParent(hwnd)
            if parent:
                hwnd = parent
            SW_MINIMIZE = 6
            ctypes.windll.user32.ShowWindow(hwnd, SW_MINIMIZE)
            return
        except Exception as exc:
            log.debug("ctypes minimize skipped: %s", exc)
        try:
            self._root.iconify()
        except Exception:
            self._root.withdraw()

    def _toggle_max(self) -> None:
        if self._is_maximized:
            if self._pre_max_geom:
                self._root.geometry(self._pre_max_geom)
            self._is_maximized = False
        else:
            self._pre_max_geom = self._root.geometry()
            x, y, w, h = self._get_work_area()
            self._root.geometry(f"{w}x{h}+{x}+{y}")
            self._is_maximized = True

    def _get_work_area(self) -> tuple[int, int, int, int]:
        try:
            import ctypes
            from ctypes import wintypes
            SPI_GETWORKAREA = 0x0030
            rect = wintypes.RECT()
            ctypes.windll.user32.SystemParametersInfoW(
                SPI_GETWORKAREA, 0, ctypes.byref(rect), 0,
            )
            return (rect.left, rect.top,
                    rect.right - rect.left, rect.bottom - rect.top)
        except Exception:
            return (0, 0,
                    self._root.winfo_screenwidth(),
                    self._root.winfo_screenheight())

    def _escape_key(self) -> None:
        if self._drawer_open:
            self._close_drawer()
        else:
            self.hide()

    def _on_configure(self, event) -> None:
        if event.widget is not self._root:
            return
        if self._drawer_open:
            self._place_drawer(open_=True, animate=False)

    # ── Status strip + waveform ───────────────────────────────────────────────

    def _build_status_strip(self, parent) -> None:
        strip = tk.Frame(parent, bg=self.BG, height=self.STATUS_H)
        strip.pack(fill="x", side="top", padx=18, pady=(14, 0))
        strip.pack_propagate(False)

        dot = tk.Canvas(strip, width=12, height=12, bg=self.BG,
                        highlightthickness=0, bd=0)
        dot.pack(side="left", pady=(2, 0))

        status_lbl = tk.Label(strip, text="Loading",
                              bg=self.BG, fg=self.FG,
                              font=("Segoe UI Semibold", 11))
        status_lbl.pack(side="left", padx=(10, 0))

        tk.Label(strip, text=" ·  ",
                 bg=self.BG, fg=self.FG_MUTE,
                 font=("Segoe UI", 10)).pack(side="left")

        hint_lbl = tk.Label(strip, text="getting ready…",
                            bg=self.BG, fg=self.FG_DIM,
                            font=("Segoe UI", 10))
        hint_lbl.pack(side="left")

        # Settings gear on the right
        gear = tk.Label(strip, text="\u2699", bg=self.BG, fg=self.FG_DIM,
                        cursor="hand2", font=("Segoe UI", 14),
                        padx=6)
        gear.pack(side="right")
        gear.bind("<Enter>", lambda _e: gear.configure(fg=self.FG))
        gear.bind("<Leave>", lambda _e: gear.configure(fg=self.FG_DIM))
        gear.bind("<Button-1>", lambda _e: self._toggle_drawer())

        self._widgets["dot"] = dot
        self._widgets["status"] = status_lbl
        self._widgets["hint"] = hint_lbl

    def _build_waveform(self, parent) -> None:
        holder = tk.Frame(parent, bg=self.BG)
        # Not packed yet — we show it only while recording/transcribing
        wave = tk.Canvas(holder, height=self.WAVE_H, bg=self.BG,
                         highlightthickness=0, bd=0)
        wave.pack(fill="x", padx=18, pady=(8, 4))
        self._widgets["wave_holder"] = holder
        self._widgets["wave"] = wave

    # ── History + preview ─────────────────────────────────────────────────────

    def _build_history(self, parent) -> None:
        # Section label
        label_row = tk.Frame(parent, bg=self.BG)
        label_row.pack(fill="x", side="top", padx=18, pady=(14, 6))
        tk.Label(label_row, text="RECENT",
                 bg=self.BG, fg=self.FG_MUTE,
                 font=("Segoe UI Semibold", 9)).pack(side="left")
        tk.Frame(label_row, height=1, bg=self.BORDER_SOFT
                 ).pack(side="left", fill="x", expand=True, padx=(10, 0), pady=(6, 0))

        # Container for list + preview
        container = tk.Frame(parent, bg=self.BG)
        container.pack(fill="both", expand=True, side="top", padx=18)

        list_frame = tk.Frame(container, bg=self.BG_CARD,
                              highlightthickness=1,
                              highlightbackground=self.BORDER_SOFT)
        list_frame.pack(fill="both", expand=True)

        import tkinter.ttk as ttk
        sb = ttk.Scrollbar(list_frame, style="Vertical.TScrollbar",
                           orient="vertical")
        sb.pack(side="right", fill="y")

        hist = tk.Listbox(
            list_frame,
            bg=self.BG_CARD, fg=self.FG,
            selectbackground=self.ACCENT, selectforeground="#ffffff",
            highlightthickness=0, bd=0, activestyle="none",
            font=("Segoe UI", 10),
            yscrollcommand=sb.set,
        )
        hist.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        sb.config(command=hist.yview)

        # Preview pane — shown when a row is clicked
        preview_holder = tk.Frame(container, bg=self.BG)
        # Not packed yet
        inner = tk.Frame(preview_holder, bg=self.BG_CARD,
                         highlightthickness=1, highlightbackground=self.BORDER_SOFT)
        inner.pack(fill="x", pady=(8, 0))

        head = tk.Frame(inner, bg=self.BG_CARD)
        head.pack(fill="x", padx=12, pady=(8, 0))
        preview_ts = tk.Label(head, text="", bg=self.BG_CARD, fg=self.FG_DIM,
                              font=("Consolas", 9))
        preview_ts.pack(side="left")

        # Copy button
        copy_btn = tk.Label(head, text="Copy", bg=self.BG_CARD, fg=self.ACCENT,
                            cursor="hand2", font=("Segoe UI Semibold", 9))
        copy_btn.pack(side="right")
        copy_btn.bind("<Enter>", lambda _e: copy_btn.configure(fg=self.ACCENT_H))
        copy_btn.bind("<Leave>", lambda _e: copy_btn.configure(fg=self.ACCENT))
        # Close preview
        close_prev = tk.Label(head, text="×", bg=self.BG_CARD, fg=self.FG_DIM,
                              cursor="hand2", font=("Segoe UI", 12))
        close_prev.pack(side="right", padx=(0, 10))
        close_prev.bind("<Enter>", lambda _e: close_prev.configure(fg=self.FG))
        close_prev.bind("<Leave>", lambda _e: close_prev.configure(fg=self.FG_DIM))

        preview_text = tk.Text(inner, height=4, bg=self.BG_CARD, fg=self.FG,
                               bd=0, highlightthickness=0,
                               wrap="word", padx=12, pady=(6, 10),
                               font=("Segoe UI", 10))
        preview_text.pack(fill="x")
        preview_text.configure(state="disabled")

        def _hide_preview():
            preview_holder.pack_forget()
            try:
                hist.selection_clear(0, "end")
            except Exception:
                pass
        close_prev.bind("<Button-1>", lambda _e: _hide_preview())

        def _select(_e=None):
            sel = hist.curselection()
            if not sel:
                return
            data = self._widgets.get("hist_data") or []
            i = sel[0]
            if 0 <= i < len(data):
                e = data[i]
                preview_ts.configure(text=(e.get("ts") or "").replace("T", " ")[:19])
                preview_text.configure(state="normal")
                preview_text.delete("1.0", "end")
                preview_text.insert("1.0", e.get("text", ""))
                preview_text.configure(state="disabled")
                if not preview_holder.winfo_ismapped():
                    preview_holder.pack(fill="x", side="top")

        def _copy_current(_e=None):
            sel = hist.curselection()
            data = self._widgets.get("hist_data") or []
            if sel and 0 <= sel[0] < len(data):
                try:
                    pyperclip.copy(data[sel[0]].get("text", ""))
                    orig = copy_btn.cget("text")
                    copy_btn.configure(text="Copied!")
                    copy_btn.after(1200, lambda: copy_btn.configure(text=orig))
                except Exception:
                    pass
        copy_btn.bind("<Button-1>", _copy_current)

        hist.bind("<<ListboxSelect>>", _select)
        hist.bind("<Double-Button-1>", _copy_current)
        hist.bind("<Return>", _copy_current)

        self._widgets["hist"] = hist
        self._widgets["hist_data"] = []
        self._widgets["preview_holder"] = preview_holder
        self._widgets["preview_ts"] = preview_ts
        self._widgets["preview_text"] = preview_text

    # ── Action row + footer ───────────────────────────────────────────────────

    def _build_actions(self, parent) -> None:
        row = tk.Frame(parent, bg=self.BG)
        row.pack(fill="x", side="top", padx=18, pady=(12, 10))

        # Primary record button
        rec_btn = tk.Label(row, text="\u25CF   Record",
                           bg=self.ACCENT, fg="#ffffff",
                           font=("Segoe UI Semibold", 10),
                           padx=18, pady=8, cursor="hand2")
        rec_btn.pack(side="left")
        rec_btn.bind("<Enter>", lambda _e: rec_btn.configure(bg=self.ACCENT_H))
        rec_btn.bind("<Leave>", lambda _e: rec_btn.configure(bg=self.ACCENT))
        rec_btn.bind("<Button-1>",
                     lambda _e: self._cb_record and self._cb_record())

        # Secondary: open full history
        hist_btn = tk.Label(row, text="Open full history \u2192",
                            bg=self.BG, fg=self.FG_DIM,
                            font=("Segoe UI", 10), cursor="hand2",
                            padx=10, pady=8)
        hist_btn.pack(side="right")
        hist_btn.bind("<Enter>", lambda _e: hist_btn.configure(fg=self.FG))
        hist_btn.bind("<Leave>", lambda _e: hist_btn.configure(fg=self.FG_DIM))
        hist_btn.bind("<Button-1>",
                      lambda _e: self._cb_history and self._cb_history())

        self._widgets["rec_btn"] = rec_btn

    def _build_footer(self, parent) -> None:
        import webbrowser
        tk.Frame(parent, height=1, bg=self.BORDER_SOFT
                 ).pack(fill="x", side="top")
        foot = tk.Frame(parent, bg=self.BG, height=self.FOOTER_H)
        foot.pack(fill="x", side="top")
        foot.pack_propagate(False)

        left = tk.Frame(foot, bg=self.BG)
        left.pack(side="left", padx=18)
        tk.Label(left, text=f"STT v{APP_VERSION}",
                 bg=self.BG, fg=self.FG_DIM,
                 font=("Segoe UI Semibold", 9)).pack(side="left", pady=8)
        tk.Label(left, text="  \u00b7  ", bg=self.BG, fg=self.FG_MUTE,
                 font=("Segoe UI", 9)).pack(side="left", pady=8)
        gh = tk.Label(left, text="GitHub", bg=self.BG, fg=self.ACCENT,
                      cursor="hand2", font=("Segoe UI", 9, "underline"))
        gh.pack(side="left", pady=8)
        gh.bind("<Enter>", lambda _e: gh.configure(fg=self.ACCENT_H))
        gh.bind("<Leave>", lambda _e: gh.configure(fg=self.ACCENT))
        gh.bind("<Button-1>", lambda _e: webbrowser.open(GITHUB_URL))

        right = tk.Frame(foot, bg=self.BG)
        right.pack(side="right", padx=18)
        tk.Label(right, text="by ", bg=self.BG, fg=self.FG_DIM,
                 font=("Segoe UI", 9)).pack(side="left", pady=8)
        ms = tk.Label(right, text="MarkSoft", bg=self.BG, fg=self.ACCENT,
                      cursor="hand2",
                      font=("Segoe UI Semibold", 9, "underline"))
        ms.pack(side="left", pady=8)
        ms.bind("<Enter>", lambda _e: ms.configure(fg=self.ACCENT_H))
        ms.bind("<Leave>", lambda _e: ms.configure(fg=self.ACCENT))
        ms.bind("<Button-1>", lambda _e: webbrowser.open(MARKSOFT_URL))

    # ── Settings drawer (slide-in from the right) ─────────────────────────────

    def _build_drawer(self, root) -> None:
        import tkinter as tk
        from tkinter import ttk

        drawer = tk.Frame(root, bg=self.BG_CARD,
                          highlightthickness=1,
                          highlightbackground=self.BORDER)
        # Off-screen right initially
        drawer.place(x=self.W, y=self.TITLE_H + 1,
                     width=self.DRAWER_W,
                     height=self.H - self.TITLE_H - 1)
        self._drawer = drawer
        self._drawer_open = False

        # Header
        header = tk.Frame(drawer, bg=self.BG_CARD, height=44)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        tk.Label(header, text="Settings",
                 bg=self.BG_CARD, fg=self.FG,
                 font=("Segoe UI Semibold", 12)
                 ).pack(side="left", padx=16, pady=10)
        close = tk.Label(header, text="\u2715", bg=self.BG_CARD, fg=self.FG_DIM,
                         cursor="hand2", font=("Segoe UI", 11), padx=12)
        close.pack(side="right", pady=4)
        close.bind("<Enter>", lambda _e: close.configure(fg=self.FG))
        close.bind("<Leave>", lambda _e: close.configure(fg=self.FG_DIM))
        close.bind("<Button-1>", lambda _e: self._close_drawer())
        tk.Frame(drawer, height=1, bg=self.BORDER_SOFT).pack(fill="x", side="top")

        # Body
        body = tk.Frame(drawer, bg=self.BG_CARD)
        body.pack(fill="both", expand=True, side="top", padx=16, pady=14)

        def section_label(txt):
            tk.Label(body, text=txt.upper(),
                     bg=self.BG_CARD, fg=self.FG_MUTE,
                     font=("Segoe UI Semibold", 9)
                     ).pack(anchor="w", pady=(10, 4))

        # HOTKEY
        section_label("Hotkey")
        hrow = tk.Frame(body, bg=self.BG_CARD)
        hrow.pack(fill="x")
        hk_val = tk.Label(hrow, text="—", bg=self.BG_CARD, fg=self.FG,
                          font=("Segoe UI Semibold", 10))
        hk_val.pack(side="left")
        hk_change = tk.Label(hrow, text="Change", bg=self.BG_CARD, fg=self.ACCENT,
                             cursor="hand2",
                             font=("Segoe UI Semibold", 9, "underline"))
        hk_change.pack(side="right")
        hk_change.bind("<Enter>", lambda _e: hk_change.configure(fg=self.ACCENT_H))
        hk_change.bind("<Leave>", lambda _e: hk_change.configure(fg=self.ACCENT))
        hk_change.bind("<Button-1>",
                       lambda _e: self._cb_hotkey and self._cb_hotkey())

        # MODEL
        section_label("Model")
        mod_var = tk.StringVar(value="base")
        mod_cb = ttk.Combobox(body, textvariable=mod_var, state="readonly",
                              values=["tiny", "base", "small"], width=20)
        mod_cb.pack(anchor="w", fill="x")
        mod_cb.bind("<<ComboboxSelected>>",
                    lambda _e: self._cb_model and self._cb_model(mod_var.get()))

        # LANGUAGE
        section_label("Language")
        lang_var = tk.StringVar(value="en")
        lang_cb = ttk.Combobox(
            body, textvariable=lang_var, state="readonly",
            values=["auto", "en", "ro", "fr", "de", "es", "it",
                    "pt", "nl", "pl", "ja", "zh"],
            width=20,
        )
        lang_cb.pack(anchor="w", fill="x")
        lang_cb.bind("<<ComboboxSelected>>",
                     lambda _e: self._cb_lang and self._cb_lang(lang_var.get()))

        # About — separator + version + link
        tk.Frame(body, height=1, bg=self.BORDER_SOFT
                 ).pack(fill="x", pady=(18, 10))
        section_label("About")
        about_row = tk.Frame(body, bg=self.BG_CARD)
        about_row.pack(fill="x")
        tk.Label(about_row, text=f"STT v{APP_VERSION}",
                 bg=self.BG_CARD, fg=self.FG,
                 font=("Segoe UI Semibold", 10)).pack(side="left")
        wn_link = tk.Label(about_row, text="What's new \u2192",
                           bg=self.BG_CARD, fg=self.ACCENT, cursor="hand2",
                           font=("Segoe UI Semibold", 9, "underline"))
        wn_link.pack(side="right")
        wn_link.bind("<Enter>", lambda _e: wn_link.configure(fg=self.ACCENT_H))
        wn_link.bind("<Leave>", lambda _e: wn_link.configure(fg=self.ACCENT))
        wn_link.bind("<Button-1>",
                     lambda _e: self._cb_whats_new and self._cb_whats_new())

        self._widgets["hk_val"] = hk_val
        self._widgets["mod_var"] = mod_var
        self._widgets["lang_var"] = lang_var

    def _place_drawer(self, *, open_: bool, animate: bool = True) -> None:
        if self._root is None or self._drawer is None:
            return
        W = self._root.winfo_width() or self.W
        H = self._root.winfo_height() or self.H
        target_x = W - self.DRAWER_W if open_ else W
        current_x = self._drawer.winfo_x() if self._drawer.winfo_ismapped() else W

        if not animate:
            self._drawer.place(x=target_x, y=self.TITLE_H + 1,
                               width=self.DRAWER_W,
                               height=H - self.TITLE_H - 1)
            self._drawer.lift()
            return

        def step(i=0, total=8):
            t = i / total
            e = 1 - (1 - t) ** 3         # ease-out cubic
            x = int(current_x + (target_x - current_x) * e)
            self._drawer.place(x=x, y=self.TITLE_H + 1,
                               width=self.DRAWER_W,
                               height=H - self.TITLE_H - 1)
            self._drawer.lift()
            if i < total:
                self._root.after(14, lambda: step(i + 1, total))
        step()

    def _toggle_drawer(self) -> None:
        if self._drawer_open:
            self._close_drawer()
        else:
            self._open_drawer()

    def _open_drawer(self) -> None:
        self._drawer_open = True
        self._place_drawer(open_=True)

    def _close_drawer(self) -> None:
        self._drawer_open = False
        self._place_drawer(open_=False)

    # ── Drawing / polling loop ────────────────────────────────────────────────

    _STATE_COPY = {
        "LOADING_MODEL": ("Loading",      "getting the model ready…"),
        "IDLE":          ("Ready",        "hold Right Alt or click record"),
        "RECORDING":     ("Recording",    "release to transcribe"),
        "TRANSCRIBING":  ("Transcribing", "almost there…"),
    }

    def _state_color(self, name: str) -> str:
        return {
            "LOADING_MODEL": self.CYAN,
            "IDLE":          self.OK,
            "RECORDING":     self.REC,
            "TRANSCRIBING":  self.AMBER,
        }.get(name, self.FG_DIM)

    def _tick(self) -> None:
        import math
        if self._root is None:
            return
        w = self._widgets
        self._wave_phase += 1

        # ── State transitions ───────────────────────────────────────────
        st = self._get_state()
        st_name = getattr(st, "name", str(st))
        if st_name != self._last_state:
            self._last_state = st_name
            color = self._state_color(st_name)
            label, hint = self._STATE_COPY.get(st_name, (st_name, ""))

            dot: tk.Canvas = w["dot"]
            dot.delete("all")
            dot.create_oval(0, 0, 12, 12, fill=color, outline="")

            w["status"].configure(text=label, fg=color)
            w["hint"].configure(
                text=hint + (f"  ·  {self._get_config().get('hotkey','')}"
                             if st_name == "IDLE" else ""),
            )

            # Record button label flips while recording
            if st_name == "RECORDING":
                w["rec_btn"].configure(text="\u25A0   Stop", bg=self.REC)
                w["rec_btn"].bind("<Leave>",
                                  lambda _e: w["rec_btn"].configure(bg=self.REC))
                w["rec_btn"].bind("<Enter>",
                                  lambda _e: w["rec_btn"].configure(bg="#ff6b60"))
            else:
                w["rec_btn"].configure(text="\u25CF   Record", bg=self.ACCENT)
                w["rec_btn"].bind("<Leave>",
                                  lambda _e: w["rec_btn"].configure(bg=self.ACCENT))
                w["rec_btn"].bind("<Enter>",
                                  lambda _e: w["rec_btn"].configure(bg=self.ACCENT_H))

            # Waveform holder visibility
            show_wave = st_name in ("RECORDING", "TRANSCRIBING")
            holder = w["wave_holder"]
            if show_wave and not holder.winfo_ismapped():
                holder.pack(fill="x", side="top", after=w["dot"].master)
            elif not show_wave and holder.winfo_ismapped():
                holder.pack_forget()

        # ── Config-driven labels (hotkey / model / language) ─────────────
        cfg = self._get_config() or {}
        hk = cfg.get("hotkey", "—")
        if hk != self._last_hotkey:
            self._last_hotkey = hk
            if "hk_val" in w:
                w["hk_val"].configure(text=hk)
            # Also refresh hint line when idle so the hotkey stays current
            if self._last_state == "IDLE":
                w["hint"].configure(text=f"hold {hk} or click record")
        model = cfg.get("model", "base")
        if model != self._last_model:
            self._last_model = model
            if "mod_var" in w:
                w["mod_var"].set(model)
        lang = cfg.get("language") or "auto"
        if lang != self._last_language:
            self._last_language = lang
            if "lang_var" in w:
                w["lang_var"].set(lang)

        # ── Waveform (rolling smooth mirror curve) ───────────────────────
        wave: tk.Canvas = w["wave"]
        if w["wave_holder"].winfo_ismapped():
            wave.delete("all")
            cw = int(wave.winfo_width()) or (self.W - 36)
            ch = int(wave.winfo_height()) or self.WAVE_H
            cy = ch // 2

            # Roll the level buffer
            live = 0.08
            if self._last_state == "RECORDING":
                live = max(0.10, min(1.0, self._get_level()))
            self._levels.pop(0)
            prev = self._levels[-1]
            self._levels.append(prev + (live - prev) * 0.3)

            col = self.REC if self._last_state == "RECORDING" else self.AMBER
            n = len(self._levels)
            step = cw / (n - 1)
            max_amp = (ch - 4) / 2
            top, bot = [], []
            phase = self._wave_phase / 4.0
            for i, v in enumerate(self._levels):
                x = i * step
                wiggle = 0.85 + 0.15 * math.sin(phase + i * 0.35)
                a = v * max_amp * wiggle
                top += [x, cy - a]
                bot += [x, cy + a]
            wave.create_line(*top, smooth=True, width=1.6,
                             fill=col, capstyle="round", joinstyle="round")
            wave.create_line(*bot, smooth=True, width=1.6,
                             fill=col, capstyle="round", joinstyle="round")

        # ── History refresh every ~1 s ───────────────────────────────────
        if self._wave_phase % 30 == 0:
            try:
                entries = self._get_recent() or []
            except Exception:
                entries = []
            hist: tk.Listbox = w["hist"]
            # Only redraw if the newest entry changed
            data = w.get("hist_data") or []
            newest_new = (entries[0].get("ts") if entries else None)
            newest_old = (data[0].get("ts") if data else None)
            if newest_new != newest_old or len(entries) != len(data):
                hist.delete(0, "end")
                for e in entries[:25]:
                    ts = (e.get("ts") or "")[11:16]      # HH:MM
                    txt = (e.get("text") or "").replace("\n", " ").strip()
                    if len(txt) > 64:
                        txt = txt[:62] + "…"
                    hist.insert("end", f"  {ts}   {txt}")
                w["hist_data"] = entries[:25]

        self._root.after(33, self._tick)

    # ── Public (thread-safe) ──────────────────────────────────────────────────

    def show(self) -> None:
        if self._root is None:
            return

        def _apply():
            # deiconify() alone is unreliable for overrideredirect windows
            # that were iconified or hidden via ShowWindow — layer ctypes.
            try:
                self._root.deiconify()
            except Exception:
                pass
            try:
                import ctypes
                hwnd = self._root.winfo_id()
                # Fall back to the parent HWND if Tk wraps us for some reason
                parent = ctypes.windll.user32.GetParent(hwnd)
                if parent:
                    hwnd = parent
                SW_RESTORE = 9
                ctypes.windll.user32.ShowWindow(hwnd, SW_RESTORE)
                try:
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
                except Exception:
                    pass
            except Exception as exc:
                log.debug("show() ctypes restore skipped: %s", exc)
            try:
                self._root.lift()
                self._root.focus_force()
            except Exception:
                pass

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

    # ── Update checking ───────────────────────────────────────────────────────

    def _show_whats_new(self, *, force: bool = False) -> None:
        """Fetch release notes for APP_VERSION and show the 'What's new' card.
        When `force=False`, we only show once per version (last_seen_version)."""
        def _worker():
            try:
                rel = _fetch_release_for_tag(f"v{APP_VERSION}")
            except Exception as exc:
                log.debug("What's-new fetch failed: %s", exc)
                rel = None
            if not rel:
                if force:
                    self._notify("Couldn't load release notes. Check your connection.")
                return

            update = {
                "tag":          rel.get("tag_name") or f"v{APP_VERSION}",
                "name":         rel.get("name") or f"v{APP_VERSION}",
                "body":         rel.get("body") or "",
                "html_url":     rel.get("html_url"),
                "published_at": rel.get("published_at", ""),
            }

            def _on_dismiss():
                # Record that this version's notes have been seen
                self.config["last_seen_version"] = APP_VERSION
                save_config(self.config)

            threading.Thread(
                target=show_update_dialog,
                args=(update,),
                kwargs={"mode": "whatsnew", "on_dismiss": _on_dismiss},
                name="whatsnew-dialog", daemon=True,
            ).start()
        threading.Thread(target=_worker, name="whatsnew-fetch",
                         daemon=True).start()

    def _maybe_show_whats_new_on_launch(self) -> None:
        """If the user just updated to a new version, surface the notes once."""
        last_seen = (self.config.get("last_seen_version") or "").strip()
        if last_seen == APP_VERSION:
            return
        # Only surface when updating, not on first-ever install
        if last_seen:
            # Give the main window a moment to appear first
            threading.Timer(3.0, lambda: self._show_whats_new(force=True)).start()
        else:
            # First launch ever — just record the version without surfacing notes
            self.config["last_seen_version"] = APP_VERSION
            save_config(self.config)

    def _check_updates(self, *, manual: bool) -> None:
        """Background check. If newer release exists, show the pop-up.
        When `manual`, always surface a result (pop-up or notification)."""
        def _worker():
            try:
                upd = check_for_update(force=manual)
            except Exception as exc:
                log.debug("Update check error: %s", exc)
                upd = None
            if upd:
                try:
                    threading.Thread(
                        target=show_update_dialog, args=(upd,),
                        name="update-dialog", daemon=True,
                    ).start()
                except Exception:
                    pass
            elif manual:
                self._notify(f"You're up to date.  (v{APP_VERSION})")
        threading.Thread(target=_worker, name="update-check",
                         daemon=True).start()

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

        def _manual_update(icon, item):
            self._check_updates(manual=True)

        def _whats_new(icon, item):
            self._show_whats_new(force=True)

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
            pystray.MenuItem("What's new in this version…", _whats_new),
            pystray.MenuItem("Check for updates…", _manual_update),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(f"Version  {APP_VERSION}", action=None, enabled=False),
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
                    on_show_whats_new=lambda: self._show_whats_new(force=True),
                )
                if self.config.get("start_minimized", False):
                    self.main_window.hide()
            except Exception as exc:
                log.warning("Main window init failed: %s", exc)
                self.main_window = None

        self._register_hotkey()
        self._load_model()

        # Non-blocking "new version?" check 8 s after startup
        threading.Timer(8.0, lambda: self._check_updates(manual=False)).start()

        # First launch after an update → surface the "What's new" card once
        self._maybe_show_whats_new_on_launch()

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