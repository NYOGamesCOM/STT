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
    "model": "base",          # "base" | "small" — Whisper model size
    "hotkey": "right alt",    # Key name as recognised by the keyboard library
    "language": None,         # None = auto-detect, or e.g. "en", "ro"
    "sample_rate": 16000,     # Hz — Whisper expects 16 kHz
    "device": "cpu",          # "cpu" | "cuda"
    "compute_type": "int8",   # "int8" | "float16" | "float32"
    "paste_method": "clipboard",  # "clipboard" | "type"
    "min_audio_seconds": 0.5,     # Ignore recordings shorter than this
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
    """Captures audio from the default microphone while active."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._frames: list[np.ndarray] = []
        self._active = False
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.debug("sounddevice status: %s", status)
        if self._active:
            with self._lock:
                self._frames.append(indata.copy())

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._frames = []
        self._active = True
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=self._callback,
            )
            self._stream.start()
            log.info("Audio recording started (%.0f Hz)", self.sample_rate)
        except Exception as exc:
            self._active = False
            log.error("Failed to open audio stream: %s", exc)
            raise

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a 1-D float32 array."""
        self._active = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                log.warning("Error closing audio stream: %s", exc)
            self._stream = None

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


# ═════════════════════════════════════════════════════════════════════════════
# Transcriber (wraps faster-whisper)
# ═════════════════════════════════════════════════════════════════════════════

class Transcriber:
    """Lazy-loading wrapper around WhisperModel."""

    def __init__(self, model_name: str, device: str = "cpu", compute_type: str = "int8"):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
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
                log.info("Loading WhisperModel '%s' (device=%s, compute=%s)…",
                         self.model_name, self.device, self.compute_type)
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
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

    # ── Transcription ─────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        """Return transcribed text. Raises if model not loaded."""
        if self._model is None:
            raise RuntimeError("Model not loaded yet.")
        if len(audio) == 0:
            return ""

        try:
            segments, _info = self._model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:
            # VAD can fail in rare edge cases — retry without it
            log.warning("Transcription with VAD failed (%s); retrying without VAD.", exc)
            segments, _info = self._model.transcribe(
                audio,
                language=language,
                beam_size=5,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()

        log.info("Transcribed: %r", text)
        return text


# ═════════════════════════════════════════════════════════════════════════════
# Text injection
# ═════════════════════════════════════════════════════════════════════════════

def inject_text(text: str, method: str = "clipboard") -> None:
    """Type/paste *text* into whatever window currently has focus."""
    if not text:
        return

    # Brief pause so the hotkey-release event settles before we send input
    time.sleep(0.15)

    if method == "clipboard":
        _inject_via_clipboard(text)
    else:
        _inject_via_type(text)


def _inject_via_clipboard(text: str) -> None:
    old: str = ""
    try:
        old = pyperclip.paste() or ""
    except Exception:
        pass

    try:
        pyperclip.copy(text)
        time.sleep(0.05)
        keyboard.send("ctrl+v")
        time.sleep(0.15)
    except Exception as exc:
        log.error("Clipboard injection failed: %s — falling back to type.", exc)
        _inject_via_type(text)
        return
    finally:
        # Restore previous clipboard contents
        try:
            pyperclip.copy(old)
        except Exception:
            pass


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

        self.recorder = AudioRecorder(sample_rate=self.config["sample_rate"])
        self.transcriber = self._make_transcriber()

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
                text = self.transcriber.transcribe(audio, language=language)
                if text:
                    inject_text(text, method=self.config.get("paste_method", "clipboard"))
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
            pystray.MenuItem(
                "Model",
                pystray.Menu(
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
            pystray.MenuItem("Quit", _quit),
        )

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("STT (Speech to text) starting… (config: %s)", CONFIG_PATH)

        self._register_hotkey()
        self._load_model()

        self.icon = pystray.Icon(
            name="STT",
            icon=_make_icon(AppState.LOADING_MODEL),
            title=_ICON_TOOLTIPS[AppState.LOADING_MODEL],
            menu=self._build_menu(),
        )

        log.info("System tray icon running.")
        self.icon.run()          # blocks until icon.stop() is called
        log.info("STT (Speech to text) shut down.")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = STTApp()
    app.run()