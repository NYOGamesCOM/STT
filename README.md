# STT (Speech to text) 🎙️

A lightweight, **fully offline** hold-to-record voice-to-text app for Windows  
(macOS support included; Linux untested but should work).

Sits in your system tray. Hold **Right Alt**, speak, release → transcribed text  
is pasted into whatever window is focused.

Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — no API  
keys, no cloud, everything runs locally.

🌐 **Project page:** [marksoft.ro/stt](https://marksoft.ro/stt)

---

## 📥 Download

Pre-built binaries for the latest release — no Python install required.
Or grab them from [marksoft.ro/stt](https://marksoft.ro/stt#download).

| Platform | Download | Run |
|---|---|---|
| **Windows 10/11 (x64)** | [**STT-windows-x64.exe**](https://github.com/NYOGamesCOM/STT/releases/latest/download/STT-windows-x64.exe) | Right-click → **Run as administrator** (global hotkeys need it) |
| **macOS (Apple Silicon)** | [**STT-macos.zip**](https://github.com/NYOGamesCOM/STT/releases/latest/download/STT-macos.zip) | Unzip → grant **Accessibility** in System Settings → Privacy & Security |
| **Linux (x64)** | [**STT-linux-x64**](https://github.com/NYOGamesCOM/STT/releases/latest/download/STT-linux-x64) | `chmod +x STT-linux-x64 && sudo ./STT-linux-x64` |

Browse every version on the [**Releases page**](https://github.com/NYOGamesCOM/STT/releases).

> First launch downloads the Whisper `base` model (~150 MB) into your
> HuggingFace cache. Everything after that is fully offline.

---

## Quick-start (from source)

### 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.11 recommended |
| pip | bundled with Python |
| **Windows**: Run as Administrator | The `keyboard` library needs elevated privileges for global hotkey hooks |
| **macOS**: Accessibility permission | System Settings → Privacy & Security → Accessibility → add Terminal / your app |

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows note:** if you see a CTranslate2 / ONNX error, make sure you have the  
> [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) installed.

### 3. Run

```bash
# Windows – must be run as Administrator for global hotkeys
python stt.py

# macOS
python3 stt.py
```

The app will:
1. Start and appear in the system tray (blue microphone icon)
2. Download the `base` Whisper model on first run (~150 MB, one-time)
3. Turn **amber** while loading the model
4. Turn **blue** (idle) once ready

---

## Usage

| Action | What happens |
|---|---|
| **Hold Right Alt** | Recording starts (icon turns red) |
| **Release Right Alt** | Recording stops; transcription runs (icon turns green) |
| Transcription complete | Text is pasted into the focused window; icon returns to blue |

### Tray menu (right-click the icon)

- **Hotkey** – shows the currently configured hotkey (read-only)
- **Switch Model → base / small** – hot-switch the Whisper model (downloads if needed)
- **Quit** – exit the app

---

## Configuration (`config.json`)

```json
{
  "model": "base",
  "hotkey": "right alt",
  "language": "en",
  "sample_rate": 16000,
  "channels": 1,
  "beam_size": 5
}
```

| Key | Values | Description |
|---|---|---|
| `model` | `"base"` / `"small"` | Whisper model size. `base` ≈ 150 MB, `small` ≈ 500 MB |
| `hotkey` | Any key name | See [keyboard key names](https://github.com/boppreh/keyboard#api) |
| `language` | `"en"`, `"fr"`, … or `null` | Force language or `null` for auto-detect |
| `sample_rate` | `16000` | Do not change unless your mic requires it |
| `channels` | `1` | Mono recommended |
| `beam_size` | `1`–`10` | Higher = more accurate but slower. `5` is a good default |

---

## Model sizes & trade-offs

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `base` | ~150 MB | Very fast | Good |
| `small` | ~500 MB | Fast | Better |
| `medium` | ~1.5 GB | Moderate | Very good |
| `large-v3` | ~3 GB | Slow | Best |

To use `medium` or `large-v3`, add them to the `Switch Model` sub-menu in  
`_build_menu()` inside `stt.py`.

---

## Packaging with PyInstaller

### Install PyInstaller

```bash
pip install pyinstaller==6.10.0
```

### Windows – single `.exe`

```bash
pyinstaller \
  --onefile \
  --noconsole \
  --name stt \
  --icon stt.ico \
  --add-data "config.json;." \
  stt.py
```

> Remove `--icon stt.ico` if you don't have an icon file.  
> The `--noconsole` flag suppresses the console window on Windows.

The executable will be at `dist\stt.exe`. Copy `config.json` next to it  
(it will be created automatically on first run if absent).

### macOS – `.app` bundle

```bash
pyinstaller \
  --onefile \
  --windowed \
  --name STT \
  --add-data "config.json:." \
  stt.py
```

Bundle is at `dist/STT.app`.  
You may need to sign it: `codesign --deep --force --sign - dist/STT.app`

### Notes on frozen paths

`stt.py` detects whether it is running as a PyInstaller bundle via  
`sys._MEIPASS` / `sys.frozen` and resolves `config.json` relative to the  
**executable** (not the bundle's temp directory) so settings persist between  
runs.

---

## Releasing (maintainers)

Pre-built binaries for Windows, macOS, and Linux are produced automatically by
[`.github/workflows/release.yml`](.github/workflows/release.yml) whenever a
`v*` tag is pushed.

```bash
# 1. Tag the commit you want to ship
git tag v0.1.0
git push origin v0.1.0

# 2. GitHub Actions builds STT on windows-latest / macos-latest / ubuntu-latest
#    and attaches the three binaries to a new release for that tag.
#    Watch progress: https://github.com/NYOGamesCOM/STT/actions
```

Resulting release assets:

| Asset | Platform | Build command |
|---|---|---|
| `STT-windows-x64.exe` | Windows 10/11 x64 | `pyinstaller --onefile --noconsole` |
| `STT-macos.zip` | macOS (arm64) | `pyinstaller --onefile --windowed` |
| `STT-linux-x64` | Linux x64 | `pyinstaller --onefile` |

The workflow can also be triggered manually from the **Actions** tab
(*Run workflow* → enter the tag name).

### Permanent "latest" download URLs

GitHub exposes a `releases/latest/download/<asset>` redirect that always points
to the newest tagged release — use these on your website so links never go
stale:

```
https://github.com/NYOGamesCOM/STT/releases/latest/download/STT-windows-x64.exe
https://github.com/NYOGamesCOM/STT/releases/latest/download/STT-macos.zip
https://github.com/NYOGamesCOM/STT/releases/latest/download/STT-linux-x64
```

Version-pinned URLs are also available at
`releases/download/v0.1.0/<asset>` if you need reproducibility.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Hotkey not detected (Windows) | Run as Administrator |
| "Accessibility" error (macOS) | Grant Accessibility permission to Terminal / the `.app` |
| Model download hangs | Check internet connection; model is cached in `~/.cache/huggingface` |
| Pasted text is garbled | Try setting `"language": "en"` in `config.json` |
| Nothing pasted | Make sure the target app accepts Ctrl+V; try clicking it first |
| `sounddevice` error on Windows | Install [portaudio](http://www.portaudio.com/) or use the conda package |

---

## Contributors

- **[NYOGamesCOM](https://github.com/NYOGamesCOM)** — creator & maintainer
- **[Claude](https://claude.com/claude-code)** (Anthropic Opus 4.7) — overlay indicator, history UI, performance pass, build/release plumbing
  - assisted by **[Cursor](https://cursor.com)** — editor & pair-programming companion

Built with ❤️ by [MarkSoft](https://marksoft.ro).

---

## License

MIT – do whatever you like.