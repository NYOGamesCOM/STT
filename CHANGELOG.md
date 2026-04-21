# Changelog

All notable user-facing changes land here. Releases are cut with
semantic-ish versioning and this file is the source of truth for what
each tag shipped. The release workflow pulls the section for the tag
into the GitHub Release body and the in-app update dialog.

## v0.2.3 · 2026-04-21 — Show-window fixes + native title bar escape hatch

### 🐛 Fixed
- Clicking **Show window** in the tray menu now reliably restores the
  main window from every hidden state (withdrawn, iconified, or
  minimised via the Win32 API) — 5-step restore: deiconify →
  `ShowWindow(SW_RESTORE/SW_SHOW)` → recentre if off-screen →
  topmost-flash → `SetForegroundWindow`.
- Debug log now records every show / restore attempt so any remaining
  edge case can be diagnosed from `stt.log`.
- Reverted the minimise button to Tk's `iconify()` — the restore path
  handles every entry state, no need for a bespoke minimise flavour.

### ✨ New
- `use_native_titlebar` config option as an escape hatch. Set it to
  `true` in `config.json` to fall back to the native Windows title bar
  (dark on Windows 11 via DWM, white on Windows 10). The custom
  title bar remains the default.

## v0.2.2 · 2026-04-21 — Reliable minimize & restore

### 🐛 Fixed
- Main window wouldn't come back after minimizing or closing — "Show window"
  (tray right-click) and double-clicking the tray icon now reliably restore
  it via the Win32 API. The custom-title-bar `−` minimise button also
  uses `SW_MINIMIZE` directly so it's consistent with the restore path.

### 💡 Heads-up
- Windows tray icons respond to **right-click** for the menu and
  **double-click** to open the main window. A single left-click does
  nothing (OS convention; can't be changed).

## v0.2.1 · 2026-04-21 — Release notes that actually read like release notes

### ✨ New
- Real changelog: every release now ships hand-written user-facing notes
- "What's new" dialog on first launch after you update to a new version
- "What's new in this version…" entry added to the tray menu
- Current version now visible in the Settings drawer (under **About**)

### 🎨 Design
- Redesigned update dialog — custom dark title bar, version diff pill
  (`v0.1.5 → v0.2.0`), relative release time ("released 2d ago"),
  section-coloured bullets (mint = new, amber = design, red = fixed)
- Tertiary "See on GitHub" link added alongside the primary buttons

## v0.2.0 · 2026-04-21 — Full UI redesign

### ✨ New
- Fully dark-themed custom title bar with drag, minimize, maximize and close
- Settings drawer slides in from the right with an ease-out animation
- Click a history row to preview the full text; explicit Copy button,
  no surprise clipboard writes

### 🎨 Design
- iOS-inspired palette — systemRed, systemOrange, systemBlue accents
- Slimmer overlay pill (200 × 32) with thinner waveform line and a single soft halo
- Compact 44 px status strip replaces the oversized status card
- Waveform only shows during record/transcribe — calmer window when idle
- Typography scale of 3 (title/body/caption), tighter 8 px grid, 460 × 600 window
- Combobox dark theme fixed (field / list / scrollbar all styled)

### 🐛 Fixed
- Taskbar icon now reliably shows the branded S on frozen Windows builds

## v0.1.5 · 2026-04-21 — Branded icons and the first update checker

### ✨ New
- In-app update checker — polls GitHub 8 s after launch and pops up a
  dark-themed dialog when a newer tag is out
- Manual "Check for updates…" item in the tray menu
- Current version visible in the tray menu (next to Quit)

### 🎨 Design
- Redesigned tray and window icons — rounded-square dark tile with an
  "S" wordmark and a state-coloured badge dot (cyan / mint / amber / red)
- Main window footer — separator, version, GitHub and MarkSoft links

## v0.1.4 · 2026-04-21 — Liquid-glass overlay

### 🎨 Design
- On-screen pill redesigned with true transparent rounded corners on Windows
- Smooth scrolling bezier waveform replaces the old discrete bars
- Pulsing status dot with a soft halo, monospace elapsed timer, breathing
  transcribe dots

## v0.1.3 · 2026-04-20 — Windows metadata

### 🎨 Design
- Embedded VERSIONINFO — Publisher and Product Name now show in
  Explorer / task manager / the SmartScreen dialog
- README walks users through the SmartScreen bypass
  ("More info → Run anyway")

## v0.1.2 · 2026-04-20 — Main desktop window

### ✨ New
- Added a proper dark-themed desktop window (was tray-only before)
- Click-to-toggle record button as an alternative to the hotkey
- Integrated recent-transcriptions list and live waveform

### 🐛 Fixed
- Frozen Windows builds were capturing silence — switched to a
  non-persistent mic stream by default, matching the proven code path

## v0.1.1 · 2026-04-20 — Reliable bundling

### 🐛 Fixed
- numpy C-extensions import error in the packaged exe
- UPX no longer compresses critical DLLs on Windows builds

## v0.1.0 · 2026-04-20 — Initial public release

- Held-key push-to-talk voice-to-text, fully offline, powered by faster-whisper
- System-tray control with live state indicator
- Configurable hotkey, model (tiny/base/small), language
