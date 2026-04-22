# Changelog

All notable user-facing changes land here. Releases are cut with
semantic-ish versioning and this file is the source of truth for what
each tag shipped. The release workflow pulls the section for the tag
into the GitHub Release body and the in-app update dialog.

## v0.3.4 · 2026-04-22 — Overlay actually visible (verified at the Win32 API level)

### 🐛 Fixed
- **Overlay indicator was invisible because of a bug in *our* click-through
  setup**, not the Tk root state. Confirmed by probing the actual HWND
  state with `ctypes` — no more guessing:
  - `Overlay._build`'s ctypes call used `top.winfo_id()` as the target,
    which returns the *inner* Tk frame HWND, not the Win32 top-level.
  - We then OR-ed `WS_EX_LAYERED` onto that inner frame.
  - A layered window with no `SetLayeredWindowAttributes` or
    `UpdateLayeredWindow` call is rendered as **fully transparent** by
    the DWM compositor — everything we drew on the canvas was silently
    dropped, even though Tk reported `mapped=True`.
  - Meanwhile, Tk's own `-alpha` attribute correctly set layered
    attributes on the *root* HWND (alpha=235, topmost, etc.) — so the
    wrapper behaved correctly, it was just invisible.
- **Fix**: apply click-through flags (`WS_EX_TRANSPARENT`,
  `WS_EX_TOOLWINDOW`, `WS_EX_NOACTIVATE`) to the root HWND via
  `GetAncestor(hwnd, GA_ROOT)`, and stop adding `WS_EX_LAYERED`
  ourselves. Verified via a Win32 API probe that the inner frame is no
  longer layered and the root keeps `alpha=235`, `topmost=True`.
- This was the bug all the way back from v0.3.0. Several "fixes" before
  this one (dropping `-transparentcolor`, un-withdrawing the root,
  adding the `lift()` dance) turned out to treat symptoms rather than
  the real cause.

## v0.3.3 · 2026-04-22 — Overlay & dialogs actually render now

### 🐛 Fixed
- **Real root cause of the invisible overlay and invisible "Open full
  history" window.** Since v0.3.0 the shared Tk interpreter root was
  `withdraw()`-n — totally hidden. On Windows, Toplevels of a withdrawn
  root (especially `overrideredirect` + layered `-alpha` windows) get
  their HWND created and Tk reports them as mapped, but DWM never
  actually composites them. The main window worked only because it
  was a simple non-layered non-overrideredirect Toplevel.
- Fix: stop withdrawing the root. Instead, size it 1 × 1, position it
  at `-32000, -32000` (far off any reasonable screen), set `-alpha 0.0`
  and `overrideredirect True`. It's invisible to the user but
  "visible" to Win32 so owned Toplevels render correctly.
- Overlay indicator is back. History window now opens where the user
  can actually see it.

### 🔭 Better diagnostics
- `show_history_window()` and the tray "History…" click now log
  breadcrumbs. Every history open path is traceable in `stt.log`.

## v0.3.2 · 2026-04-22 — Liquid-glass overlay, dialog focus, drawer drift

### 🎨 Design
- Overlay now has the Apple-ish liquid-glass look: a slightly-lighter
  pill on the dark canvas (so the pill shape is actually visible), a
  1-px top-half highlight for the glass shine, a soft outer edge ring,
  and a faint red aura around the pill rim while recording.
- Overlay alpha lowered to 0.92 for a touch of real transparency.
- Pill grown by a hair (210 × 36) to make room for the new layering.

### 🐛 Fixed
- **"Open full history …" button did nothing.** It was firing correctly,
  but the new history window opened *behind* the main window. Now
  `lift()` + `focus_force()` + a 300 ms topmost flash guarantee it
  surfaces. Same fix applied to the Set-Hotkey dialog.
- **Settings drawer drifting into view when you dragged the window.**
  The drawer's off-screen position was pinned to the initial window
  width; resizing wider revealed it. Now the drawer always repositions
  to the current right edge on every `<Configure>` event, whether it's
  open or closed.
- **Overlay topmost kept dropping** after the layered-window attribute
  was set while withdrawn. Now re-asserted on every show transition.

## v0.3.1 · 2026-04-22 — Overlay visible again

### 🐛 Fixed
- **Bottom-centre overlay indicator is back.** v0.3.0 converted the
  overlay from its own `Tk()` root to a `Toplevel` of the shared UI
  root — which silently broke `-transparentcolor` (Windows' transparent
  key attribute is only reliable on top-level `Tk()` windows, not on
  `Toplevel` children). On affected machines the pill rendered as a
  hot-pink rectangle that didn't look like an indicator at all; on
  others it just didn't show. Dropped the transparent-corner trick in
  favour of a solid dark background — the pill still looks clean, just
  with a tight rectangular boundary indistinguishable from the pill at
  96 % alpha.
- Overlay now logs every show/hide transition and its initial geometry
  to `stt.log` so future overlay bugs can be diagnosed from the log.

## v0.3.0 · 2026-04-21 — UI architecture rewrite (one Tk, one thread)

This release fixes the two bugs that plagued v0.2.x — "Show window" that
didn't open the window, and random crashes when opening the UI — by
rewriting the Tkinter architecture from the ground up.

### 🐛 Fixed
- **Show window reliably works.** The tray menu's "Show window" (and
  double-clicking the icon) now always restores the main window, no
  matter how it was hidden.
- **No more random UI crashes.** Opening history / update / what's-new
  dialogs while the main window is alive no longer destabilises Tk.

### ✨ Changed (under the hood)
- All Tk work now runs on a single dedicated UI thread owned by a new
  `UIManager`. Every window (overlay, main window, hotkey dialog,
  history, update, what's-new) is a `Toplevel` of the shared hidden
  root. Cross-thread work goes through a dispatch queue.
- Dropped the custom dark title bar on the main window — it was the
  source of the iconify / restore / taskbar-visibility flakiness. The
  main window now uses the native Windows title bar, darkened on
  Windows 11 via DWM. White on Windows 10 — acceptable tradeoff for
  everything actually working reliably.
- Removed ~150 lines of Win32 ctypes gymnastics (`SW_RESTORE`,
  `SW_MINIMIZE`, `WS_EX_APPWINDOW`, topmost-flash, `SetForegroundWindow`,
  manual maximize toggle, custom drag handlers) that existed only to
  paper over the multi-threaded Tk issues.
- `use_native_titlebar` config key removed (the native bar is always used now).

### 🐛 Also fixed along the way
- Preview pane's `tk.Text(pady=(6, 10))` was an invalid argument (tuples
  aren't accepted as internal padding on Text widgets). Previously
  silently swallowed by the broken threading; now a proper int.

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
