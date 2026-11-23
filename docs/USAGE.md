# MUR Operator Guide

This guide covers everything needed to set up, operate, and troubleshoot **MUR** during live performances.

---

## 1. Quick Start & Startup Modes

MUR builds as a single binary (`mur`) with no external runtime dependencies:

```sh
cargo run --release -- [OPTIONS]
```

### Command-Line Options

| Option | Argument | Behavior |
| --- | --- | --- |
| `--folder <PATH>` | Directory path | Scans `<PATH>` (non-recursively) for `.mov`, `.png`, `.jpg`, and `.jpeg` files sorted by filename |
| `--project <PATH>` | `.toml` or `.json` file | Applies project clips, layer settings, master effects, output display, and preferred MIDI port |
| `--state <PATH>` | `.json` file | Overrides the default auto-saved `state.json` location for both restore and auto-save |
| `--no-restore` | *(flag)* | Skips loading the previous session's `state.json` on startup |
| `-h`, `--help` | *(flag)* | Prints CLI usage |
| `-V`, `--version` | *(flag)* | Prints version information |

### Startup & Auto-Restore Order

1. Unless `--no-restore` is passed, MUR loads the persisted session snapshot from `--state` (or the OS default `state.json`).
2. If `--project` is supplied, its settings and clips are layered on top.
3. If `--folder` is supplied, supported media files in that directory are appended to the clip library.

> [!IMPORTANT]
> **Safe Crash/Restart Recovery**: Restored clips and layers always come back in the `STOPPED` state (with their restored opacity, blend mode, speed, playback mode, and playhead position ready). Live output never starts moving on its own after a restart until you trigger or unpause a layer.

---

## 2. Control UI & Media Loading

The control window (`MUR — Live VJ Control`, minimum `960×640`, default `1440×900`) is divided into four areas plus overlays:

### Status Bar (Top)

- **DISPLAY**: Select `Auto / None` or any detected monitor (`Name (W×H) [Primary]`), toggle `Output ON` / `Output OFF`, and monitor live status pills:
  - `● LIVE <monitor> <W>×<H>` (green) — Output active.
  - `WAITING <monitor>` (amber) — Output enabled, waiting for the selected display to reconnect.
  - `OFF` (gray) — Output disabled.
  - `OUTPUT ERROR` (red) — Hover for error details.
- **MIDI**: Select `Auto (LCXL3)` or a specific MIDI input port, click `Rescan` to re-enumerate immediately, and check the connection pill (`● <port>`, `Scanning…`, `Reconnecting <port>`, or `Disconnected`).
- **Performance & Tools (Right)**: Shows `drop: N` warning badge if frames missed the `16.67 ms` (`60 fps`) deadline, smoothed FPS and frame time (`ms`), plus `Logs` (`G`) and `Shortcuts (?)` (`H` or `?`) drawer toggles.

### Stage (Upper Center)

- **Live Master Mirror (`MASTER · 1920×1080`)**: Real-time GPU-shared 16:9 preview of the exact output frame (with zero CPU readback). Displays a `BLACKOUT` badge when blackout is active or a bottom banner when waiting on a monitor / reporting an output error.
- **Master FX Strip**:
  - Toggle pills and sliders (`0.00..=1.00`) for `RGB SHIFT`, `GLITCH`, and `STROBE`.
  - `Speed` (`1.00..=30.00 Hz`) for strobe frequency.
  - Color grading sliders: `Brightness` (`-1.00..=1.00`), `Contrast` (`0.00..=2.00`), `Saturation` (`0.00..=2.00`).
  - `DIMMER` (`0.00..=1.00`) overall master output gain.
  - `Reset FX` restores all effects and color grading to neutral (`Brightness = 0.0`, `Contrast = 1.0`, `Saturation = 1.0`, `Speed = 8.0 Hz`) while **preserving `DIMMER` and `BLACKOUT`** so a reset mid-set never causes a sudden flash or silence.
  - `BLACKOUT` toggles immediate solid black output without altering other sliders.

### 4-Layer Mixer Rack (Middle)

Four side-by-side layer strips (`L1` through `L4`, composited bottom-to-top from `L1` to `L4`):

- Click the `L1`–`L4` badge to focus a layer for keyboard commands and clip-grid selection (`FOCUS` indicator).
- Transport controls: `Play`/`Pause`, `Retrigger`, `Loop`/`1-Shot`, and `Clear`.
- Mix controls: `Opacity` (`0.00..=1.00`), `Speed` (`-2.00x..=4.00x` with a `1x` reset button), and `Blend` (`Normal`, `Add`, `Multiply`, `Screen`, `Difference`).
- Frame progress bar (`current / total`), clip metadata (`W×H @ fps`), format badge, `Next ▸ <clip>` indicator when a different clip is selected, and an inline red error box if a clip or I/O failure isolates that layer.

### Media Library & Clip Grid (Bottom)

- **Loading Clips**:
  - Click `+ Files` to pick `.mov`, `.png`, `.jpg`, `.jpeg`, `.toml`, or `.json` files.
  - Click `+ Folder` to scan a directory for supported media.
  - **Drag & drop** media files, folders, or `.toml`/`.json` project files anywhere onto the control window.
- **Clip Tiles**:
  - Top-left badge identifies the clip type (`HAP`, `HAP A`, `HAP Q`, or `IMG`); the pixel/compression format (`DXT1`, `DXT5`, `YCoCg`, `RGBA`) appears in the hover tooltip and layer strip.
  - Top-right cyan pill lists any layers currently bound to that clip (including paused or restored stopped layers) (e.g. `L1 L3`).
  - **Single-click** thumbnail: selects the clip as `Next ▸` on the currently focused layer.
  - **Double-click** thumbnail: fires the clip immediately on the currently focused layer.
  - **`1` `2` `3` `4` buttons**: fires the clip directly on Layer 1–4.
  - **`×` button**: unloads an unused clip in one click. If the clip is currently live on any layer, clicking `×` arms a 3-second confirmation (`×?` in amber); click `×?` again within 3 seconds to unload the clip and clear those layers.

### Non-Blocking Overlays

- **Toasts** (bottom-right): Up to 8 transient notifications (`INFO` 3.5s, `WARN` 6s, `ERROR` 10s) with `Clear notifications`.
- **Log Drawer** (`G`): Floating window showing up to 200 recent in-memory log entries, newest first.
- **Shortcuts Overlay** (`H` or `?`): Centered keyboard reference table.
- **Exit Confirmation Guard**: Modal dialog (`Exit MUR?` / `Live VJ output will stop.`) triggered by `Cmd+Q`, `Ctrl+Q`, `Alt+F4`, the window close button, or macOS Dock/menu Quit. All live controls behind it are disabled and **`Enter` is intentionally swallowed** so live clip triggering cannot accidentally exit; confirm by clicking **Quit MUR** or repeating a quit shortcut (`Cmd + Q` / `Ctrl + Q` / `Alt + F4`), or cancel with `Escape` / **Cancel (Esc)**.

---

## 3. Dual-Display Output Setup (macOS & Windows)

MUR renders internally at **1920×1080 @ 60 fps** and scales the master output aspect-fit (with black letterbox/pillarbox bars if needed) to the selected output window.

### Single-Display Safety

- On first launch with only one monitor connected, MUR leaves output **disabled** (`Auto / None`) so the borderless output window never covers the control UI.
- When an external monitor is connected at first launch, `Auto / None` selects the first non-primary display and enables output automatically.
- If you explicitly choose the same monitor that holds the control window, MUR intentionally keeps the output window at normal window level instead of elevating it above all other windows.

### macOS Behavior

- **Space-Free Overlay**: Instead of macOS native fullscreen Spaces (which animate slowly and can blank secondary displays), MUR places a borderless `NSWindow` configured with `CanJoinAllSpaces | Stationary | IgnoresCycle | FullScreenAuxiliary` at level `NSMainMenuWindowLevel + 1` and `hidesOnDeactivate = false`. It covers the Menu Bar and Dock of the target display, handles Retina HiDPI scaling cleanly, and stays live while you interact with the control window.
- **Quit Guard**: `Cmd+Q`, the app menu `Quit` item, and Dock `Quit` are intercepted via `NSApplication` menu retargeting and `applicationShouldTerminate:` so accidental quits—and OS logout/shutdown attempts while a live set is running—bring up the exit confirmation dialog instead of terminating immediately.

### Windows Behavior

- **DPI-Aware Fullscreen**: The process runs per-monitor DPI-aware (v2) and places a borderless `Fullscreen::Borderless` window verified against `GetMonitorInfoW` physical bounds (`AlwaysOnTop` when on a separate monitor from the control UI).
- **Quit Guard**: `Alt+F4` and the window close button route through `CloseRequested` to the exit confirmation dialog.

### Monitor Hot-Unplug Auto-Recovery

- MUR polls connected displays every `750 ms`.
- If the active output cable (HDMI / USB-C / DisplayPort) is unplugged mid-show, MUR safely closes the output window without crashing and transitions to `WaitingForMonitor`.
- When the display reconnects (matching by stable monitor ID, name, or desktop position), MUR automatically recreates and places the output window and resumes presenting.

---

## 4. Novation Launch Control XL 3 Mapping (MIDI Channel 1)

MUR uses a fixed **Custom Mode** template on **MIDI Channel 1**. Messages on Channels 2–16, Note Off, and Note On with velocity `0` are ignored.

When set to `Auto (LCXL3)`, auto-detection matches Launch Control / Novation port hints, skips Launch Control XL 3 `"DAW"` integration ports (word-by-word), prefers physical `"MIDI"` ports over virtual loopbacks, and **does not auto-connect to unknown third-party MIDI gear**. To use another controller, select its port explicitly from the top-bar `MIDI` dropdown.

### Hardware Template Requirements

Configure a Custom Mode slot in **Novation Components** with:
- **MIDI Channel**: `1` for every fader, encoder, and button
- **Encoders**: `Absolute` (`0–127`)
- **Buttons**: `Momentary` (`Note On` on press, `Note Off` on release; MUR manages Toggle vs Trigger behavior internally)
- **LED Colors**: Configured statically in Novation Components (MUR does not send MIDI Out LED feedback)

### Left 4 Strips (Columns 1–4 → Layers 1–4)

| Control | Message | Number | Function | Value / Behavior |
| --- | --- | --- | --- | --- |
| Faders 1–4 | CC | `1`–`4` | Layer 1–4 Opacity | `0–127` → `0.0–1.0` |
| Encoders Row 1 (1–4) | CC | `11`–`14` | Layer 1–4 Playback Speed | `0` = `-2.0x`, `64` = `+1.0x`, `127` = `+4.0x` |
| Encoders Row 2 (9–12) | CC | `21`–`24` | Layer 1–4 Blend Mode | `0–25` Normal · `26–51` Add · `52–76` Multiply · `77–102` Screen · `103–127` Difference |
| Encoders Row 3 (17–20) | CC | `31`–`34` | Layer 1–4 Clip Selection | Spread evenly across `≤128` clips; direct index `0–127` when `>128` clips |
| Buttons Row 1 (1–4) | Note | `36`–`39` | Layer 1–4 Trigger / Retrigger | Momentary: fires selected clip (or restarts active clip from start/end) |
| Buttons Row 2 (9–12) | Note | `44`–`47` | Layer 1–4 Pause / Resume | Toggle per press |

### Right 4 Strips (Columns 5–8 → Master Effects & Output)

| Control | Message | Number | Function | Value / Behavior |
| --- | --- | --- | --- | --- |
| Fader 5 | CC | `5` | RGB Shift Intensity | `0–127` → `0.0–1.0` |
| Fader 6 | CC | `6` | Glitch Intensity | `0–127` → `0.0–1.0` |
| Fader 7 | CC | `7` | Strobe Intensity | `0–127` → `0.0–1.0` |
| Fader 8 | CC | `8` | Master Dimmer | `0–127` → `0.0–1.0` |
| Encoder Row 1 (5) | CC | `15` | Brightness | `0` = `-1.0`, `64` = `0.0` (neutral), `127` = `+1.0` |
| Encoder Row 1 (6) | CC | `16` | Contrast | `0` = `0.0`, `64` = `1.0` (neutral), `127` = `2.0` |
| Encoder Row 1 (7) | CC | `17` | Saturation | `0` = `0.0`, `64` = `1.0` (neutral), `127` = `2.0` |
| Encoder Row 1 (8) | CC | `18` | Strobe Speed | `0–127` → `1.0–30.0 Hz` (exponential curve) |
| Encoders Row 2–3 (13–16, 21–24) | CC | `25`–`28`, `35`–`38` | Reserved | Unassigned |
| Button Row 1 (5) | Note | `40` | RGB Shift ON / OFF | Toggle per press |
| Button Row 1 (6) | Note | `41` | Glitch ON / OFF | Toggle per press |
| Button Row 1 (7) | Note | `42` | Strobe ON / OFF | Toggle per press |
| Button Row 1 (8) | Note | `43` | Master Blackout | Toggle per press |
| Button Row 2 (13) | Note | `48` | Reset All Effects | Restores effects & color grading to neutral (keeps dimmer & blackout) |
| Button Row 2 (14) | Note | `49` | Retrigger All Active Layers | Restarts every layer that currently has an active clip |
| Button Row 2 (15) | Note | `50` | Toggle Focused Layer Playback Mode | Switches the currently focused layer between `Loop` and `OneShot` |
| Button Row 2 (16) | Note | `51` | Reset All Layer Speeds | Sets all 4 layers back to `1.0x` |

> [!NOTE]
> Notes `48`–`51` finalize the reserved Row 2 buttons in [`SPEC.md`](SPEC.md) §6.3.

---

## 5. Keyboard Fallback Reference

Every MIDI-controlled function can also be performed from the keyboard (matching `SHORTCUT_LEGEND` in [`src/ui/shortcuts.rs`](../src/ui/shortcuts.rs)). Continuous adjustments support key repeat; triggers, toggles, and exit actions do not repeat while held.

| Keys | Action |
| --- | --- |
| `1 / 2 / 3 / 4` | Focus layer 1-4 |
| `Shift + 1-4` | Trigger layer 1-4 (selected clip) |
| `Alt + 1-4` | Pause / resume layer 1-4 |
| `Tab / Shift + Tab` | Focus next / previous layer |
| `R` | Retrigger focused layer |
| `Shift + R` | Retrigger all active layers |
| `Space` | Pause / resume focused layer |
| `L` | Toggle loop / one-shot on focused layer |
| `X` | Clear focused layer |
| `Q / W` | Layer opacity - / + (`±0.05`) |
| `A / D` | Layer speed - / + (`±0.1`) |
| `S` | Reset focused layer speed to 1x |
| `Shift + S` | Reset every layer speed to 1x |
| `B / Shift + B` | Next / previous blend mode |
| `[ / ]` | Previous / next clip in the grid |
| `Enter` | Trigger selected clip on focused layer |
| `F1` | Toggle RGB shift |
| `Shift + F1 / Alt + F1` | RGB shift intensity + / - (`±0.1`) |
| `F2` | Toggle glitch |
| `Shift + F2 / Alt + F2` | Glitch intensity + / - (`±0.1`) |
| `F3` | Toggle strobe |
| `Shift + F3 / Alt + F3` | Strobe intensity + / - (`±0.1`) |
| `F4 or \`` | Toggle master blackout |
| `F5 / F6` | Master dimmer - / + (`±0.05`) |
| `F7 / Shift + F7` | Brightness + / - (`±0.05`) |
| `F8 / Shift + F8` | Contrast + / - (`±0.05`) |
| `F9 / Shift + F9` | Saturation + / - (`±0.05`) |
| `F10 / Shift + F10` | Strobe speed + / - (`±2.0 Hz`) |
| `F12 (Ctrl/Cmd + F12 reserved)` | Reset all effects |
| `Ctrl + 1 / 2 / 3 / 4` | Toggle RGB shift / glitch / strobe / blackout |
| `Ctrl + 5-9, Ctrl + 0` | Dimmer (`5`), brightness (`6`), contrast (`7`), saturation (`8`), strobe speed (`9`), reset (`0`) (`Shift` = `-`) |
| `Ctrl + O or Cmd + O` | Toggle output on / off |
| `H or ?` | Show / hide this shortcut overlay |
| `G` | Show / hide the log drawer |
| `Escape` | Close overlays |
| `Cmd + Q / Ctrl + Q / Alt + F4` | Request exit (confirmation guard) |
| `Repeat quit key / Escape (in dialog)` | Confirm / cancel exit |

> [!TIP]
> On macOS laptops where `F1`–`F12` default to system brightness/media keys, press `fn + F1..F12` (or enable *"Use F1, F2, etc. keys as standard function keys"* in **System Settings → Keyboard → Keyboard Shortcuts → Function Keys**), or use `Ctrl + 1..0` (`Ctrl + Shift + 5..9` to decrease) and `` ` `` (backtick) for Blackout without needing `fn`.
>
> **Crash-Recovered Sessions**: Restored layers start in `Stopped` state with their `playhead_frame` preserved so live output starts dark; press `Space`, `Alt + 1-4`, MIDI Notes `44–47`, or the layer's `Play` button to resume from the preserved `playhead_frame`, or `Shift + R` (MIDI Note `49`) to retrigger all active layers together from their start frame.

---

## 6. Project File Format (TOML & JSON)

Project files can be passed via `--project <PATH>` or dragged onto the control window. All relative paths (`clip_folder`, `clips`, and `layers[*].clip_path`) are resolved relative to the **project file's parent directory**. If `clips` is non-empty, MUR loads those exact clips in grid order instead of scanning `clip_folder`.

All top-level, `[effects]`, `[output]`, and `[[layers]]` fields are optional (`#[serde(default)]`) and fall back to neutral defaults when omitted. In `[[layers]]`, omitting `clip_path` updates that layer's mix/playback parameters (`playback_mode`, `speed`, `opacity`, `blend_mode`) while **preserving any clip currently active on that layer**. `output.monitor_name` matches either a plain display name (e.g., `"DELL U2720Q"`) or a full monitor ID (`name|WxH|x,y` or `name|WxH|x,y|cg:<id>` on macOS).

### Schema & Clamped Ranges

| Section / Field | Type | Default | Valid / Clamped Range |
| --- | --- | --- | --- |
| `clip_folder` | `PathBuf?` | `None` | Directory path (ignored when `clips` is non-empty) |
| `clips` | `Vec<PathBuf>` | `[]` | Explicit clip paths in grid order |
| `preferred_midi_port` | `String?` | `None` | Exact or substring port name (`None` = `Auto (LCXL3)`) |
| `output.monitor_name` | `String?` | `None` | Display name or full monitor ID (`name\|WxH\|x,y[\|cg:<id>]`) |
| `output.enabled` | `bool` | `false` | Enables or disables the external output window |
| `effects.rgb_shift_enabled` / `glitch_enabled` / `strobe_enabled` / `blackout` | `bool` | `false` | Effect / blackout enable toggles |
| `effects.rgb_shift_intensity` / `glitch_intensity` / `strobe_intensity` | `f32` | `0.0` | Clamped to `0.0..=1.0` |
| `effects.strobe_speed_hz` | `f32` | `8.0` | Clamped to `1.0..=30.0` |
| `effects.brightness` | `f32` | `0.0` | Clamped to `-1.0..=1.0` |
| `effects.contrast` / `effects.saturation` | `f32` | `1.0` | Clamped to `0.0..=2.0` |
| `effects.master_dimmer` | `f32` | `1.0` | Clamped to `0.0..=1.0` |
| `layers[*].clip_path` | `PathBuf?` | `None` | Omitting keeps the layer's active clip intact |
| `layers[*].playback_mode` | `String` | `"Loop"` | `"Loop"` \| `"OneShot"` |
| `layers[*].speed` | `f32` | `1.0` | Clamped to `-2.0..=4.0` |
| `layers[*].opacity` | `f32` | `1.0` | Clamped to `0.0..=1.0` |
| `layers[*].blend_mode` | `String` | `"Normal"` | `"Normal"` \| `"Add"` \| `"Multiply"` \| `"Screen"` \| `"Difference"` |
| `layers[*].playhead_frame` | `f64` | `0.0` | Non-negative frame offset (`>= 0.0`, finite) |

### TOML Example (`show.toml`)

```toml
# Used when `clips` is omitted or empty
clip_folder = "media"

# Optional explicit clip list (overrides `clip_folder` scan)
clips = [
  "media/01_intro.mov",
  "media/02_grid_alpha.mov",
  "media/logo.png",
]

preferred_midi_port = "Launch Control XL 3"

[output]
monitor_name = "DELL U2720Q"
enabled = true

[effects]
rgb_shift_enabled = false
rgb_shift_intensity = 0.0
glitch_enabled = false
glitch_intensity = 0.0
strobe_enabled = false
strobe_intensity = 0.0
strobe_speed_hz = 8.0
brightness = 0.0
contrast = 1.0
saturation = 1.0
master_dimmer = 1.0
blackout = false

# Up to 4 entries (Layers 1..4). Layers bind in Stopped state on load.
[[layers]]
clip_path = "media/01_intro.mov"
playback_mode = "Loop" # "Loop" | "OneShot"
speed = 1.0
opacity = 1.0
blend_mode = "Normal" # "Normal" | "Add" | "Multiply" | "Screen" | "Difference"
playhead_frame = 0.0

[[layers]]
clip_path = "media/02_grid_alpha.mov"
playback_mode = "OneShot"
speed = 1.0
opacity = 0.85
blend_mode = "Screen"
playhead_frame = 0.0
```

### JSON Example (`show.json`)

```json
{
  "clip_folder": "media",
  "clips": [
    "media/01_intro.mov",
    "media/02_grid_alpha.mov",
    "media/logo.png"
  ],
  "preferred_midi_port": "Launch Control XL 3",
  "output": {
    "monitor_name": "DELL U2720Q",
    "enabled": true
  },
  "effects": {
    "rgb_shift_enabled": false,
    "rgb_shift_intensity": 0.0,
    "glitch_enabled": false,
    "glitch_intensity": 0.0,
    "strobe_enabled": false,
    "strobe_intensity": 0.0,
    "strobe_speed_hz": 8.0,
    "brightness": 0.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "master_dimmer": 1.0,
    "blackout": false
  },
  "layers": [
    {
      "clip_path": "media/01_intro.mov",
      "playback_mode": "Loop",
      "speed": 1.0,
      "opacity": 1.0,
      "blend_mode": "Normal",
      "playhead_frame": 0.0
    },
    {
      "clip_path": "media/02_grid_alpha.mov",
      "playback_mode": "OneShot",
      "speed": 1.0,
      "opacity": 0.85,
      "blend_mode": "Screen",
      "playhead_frame": 0.0
    }
  ]
}
```

---

## 7. Media Encoding Guide (FFmpeg)

MUR expects pre-encoded HAP `.mov` clips (no H.264/H.265/ProRes decoding at runtime) or PNG/JPEG stills.

### Recommended FFmpeg Commands

Use `-chunks 4` (or `8` for 4K sources) with Snappy compression (FFmpeg's default compressor for HAP) and strip audio (`-an`) since MUR ignores non-video tracks:

- **HAP (`Hap` / BC1 / DXT1 — Opaque, smallest & fastest)**:

  ```sh
  ffmpeg -i source.mp4 -c:v hap -format hap -chunks 4 -an clip_hap.mov
  ```

- **HAP Alpha (`HapAlpha` / BC3 / DXT5 — Transparent overlays)**:

  ```sh
  ffmpeg -i source_with_alpha.mov -c:v hap -format hap_alpha -chunks 4 -an clip_alpha.mov
  ```

- **HAP Q (`HapQ` / DXT5-YCoCg — Highest color quality, opaque)**:

  ```sh
  ffmpeg -i source.mp4 -c:v hap -format hap_q -chunks 4 -an clip_hapq.mov
  ```

### Aspect-Ratio Fit & Margin Behavior

MUR scales each layer to fit inside the `1920×1080` canvas while preserving its native aspect ratio. How the letterbox/pillarbox margin behaves depends on whether the clip carries alpha:

- **Opaque clips** (`Hap`, `HapQ`, and opaque PNG/JPEG): Margins outside the fitted rectangle are filled with **opaque black** (`alpha = 1.0` scaled by layer opacity).
- **Alpha clips** (`HapAlpha` and PNGs with any pixel `alpha < 255`): Margins outside the fitted rectangle remain **transparent** (`alpha = 0.0`), so non-16:9 overlays, square logos, or vertical graphics never cover lower layers with black bars.
- **Blend modes do not apply to margins**: The margin strip is always composited as `Normal`, so the same 4:3 opaque clip paints black bars whether its layer uses `Normal`, `Add`, `Multiply`, `Screen`, or `Difference`. Only the fitted picture area uses the layer's blend mode.

---

## 8. Troubleshooting & Logs

### Log & State File Locations

`AppPaths::discover()` resolves OS-specific directories via `directories::ProjectDirs::from("io", "potistudio", "MUR")`:

- **macOS**:
  - Data / State: `~/Library/Application Support/io.potistudio.MUR/state.json`
  - Logs: `~/Library/Application Support/io.potistudio.MUR/logs/mur.log.YYYY-MM-DD`
- **Windows**:
  - Data / State: `%APPDATA%\potistudio\MUR\data\state.json`
  - Logs: `%APPDATA%\potistudio\MUR\data\logs\mur.log.YYYY-MM-DD`
- **Fallback** (if no home directory is reported): `<OS temp dir>/mur/state.json` and `<OS temp dir>/mur/logs/`.

Logs rotate daily (`mur.log.YYYY-MM-DD`) and retain up to **14 files**. Default verbosity is `info`; set `RUST_LOG=debug` (or `RUST_LOG=mur=debug`) before launch for verbose diagnostics.

### Common Scenarios

- **MIDI Controller Not Responding**:
  - Verify the controller is in the **Custom Mode** slot configured for **MIDI Channel 1**, `Absolute` encoders, and `Momentary` buttons.
  - Check the MIDI status pill in the top bar. If another app holds the port exclusively (common on Windows), close that app and click `Rescan`.
  - If an explicit port was saved or set via `preferred_midi_port`, MUR waits for that exact device (`Reconnecting`) instead of switching to a virtual/loopback port. Switch the dropdown back to `Auto (LCXL3)` to auto-detect any Launch Control port.
- **External Monitor Unplugged or Blank**:
  - Check the `DISPLAY` status pill. If it shows `WAITING <monitor>`, reconnect the cable or select another display from the dropdown.
  - If you only have one display connected, MUR keeps output disabled by default; manually select your monitor and click `Output OFF` → `Output ON` to test on a single screen.
- **Clip Shows `Error` in the Grid or Stops a Layer**:
  - Hover over the clip tile or check the `Logs` drawer (`G`) for the exact reason (e.g., unsupported codec such as H.264 or `HapM` / HAP Q Alpha, corrupt `moov` atom, dimensions exceeding `8192×8192`, or file deleted/moved during playback). Other clips and layers keep running unaffected.
- **GPU Device Lost / Driver Reset**:
  - MUR catches `DeviceLost` / `SurfaceLost` errors automatically, drops stale GPU surfaces, and attempts to rebuild the `GpuStack` every 2 seconds while keeping the session state intact.
