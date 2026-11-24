# MUR

Live performance VJ software written in pure Rust for macOS and Windows.

MUR combines zero-dependency QuickTime HAP playback, a 4-layer GPU compositor, real-time master shader effects, fixed-template MIDI hardware control, and stage-hardened error isolation to sustain **1920×1080 @ 60 fps** output on an external display.

## Key Features

- **Pure-Rust Media Playback**: Native QuickTime (`.mov`) parser and HAP (`Hap` / `Hap Alpha` / `Hap Q`) + PNG/JPEG decoder with Snappy decompression—no external FFmpeg binaries or shared libraries required at runtime.
- **4-Layer Real-Time Compositor**: Independent `Loop` and `OneShot` playback per layer, variable speed (`-2.0x` reverse to `+4.0x` forward), per-layer opacity (`0.0..=1.0`), and 5 GPU blend modes (`Normal`, `Add`, `Multiply`, `Screen`, `Difference`) with aspect-ratio-preserving fit.
- **Master Shader Effects & Output Safety**: Single-pass GPU `RGB Shift`, `Glitch`, `Strobe` (`1.0..=30.0 Hz`), color grading (`Brightness`, `Contrast`, `Saturation`), `Master Dimmer`, and instant `Blackout`.
- **Fixed-Template MIDI + Full Fallback**: Plug-and-play **Novation Launch Control XL 3** mapping on MIDI Channel 1 with background hot-plug auto-reconnect, paired with complete mouse and keyboard control for every function.
- **Dual-Window Live Setup**: Operator booth UI (`egui` + `wgpu`) with a live 16:9 mirror preview alongside an independent borderless external output window (`1920×1080` internal render target scaled to any display, Space-free overlay over macOS Menu Bar & Dock, DPI-aware borderless fullscreen on Windows).
- **Stage-Ready Resilience**: Per-layer error isolation, automatic monitor hot-unplug recovery (`WaitingForMonitor`), GPU device-loss rebuild, panic-free Clippy deny rules, `catch_unwind` worker supervision, atomic debounced auto-save/restore (`state.json`), and an exit confirmation guard (`Cmd+Q`, `Ctrl+Q`, `Alt+F4`, Dock Quit, close button, and macOS OS logout/shutdown via `applicationShouldTerminate:`; `Enter` is swallowed—confirm by repeating the quit shortcut or clicking **Quit MUR**, or cancel with `Escape`).

## Prerequisites & Build

- **Rust**: `1.100.0` or later (Edition 2024; see [`rust-toolchain.toml`](rust-toolchain.toml))
- **OS**: macOS (Metal) or Windows (DX12 / Vulkan) with a `wgpu`-capable GPU

```sh
git clone https://github.com/potistudio/MUR.git
cd MUR
cargo build --release
```

To run the optimized release build directly:

```sh
cargo run --release
```

## CLI Usage

```sh
mur [OPTIONS]
```

| Flag | Description |
| --- | --- |
| `--folder <PATH>` | Scan a folder for supported media clips (`.mov`, `.png`, `.jpg`, `.jpeg`) at startup |
| `--project <PATH>` | Load a project configuration file (`.toml` or `.json`) at startup |
| `--state <PATH>` | Use an explicit session state JSON file instead of the OS default |
| `--no-restore` | Start with a clean session instead of restoring the previous auto-saved state |
| `-h`, `--help` | Print help information |
| `-V`, `--version` | Print version information |

Examples:

```sh
# Restore previous session (default)
cargo run --release

# Start clean and load a show folder
cargo run --release -- --no-restore --folder ./media

# Launch a TOML project file
cargo run --release -- --project ./shows/club-set.toml
```

## Supported Media Formats

| Format | Container / Extension | Codec / Pixel Format | Alpha Support | Notes |
| --- | --- | --- | --- | --- |
| **HAP** | `.mov` (`Hap1`) | DXT1 / BC1 | No (opaque black margins) | Fastest GPU upload; recommended for opaque backgrounds |
| **HAP Alpha** | `.mov` (`Hap5`) | DXT5 / BC3 | Yes (transparent margins) | Recommended for overlays and masked visuals |
| **HAP Q** | `.mov` (`HapY`) | DXT5-YCoCg | No (opaque black margins) | High color fidelity; YCoCg → RGB decoded in GPU shader |
| **Still Images** | `.png`, `.jpg`, `.jpeg` | RGBA8 | Yes (PNG with alpha) | Resident in memory; aspect-fit margins match alpha presence |

> [!NOTE]
> Non-video tracks (audio, timecode) in `.mov` files are ignored. Maximum frame dimension is `8192×8192`. `Hap Q Alpha` (`HapM`) and `Hap Alpha-only` (`HapA`) are not supported.

### Preparing Clips with FFmpeg

Convert standard video files into HAP `.mov` clips ahead of your set using FFmpeg (`libsnappy` + chunked encoding recommended for multi-threaded/fast seeking):

```sh
# Opaque HAP (BC1 / Hap1)
ffmpeg -i input.mp4 -c:v hap -format hap -chunks 4 -an output_hap.mov

# HAP Alpha with transparency (BC3 / Hap5)
ffmpeg -i input_alpha.mov -c:v hap -format hap_alpha -chunks 4 -an output_alpha.mov

# High-quality HAP Q (YCoCg / HapY)
ffmpeg -i input.mp4 -c:v hap -format hap_q -chunks 4 -an output_hapq.mov
```

## Documentation

- [Operator Guide (USAGE.md)](docs/USAGE.md) — Startup modes, dual-display setup, Novation Launch Control XL 3 MIDI tables, keyboard shortcuts, TOML/JSON project schema, FFmpeg encoding tips, and troubleshooting.
- [Developer Architecture (ARCHITECTURE.md)](docs/ARCHITECTURE.md) — System topology, module map, pure-Rust `.mov`/HAP streaming pipeline, `wgpu` rendering passes, and platform/resilience guarantees.
- [Specification (SPEC.md)](docs/SPEC.md) — Original product and engineering specification (Japanese).

## License

This project is licensed under the [MIT License](LICENSE).
