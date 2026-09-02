# Edge TTS Studio

A lightweight Rust desktop GUI for macOS. It fetches Microsoft Edge Read Aloud
voices, accepts multiline text, and writes synthesized speech to an MP3 file.
The interface defaults to Chinese, can switch to English, supports bilingual
voice search, one-click macOS voice previews, speech-rate and volume controls,
and keeps long documents inside a dedicated scrollable editor. It can also
import SRT (including `.str`-named files), WebVTT, ASS/SSA, and LRC subtitles and build an MP3 whose silence
and speech follow the authored cue timings.

## Project structure

```text
RustEdgeTTS/
├── assets/
│   ├── AppIcon.svg
│   └── AppIcon.icns
├── Cargo.toml
├── Cargo.lock
├── README.md
├── examples/
│   ├── edge_smoke.rs
│   └── subtitle_smoke.rs
└── src/
    ├── main.rs                 # egui UI and Tokio channel worker
    ├── subtitle_pipeline.rs    # asynchronous per-cue TTS orchestration
    ├── subtitles.rs            # SRT/VTT/ASS/SSA/LRC and text encodings
    ├── system_proxy.rs
    └── timeline_audio.rs       # streaming silence/timeline MP3 assembly
```

## Run on macOS

Install Rust 1.95 or newer from <https://rustup.rs>, then run:

```bash
cargo run --release
```

If your network cannot reach crates.io directly, use a mirror only for that
command (this does not modify the project configuration):

```bash
cargo \
  --config 'source.crates-io.replace-with="rsproxy-sparse"' \
  --config 'source.rsproxy-sparse.registry="sparse+https://rsproxy.cn/index/"' \
  run --release
```

The first build downloads and compiles Rust dependencies. Both Apple Silicon
(`aarch64-apple-darwin`) and Intel (`x86_64-apple-darwin`) are supported by the
selected dependencies. The app uses macOS's native save dialog and loads a
system CJK font fallback so Chinese input renders correctly. It also reads the
macOS system HTTPS proxy at startup when no proxy environment variable was
provided, which is important for apps launched directly from Finder.

## Verification

Compile, run unit tests, and perform a real Edge TTS network/synthesis check:

```bash
cargo check
cargo test
cargo run --example edge_smoke
cargo run --example subtitle_smoke
```

The live smoke tests write their MP3 results into the macOS temporary directory
and print the exact locations. `subtitle_smoke` also decodes the final file and
checks that its duration matches the eight-second test timeline.

## Optional `.app` bundle

```bash
cargo install cargo-bundle
cargo bundle --release
open "target/release/bundle/osx/Edge TTS Studio.app"
```

`cargo-bundle` is only packaging tooling; it is not needed for development.

## Architecture

The main thread owns `eframe`/`egui`; all voice discovery, synthesis, cache I/O,
and MP3 writing run in the background. A dedicated OS thread owns a two-thread
Tokio runtime and a reusable `EdgeTtsClient`. Two `tokio::sync::mpsc` channels
carry commands to that worker and results back to the UI. The UI uses
`try_recv`, so neither voice discovery nor synthesis can block rendering.

Subtitle files are decoded as UTF-8, UTF-16, or legacy Chinese GBK. Each cue is
synthesized separately. The pipeline streams zero-valued PCM through silent
gaps and encodes the combined 24 kHz mono timeline with the bundled pure-Rust
MP3 codec, so the app does not require `ffmpeg` or `lame`. If a spoken cue is
longer than its time slot, the app first requests a faster version of that cue;
only a still-overlong result is faded and clipped to protect later timestamps.

The voice catalogue is cached as JSON under the user's macOS cache directory.
If a later refresh fails, the app can still show the last successful list.

## Privacy and service note

This is an online TTS client. The entered text, chosen voice, and synthesis
settings are sent to Microsoft's Edge Read Aloud service. No API key is needed,
but this is an unofficial service endpoint and availability can change.
