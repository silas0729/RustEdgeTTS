# Edge TTS Studio

A lightweight Rust desktop GUI for macOS. It fetches Microsoft Edge Read Aloud
voices, accepts multiline text, and writes synthesized speech to an MP3 file.
The interface defaults to Chinese, can switch to English, supports bilingual
voice search, one-click macOS voice previews, speech-rate and volume controls,
and keeps long documents inside a dedicated scrollable editor. It can also
import SRT (including `.str`-named files), WebVTT, ASS/SSA, and LRC subtitles and build an MP3 whose silence
and speech follow the authored cue timings.

The second workspace turns an MP3 back into SRT or WebVTT subtitles with a
local, quantized Whisper Large-v3 Turbo model. Chinese/English mixed recognition
is the default, with dedicated Chinese-only and English-only modes also
available. Inference runs through Rust/Candle with Metal acceleration on macOS;
it does not call Python, `whisper.cpp`, FFmpeg, or a cloud transcription API.

## Project structure

```text
RustEdgeTTS/
├── assets/
│   ├── AppIcon.svg
│   ├── AppIcon.icns
│   └── whisper-large-v3-turbo-config.json
├── Cargo.toml
├── Cargo.lock
├── README.md
├── examples/
│   ├── edge_smoke.rs
│   ├── asr_smoke.rs
│   └── subtitle_smoke.rs
└── src/
    ├── asr.rs                  # local Whisper inference and SRT/VTT export
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
cargo run --release --example asr_smoke
```

The live smoke tests write their MP3 results into the macOS temporary directory
and print the exact locations. `subtitle_smoke` also decodes the final file and
checks that its duration matches the eight-second test timeline.
`asr_smoke` consumes the MP3 produced by `edge_smoke` unless an MP3 path is
passed explicitly. Its first run downloads about 478 MB of model data from
Hugging Face into the user's Application Support cache; later runs reuse those
files and transcribe offline.

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

Local transcription reuses the same background channel architecture. MP3 is
decoded to mono PCM in Rust, resampled by the recognition pipeline, and sent to
the quantized multilingual Whisper model. Long audio is split at quiet points
into windows shorter than Whisper's internal 30-second boundary so a mistaken
no-speech decision cannot discard a complete block. Invalid padded tail
segments are ignored, while valid word timestamps are grouped at Chinese and
English sentence boundaries into readable subtitle cues. The small public model
configuration is embedded in the binary so a transient network failure after
the large weight download cannot invalidate the first run.

## Privacy and service note

This is an online TTS client. The entered text, chosen voice, and synthesis
settings are sent to Microsoft's Edge Read Aloud service. No API key is needed,
but this is an unofficial service endpoint and availability can change.

Audio-to-subtitle recognition is local after the one-time model download: MP3
content and generated subtitles are not uploaded. Whisper may still make
recognition errors or hallucinate text, so important subtitles should be
proofread before publication. Whisper Large-v3 Turbo and its model
configuration are MIT-licensed; model files are downloaded from their
Hugging Face repositories rather than redistributed inside this app.
