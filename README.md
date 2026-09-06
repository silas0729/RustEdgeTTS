# Edge TTS Studio

A lightweight Rust desktop GUI for macOS and Windows. It combines Microsoft Edge Read Aloud
voices with an optional, fully local Qwen3-TTS engine, accepts multiline text, and writes
synthesized speech to an MP3 file.
The interface defaults to Chinese, can switch to English, supports bilingual
voice search, one-click voice previews, speech-rate and volume controls,
and keeps long documents inside a dedicated scrollable editor. It can also
import SRT (including `.str`-named files), WebVTT, ASS/SSA, and LRC subtitles and build an MP3 whose silence
and speech follow the authored cue timings.

The Qwen3 local mode lets the user explicitly select either the official
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` lightweight checkpoint or the
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` higher-quality checkpoint. Both expose
the same nine preset voices and support Chinese, English, mixed Chinese/English
text, Japanese, and Korean voice presets. The first generation or preview
downloads about 2.4 GB for 0.6B or 4.5 GB for 1.7B. Each main model has its own
cache, while the shared 12Hz decoder and tokenizer are hard-linked on supported
filesystems to avoid wasting roughly another 660 MB. Only the selected version
is held in memory. No Python, PyTorch, ONNX runtime, or external audio encoder
is used.

Automatic downloading is optional. After selecting Qwen 0.6B or Qwen 1.7B,
click **Offline model** to import a complete folder downloaded from the official
[0.6B model repository](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
or [1.7B model repository](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).
The app validates that the folder is the selected CustomVoice version and
requires `model.safetensors`, `config.json`,
`speech_tokenizer/model.safetensors`, plus either `tokenizer.json` or both
`vocab.json` and `merges.txt`. Same-volume files are hard-linked into the app
cache; cross-volume files are copied with progress on the background worker.
If automatic downloading fails, the app opens this download/import guide
automatically and also offers the corresponding official ModelScope page for
users in mainland China.

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
│   ├── qwen_smoke.rs
│   ├── asr_smoke.rs
│   └── subtitle_smoke.rs
└── src/
    ├── asr.rs                  # local Whisper inference and SRT/VTT export
    ├── main.rs                 # egui UI and Tokio channel worker
    ├── qwen_local.rs           # local Qwen3-TTS model and language detection
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
cargo run --release --example qwen_smoke -- 0.6
cargo run --release --example qwen_smoke -- 1.7
# Optionally verify importing an already downloaded folder:
cargo run --release --example qwen_smoke -- 1.7 /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

The live smoke tests write their MP3 results into the macOS temporary directory
and print the exact locations. `subtitle_smoke` also decodes the final file and
checks that its duration matches the eight-second test timeline.
`asr_smoke` consumes the MP3 produced by `edge_smoke` unless an MP3 path is
passed explicitly. Its first run downloads about 478 MB of model data from
Hugging Face into the user's Application Support cache; later runs reuse those
files and transcribe offline.

Qwen3-TTS is substantially larger than the other dependencies, so the normal
test suite validates model-version routing, mixed-language detection, long-text
segmentation, PCM speed/volume processing, and MP3 encoding without downloading
both models. A real first-run synthesis is initiated from the app and requires
enough free disk space and memory for the selected official checkpoint. The
1.7B version uses materially more unified memory than 0.6B, so 0.6B remains the
safer default on lower-memory Macs.

## Build the macOS app and installer image

```bash
cargo install cargo-bundle
./scripts/package_macos.sh
```

Open `target/release/bundle/dmg/Edge TTS Studio.dmg`, then drag the app onto the
`Applications` shortcut. A macOS app does not run a traditional installer; the
copy into `/Applications` is the installation step. The local package uses an
ad-hoc signature. Public distribution additionally requires a paid Developer ID
certificate and Apple notarization.

## Build on Windows

Install the stable Rust MSVC toolchain and Visual Studio Build Tools with the
"Desktop development with C++" workload, then run in PowerShell:

```powershell
cargo build --release --locked
```

The portable program is written to
`target\release\edge-tts-studio.exe`. Windows uses CPU inference for the local
Whisper and Qwen3-TTS models; macOS continues to use Metal acceleration. A distributable
installer and SmartScreen reputation require separate Windows packaging and
code signing.

## Architecture

The main thread owns `eframe`/`egui`; all voice discovery, Edge/Qwen synthesis,
model loading, cache I/O, and MP3 writing run in the background. A dedicated OS
thread owns a two-thread Tokio runtime, a reusable `EdgeTtsClient`, and an
on-demand Qwen model. When the requested Qwen version differs from the loaded
one, the worker releases the old model before loading the selected version.
Two `tokio::sync::mpsc` channels
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

On macOS, Qwen3-TTS uses Candle's Metal backend. Windows and Linux use its CPU
backend. The worker releases Qwen before loading Whisper (and vice versa), so
the two large local models do not occupy memory at the same time. Qwen output is
resampled in Rust for the selected speed and volume, then encoded as 24 kHz mono
MP3. In subtitle mode every cue remains anchored to its authored start time.

## Privacy and service note

Edge mode is an online TTS client. The entered text, chosen voice, and synthesis
settings are sent to Microsoft's Edge Read Aloud service. No API key is needed,
but this is an unofficial service endpoint and availability can change. Qwen3
mode downloads model files once, then performs synthesis locally without sending
the entered text to a TTS service.

Audio-to-subtitle recognition is local after the one-time model download: MP3
content and generated subtitles are not uploaded. Whisper may still make
recognition errors or hallucinate text, so important subtitles should be
proofread before publication. Whisper Large-v3 Turbo and its model
configuration are MIT-licensed; model files are downloaded from their
Hugging Face repositories rather than redistributed inside this app.

The official Qwen3-TTS 0.6B and 1.7B CustomVoice checkpoints are Apache-2.0
licensed and downloaded from their Qwen Hugging Face repositories rather than
bundled in the installer. The
`speakers-qwen3-tts` Rust inference backend is a community Candle implementation,
not an official Qwen Rust SDK. See `SERVICE_AND_VOICE_NOTICE.md` for the separate
model, voice, generated-content, and noncommercial-project boundaries.

## License and commercial use

This project is **source-available, not OSI open source**. Project-authored code
is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE.md). It is intended for personal
learning, experiments, and noncommercial research. Commercial use—including
business operations, client work, paid content, monetized media, hosted
services, SaaS/API access, advertising, or indirect commercial benefit—is not
granted by the public license.

See the [Chinese license summary](LICENSE.zh-CN.md),
[commercial licensing policy](COMMERCIAL_LICENSE.md), and
[service and voice notice](SERVICE_AND_VOICE_NOTICE.md). A separately signed
commercial source-code license, if offered, cannot grant Microsoft Edge Read
Aloud voice, service, trademark, endpoint, or generated-audio rights. Those
permissions must be obtained independently from Microsoft and any other
applicable rights holders, or the integration must be replaced with a properly
licensed commercial TTS provider.
