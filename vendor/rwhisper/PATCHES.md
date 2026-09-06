# Local rwhisper patch

This directory vendors `rwhisper` 0.4.1 (MIT OR Apache-2.0) from crates.io.

The only behavioral change is a bounds-safe lookup for DTW token timestamps in
`src/model.rs`. On some long multilingual recordings, Whisper produces fewer
timestamp entries than decoded tokens. Upstream indexed this vector directly,
which could panic and terminate transcription. Trailing tokens now reuse the
last available timestamp; empty timestamp output remains untimed.
