#![allow(dead_code)]

use std::path::PathBuf;

#[path = "../src/asr.rs"]
mod asr;
#[path = "../src/subtitles.rs"]
mod subtitles;
#[path = "../src/system_proxy.rs"]
mod system_proxy;
#[path = "../src/timeline_audio.rs"]
mod timeline_audio;

use asr::{RecognitionLanguage, SubtitleExportFormat};
use rwhisper::ModelLoadingProgress;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    system_proxy::configure_from_macos_settings();
    let mut input_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("edge-tts-studio-smoke-test.mp3"));
    if !input_path.is_file() {
        return Err(format!(
            "MP3 not found: {}. Run `cargo run --example edge_smoke` first or pass an MP3 path.",
            input_path.display()
        )
        .into());
    }

    if let Some(seconds) = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
    {
        let start_seconds = std::env::args()
            .nth(3)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let bytes = tokio::fs::read(&input_path).await?;
        let samples = timeline_audio::decode_mp3_mono_preserving_silence(&bytes)?;
        let sample_rate = timeline_audio::TIMELINE_SAMPLE_RATE as usize;
        let start = start_seconds.saturating_mul(sample_rate).min(samples.len());
        let end = start
            .saturating_add(seconds.saturating_mul(sample_rate))
            .min(samples.len());
        let mut encoder = timeline_audio::TimelineMp3Encoder::new();
        encoder.write_clip(&samples[start..end])?;
        input_path = std::env::temp_dir().join("edge-tts-studio-asr-smoke-slice.mp3");
        tokio::fs::write(&input_path, encoder.finish()?).await?;
        println!("Testing {seconds}s from {start_seconds}s in the input audio");
    }

    let model = asr::load_model(
        RecognitionLanguage::MixedChineseEnglish,
        |progress| match progress {
            ModelLoadingProgress::Downloading { source, progress } => println!(
                "Preparing {source}: {:.1}/{:.1} MB",
                progress.progress as f64 / 1_048_576.0,
                progress.size as f64 / 1_048_576.0
            ),
            ModelLoadingProgress::Loading { progress } => {
                println!("Loading model: {:.0}%", progress * 100.0)
            }
        },
    )
    .await?;

    let report = asr::transcribe_mp3(&model, &input_path, |progress, remaining| {
        println!(
            "Transcribing: {:.0}% · about {remaining}s remaining",
            progress * 100.0
        );
    })
    .await?;
    let output_path = std::env::temp_dir().join("edge-tts-studio-asr-smoke.srt");
    tokio::fs::write(
        &output_path,
        asr::render_subtitles(&report.cues, SubtitleExportFormat::Srt),
    )
    .await?;
    println!(
        "Local transcription passed: {} cues from {:.2}s, output {}",
        report.cues.len(),
        report.audio_duration_ms as f64 / 1_000.0,
        output_path.display()
    );
    for cue in &report.cues {
        println!("{} --> {}  {}", cue.start_ms, cue.end_ms, cue.text);
    }
    Ok(())
}
