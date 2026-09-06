#![allow(dead_code)]

use edge_tts_rust::EdgeTtsClient;

#[path = "../src/subtitle_pipeline.rs"]
mod subtitle_pipeline;
#[path = "../src/subtitles.rs"]
mod subtitles;
#[path = "../src/system_proxy.rs"]
mod system_proxy;
#[path = "../src/timeline_audio.rs"]
mod timeline_audio;

use subtitles::{SubtitleCue, format_timestamp};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    system_proxy::configure_from_macos_settings();
    let client = EdgeTtsClient::new()?;
    let cues = vec![
        SubtitleCue {
            start_ms: 1_000,
            end_ms: 4_000,
            text: "你好，这是第一条字幕。".to_owned(),
        },
        SubtitleCue {
            start_ms: 5_000,
            end_ms: 8_000,
            text: "This is the second subtitle cue.".to_owned(),
        },
    ];

    let report = subtitle_pipeline::generate_subtitle_audio(
        &client,
        &cues,
        "zh-CN-XiaoxiaoNeural",
        0,
        0,
        |current, total| println!("Synthesizing cue {current}/{total}"),
    )
    .await?;
    if report.mp3.len() < 4_000 {
        return Err("The generated subtitle MP3 was unexpectedly small.".into());
    }

    let decoded = timeline_audio::decode_mp3_mono_preserving_silence(&report.mp3)?;
    let duration_ms =
        decoded.len() as u64 * 1_000 / u64::from(timeline_audio::TIMELINE_SAMPLE_RATE);
    if !(7_900..=8_200).contains(&duration_ms) {
        return Err(format!("Unexpected timeline duration: {duration_ms} ms").into());
    }
    let peak = |start_ms: usize, end_ms: usize| {
        let start = start_ms * timeline_audio::TIMELINE_SAMPLE_RATE as usize / 1_000;
        let end =
            (end_ms * timeline_audio::TIMELINE_SAMPLE_RATE as usize / 1_000).min(decoded.len());
        decoded[start..end]
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0_f32, f32::max)
    };
    if peak(0, 800) > 0.01 || peak(4_200, 4_800) > 0.01 {
        return Err("Authored subtitle gaps did not remain silent.".into());
    }
    if peak(1_000, 4_000) < 0.02 || peak(5_000, 8_000) < 0.02 {
        return Err("Expected speech was not found inside both subtitle cues.".into());
    }

    let output_path = std::env::temp_dir().join("edge-tts-subtitle-smoke.mp3");
    tokio::fs::write(&output_path, &report.mp3).await?;
    println!(
        "Subtitle timeline passed: {} bytes, duration {}, overflowed {}, output {}",
        report.mp3.len(),
        format_timestamp(duration_ms),
        report.overflow_count,
        output_path.display()
    );
    Ok(())
}
