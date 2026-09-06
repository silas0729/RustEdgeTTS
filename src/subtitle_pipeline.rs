use edge_tts_rust::{EdgeTtsClient, SpeakOptions};

use crate::{subtitles::SubtitleCue, timeline_audio};

pub struct SubtitleGenerationReport {
    pub mp3: Vec<u8>,
    pub overflow_count: usize,
}

async fn synthesize_clip(
    client: &EdgeTtsClient,
    text: &str,
    voice: &str,
    rate_percent: i32,
    volume_percent: i32,
) -> Result<Vec<f32>, String> {
    let result = client
        .synthesize(
            text.to_owned(),
            SpeakOptions {
                voice: voice.to_owned(),
                rate: format!("{rate_percent:+}%"),
                volume: format!("{volume_percent:+}%"),
                ..SpeakOptions::default()
            },
        )
        .await
        .map_err(|error| format!("Edge TTS synthesis failed: {error}"))?;

    timeline_audio::decode_mp3_mono(&result.audio)
}

/// Synthesize each cue separately and stream it into a constant-rate MP3
/// timeline. Gaps remain zero PCM, so memory use stays bounded even for long
/// subtitle files.
pub async fn generate_subtitle_audio(
    client: &EdgeTtsClient,
    cues: &[SubtitleCue],
    voice: &str,
    rate_percent: i32,
    volume_percent: i32,
    mut on_progress: impl FnMut(usize, usize),
) -> Result<SubtitleGenerationReport, String> {
    if cues.is_empty() {
        return Err("The imported subtitle has no timed text cues.".to_owned());
    }

    let total = cues.len();
    let timeline_end_ms = cues.iter().map(|cue| cue.end_ms).max().unwrap_or(0);
    let mut encoder = timeline_audio::TimelineMp3Encoder::new();
    let mut overflow_count = 0;

    for (index, cue) in cues.iter().enumerate() {
        let current = index + 1;
        on_progress(current, total);

        let cue_start_sample = timeline_audio::milliseconds_to_samples(cue.start_ms);
        encoder.write_silence_until(cue_start_sample)?;

        // When subtitle entries overlap, the earlier voice gets the interval up
        // to the next cue. Every later cue therefore remains anchored to its
        // authored start time instead of being shifted by one long line.
        let cue_slot_end_ms = cues
            .get(index + 1)
            .map(|next| cue.end_ms.min(next.start_ms))
            .unwrap_or(cue.end_ms)
            .max(cue.start_ms + 1);
        let available_samples =
            timeline_audio::milliseconds_to_samples(cue_slot_end_ms.saturating_sub(cue.start_ms))
                as usize;

        let clip = synthesize_clip(client, &cue.text, voice, rate_percent, volume_percent)
            .await
            .map_err(|error| format!("Subtitle {current}/{total}: {error}"))?;

        if clip.len() > available_samples {
            // Keep the user's selected rate and every spoken word. When speech
            // is longer than its authored slot, later cues are written after it
            // rather than being auto-sped-up, overlapped, or truncated.
            overflow_count += 1;
        }
        encoder.write_clip(&clip)?;
    }

    encoder.write_silence_until(timeline_audio::milliseconds_to_samples(timeline_end_ms))?;
    Ok(SubtitleGenerationReport {
        mp3: encoder.finish()?,
        overflow_count,
    })
}
