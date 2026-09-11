use std::{
    fs::File,
    io::ErrorKind,
    ops::Range,
    path::{Path, PathBuf},
    time::Instant,
};

use directories::BaseDirs;
use futures_util::StreamExt;
use rodio::buffer::SamplesBuffer;
use rwhisper::{ModelLoadingProgress, Whisper, WhisperLanguage, WhisperSource};
use symphonia::core::{
    audio::SampleBuffer,
    codecs::DecoderOptions,
    errors::Error as SymphoniaError,
    formats::FormatOptions,
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
    probe::Hint,
};

use crate::{
    subtitles::SubtitleCue,
    timeline_audio::{TIMELINE_SAMPLE_RATE, decode_mp3_mono_preserving_silence},
};

const MAX_CUE_CHARACTERS: usize = 28;
const ANALYSIS_FRAME_MS: usize = 20;
const TARGET_WINDOW_MS: usize = 22_000;
const MAX_WINDOW_MS: usize = 26_000;
const MAX_CUE_DURATION_MS: u64 = 5_500;
const MIN_SENTENCE_CUE_MS: u64 = 700;
const WHISPER_SAMPLE_RATE: u64 = 16_000;
const BUNDLED_MODEL_CONFIG: &str = include_str!("../assets/whisper-large-v3-turbo-config.json");

pub fn model_directory() -> Result<PathBuf, String> {
    BaseDirs::new()
        .map(|dirs| {
            dirs.data_dir()
                .join("kalosm/cache/Demonthos/candle-quantized-whisper-large-v3-turbo/main")
        })
        .ok_or_else(|| "Could not determine the local Whisper model directory.".to_owned())
}

#[derive(Debug)]
struct TimedChunk {
    start_ms: u64,
    end_ms: u64,
    text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecognitionLanguage {
    Chinese,
    MixedChineseEnglish,
    English,
}

impl RecognitionLanguage {
    fn whisper_language(self) -> Option<WhisperLanguage> {
        match self {
            Self::Chinese => Some(WhisperLanguage::Chinese),
            // rwhisper 0.4.1 treats `None` as English rather than performing
            // language detection. A Chinese language token still keeps the
            // multilingual vocabulary available, so it is the reliable prompt
            // for alternating Chinese and English speech.
            Self::MixedChineseEnglish => Some(WhisperLanguage::Chinese),
            Self::English => Some(WhisperLanguage::English),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubtitleExportFormat {
    Srt,
    WebVtt,
}

impl SubtitleExportFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::WebVtt => "vtt",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Srt => "SRT",
            Self::WebVtt => "WebVTT",
        }
    }
}

pub struct TranscriptionReport {
    pub cues: Vec<SubtitleCue>,
    pub audio_duration_ms: u64,
}

/// Load the multilingual, quantized Whisper Large-v3 Turbo model. All model
/// inference is implemented by Rust/Candle; Metal is selected automatically on
/// supported Macs. The model files are downloaded once and then reused from
/// the local Kalosm cache.
pub async fn load_model(
    language: RecognitionLanguage,
    on_progress: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
) -> Result<Whisper, String> {
    ensure_model_config().await?;
    Whisper::builder()
        .with_source(WhisperSource::QuantizedLargeV3Turbo)
        .with_language(language.whisper_language())
        .build_with_loading_handler(on_progress)
        .await
        .map_err(|error| format!("Could not load the local Whisper model: {error}"))
}

async fn ensure_model_config() -> Result<(), String> {
    let path = model_directory()
        .unwrap_or_else(|_| {
            PathBuf::from("kalosm/cache/Demonthos/candle-quantized-whisper-large-v3-turbo/main")
        })
        .join("config.json");
    if path.is_file() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| format!("Could not create the local model cache: {error}"))?;
    }
    tokio::fs::write(&path, BUNDLED_MODEL_CONFIG)
        .await
        .map_err(|error| format!("Could not prepare the local model configuration: {error}"))
}

pub async fn transcribe_media(
    model: &Whisper,
    input_path: &Path,
    mut on_progress: impl FnMut(f32, u64) + Send,
) -> Result<TranscriptionReport, String> {
    let decode_path = input_path.to_path_buf();
    let samples = tokio::task::spawn_blocking(move || decode_media_mono(&decode_path))
        .await
        .map_err(|error| format!("The media decoding task failed: {error}"))??;
    if samples.is_empty() {
        return Err("The selected media file does not contain a decodable audio track.".to_owned());
    }

    let audio_duration_ms = samples.len() as u64 * 1_000 / u64::from(TIMELINE_SAMPLE_RATE);
    // Whisper internally works in 30-second blocks and can occasionally mark a
    // complete block as no-speech after a language/context transition. Split
    // before that boundary, preferably at the quietest nearby frame, and run
    // every window independently so one bad block cannot remove 30 seconds of
    // otherwise valid speech.
    let windows = split_audio_windows(&samples, TIMELINE_SAMPLE_RATE);
    let total_windows = windows.len();
    let started_at = Instant::now();
    let mut timed_chunks = Vec::new();
    for (index, range) in windows.into_iter().enumerate() {
        let source = SamplesBuffer::new(1, TIMELINE_SAMPLE_RATE, samples[range.clone()].to_vec());
        let window_start_ms = range.start as u64 * 1_000 / u64::from(TIMELINE_SAMPLE_RATE);
        let window_end_ms = range.end as u64 * 1_000 / u64::from(TIMELINE_SAMPLE_RATE);
        let mut stream = model.transcribe(source).timestamped();
        while let Some(segment) = stream.next().await {
            let segment_range = segment.sample_range();
            // rwhisper may emit a padded tail segment whose start is beyond
            // the real input and whose range is even reversed (for example
            // 480000..400000). Its text is a Whisper silence hallucination,
            // not speech from the file.
            let Some((segment_start_ms, segment_end_ms)) =
                valid_segment_bounds(window_start_ms, window_end_ms, segment_range)
            else {
                continue;
            };
            let mut found_timestamp = false;
            for chunk in segment.chunks() {
                let Some(timestamp) = chunk.timestamp() else {
                    continue;
                };
                found_timestamp = true;
                let start_ms = segment_start_ms + (timestamp.start.max(0.0) * 1_000.0) as u64;
                let end_ms = segment_start_ms + (timestamp.end.max(0.0) * 1_000.0) as u64;
                if start_ms < window_end_ms {
                    timed_chunks.push(TimedChunk {
                        start_ms,
                        end_ms: end_ms.max(start_ms.saturating_add(80)).min(window_end_ms),
                        text: chunk.text().to_owned(),
                    });
                }
            }
            if !found_timestamp && !segment.text().trim().is_empty() {
                timed_chunks.push(TimedChunk {
                    start_ms: segment_start_ms,
                    end_ms: segment_end_ms.max(segment_start_ms.saturating_add(200)),
                    text: segment.text().to_owned(),
                });
            }
        }

        let completed = index + 1;
        let progress = completed as f32 / total_windows as f32;
        let elapsed = started_at.elapsed().as_secs_f64();
        let remaining =
            (elapsed / completed as f64 * (total_windows - completed) as f64).round() as u64;
        on_progress(progress, remaining);
    }

    let mut cues = group_timed_chunks_into_cues(timed_chunks);

    normalize_cue_boundaries(&mut cues, audio_duration_ms);
    if cues.is_empty() {
        return Err("No speech was recognized in this audio file.".to_owned());
    }
    Ok(TranscriptionReport {
        cues,
        audio_duration_ms,
    })
}

/// Decode an audio file or the audio track embedded in a common video
/// container. MP3 keeps the project's gap-preserving decoder; MP4/MOV/MKV and
/// other supported containers go through Symphonia and are mixed to mono once.
pub(crate) fn decode_media_mono(path: &Path) -> Result<Vec<f32>, String> {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mp3"))
    {
        let bytes =
            std::fs::read(path).map_err(|error| format!("Could not read the MP3 file: {error}"))?;
        return decode_mp3_mono_preserving_silence(&bytes)
            .map_err(|error| format!("Could not decode the MP3 file: {error}"));
    }

    let file = File::open(path)
        .map_err(|error| format!("Could not open the selected media file: {error}"))?;
    let stream = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|error| format!("Could not read the media container: {error}"))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        // Compressed AAC tracks often declare their sample rate before the
        // decoder has discovered the channel layout. Video tracks do not have
        // an audio sample rate, so this reliably selects the embedded audio.
        .find(|track| track.codec_params.sample_rate.is_some())
        .ok_or_else(|| {
            "No supported audio track was found in the selected media file.".to_owned()
        })?;
    let track_id = track.id;
    let codec_params = track.codec_params.clone();
    let mut source_rate = codec_params
        .sample_rate
        .ok_or_else(|| "The media audio track does not declare a sample rate.".to_owned())?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|error| format!("The media audio codec is not supported: {error}"))?;
    let mut mono = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error)) if error.kind() == ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err("The media audio track changes format part-way through and cannot be transcribed.".to_owned());
            }
            Err(error) => return Err(format!("Could not read the media audio track: {error}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(error)) if error.kind() == ErrorKind::UnexpectedEof => {
                break;
            }
            Err(error) => return Err(format!("Could not decode the media audio track: {error}")),
        };
        let spec = *decoded.spec();
        if spec.rate != source_rate {
            if mono.is_empty() {
                source_rate = spec.rate;
            } else {
                return Err(
                    "The media audio track changes sample rate part-way through.".to_owned(),
                );
            }
        }
        let channels = spec.channels.count().max(1);
        let mut samples = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        samples.copy_interleaved_ref(decoded);
        if channels == 1 {
            mono.extend_from_slice(samples.samples());
        } else {
            mono.reserve(samples.samples().len() / channels);
            for frame in samples.samples().chunks_exact(channels) {
                mono.push(frame.iter().copied().sum::<f32>() / channels as f32);
            }
        }
    }

    if source_rate != TIMELINE_SAMPLE_RATE {
        mono = crate::timeline_audio::resample_linear(&mono, source_rate, TIMELINE_SAMPLE_RATE);
    }
    Ok(mono)
}

/// Kept for the public smoke example and downstream callers.
#[allow(dead_code)]
pub async fn transcribe_mp3(
    model: &Whisper,
    input_path: &Path,
    on_progress: impl FnMut(f32, u64) + Send,
) -> Result<TranscriptionReport, String> {
    transcribe_media(model, input_path, on_progress).await
}

fn split_audio_windows(samples: &[f32], sample_rate: u32) -> Vec<Range<usize>> {
    if samples.is_empty() {
        return Vec::new();
    }
    let frame_samples = (sample_rate as usize * ANALYSIS_FRAME_MS / 1_000).max(1);
    let energies: Vec<f32> = samples
        .chunks(frame_samples)
        .map(|frame| {
            (frame.iter().map(|sample| sample * sample).sum::<f32>() / frame.len() as f32).sqrt()
        })
        .collect();
    let target_frames = (TARGET_WINDOW_MS / ANALYSIS_FRAME_MS).max(1);
    let max_frames = (MAX_WINDOW_MS / ANALYSIS_FRAME_MS).max(target_frames);
    let mut windows = Vec::new();
    let mut start_frame = 0usize;

    while energies.len().saturating_sub(start_frame) > max_frames {
        let search_start = (start_frame + target_frames).min(energies.len());
        let search_end = (start_frame + max_frames).min(energies.len());
        let split_frame = (search_start..search_end)
            .min_by(|left, right| energies[*left].total_cmp(&energies[*right]))
            .unwrap_or(search_end);
        windows.push(
            (start_frame * frame_samples).min(samples.len())
                ..(split_frame * frame_samples).min(samples.len()),
        );
        start_frame = split_frame;
    }
    windows.push((start_frame * frame_samples).min(samples.len())..samples.len());
    windows.retain(|range| !range.is_empty());
    windows
}

fn valid_segment_bounds(
    window_start_ms: u64,
    window_end_ms: u64,
    sample_range: Range<usize>,
) -> Option<(u64, u64)> {
    if sample_range.start >= sample_range.end {
        return None;
    }
    let start_ms = window_start_ms + sample_range.start as u64 * 1_000 / WHISPER_SAMPLE_RATE;
    let end_ms = (window_start_ms + sample_range.end as u64 * 1_000 / WHISPER_SAMPLE_RATE)
        .min(window_end_ms);
    (start_ms < end_ms).then_some((start_ms, end_ms))
}

fn group_timed_chunks_into_cues(chunks: Vec<TimedChunk>) -> Vec<SubtitleCue> {
    let mut cues = Vec::new();
    let mut current_text = String::new();
    let mut current_start = 0;
    let mut current_end = 0;

    for chunk in chunks {
        let chunk_text = normalize_recognized_text(&chunk.text);
        let mut flush_after_word = false;
        if chunk_text.is_empty() {
            continue;
        }
        if chunk_text.chars().count() > MAX_CUE_CHARACTERS
            || chunk.end_ms.saturating_sub(chunk.start_ms) > MAX_CUE_DURATION_MS
        {
            if !current_text.is_empty() && continues_ascii_word(&current_text, &chunk_text) {
                append_recognized_text(&mut current_text, &chunk.text);
                cues.extend(cues_from_utterance(
                    current_start,
                    chunk.end_ms.max(current_end),
                    &current_text,
                ));
                current_text.clear();
                continue;
            }
            push_timed_cue(&mut cues, current_start, current_end, &mut current_text);
            cues.extend(cues_from_utterance(
                chunk.start_ms,
                chunk.end_ms,
                &chunk.text,
            ));
            continue;
        }
        if current_text.is_empty() {
            current_start = chunk.start_ms;
            current_end = chunk.end_ms;
        } else {
            let projected_characters = current_text.chars().count() + chunk_text.chars().count();
            let projected_duration = chunk.end_ms.saturating_sub(current_start);
            if projected_characters > MAX_CUE_CHARACTERS || projected_duration > MAX_CUE_DURATION_MS
            {
                if continues_ascii_word(&current_text, &chunk_text) {
                    flush_after_word = true;
                } else {
                    push_timed_cue(&mut cues, current_start, current_end, &mut current_text);
                    current_start = chunk.start_ms;
                }
            }
            current_end = current_end.max(chunk.end_ms);
        }
        append_recognized_text(&mut current_text, &chunk.text);

        let duration = current_end.saturating_sub(current_start);
        if flush_after_word || (duration >= MIN_SENTENCE_CUE_MS && ends_sentence(&current_text)) {
            push_timed_cue(&mut cues, current_start, current_end, &mut current_text);
        }
    }
    push_timed_cue(&mut cues, current_start, current_end, &mut current_text);
    cues
}

fn push_timed_cue(cues: &mut Vec<SubtitleCue>, start_ms: u64, end_ms: u64, text: &mut String) {
    let normalized = normalize_recognized_text(text);
    text.clear();
    if normalized.is_empty() {
        return;
    }
    cues.push(SubtitleCue {
        start_ms,
        end_ms: end_ms.max(start_ms.saturating_add(200)),
        text: normalized,
    });
}

fn ends_sentence(text: &str) -> bool {
    text.trim_end()
        .chars()
        .next_back()
        .is_some_and(|character| matches!(character, '。' | '！' | '？' | '；' | '!' | '?' | ';'))
}

fn continues_ascii_word(left: &str, right: &str) -> bool {
    left.chars()
        .next_back()
        .is_some_and(|character| character.is_ascii_alphanumeric())
        && right
            .chars()
            .next()
            .is_some_and(|character| character.is_ascii_alphanumeric())
}

fn cues_from_utterance(start_ms: u64, end_ms: u64, text: &str) -> Vec<SubtitleCue> {
    let text = normalize_recognized_text(text);
    if text.is_empty() || end_ms <= start_ms {
        return Vec::new();
    }
    let sentences = split_sentences(&text);
    let sentence_count = sentences.len();
    let total_weight: usize = sentences
        .iter()
        .map(|sentence| sentence.chars().count().max(1))
        .sum();
    let duration = end_ms - start_ms;
    let mut elapsed_weight = 0usize;
    sentences
        .into_iter()
        .enumerate()
        .map(|(index, sentence)| {
            let cue_start = start_ms + duration * elapsed_weight as u64 / total_weight as u64;
            elapsed_weight += sentence.chars().count().max(1);
            let cue_end = if index + 1 == sentence_count {
                end_ms
            } else {
                start_ms + duration * elapsed_weight as u64 / total_weight as u64
            };
            SubtitleCue {
                start_ms: cue_start,
                end_ms: cue_end.max(cue_start + 200),
                text: sentence,
            }
        })
        .collect()
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        current.push(character);
        let character_count = current.chars().count();
        let should_split = matches!(character, '。' | '！' | '？' | '；' | '!' | '?' | ';')
            || (character == '.' && character_count >= 12)
            || (character_count >= MAX_CUE_CHARACTERS && !character.is_ascii_alphanumeric());
        if should_split {
            let sentence = current.trim().to_owned();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            current.clear();
        }
    }
    let remainder = current.trim();
    if !remainder.is_empty() {
        sentences.push(remainder.to_owned());
    }
    if sentences.is_empty() && !text.trim().is_empty() {
        sentences.push(text.trim().to_owned());
    }
    sentences
}

fn normalize_cue_boundaries(cues: &mut [SubtitleCue], audio_duration_ms: u64) {
    for index in 0..cues.len() {
        cues[index].end_ms = cues[index].end_ms.min(audio_duration_ms);
        if let Some(next) = cues.get(index + 1) {
            cues[index].end_ms = cues[index]
                .end_ms
                .min(next.start_ms.saturating_sub(20))
                .max(cues[index].start_ms.saturating_add(120));
        }
    }
}

pub fn render_subtitles(cues: &[SubtitleCue], format: SubtitleExportFormat) -> String {
    match format {
        SubtitleExportFormat::Srt => render_srt(cues),
        SubtitleExportFormat::WebVtt => render_webvtt(cues),
    }
}

fn append_recognized_text(target: &mut String, next: &str) {
    if target.is_empty() {
        target.push_str(next.trim_start());
    } else {
        target.push_str(next);
    }
}

fn normalize_recognized_text(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut pending_space = false;
    for character in text.trim().chars() {
        if character.is_whitespace() {
            pending_space = !output.is_empty();
            continue;
        }
        if pending_space
            && output
                .chars()
                .last()
                .is_some_and(|previous| needs_word_space(previous, character))
        {
            output.push(' ');
        }
        pending_space = false;
        output.push(character);
    }
    output
}

fn needs_word_space(previous: char, next: char) -> bool {
    previous.is_ascii_alphanumeric() && next.is_ascii_alphanumeric()
}

fn render_srt(cues: &[SubtitleCue]) -> String {
    let mut output = String::new();
    for (index, cue) in cues.iter().enumerate() {
        output.push_str(&(index + 1).to_string());
        output.push('\n');
        output.push_str(&format!(
            "{} --> {}\n",
            subtitle_timestamp(cue.start_ms, ','),
            subtitle_timestamp(cue.end_ms, ',')
        ));
        output.push_str(&cue.text);
        output.push_str("\n\n");
    }
    output
}

fn render_webvtt(cues: &[SubtitleCue]) -> String {
    let mut output = String::from("WEBVTT\n\n");
    for cue in cues {
        output.push_str(&format!(
            "{} --> {}\n",
            subtitle_timestamp(cue.start_ms, '.'),
            subtitle_timestamp(cue.end_ms, '.')
        ));
        output.push_str(&cue.text);
        output.push_str("\n\n");
    }
    output
}

fn subtitle_timestamp(milliseconds: u64, separator: char) -> String {
    let hours = milliseconds / 3_600_000;
    let minutes = (milliseconds / 60_000) % 60;
    let seconds = (milliseconds / 1_000) % 60;
    let millis = milliseconds % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}{separator}{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn decodes_aac_audio_from_an_mp4_container() {
        let source = Path::new("/System/Library/Sounds/Basso.aiff");
        if !source.is_file() {
            return;
        }
        let output = std::env::temp_dir().join(format!(
            "edge-tts-studio-media-decode-{}.mp4",
            std::process::id()
        ));
        let status = std::process::Command::new("/usr/bin/afconvert")
            .arg(source)
            .arg(&output)
            .args(["-f", "m4af", "-d", "aac"])
            .status()
            .expect("macOS includes afconvert");
        assert!(status.success());
        let samples = decode_media_mono(&output).expect("decode AAC from ISO MP4");
        let _ = std::fs::remove_file(output);
        assert!(samples.len() > TIMELINE_SAMPLE_RATE as usize / 4);
    }

    #[test]
    fn splits_a_mixed_utterance_at_sentence_boundaries() {
        let cues = cues_from_utterance(100, 3_000, "你好，世界。Hello world!");
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "你好，世界。");
        assert_eq!(cues[0].start_ms, 100);
        assert_eq!(cues[1].text, "Hello world!");
        assert_eq!(cues[1].end_ms, 3_000);
    }

    #[test]
    fn audio_windows_cover_every_sample_without_crossing_whispers_boundary() {
        let samples = vec![0.0; 74_640];
        let windows = split_audio_windows(&samples, 1_000);
        assert!(windows.len() >= 3);
        assert_eq!(windows.first().map(|range| range.start), Some(0));
        assert_eq!(windows.last().map(|range| range.end), Some(samples.len()));
        assert!(windows.windows(2).all(|pair| pair[0].end == pair[1].start));
        assert!(windows.iter().all(|range| range.len() <= 26_000));
    }

    #[test]
    fn groups_valid_mixed_language_timestamps() {
        let cues = group_timed_chunks_into_cues(vec![
            TimedChunk {
                start_ms: 100,
                end_ms: 1_100,
                text: "你好。".to_owned(),
            },
            TimedChunk {
                start_ms: 1_200,
                end_ms: 2_400,
                text: " Hello world!".to_owned(),
            },
        ]);
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "你好。");
        assert_eq!(cues[1].text, "Hello world!");
    }

    #[test]
    fn does_not_split_an_ascii_word_between_token_chunks() {
        let cues = group_timed_chunks_into_cues(vec![
            TimedChunk {
                start_ms: 0,
                end_ms: 5_400,
                text: "欢迎在Git".to_owned(),
            },
            TimedChunk {
                start_ms: 5_420,
                end_ms: 5_900,
                text: "Hub查看".to_owned(),
            },
        ]);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "欢迎在GitHub查看");
    }

    #[test]
    fn keeps_ascii_word_whole_when_the_following_chunk_is_oversized() {
        let cues = group_timed_chunks_into_cues(vec![
            TimedChunk {
                start_ms: 0,
                end_ms: 4_000,
                text: "我决定把项目上传G".to_owned(),
            },
            TimedChunk {
                start_ms: 4_020,
                end_ms: 9_000,
                text: "itHub链接放在简介区大家可以直接下载查看完整源代码".to_owned(),
            },
        ]);
        assert!(!cues.is_empty());
        assert!(
            cues.windows(2)
                .all(|pair| !continues_ascii_word(&pair[0].text, &pair[1].text))
        );
        assert!(cues.iter().any(|cue| cue.text.contains("GitHub")));
    }

    #[test]
    fn sentence_length_limit_keeps_ascii_words_whole() {
        let parts = split_sentences(
            "这是较长的一段中文内容用于填充字幕长度然后继续补充更多内容上传GitHub链接继续查看",
        );
        assert!(parts.len() >= 2);
        assert!(
            parts
                .windows(2)
                .all(|pair| { !continues_ascii_word(pair[0].as_str(), pair[1].as_str()) })
        );
    }

    #[test]
    fn rejects_padded_or_reversed_segments_after_the_real_audio() {
        assert_eq!(
            valid_segment_bounds(0, 25_000, 0..400_000),
            Some((0, 25_000))
        );
        assert_eq!(
            valid_segment_bounds(
                0,
                25_000,
                Range {
                    start: 480_000,
                    end: 400_000,
                },
            ),
            None
        );
        assert_eq!(valid_segment_bounds(0, 25_000, 480_000..500_000), None);
    }

    #[test]
    fn renders_valid_srt_and_webvtt_timestamps() {
        let cues = vec![SubtitleCue {
            start_ms: 3_723_004,
            end_ms: 3_724_500,
            text: "测试 subtitle".to_owned(),
        }];
        assert!(render_srt(&cues).contains("01:02:03,004 --> 01:02:04,500"));
        assert!(render_webvtt(&cues).contains("01:02:03.004 --> 01:02:04.500"));
    }

    #[test]
    fn removes_whitespace_around_chinese_but_keeps_english_words() {
        assert_eq!(
            normalize_recognized_text("你 好 hello world"),
            "你好hello world"
        );
    }
}
