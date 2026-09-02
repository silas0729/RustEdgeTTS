use rusty_mp3::{Error as Mp3Error, Mp3Decoder, Mp3Encoder, Mp3EncoderConfig};

pub const TIMELINE_SAMPLE_RATE: u32 = 24_000;
const TIMELINE_BITRATE_KBPS: u32 = 48;
const SILENCE_CHUNK_SAMPLES: usize = TIMELINE_SAMPLE_RATE as usize;

pub fn decode_mp3_mono(mp3: &[u8]) -> Result<Vec<f32>, String> {
    let mut pcm = decode_mp3_mono_preserving_silence(mp3)?;
    trim_edge_silence(&mut pcm);
    Ok(pcm)
}

pub fn decode_mp3_mono_preserving_silence(mp3: &[u8]) -> Result<Vec<f32>, String> {
    let mut decoder = Mp3Decoder::new();
    decoder.push(mp3);
    decoder.flush();

    let mut source_rate = None;
    let mut pcm = Vec::new();
    loop {
        match decoder.next_frame() {
            Ok(frame) => {
                let channels = usize::from(frame.channels.max(1));
                if let Some(expected) = source_rate {
                    if frame.sample_rate != expected {
                        return Err(
                            "The TTS response changed sample rate inside one clip.".to_owned()
                        );
                    }
                } else {
                    source_rate = Some(frame.sample_rate);
                }

                if channels == 1 {
                    pcm.extend_from_slice(&frame.samples);
                } else {
                    for sample in frame.samples.chunks_exact(channels) {
                        pcm.push(sample.iter().copied().sum::<f32>() / channels as f32);
                    }
                }
            }
            Err(Mp3Error::Again | Mp3Error::Eof) => break,
            Err(error) => return Err(format!("Could not decode a synthesized MP3 clip: {error}")),
        }
    }

    let source_rate =
        source_rate.ok_or_else(|| "The TTS service returned no MP3 frames.".to_owned())?;
    if source_rate != TIMELINE_SAMPLE_RATE {
        pcm = resample_linear(&pcm, source_rate, TIMELINE_SAMPLE_RATE);
    }
    Ok(pcm)
}

/// Trim encoder padding and long near-silent tails while preserving a small
/// attack/release margin so consonants are not clipped.
fn trim_edge_silence(pcm: &mut Vec<f32>) {
    const THRESHOLD: f32 = 0.0015;
    const LEADING_MARGIN_MS: usize = 8;
    const TRAILING_MARGIN_MS: usize = 45;

    let Some(first_signal) = pcm.iter().position(|sample| sample.abs() >= THRESHOLD) else {
        pcm.clear();
        return;
    };
    let Some(last_signal) = pcm.iter().rposition(|sample| sample.abs() >= THRESHOLD) else {
        pcm.clear();
        return;
    };
    let leading_margin = TIMELINE_SAMPLE_RATE as usize * LEADING_MARGIN_MS / 1_000;
    let trailing_margin = TIMELINE_SAMPLE_RATE as usize * TRAILING_MARGIN_MS / 1_000;
    let start = first_signal.saturating_sub(leading_margin);
    let end = (last_signal + 1 + trailing_margin).min(pcm.len());
    pcm.drain(end..);
    pcm.drain(..start);
}

fn resample_linear(input: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if input.is_empty() || source_rate == target_rate {
        return input.to_vec();
    }
    let output_len =
        ((input.len() as u64 * u64::from(target_rate)) / u64::from(source_rate)).max(1) as usize;
    let ratio = source_rate as f64 / target_rate as f64;
    let mut output = Vec::with_capacity(output_len);
    for output_index in 0..output_len {
        let source_position = output_index as f64 * ratio;
        let left = source_position.floor() as usize;
        let right = (left + 1).min(input.len() - 1);
        let fraction = (source_position - left as f64) as f32;
        output.push(input[left] * (1.0 - fraction) + input[right] * fraction);
    }
    output
}

pub fn milliseconds_to_samples(milliseconds: u64) -> u64 {
    milliseconds.saturating_mul(u64::from(TIMELINE_SAMPLE_RATE)) / 1_000
}

pub fn fade_out_for_truncation(samples: &mut [f32]) {
    let fade_samples = (TIMELINE_SAMPLE_RATE as usize * 35 / 1_000).min(samples.len());
    if fade_samples == 0 {
        return;
    }
    let start = samples.len() - fade_samples;
    for (index, sample) in samples[start..].iter_mut().enumerate() {
        let gain = 1.0 - (index + 1) as f32 / fade_samples as f32;
        *sample *= gain;
    }
}

pub struct TimelineMp3Encoder {
    encoder: Mp3Encoder,
    output: Vec<u8>,
    written_samples: u64,
}

impl TimelineMp3Encoder {
    pub fn new() -> Self {
        Self {
            encoder: Mp3Encoder::new(Mp3EncoderConfig {
                bitrate_kbps: TIMELINE_BITRATE_KBPS,
                vbr_quality: None,
            }),
            output: Vec::new(),
            written_samples: 0,
        }
    }

    pub fn write_silence_until(&mut self, target_sample: u64) -> Result<(), String> {
        let mut remaining = target_sample.saturating_sub(self.written_samples);
        let zeros = vec![0.0_f32; SILENCE_CHUNK_SAMPLES];
        while remaining > 0 {
            let chunk_len = remaining.min(SILENCE_CHUNK_SAMPLES as u64) as usize;
            self.push(&zeros[..chunk_len])?;
            remaining -= chunk_len as u64;
        }
        Ok(())
    }

    pub fn write_clip(&mut self, samples: &[f32]) -> Result<(), String> {
        self.push(samples)
    }

    fn push(&mut self, samples: &[f32]) -> Result<(), String> {
        if samples.is_empty() {
            return Ok(());
        }
        self.encoder
            .push_pcm_f32(samples, 1, TIMELINE_SAMPLE_RATE)
            .map_err(|error| format!("Could not encode the subtitle timeline: {error}"))?;
        self.written_samples = self.written_samples.saturating_add(samples.len() as u64);
        self.drain_packets(false)
    }

    fn drain_packets(&mut self, finishing: bool) -> Result<(), String> {
        loop {
            match self.encoder.next_packet() {
                Ok(packet) => self.output.extend_from_slice(&packet),
                Err(Mp3Error::Again) if !finishing => return Ok(()),
                Err(Mp3Error::Eof) if finishing => return Ok(()),
                Err(Mp3Error::Again) => {
                    return Err("The MP3 encoder requested more audio after finishing.".to_owned());
                }
                Err(Mp3Error::Eof) => return Ok(()),
                Err(error) => {
                    return Err(format!(
                        "Could not collect the encoded MP3 timeline: {error}"
                    ));
                }
            }
        }
    }

    pub fn finish(mut self) -> Result<Vec<u8>, String> {
        // Ensure even a completely silent/empty first cue initializes the MP3
        // header, then finish and drain the encoder's final padded frame.
        if self.written_samples == 0 {
            self.push(&[0.0])?;
        }
        self.encoder.finish();
        self.drain_packets(true)?;
        if self.output.is_empty() {
            Err("The MP3 encoder produced an empty subtitle track.".to_owned())
        } else {
            Ok(self.output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_and_decodes_a_timed_mono_track() {
        let mut encoder = TimelineMp3Encoder::new();
        encoder.write_silence_until(12_000).unwrap();
        let tone: Vec<f32> = (0..12_000)
            .map(|index| {
                0.2 * (std::f32::consts::TAU * 440.0 * index as f32 / TIMELINE_SAMPLE_RATE as f32)
                    .sin()
            })
            .collect();
        encoder.write_clip(&tone).unwrap();
        encoder.write_silence_until(36_000).unwrap();
        let mp3 = encoder.finish().unwrap();
        assert!(mp3.len() > 1_000);

        let decoded = decode_mp3_mono_preserving_silence(&mp3).unwrap();
        assert!(decoded.len() >= 36_000);
        assert!(decoded.iter().any(|sample| sample.abs() > 0.02));
    }

    #[test]
    fn fades_a_truncated_clip_to_zero() {
        let mut samples = vec![1.0; 2_000];
        fade_out_for_truncation(&mut samples);
        assert_eq!(samples[0], 1.0);
        assert!(samples.last().unwrap().abs() < f32::EPSILON);
    }
}
