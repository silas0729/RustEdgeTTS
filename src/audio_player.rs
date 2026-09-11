use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use rodio::{OutputStream, Sink, Source, source::SeekError};

pub struct GeneratedAudioPlayer {
    path: PathBuf,
    samples: Arc<[f32]>,
    sample_rate: u32,
    duration: Duration,
    sink: Sink,
    _stream: OutputStream,
    has_started: bool,
}

impl GeneratedAudioPlayer {
    pub fn new(path: PathBuf, samples: Vec<f32>, sample_rate: u32) -> Result<Self, String> {
        if samples.is_empty() || sample_rate == 0 {
            return Err("生成的音频没有可播放的采样。".to_owned());
        }
        let duration = Duration::from_secs_f64(samples.len() as f64 / f64::from(sample_rate));
        let samples: Arc<[f32]> = samples.into();
        let (stream, stream_handle) = OutputStream::try_default()
            .map_err(|error| format!("无法打开系统音频输出设备：{error}"))?;
        let sink = Sink::try_new(&stream_handle)
            .map_err(|error| format!("无法创建音频播放器：{error}"))?;
        sink.append(SharedSamples::new(Arc::clone(&samples), sample_rate));
        sink.pause();
        Ok(Self {
            path,
            samples,
            sample_rate,
            duration,
            sink,
            _stream: stream,
            has_started: false,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn duration_seconds(&self) -> f32 {
        self.duration.as_secs_f32()
    }

    pub fn position_seconds(&self) -> f32 {
        if self.sink.empty() && self.has_started {
            self.duration_seconds()
        } else {
            self.sink
                .get_pos()
                .as_secs_f32()
                .clamp(0.0, self.duration_seconds())
        }
    }

    pub fn is_playing(&self) -> bool {
        !self.sink.is_paused() && !self.sink.empty()
    }

    pub fn toggle(&mut self) {
        if self.is_playing() {
            self.sink.pause();
            return;
        }
        if self.sink.empty() {
            self.reset_source();
        }
        self.has_started = true;
        self.sink.play();
    }

    pub fn restart(&mut self) {
        self.reset_source();
        self.has_started = true;
        self.sink.play();
    }

    pub fn seek(&mut self, seconds: f32) -> Result<(), String> {
        let target = seconds.clamp(0.0, self.duration_seconds());
        let was_playing = self.is_playing();
        if self.sink.empty() {
            self.reset_source();
        }
        self.sink
            .try_seek(Duration::from_secs_f32(target))
            .map_err(|error| format!("无法跳转播放位置：{error}"))?;
        self.has_started = target > 0.0 || self.has_started;
        if was_playing && target < self.duration_seconds() {
            self.sink.play();
        }
        Ok(())
    }

    fn reset_source(&mut self) {
        self.sink.stop();
        self.sink.append(SharedSamples::new(
            Arc::clone(&self.samples),
            self.sample_rate,
        ));
        self.sink.pause();
    }
}

struct SharedSamples {
    samples: Arc<[f32]>,
    position: usize,
    sample_rate: u32,
}

impl SharedSamples {
    fn new(samples: Arc<[f32]>, sample_rate: u32) -> Self {
        Self {
            samples,
            position: 0,
            sample_rate,
        }
    }
}

impl Iterator for SharedSamples {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let sample = self.samples.get(self.position).copied()?;
        self.position += 1;
        Some(sample)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.samples.len().saturating_sub(self.position);
        (remaining, Some(remaining))
    }
}

impl Source for SharedSamples {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        Some(Duration::from_secs_f64(
            self.samples.len() as f64 / f64::from(self.sample_rate),
        ))
    }

    fn try_seek(&mut self, pos: Duration) -> Result<(), SeekError> {
        let sample = (pos.as_secs_f64() * f64::from(self.sample_rate)) as usize;
        self.position = sample.min(self.samples.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_samples_seek_without_copying_the_audio() {
        let data: Arc<[f32]> = (0..100).map(|value| value as f32).collect();
        let mut source = SharedSamples::new(data, 10);
        source.try_seek(Duration::from_secs(5)).unwrap();
        assert_eq!(source.next(), Some(50.0));
        source.try_seek(Duration::from_secs(99)).unwrap();
        assert_eq!(source.next(), None);
    }
}
