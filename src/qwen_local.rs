use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
};

use directories::{BaseDirs, ProjectDirs};
use qwen3_tts::{AudioBuffer, Language, Qwen3TTS, Speaker, SynthesisOptions};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenModelVersion {
    Small0_6B,
    Large1_7B,
}

impl QwenModelVersion {
    pub const DEFAULT: Self = Self::Small0_6B;

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Small0_6B => "0.6B",
            Self::Large1_7B => "1.7B",
        }
    }

    pub fn model_id(self) -> &'static str {
        match self {
            Self::Small0_6B => "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            Self::Large1_7B => "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        }
    }

    pub fn download_size_label(self) -> &'static str {
        match self {
            Self::Small0_6B => "2.4 GB",
            Self::Large1_7B => "4.5 GB",
        }
    }

    fn cache_folder(self) -> &'static str {
        match self {
            Self::Small0_6B => "qwen3-tts-12hz-0.6b-customvoice",
            Self::Large1_7B => "qwen3-tts-12hz-1.7b-customvoice",
        }
    }

    fn hugging_face_repo_folder(self) -> &'static str {
        match self {
            Self::Small0_6B => "models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots",
            Self::Large1_7B => "models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots",
        }
    }

    fn minimum_main_model_size(self) -> u64 {
        match self {
            Self::Small0_6B => 1_500_000_000,
            Self::Large1_7B => 3_500_000_000,
        }
    }

    fn other(self) -> Self {
        match self {
            Self::Small0_6B => Self::Large1_7B,
            Self::Large1_7B => Self::Small0_6B,
        }
    }
}

pub struct DownloadProgress {
    pub file: &'static str,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenVoice {
    Vivian,
    Serena,
    UncleFu,
    Dylan,
    Eric,
    Ryan,
    Aiden,
    OnoAnna,
    Sohee,
}

impl QwenVoice {
    pub const ALL: [Self; 9] = [
        Self::Vivian,
        Self::Serena,
        Self::UncleFu,
        Self::Dylan,
        Self::Eric,
        Self::Ryan,
        Self::Aiden,
        Self::OnoAnna,
        Self::Sohee,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::Vivian => "Vivian",
            Self::Serena => "Serena",
            Self::UncleFu => "Uncle_Fu",
            Self::Dylan => "Dylan",
            Self::Eric => "Eric",
            Self::Ryan => "Ryan",
            Self::Aiden => "Aiden",
            Self::OnoAnna => "Ono_Anna",
            Self::Sohee => "Sohee",
        }
    }

    pub fn chinese_label(self) -> &'static str {
        match self {
            Self::Vivian => "Vivian · 明亮女声 · 普通话",
            Self::Serena => "Serena · 温柔女声 · 普通话",
            Self::UncleFu => "Uncle Fu · 沉稳男声 · 普通话",
            Self::Dylan => "Dylan · 北京男声 · 北京话",
            Self::Eric => "Eric · 活力男声 · 四川话",
            Self::Ryan => "Ryan · 动感男声 · 英语",
            Self::Aiden => "Aiden · 阳光男声 · 美式英语",
            Self::OnoAnna => "Ono Anna · 活泼女声 · 日语",
            Self::Sohee => "Sohee · 温暖女声 · 韩语",
        }
    }

    pub fn english_label(self) -> &'static str {
        match self {
            Self::Vivian => "Vivian · Bright female · Chinese",
            Self::Serena => "Serena · Warm female · Chinese",
            Self::UncleFu => "Uncle Fu · Mellow male · Chinese",
            Self::Dylan => "Dylan · Young male · Beijing dialect",
            Self::Eric => "Eric · Lively male · Sichuan dialect",
            Self::Ryan => "Ryan · Dynamic male · English",
            Self::Aiden => "Aiden · Sunny male · American English",
            Self::OnoAnna => "Ono Anna · Playful female · Japanese",
            Self::Sohee => "Sohee · Warm female · Korean",
        }
    }

    pub fn matches(self, filter: &str) -> bool {
        let query = filter.trim().to_lowercase();
        query.is_empty()
            || self.id().to_lowercase().contains(&query)
            || self.chinese_label().to_lowercase().contains(&query)
            || self.english_label().to_lowercase().contains(&query)
    }

    fn speaker(self) -> Speaker {
        match self {
            Self::Vivian => Speaker::Vivian,
            Self::Serena => Speaker::Serena,
            Self::UncleFu => Speaker::UncleFu,
            Self::Dylan => Speaker::Dylan,
            Self::Eric => Speaker::Eric,
            Self::Ryan => Speaker::Ryan,
            Self::Aiden => Speaker::Aiden,
            Self::OnoAnna => Speaker::OnoAnna,
            Self::Sohee => Speaker::Sohee,
        }
    }

    fn default_language(self) -> Language {
        match self {
            Self::OnoAnna => Language::Japanese,
            Self::Sohee => Language::Korean,
            Self::Ryan | Self::Aiden => Language::English,
            _ => Language::Chinese,
        }
    }
}

pub struct LocalQwenModel {
    model: Qwen3TTS,
    version: QwenModelVersion,
    device_label: String,
}

impl LocalQwenModel {
    /// Downloads missing official weights into the application cache, then
    /// loads them with Candle. Calling this on the dedicated worker thread keeps
    /// both the synchronous download and model initialization away from egui.
    pub fn load(
        version: QwenModelVersion,
        mut on_progress: impl FnMut(DownloadProgress),
    ) -> Result<Self, String> {
        let model_dir = prepare_model_files(version, &mut on_progress)?;
        let device = qwen3_tts::auto_device()
            .map_err(|error| format!("无法初始化 Qwen3-TTS 推理设备：{error:#}"))?;
        let device_label = if device.is_metal() {
            "Apple Metal".to_owned()
        } else if device.is_cuda() {
            "NVIDIA CUDA".to_owned()
        } else {
            "CPU".to_owned()
        };
        let model_path = model_dir
            .to_str()
            .ok_or_else(|| "Qwen3-TTS 模型缓存路径不是有效的 Unicode。".to_owned())?;
        let model = Qwen3TTS::from_pretrained(model_path, device)
            .map_err(|error| format!("无法加载 Qwen3-TTS 模型：{error:#}"))?;
        Ok(Self {
            model,
            version,
            device_label,
        })
    }

    pub fn version(&self) -> QwenModelVersion {
        self.version
    }

    pub fn device_label(&self) -> &str {
        &self.device_label
    }

    pub fn synthesize(&self, text: &str, voice: QwenVoice) -> Result<AudioBuffer, String> {
        let language = detect_language(text, voice);
        let options = SynthesisOptions {
            // 2048 frames is about 2m40s at 12.5 Hz. Long input is split by the
            // caller, so this is a generous per-segment safety ceiling.
            max_length: 2048,
            ..SynthesisOptions::default()
        };
        self.model
            .synthesize_with_voice(text, voice.speaker(), language, Some(options))
            .map_err(|error| format!("Qwen3-TTS 本地合成失败：{error:#}"))
    }
}

struct ModelFile {
    label: &'static str,
    repository: &'static str,
    remote_path: &'static str,
    local_path: &'static str,
    minimum_size: u64,
}

fn prepare_model_files(
    version: QwenModelVersion,
    on_progress: &mut impl FnMut(DownloadProgress),
) -> Result<PathBuf, String> {
    let project_dirs = ProjectDirs::from("com", "Aura Labs", "Edge TTS Studio")
        .ok_or_else(|| "无法确定 Qwen3-TTS 模型缓存目录。".to_owned())?;
    let models_dir = project_dirs.cache_dir().join("models");
    let model_dir = models_dir.join(version.cache_folder());
    std::fs::create_dir_all(&model_dir)
        .map_err(|error| format!("无法创建 Qwen3-TTS 模型缓存目录：{error}"))?;

    reuse_hugging_face_main_model(&model_dir, version);
    reuse_shared_assets(&models_dir, &model_dir, version);
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(30 * 60))
        .user_agent("Edge-TTS-Studio/0.1 Qwen3-model-downloader")
        .build()
        .map_err(|error| format!("无法初始化 Qwen3-TTS 下载器：{error}"))?;

    let model_files = [
        ModelFile {
            label: "main-model",
            repository: version.model_id(),
            remote_path: "model.safetensors",
            local_path: "model.safetensors",
            minimum_size: version.minimum_main_model_size(),
        },
        ModelFile {
            label: "model-config",
            repository: version.model_id(),
            remote_path: "config.json",
            local_path: "config.json",
            minimum_size: 100,
        },
        ModelFile {
            label: "audio-decoder",
            repository: "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            remote_path: "model.safetensors",
            local_path: "speech_tokenizer/model.safetensors",
            minimum_size: 500_000_000,
        },
        ModelFile {
            label: "text-tokenizer",
            repository: "Qwen/Qwen2-0.5B",
            remote_path: "tokenizer.json",
            local_path: "tokenizer.json",
            minimum_size: 1_000_000,
        },
    ];

    for file in &model_files {
        let destination = model_dir.join(file.local_path);
        download_file(&client, file, &destination, on_progress)?;
    }
    Ok(model_dir)
}

fn reuse_hugging_face_main_model(model_dir: &Path, version: QwenModelVersion) {
    let destination = model_dir.join("model.safetensors");
    if valid_file(&destination, version.minimum_main_model_size()) {
        return;
    }
    let Some(base_dirs) = BaseDirs::new() else {
        return;
    };
    let snapshots = base_dirs
        .home_dir()
        .join(".cache/huggingface/hub")
        .join(version.hugging_face_repo_folder());
    let Ok(entries) = std::fs::read_dir(snapshots) else {
        return;
    };
    let Some(existing) = entries
        .flatten()
        .map(|entry| entry.path().join("model.safetensors"))
        .find(|path| valid_file(path, version.minimum_main_model_size()))
        .and_then(|path| std::fs::canonicalize(path).ok())
    else {
        return;
    };
    try_reuse_file(&existing, &destination, version.minimum_main_model_size());
}

fn reuse_shared_assets(models_dir: &Path, model_dir: &Path, version: QwenModelVersion) {
    let other_dir = models_dir.join(version.other().cache_folder());
    for (relative_path, minimum_size) in [
        ("speech_tokenizer/model.safetensors", 500_000_000),
        ("tokenizer.json", 1_000_000),
    ] {
        try_reuse_file(
            &other_dir.join(relative_path),
            &model_dir.join(relative_path),
            minimum_size,
        );
    }
}

fn try_reuse_file(source: &Path, destination: &Path, minimum_size: u64) {
    if valid_file(destination, minimum_size) || !valid_file(source, minimum_size) {
        return;
    }
    if let Some(parent) = destination.parent()
        && std::fs::create_dir_all(parent).is_err()
    {
        return;
    }
    let _ = std::fs::remove_file(destination);
    // APFS hard links let the two selectable model caches share the decoder
    // and tokenizer without consuming another ~660 MB. A failure is harmless;
    // the resumable downloader below will fetch an independent copy.
    let _ = std::fs::hard_link(source, destination);
}

fn download_file(
    client: &reqwest::blocking::Client,
    file: &ModelFile,
    destination: &Path,
    on_progress: &mut impl FnMut(DownloadProgress),
) -> Result<(), String> {
    if valid_file(destination, file.minimum_size) {
        let size = std::fs::metadata(destination)
            .map(|meta| meta.len())
            .unwrap_or(0);
        on_progress(DownloadProgress {
            file: file.label,
            downloaded_bytes: size,
            total_bytes: size,
        });
        return Ok(());
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("无法创建 {} 的缓存目录：{error}", file.label))?;
    }
    let part_path = destination.with_extension("download");
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        file.repository, file.remote_path
    );
    let mut last_error = String::new();

    for attempt in 1..=3 {
        let existing_bytes = std::fs::metadata(&part_path)
            .map(|meta| meta.len())
            .unwrap_or(0);
        let mut request = client.get(&url);
        if existing_bytes > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={existing_bytes}-"));
        }
        let mut response = match request.send() {
            Ok(response) => response,
            Err(error) => {
                last_error = error.to_string();
                std::thread::sleep(std::time::Duration::from_secs(attempt * 2));
                continue;
            }
        };
        let status = response.status();
        if !status.is_success() {
            last_error = format!("HTTP {status}");
            std::thread::sleep(std::time::Duration::from_secs(attempt * 2));
            continue;
        }
        let resumed = status == reqwest::StatusCode::PARTIAL_CONTENT && existing_bytes > 0;
        let base = if resumed { existing_bytes } else { 0 };
        let total = response
            .content_length()
            .map(|length| length.saturating_add(base))
            .unwrap_or(0);
        let mut output = if resumed {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&part_path)
        } else {
            std::fs::File::create(&part_path)
        }
        .map_err(|error| format!("无法写入 {}：{error}", file.label))?;
        let mut downloaded = base;
        let mut last_reported = base.saturating_sub(8 * 1024 * 1024);
        let mut buffer = vec![0_u8; 256 * 1024];
        let mut stream_error = None;
        loop {
            let bytes_read = match response.read(&mut buffer) {
                Ok(0) => break,
                Ok(bytes_read) => bytes_read,
                Err(error) => {
                    stream_error = Some(error.to_string());
                    break;
                }
            };
            output
                .write_all(&buffer[..bytes_read])
                .map_err(|error| format!("无法写入 {}：{error}", file.label))?;
            downloaded = downloaded.saturating_add(bytes_read as u64);
            if downloaded.saturating_sub(last_reported) >= 8 * 1024 * 1024
                || (total > 0 && downloaded >= total)
            {
                on_progress(DownloadProgress {
                    file: file.label,
                    downloaded_bytes: downloaded,
                    total_bytes: total,
                });
                last_reported = downloaded;
            }
        }
        output
            .flush()
            .map_err(|error| format!("无法完成 {} 写入：{error}", file.label))?;
        drop(output);
        if let Some(error) = stream_error {
            last_error = error;
            std::thread::sleep(std::time::Duration::from_secs(attempt * 2));
            continue;
        }
        if total > 0 && downloaded < total {
            last_error = format!("下载不完整：{downloaded}/{total} 字节");
            continue;
        }
        if downloaded < file.minimum_size {
            last_error = format!("文件异常小：{downloaded} 字节");
            let _ = std::fs::remove_file(&part_path);
            continue;
        }
        let _ = std::fs::remove_file(destination);
        std::fs::rename(&part_path, destination)
            .map_err(|error| format!("无法保存 {}：{error}", file.label))?;
        return Ok(());
    }
    Err(format!(
        "下载 {} 失败（已重试 3 次）：{last_error}",
        file.label
    ))
}

fn valid_file(path: &Path, minimum_size: u64) -> bool {
    std::fs::metadata(path)
        .map(|metadata| metadata.is_file() && metadata.len() >= minimum_size)
        .unwrap_or(false)
}

fn detect_language(text: &str, voice: QwenVoice) -> Language {
    if text.chars().any(is_japanese_kana) {
        Language::Japanese
    } else if text.chars().any(is_hangul) {
        Language::Korean
    } else if text.chars().any(is_han) {
        // Qwen handles Chinese sentences containing English words best when the
        // language token remains Chinese instead of switching per word.
        Language::Chinese
    } else {
        voice.default_language()
    }
}

fn is_han(character: char) -> bool {
    matches!(character as u32, 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF)
}

fn is_japanese_kana(character: char) -> bool {
    matches!(character as u32, 0x3040..=0x30FF | 0x31F0..=0x31FF)
}

fn is_hangul(character: char) -> bool {
    matches!(character as u32, 0x1100..=0x11FF | 0x3130..=0x318F | 0xAC00..=0xD7AF)
}

/// Keep local generation stable for long pasted articles. Splits prefer hard
/// line breaks and sentence punctuation, with a character-count fallback.
pub fn split_for_synthesis(text: &str) -> Vec<String> {
    const MAX_CHARS: usize = 260;
    let mut chunks = Vec::new();
    let mut current = String::new();

    for character in text.trim().chars() {
        current.push(character);
        let sentence_end = matches!(
            character,
            '。' | '！' | '？' | '；' | '.' | '!' | '?' | ';' | '\n'
        );
        if (sentence_end && current.chars().count() >= 40) || current.chars().count() >= MAX_CHARS {
            push_nonempty(&mut chunks, &mut current);
        }
    }
    push_nonempty(&mut chunks, &mut current);
    chunks
}

fn push_nonempty(chunks: &mut Vec<String>, current: &mut String) {
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        chunks.push(trimmed.to_owned());
    }
    current.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_chinese_and_english_uses_chinese_language_token() {
        assert_eq!(
            detect_language("欢迎使用 Qwen3 TTS studio", QwenVoice::Ryan),
            Language::Chinese
        );
    }

    #[test]
    fn speaker_language_is_used_for_latin_only_text() {
        assert_eq!(
            detect_language("Hello from Qwen", QwenVoice::Ryan),
            Language::English
        );
        assert_eq!(
            detect_language("Hello from Qwen", QwenVoice::OnoAnna),
            Language::Japanese
        );
    }

    #[test]
    fn long_text_is_split_without_losing_content() {
        let source = "第一句。".repeat(100);
        let chunks = split_for_synthesis(&source);
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat(), source);
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 260));
    }

    #[test]
    fn model_versions_have_independent_ids_and_cache_sizes() {
        assert_ne!(
            QwenModelVersion::Small0_6B.model_id(),
            QwenModelVersion::Large1_7B.model_id()
        );
        assert!(
            QwenModelVersion::Large1_7B.minimum_main_model_size()
                > QwenModelVersion::Small0_6B.minimum_main_model_size()
        );
        assert_eq!(QwenModelVersion::DEFAULT, QwenModelVersion::Small0_6B);
    }
}
