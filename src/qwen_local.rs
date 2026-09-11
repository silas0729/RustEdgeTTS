use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
};

use directories::{BaseDirs, ProjectDirs};
use qwen3_tts::{AudioBuffer, Language, Qwen3TTS, Speaker, SynthesisOptions, VoiceClonePrompt};

use crate::download_control::{DownloadControl, model_download_control};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenModelKind {
    CustomVoice,
    VoiceClone,
}

impl QwenModelKind {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::CustomVoice => "CustomVoice",
            Self::VoiceClone => "Base",
        }
    }

    fn config_type(self) -> &'static str {
        match self {
            Self::CustomVoice => "custom_voice",
            Self::VoiceClone => "base",
        }
    }
}

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

    pub fn model_id(self, kind: QwenModelKind) -> &'static str {
        match (self, kind) {
            (Self::Small0_6B, QwenModelKind::CustomVoice) => "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            (Self::Large1_7B, QwenModelKind::CustomVoice) => "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            (Self::Small0_6B, QwenModelKind::VoiceClone) => "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            (Self::Large1_7B, QwenModelKind::VoiceClone) => "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        }
    }

    pub fn model_folder_name(self, kind: QwenModelKind) -> &'static str {
        match (self, kind) {
            (Self::Small0_6B, QwenModelKind::CustomVoice) => "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            (Self::Large1_7B, QwenModelKind::CustomVoice) => "Qwen3-TTS-12Hz-1.7B-CustomVoice",
            (Self::Small0_6B, QwenModelKind::VoiceClone) => "Qwen3-TTS-12Hz-0.6B-Base",
            (Self::Large1_7B, QwenModelKind::VoiceClone) => "Qwen3-TTS-12Hz-1.7B-Base",
        }
    }

    pub fn hugging_face_url(self, kind: QwenModelKind) -> &'static str {
        match (self, kind) {
            (Self::Small0_6B, QwenModelKind::CustomVoice) => {
                "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
            }
            (Self::Large1_7B, QwenModelKind::CustomVoice) => {
                "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            }
            (Self::Small0_6B, QwenModelKind::VoiceClone) => {
                "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            }
            (Self::Large1_7B, QwenModelKind::VoiceClone) => {
                "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            }
        }
    }

    pub fn model_scope_url(self, kind: QwenModelKind) -> &'static str {
        match (self, kind) {
            (Self::Small0_6B, QwenModelKind::CustomVoice) => {
                "https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
            }
            (Self::Large1_7B, QwenModelKind::CustomVoice) => {
                "https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            }
            (Self::Small0_6B, QwenModelKind::VoiceClone) => {
                "https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            }
            (Self::Large1_7B, QwenModelKind::VoiceClone) => {
                "https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            }
        }
    }

    pub fn download_size_label(self) -> &'static str {
        match self {
            Self::Small0_6B => "2.4 GB",
            Self::Large1_7B => "4.5 GB",
        }
    }

    fn cache_folder(self, kind: QwenModelKind) -> &'static str {
        match (self, kind) {
            (Self::Small0_6B, QwenModelKind::CustomVoice) => "qwen3-tts-12hz-0.6b-customvoice",
            (Self::Large1_7B, QwenModelKind::CustomVoice) => "qwen3-tts-12hz-1.7b-customvoice",
            (Self::Small0_6B, QwenModelKind::VoiceClone) => "qwen3-tts-12hz-0.6b-base",
            (Self::Large1_7B, QwenModelKind::VoiceClone) => "qwen3-tts-12hz-1.7b-base",
        }
    }

    fn hugging_face_repo_folder(self, kind: QwenModelKind) -> &'static str {
        match (self, kind) {
            (Self::Small0_6B, QwenModelKind::CustomVoice) => {
                "models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots"
            }
            (Self::Large1_7B, QwenModelKind::CustomVoice) => {
                "models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots"
            }
            (Self::Small0_6B, QwenModelKind::VoiceClone) => {
                "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots"
            }
            (Self::Large1_7B, QwenModelKind::VoiceClone) => {
                "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots"
            }
        }
    }

    fn minimum_main_model_size(self) -> u64 {
        match self {
            Self::Small0_6B => 1_500_000_000,
            Self::Large1_7B => 3_500_000_000,
        }
    }

    fn expected_hidden_size(self) -> u64 {
        match self {
            Self::Small0_6B => 1024,
            Self::Large1_7B => 2048,
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

/// Generation controls exposed by the Qwen3-TTS local model.
///
/// The values mirror `speakers-qwen3-tts::SynthesisOptions`; validation is
/// performed at the boundary so malformed UI values can never allocate an
/// unbounded Metal KV cache or pass NaN into the sampler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QwenGenerationSettings {
    pub max_length: usize,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f64,
    pub min_new_tokens: usize,
    pub seed: Option<u64>,
}

impl Default for QwenGenerationSettings {
    fn default() -> Self {
        let defaults = SynthesisOptions::default();
        Self {
            // The app historically sized this cache to the current text and
            // capped long requests at 1024 frames to keep Metal memory cool.
            // Users can raise the value explicitly in the advanced panel.
            max_length: 1_024,
            temperature: defaults.temperature,
            top_k: defaults.top_k,
            top_p: defaults.top_p,
            repetition_penalty: defaults.repetition_penalty,
            min_new_tokens: defaults.min_new_tokens,
            // Keep the app's historical deterministic default while still
            // exposing a random-seed toggle in the UI.
            seed: Some(42),
        }
    }
}

impl QwenGenerationSettings {
    pub fn validated(self) -> Self {
        Self {
            max_length: self.max_length.clamp(128, 4_096),
            temperature: finite_clamp(self.temperature, 0.0, 2.0, 0.9),
            top_k: self.top_k.clamp(1, 200),
            top_p: finite_clamp(self.top_p, 0.01, 1.0, 0.9),
            repetition_penalty: finite_clamp(self.repetition_penalty, 0.5, 3.0, 1.05),
            min_new_tokens: self.min_new_tokens.clamp(0, 64),
            seed: self.seed,
        }
    }

    fn options_for(self, _text: &str) -> SynthesisOptions {
        let settings = self.validated();
        SynthesisOptions {
            max_length: settings
                .max_length
                .max(settings.min_new_tokens.saturating_add(1).max(128)),
            temperature: settings.temperature,
            top_k: settings.top_k,
            top_p: settings.top_p,
            repetition_penalty: settings.repetition_penalty,
            min_new_tokens: settings.min_new_tokens,
            seed: settings.seed,
            ..SynthesisOptions::default()
        }
    }
}

fn finite_clamp(value: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
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

/// The language token is chosen once for an entire preview, article, or
/// subtitle track. Keeping it fixed prevents the same preset speaker from
/// changing accent/timbre when adjacent cues alternate between Chinese and
/// English.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenSynthesisLanguage {
    Chinese,
    English,
    Japanese,
    Korean,
}

impl QwenSynthesisLanguage {
    fn engine_language(self) -> Language {
        match self {
            Self::Chinese => Language::Chinese,
            Self::English => Language::English,
            Self::Japanese => Language::Japanese,
            Self::Korean => Language::Korean,
        }
    }
}

pub struct LocalQwenModel {
    model: Qwen3TTS,
    version: QwenModelVersion,
    kind: QwenModelKind,
    device_label: String,
}

pub struct LocalVoiceClonePrompt {
    prompt: VoiceClonePrompt,
}

impl LocalQwenModel {
    /// Downloads missing official weights into the application cache, then
    /// loads them with Candle. Calling this on the dedicated worker thread keeps
    /// both the synchronous download and model initialization away from egui.
    pub fn load(
        version: QwenModelVersion,
        kind: QwenModelKind,
        on_progress: impl FnMut(DownloadProgress),
    ) -> Result<Self, String> {
        Self::load_internal(version, kind, true, on_progress)
    }

    pub fn redownload(
        version: QwenModelVersion,
        kind: QwenModelKind,
        on_progress: impl FnMut(DownloadProgress),
    ) -> Result<Self, String> {
        let model_dir = model_directory(version, kind)?;
        if model_dir.exists() {
            std::fs::remove_dir_all(&model_dir)
                .map_err(|error| format!("无法清理所选 Qwen3-TTS 模型缓存：{error}"))?;
        }
        Self::load_internal(version, kind, false, on_progress)
    }

    fn load_internal(
        version: QwenModelVersion,
        kind: QwenModelKind,
        reuse_existing: bool,
        mut on_progress: impl FnMut(DownloadProgress),
    ) -> Result<Self, String> {
        let control = model_download_control();
        let model_dir = {
            let _download_session = control.begin();
            prepare_model_files(version, kind, reuse_existing, control, &mut on_progress)?
        };
        // Downloading and loading are separate phases. Report the latter as an
        // indeterminate step so the UI does not leave a completed 100% download
        // bar on screen while Candle is still mapping the model weights.
        on_progress(DownloadProgress {
            file: "model-loading",
            downloaded_bytes: 0,
            total_bytes: 0,
        });
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
            kind,
            device_label,
        })
    }

    pub fn version(&self) -> QwenModelVersion {
        self.version
    }

    pub fn device_label(&self) -> &str {
        &self.device_label
    }

    pub fn kind(&self) -> QwenModelKind {
        self.kind
    }

    #[allow(dead_code)]
    pub fn synthesize(
        &self,
        text: &str,
        voice: QwenVoice,
        language: QwenSynthesisLanguage,
    ) -> Result<AudioBuffer, String> {
        self.synthesize_with_settings(text, voice, language, QwenGenerationSettings::default())
    }

    pub fn synthesize_with_settings(
        &self,
        text: &str,
        voice: QwenVoice,
        language: QwenSynthesisLanguage,
        settings: QwenGenerationSettings,
    ) -> Result<AudioBuffer, String> {
        let mut options = settings.options_for(text);
        let first_attempt = self.model.synthesize_with_voice(
            text,
            voice.speaker(),
            language.engine_language(),
            Some(options.clone()),
        );
        match first_attempt {
            Ok(audio) => Ok(audio),
            Err(error) if is_metal_buffer_error(&format!("{error:#}")) => {
                // A failed large allocation does not make the loaded model
                // unusable. Retry once with the smallest still-practical cache;
                // the normal estimate includes roughly 2x duration headroom.
                options.max_length = (options.max_length / 2).max(128);
                self.model
                    .synthesize_with_voice(
                        text,
                        voice.speaker(),
                        language.engine_language(),
                        Some(options),
                    )
                    .map_err(|retry_error| {
                        format!(
                            "Qwen3-TTS 本地合成失败：Apple Metal 内存不足，缩小缓存重试后仍失败。请关闭占用内存较大的程序，或改用 Qwen 0.6B 后重试。详情：{retry_error:#}"
                        )
                    })
            }
            Err(error) => Err(format!("Qwen3-TTS 本地合成失败：{error:#}")),
        }
    }

    pub fn create_voice_clone_prompt(
        &self,
        reference_audio: &AudioBuffer,
        reference_text: Option<&str>,
    ) -> Result<LocalVoiceClonePrompt, String> {
        if self.kind != QwenModelKind::VoiceClone || !self.model.supports_voice_cloning() {
            return Err("当前加载的不是 Qwen3-TTS Base 音色克隆模型。".to_owned());
        }
        self.model
            .create_voice_clone_prompt(reference_audio, reference_text)
            .map(|prompt| LocalVoiceClonePrompt { prompt })
            .map_err(|error| format!("无法从参考音频创建克隆提示：{error:#}"))
    }

    #[allow(dead_code)]
    pub fn synthesize_voice_clone(
        &self,
        text: &str,
        prompt: &LocalVoiceClonePrompt,
        language: QwenSynthesisLanguage,
    ) -> Result<AudioBuffer, String> {
        self.synthesize_voice_clone_with_settings(
            text,
            prompt,
            language,
            QwenGenerationSettings::default(),
        )
    }

    pub fn synthesize_voice_clone_with_settings(
        &self,
        text: &str,
        prompt: &LocalVoiceClonePrompt,
        language: QwenSynthesisLanguage,
        settings: QwenGenerationSettings,
    ) -> Result<AudioBuffer, String> {
        let mut options = settings.options_for(text);
        let first_attempt = self.model.synthesize_voice_clone(
            text,
            &prompt.prompt,
            language.engine_language(),
            Some(options.clone()),
        );
        match first_attempt {
            Ok(audio) => Ok(audio),
            Err(error) if is_metal_buffer_error(&format!("{error:#}")) => {
                options.max_length = (options.max_length / 2).max(128);
                self.model
                    .synthesize_voice_clone(
                        text,
                        &prompt.prompt,
                        language.engine_language(),
                        Some(options),
                    )
                    .map_err(|retry_error| {
                        format!(
                            "Qwen3-TTS 音色克隆失败：Apple Metal 内存不足，缩小缓存重试后仍失败。请关闭占用内存较大的程序，或改用 0.6B Base。详情：{retry_error:#}"
                        )
                    })
            }
            Err(error) => Err(format!("Qwen3-TTS 音色克隆失败：{error:#}")),
        }
    }
}

fn is_metal_buffer_error(error: &str) -> bool {
    let error = error.to_ascii_lowercase();
    error.contains("metal") && error.contains("buffer")
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
    kind: QwenModelKind,
    reuse_existing: bool,
    control: &DownloadControl,
    on_progress: &mut impl FnMut(DownloadProgress),
) -> Result<PathBuf, String> {
    let project_dirs = ProjectDirs::from("com", "Aura Labs", "Edge TTS Studio")
        .ok_or_else(|| "无法确定 Qwen3-TTS 模型缓存目录。".to_owned())?;
    let models_dir = project_dirs.cache_dir().join("models");
    let model_dir = models_dir.join(version.cache_folder(kind));
    std::fs::create_dir_all(&model_dir)
        .map_err(|error| format!("无法创建 Qwen3-TTS 模型缓存目录：{error}"))?;

    if reuse_existing {
        reuse_hugging_face_main_model(&model_dir, version, kind);
        reuse_shared_assets(&models_dir, &model_dir, version, kind);
    }
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(30 * 60))
        .user_agent("Edge-TTS-Studio/0.1 Qwen3-model-downloader")
        .build()
        .map_err(|error| format!("无法初始化 Qwen3-TTS 下载器：{error}"))?;

    let model_files = [
        ModelFile {
            label: "main-model",
            repository: version.model_id(kind),
            remote_path: "model.safetensors",
            local_path: "model.safetensors",
            minimum_size: version.minimum_main_model_size(),
        },
        ModelFile {
            label: "model-config",
            repository: version.model_id(kind),
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
        control.checkpoint()?;
        if file.label == "text-tokenizer" && has_usable_text_tokenizer(&model_dir) {
            continue;
        }
        let destination = model_dir.join(file.local_path);
        download_file(&client, file, &destination, control, on_progress)?;
    }
    validate_ready_model_dir(&model_dir, version, kind)?;
    Ok(model_dir)
}

/// Installs a complete model folder selected by the user into the app cache.
/// Files are hard-linked when possible, otherwise copied with progress. This
/// makes future launches independent from the originally selected directory.
pub fn import_offline_model(
    version: QwenModelVersion,
    kind: QwenModelKind,
    selected_dir: &Path,
    mut on_progress: impl FnMut(DownloadProgress),
) -> Result<PathBuf, String> {
    let source_dir = locate_model_source(selected_dir, version, kind)?;
    validate_model_config(&source_dir.join("config.json"), version, kind)?;

    let project_dirs = ProjectDirs::from("com", "Aura Labs", "Edge TTS Studio")
        .ok_or_else(|| "无法确定 Qwen3-TTS 模型缓存目录。".to_owned())?;
    let models_dir = project_dirs.cache_dir().join("models");
    let destination_dir = models_dir.join(version.cache_folder(kind));
    std::fs::create_dir_all(&destination_dir)
        .map_err(|error| format!("无法创建本地模型目录：{error}"))?;

    import_file(
        &source_dir.join("model.safetensors"),
        &destination_dir.join("model.safetensors"),
        "main-model",
        version.minimum_main_model_size(),
        &mut on_progress,
    )?;
    import_file(
        &source_dir.join("config.json"),
        &destination_dir.join("config.json"),
        "model-config",
        100,
        &mut on_progress,
    )?;

    let source_decoder = source_dir.join("speech_tokenizer/model.safetensors");
    if valid_file(&source_decoder, 500_000_000) {
        import_file(
            &source_decoder,
            &destination_dir.join("speech_tokenizer/model.safetensors"),
            "audio-decoder",
            500_000_000,
            &mut on_progress,
        )?;
    }

    let source_tokenizer = source_dir.join("tokenizer.json");
    if valid_file(&source_tokenizer, 1_000_000) {
        import_file(
            &source_tokenizer,
            &destination_dir.join("tokenizer.json"),
            "text-tokenizer",
            1_000_000,
            &mut on_progress,
        )?;
    } else {
        for (name, minimum_size) in [
            ("vocab.json", 100_000),
            ("merges.txt", 100_000),
            ("tokenizer_config.json", 100),
        ] {
            let source = source_dir.join(name);
            if valid_file(&source, minimum_size) {
                import_file(
                    &source,
                    &destination_dir.join(name),
                    "text-tokenizer",
                    minimum_size,
                    &mut on_progress,
                )?;
            }
        }
    }

    reuse_shared_assets(&models_dir, &destination_dir, version, kind);
    validate_ready_model_dir(&destination_dir, version, kind)?;
    Ok(destination_dir)
}

fn locate_model_source(
    selected_dir: &Path,
    version: QwenModelVersion,
    kind: QwenModelKind,
) -> Result<PathBuf, String> {
    let direct = selected_dir.to_path_buf();
    let nested = selected_dir.join(version.model_folder_name(kind));
    let source = if direct.join("model.safetensors").is_file() {
        direct
    } else if nested.join("model.safetensors").is_file() {
        nested
    } else {
        return Err(format!(
            "所选文件夹中没有找到 {} 的 model.safetensors。请选择完整模型文件夹，而不是单个文件。",
            version.model_folder_name(kind)
        ));
    };
    Ok(source)
}

fn validate_model_config(
    config_path: &Path,
    version: QwenModelVersion,
    kind: QwenModelKind,
) -> Result<(), String> {
    let bytes =
        std::fs::read(config_path).map_err(|error| format!("无法读取模型 config.json：{error}"))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("模型 config.json 格式无效：{error}"))?;
    validate_model_config_value(&value, version, kind)
}

fn validate_model_config_value(
    value: &serde_json::Value,
    version: QwenModelVersion,
    kind: QwenModelKind,
) -> Result<(), String> {
    let model_type = value
        .get("tts_model_type")
        .and_then(serde_json::Value::as_str);
    if model_type != Some(kind.config_type()) {
        return Err(format!(
            "所选目录不是 Qwen3-TTS {} 模型。",
            kind.short_label()
        ));
    }
    let hidden_size = value
        .pointer("/talker_config/hidden_size")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "config.json 缺少 talker_config.hidden_size。".to_owned())?;
    if hidden_size != version.expected_hidden_size() {
        let detected = if hidden_size == QwenModelVersion::Small0_6B.expected_hidden_size() {
            "0.6B"
        } else if hidden_size == QwenModelVersion::Large1_7B.expected_hidden_size() {
            "1.7B"
        } else {
            "未知版本"
        };
        return Err(format!(
            "模型版本不匹配：当前选择 {}，但文件夹内检测为 {detected}。",
            version.short_label()
        ));
    }
    Ok(())
}

fn validate_ready_model_dir(
    model_dir: &Path,
    version: QwenModelVersion,
    kind: QwenModelKind,
) -> Result<(), String> {
    validate_model_config(&model_dir.join("config.json"), version, kind)?;
    if !valid_file(
        &model_dir.join("model.safetensors"),
        version.minimum_main_model_size(),
    ) {
        return Err("主模型 model.safetensors 缺失或不完整。".to_owned());
    }
    if !valid_file(
        &model_dir.join("speech_tokenizer/model.safetensors"),
        500_000_000,
    ) {
        return Err(
            "缺少 speech_tokenizer/model.safetensors；请下载完整模型仓库后再导入。".to_owned(),
        );
    }
    if !has_usable_text_tokenizer(model_dir) {
        return Err(
            "缺少文本分词器；需要 tokenizer.json，或同时提供 vocab.json 与 merges.txt。".to_owned(),
        );
    }
    Ok(())
}

fn has_usable_text_tokenizer(model_dir: &Path) -> bool {
    valid_file(&model_dir.join("tokenizer.json"), 1_000_000)
        || (valid_file(&model_dir.join("vocab.json"), 100_000)
            && valid_file(&model_dir.join("merges.txt"), 100_000))
}

fn import_file(
    source: &Path,
    destination: &Path,
    label: &'static str,
    minimum_size: u64,
    on_progress: &mut impl FnMut(DownloadProgress),
) -> Result<(), String> {
    if !valid_file(source, minimum_size) {
        return Err(format!("{} 缺失或不完整：{}", label, source.display()));
    }
    let source_size = std::fs::metadata(source)
        .map_err(|error| format!("无法读取 {}：{error}", source.display()))?
        .len();
    if source == destination || valid_file(destination, minimum_size) {
        on_progress(DownloadProgress {
            file: label,
            downloaded_bytes: source_size,
            total_bytes: source_size,
        });
        return Ok(());
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("无法创建本地模型目录：{error}"))?;
    }
    let part_path = destination.with_extension("import");
    let _ = std::fs::remove_file(&part_path);
    let resolved_source = std::fs::canonicalize(source).unwrap_or_else(|_| source.to_path_buf());
    if std::fs::hard_link(&resolved_source, &part_path).is_err() {
        let mut input = std::fs::File::open(source)
            .map_err(|error| format!("无法打开 {}：{error}", source.display()))?;
        let mut output = std::fs::File::create(&part_path)
            .map_err(|error| format!("无法写入 {}：{error}", part_path.display()))?;
        let mut copied = 0_u64;
        let mut last_reported = 0_u64;
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let count = input
                .read(&mut buffer)
                .map_err(|error| format!("读取离线模型失败：{error}"))?;
            if count == 0 {
                break;
            }
            output
                .write_all(&buffer[..count])
                .map_err(|error| format!("复制离线模型失败：{error}"))?;
            copied = copied.saturating_add(count as u64);
            if copied.saturating_sub(last_reported) >= 8 * 1024 * 1024 || copied == source_size {
                on_progress(DownloadProgress {
                    file: label,
                    downloaded_bytes: copied,
                    total_bytes: source_size,
                });
                last_reported = copied;
            }
        }
        output
            .flush()
            .map_err(|error| format!("写入离线模型失败：{error}"))?;
    } else {
        on_progress(DownloadProgress {
            file: label,
            downloaded_bytes: source_size,
            total_bytes: source_size,
        });
    }
    if !valid_file(&part_path, minimum_size) {
        let _ = std::fs::remove_file(&part_path);
        return Err(format!("导入后的 {label} 不完整。"));
    }
    let _ = std::fs::remove_file(destination);
    std::fs::rename(&part_path, destination)
        .map_err(|error| format!("无法安装 {label}：{error}"))?;
    Ok(())
}

fn reuse_hugging_face_main_model(model_dir: &Path, version: QwenModelVersion, kind: QwenModelKind) {
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
        .join(version.hugging_face_repo_folder(kind));
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

fn reuse_shared_assets(
    models_dir: &Path,
    model_dir: &Path,
    version: QwenModelVersion,
    kind: QwenModelKind,
) {
    for candidate_version in [version, version.other()] {
        for candidate_kind in [QwenModelKind::CustomVoice, QwenModelKind::VoiceClone] {
            if candidate_version == version && candidate_kind == kind {
                continue;
            }
            let other_dir = models_dir.join(candidate_version.cache_folder(candidate_kind));
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
    control: &DownloadControl,
    on_progress: &mut impl FnMut(DownloadProgress),
) -> Result<(), String> {
    control.checkpoint()?;
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
        control.checkpoint()?;
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
            control.checkpoint()?;
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

pub fn models_directory() -> Result<PathBuf, String> {
    ProjectDirs::from("com", "Aura Labs", "Edge TTS Studio")
        .map(|project_dirs| project_dirs.cache_dir().join("models"))
        .ok_or_else(|| "无法确定 Qwen3-TTS 模型缓存目录。".to_owned())
}

fn model_directory(version: QwenModelVersion, kind: QwenModelKind) -> Result<PathBuf, String> {
    Ok(models_directory()?.join(version.cache_folder(kind)))
}

fn valid_file(path: &Path, minimum_size: u64) -> bool {
    std::fs::metadata(path)
        .map(|metadata| metadata.is_file() && metadata.len() >= minimum_size)
        .unwrap_or(false)
}

pub fn synthesis_language<'a>(
    texts: impl IntoIterator<Item = &'a str>,
    voice: QwenVoice,
) -> QwenSynthesisLanguage {
    let mut has_han = false;
    let mut has_kana = false;
    let mut has_hangul = false;
    let mut has_latin = false;

    for character in texts.into_iter().flat_map(str::chars) {
        has_han |= is_han(character);
        has_kana |= is_japanese_kana(character);
        has_hangul |= is_hangul(character);
        has_latin |= character.is_ascii_alphabetic();
    }

    // Kana/Hangul identify Japanese and Korean more specifically because both
    // languages may also contain Han characters. A track containing any Han
    // text is treated as Chinese for every cue, including its English-only
    // cues. A Latin-only track is always English regardless of speaker origin.
    if has_kana {
        QwenSynthesisLanguage::Japanese
    } else if has_hangul {
        QwenSynthesisLanguage::Korean
    } else if has_han {
        QwenSynthesisLanguage::Chinese
    } else if has_latin {
        QwenSynthesisLanguage::English
    } else {
        match voice.default_language() {
            Language::Japanese => QwenSynthesisLanguage::Japanese,
            Language::Korean => QwenSynthesisLanguage::Korean,
            Language::English => QwenSynthesisLanguage::English,
            _ => QwenSynthesisLanguage::Chinese,
        }
    }
}

/// Voice cloning has no preset speaker language to use as a fallback. The
/// complete job is still inspected once so mixed Chinese/English content uses
/// one stable language token for every chunk and subtitle cue.
pub fn synthesis_language_for_clone<'a>(
    texts: impl IntoIterator<Item = &'a str>,
) -> QwenSynthesisLanguage {
    synthesis_language_with_fallback(texts, QwenSynthesisLanguage::Chinese)
}

fn synthesis_language_with_fallback<'a>(
    texts: impl IntoIterator<Item = &'a str>,
    fallback: QwenSynthesisLanguage,
) -> QwenSynthesisLanguage {
    let mut has_han = false;
    let mut has_kana = false;
    let mut has_hangul = false;
    let mut has_latin = false;

    for character in texts.into_iter().flat_map(str::chars) {
        has_han |= is_han(character);
        has_kana |= is_japanese_kana(character);
        has_hangul |= is_hangul(character);
        has_latin |= character.is_ascii_alphabetic();
    }

    if has_kana {
        QwenSynthesisLanguage::Japanese
    } else if has_hangul {
        QwenSynthesisLanguage::Korean
    } else if has_han {
        QwenSynthesisLanguage::Chinese
    } else if has_latin {
        QwenSynthesisLanguage::English
    } else {
        fallback
    }
}

/// Decode a local voice-cloning reference. MP3 uses the same Rust decoder as
/// generated audio; WAV uses the Qwen crate's hound-backed loader. Nothing is
/// uploaded or copied outside the worker process.
pub fn load_reference_audio(path: &Path) -> Result<AudioBuffer, String> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let mut audio = match extension.as_str() {
        "wav" => {
            AudioBuffer::load(path).map_err(|error| format!("无法读取 WAV 参考音频：{error:#}"))?
        }
        "mp3" => {
            let bytes =
                std::fs::read(path).map_err(|error| format!("无法读取 MP3 参考音频：{error}"))?;
            let samples = crate::timeline_audio::decode_mp3_mono_preserving_silence(&bytes)
                .map_err(|error| format!("无法解码 MP3 参考音频：{error}"))?;
            AudioBuffer::new(samples, crate::timeline_audio::TIMELINE_SAMPLE_RATE)
        }
        _ => return Err("参考音频仅支持 WAV 或 MP3。".to_owned()),
    };

    if audio.sample_rate == 0 || audio.samples.is_empty() {
        return Err("参考音频没有可用的声音采样。".to_owned());
    }
    if audio.samples.iter().any(|sample| !sample.is_finite()) {
        return Err("参考音频包含无效采样，请重新导出为标准 WAV 或 MP3。".to_owned());
    }
    if audio.sample_rate != crate::timeline_audio::TIMELINE_SAMPLE_RATE {
        audio.samples = crate::timeline_audio::resample_linear(
            &audio.samples,
            audio.sample_rate,
            crate::timeline_audio::TIMELINE_SAMPLE_RATE,
        );
        audio.sample_rate = crate::timeline_audio::TIMELINE_SAMPLE_RATE;
    }
    if audio.samples.iter().all(|sample| sample.abs() < 0.0005) {
        return Err("参考音频几乎完全静音，请选择清晰的人声录音。".to_owned());
    }
    for sample in &mut audio.samples {
        *sample = sample.clamp(-1.0, 1.0);
    }
    let duration = audio.duration();
    if !(1.0..=60.0).contains(&duration) {
        return Err(format!(
            "参考音频时长为 {duration:.1} 秒；请选择 1–60 秒的单人清晰人声（推荐 3–15 秒）。"
        ));
    }
    Ok(audio)
}

#[allow(dead_code)]
fn synthesis_frame_budget(text: &str) -> usize {
    let cjk_characters = text
        .chars()
        .filter(|character| {
            is_han(*character) || is_japanese_kana(*character) || is_hangul(*character)
        })
        .count();
    let latin_words = text
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .count();
    let punctuation = text
        .chars()
        .filter(|character| {
            character.is_ascii_punctuation() || "，。！？；：、…".contains(*character)
        })
        .count();

    // Codec frames are roughly 80 ms. These deliberately conservative factors
    // allow substantially slower-than-normal speech while avoiding the fixed
    // 2048-frame allocation. Long article input is split by the caller.
    cjk_characters
        .saturating_mul(5)
        .saturating_add(latin_words.saturating_mul(10))
        .saturating_add(punctuation.saturating_mul(2))
        .saturating_add(64)
        .clamp(128, 1_024)
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
    fn mixed_chinese_and_english_track_uses_one_chinese_language_token() {
        assert_eq!(
            synthesis_language(
                ["欢迎使用 Qwen3 TTS studio", "Hello from the next cue"],
                QwenVoice::Ryan,
            ),
            QwenSynthesisLanguage::Chinese
        );
    }

    #[test]
    fn latin_only_track_stays_english_even_with_a_japanese_voice() {
        assert_eq!(
            synthesis_language(["Hello from Qwen"], QwenVoice::OnoAnna),
            QwenSynthesisLanguage::English
        );
    }

    #[test]
    fn short_subtitles_use_a_bounded_metal_frame_budget() {
        let short = synthesis_frame_budget("大家好，我是一名独立 iOS 开发者");
        let long = synthesis_frame_budget(&"中文语音。".repeat(100));

        assert!((128..512).contains(&short));
        assert_eq!(long, 1_024);
        assert!(short < SynthesisOptions::default().max_length);
    }

    #[test]
    fn generation_settings_are_finite_and_bounded() {
        let settings = QwenGenerationSettings {
            max_length: 0,
            temperature: f64::NAN,
            top_k: usize::MAX,
            top_p: f64::INFINITY,
            repetition_penalty: -10.0,
            min_new_tokens: usize::MAX,
            seed: None,
        }
        .validated();
        assert_eq!(settings.max_length, 128);
        assert_eq!(settings.temperature, 0.9);
        assert_eq!(settings.top_k, 200);
        assert_eq!(settings.top_p, 0.9);
        assert_eq!(settings.repetition_penalty, 0.5);
        assert_eq!(settings.min_new_tokens, 64);
    }

    #[test]
    fn custom_max_length_is_forwarded_to_generation_options() {
        let settings = QwenGenerationSettings {
            max_length: 2_048,
            ..QwenGenerationSettings::default()
        };
        assert_eq!(settings.options_for("短文本").max_length, 2_048);
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
            QwenModelVersion::Small0_6B.model_id(QwenModelKind::CustomVoice),
            QwenModelVersion::Large1_7B.model_id(QwenModelKind::CustomVoice)
        );
        assert!(
            QwenModelVersion::Small0_6B
                .model_id(QwenModelKind::VoiceClone)
                .ends_with("-Base")
        );
        assert!(
            QwenModelVersion::Large1_7B.minimum_main_model_size()
                > QwenModelVersion::Small0_6B.minimum_main_model_size()
        );
        assert_eq!(QwenModelVersion::DEFAULT, QwenModelVersion::Small0_6B);
    }

    #[test]
    fn offline_config_must_match_the_selected_custom_voice_version() {
        let small = serde_json::json!({
            "tts_model_type": "custom_voice",
            "talker_config": { "hidden_size": 1024 }
        });
        let large = serde_json::json!({
            "tts_model_type": "custom_voice",
            "talker_config": { "hidden_size": 2048 }
        });
        let base = serde_json::json!({
            "tts_model_type": "base",
            "talker_config": { "hidden_size": 1024 }
        });

        assert!(
            validate_model_config_value(
                &small,
                QwenModelVersion::Small0_6B,
                QwenModelKind::CustomVoice
            )
            .is_ok()
        );
        assert!(
            validate_model_config_value(
                &large,
                QwenModelVersion::Large1_7B,
                QwenModelKind::CustomVoice
            )
            .is_ok()
        );
        assert!(
            validate_model_config_value(
                &small,
                QwenModelVersion::Large1_7B,
                QwenModelKind::CustomVoice
            )
            .is_err()
        );
        assert!(
            validate_model_config_value(
                &base,
                QwenModelVersion::Small0_6B,
                QwenModelKind::VoiceClone
            )
            .is_ok()
        );
        assert!(
            validate_model_config_value(
                &base,
                QwenModelVersion::Small0_6B,
                QwenModelKind::CustomVoice
            )
            .is_err()
        );
    }

    #[test]
    fn clone_language_is_locked_for_a_mixed_track() {
        assert_eq!(
            synthesis_language_for_clone(["第一句中文", "The second cue is English"]),
            QwenSynthesisLanguage::Chinese
        );
    }
}
