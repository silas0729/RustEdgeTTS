#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use std::{
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant},
};

use directories::ProjectDirs;
use edge_tts_rust::{EdgeTtsClient, SpeakOptions};
use eframe::egui;
use rwhisper::{ModelLoadingProgress, Whisper};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

mod asr;
mod audio_player;
mod download_control;
mod indextts;
mod qwen_local;
mod subtitle_pipeline;
mod subtitles;
mod system_proxy;
mod timeline_audio;

use asr::{RecognitionLanguage, SubtitleExportFormat};
use audio_player::GeneratedAudioPlayer;
use download_control::{DownloadState, is_download_cancelled, model_download_control};
use indextts::{IndexTtsProgress, IndexTtsRuntime, IndexTtsSettings};
use qwen_local::{
    LocalQwenModel, LocalVoiceClonePrompt, QwenGenerationSettings, QwenModelKind, QwenModelVersion,
    QwenSynthesisLanguage, QwenVoice,
};
use subtitles::{SubtitleCue, SubtitleTrack, format_timestamp, parse_subtitle};

const APP_NAME: &str = "Edge TTS Studio";
const DEFAULT_TEXT: &str = "你好！欢迎使用 Edge TTS 语音工作室。\n\n请在这里输入或粘贴中英文文本，选择喜欢的音色，然后生成 MP3 音频。";
const PREFERRED_VOICE: &str = "zh-CN-XiaoxiaoNeural";

const BACKGROUND: egui::Color32 = egui::Color32::from_rgb(244, 247, 252);
const CARD_BACKGROUND: egui::Color32 = egui::Color32::from_rgb(255, 255, 255);
const EDITOR_BACKGROUND: egui::Color32 = egui::Color32::from_rgb(248, 250, 253);
const PRIMARY: egui::Color32 = egui::Color32::from_rgb(75, 96, 235);
const PRIMARY_SOFT: egui::Color32 = egui::Color32::from_rgb(235, 239, 255);
const TEXT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(31, 38, 51);
const TEXT_SECONDARY: egui::Color32 = egui::Color32::from_rgb(103, 112, 132);
const BORDER: egui::Color32 = egui::Color32::from_rgb(226, 231, 240);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UiLanguage {
    Chinese,
    English,
}

impl UiLanguage {
    fn text<'a>(self, chinese: &'a str, english: &'a str) -> &'a str {
        match self {
            Self::Chinese => chinese,
            Self::English => english,
        }
    }
}

/// Commands travel from the synchronous egui thread to the Tokio worker.
enum WorkerCommand {
    FetchVoices,
    Preview {
        text: String,
        voice: VoiceSelection,
        rate_percent: i32,
        volume_percent: i32,
        qwen_settings: QwenGenerationSettings,
        indextts_settings: IndexTtsSettings,
    },
    Generate {
        text: String,
        voice: VoiceSelection,
        rate_percent: i32,
        volume_percent: i32,
        qwen_settings: QwenGenerationSettings,
        indextts_settings: IndexTtsSettings,
        output_path: PathBuf,
    },
    GenerateSubtitles {
        cues: Vec<SubtitleCue>,
        voice: VoiceSelection,
        rate_percent: i32,
        volume_percent: i32,
        qwen_settings: QwenGenerationSettings,
        indextts_settings: IndexTtsSettings,
        output_path: PathBuf,
    },
    ImportQwenModel {
        version: QwenModelVersion,
        kind: QwenModelKind,
        source_dir: PathBuf,
    },
    PrepareQwenModel {
        version: QwenModelVersion,
        kind: QwenModelKind,
    },
    RedownloadQwenModel {
        version: QwenModelVersion,
        kind: QwenModelKind,
    },
    ImportIndexTtsModel {
        source_dir: PathBuf,
    },
    PrepareIndexTts,
    TranscribeMedia {
        input_path: PathBuf,
        language: RecognitionLanguage,
    },
    SaveTranscription {
        output_path: PathBuf,
        cues: Vec<SubtitleCue>,
        format: SubtitleExportFormat,
    },
    LoadGeneratedAudio {
        input_path: PathBuf,
    },
}

/// Results travel back to egui. The UI polls this channel without blocking.
enum WorkerEvent {
    VoicesLoaded {
        voices: Vec<VoiceChoice>,
        from_cache: bool,
        warning: Option<String>,
    },
    VoicesFailed(String),
    QwenModelPreparing {
        version: QwenModelVersion,
        kind: QwenModelKind,
    },
    QwenModelDownload {
        version: QwenModelVersion,
        kind: QwenModelKind,
        file: String,
        downloaded_bytes: u64,
        total_bytes: u64,
    },
    QwenModelReady {
        version: QwenModelVersion,
        kind: QwenModelKind,
        device: String,
    },
    QwenModelFailed(String),
    QwenModelImportProgress {
        version: QwenModelVersion,
        kind: QwenModelKind,
        file: String,
        copied_bytes: u64,
        total_bytes: u64,
    },
    QwenModelImported {
        version: QwenModelVersion,
        kind: QwenModelKind,
        device: String,
        model_dir: PathBuf,
    },
    QwenModelImportFailed {
        version: QwenModelVersion,
        kind: QwenModelKind,
        error: String,
    },
    QwenModelReleased,
    QwenClonePromptPreparing,
    QwenClonePromptReady,
    QwenProgress {
        current: usize,
        total: usize,
    },
    IndexTtsPreparing {
        phase: String,
        downloaded_bytes: u64,
        total_bytes: u64,
    },
    IndexTtsReady,
    IndexTtsFailed(String),
    IndexTtsImportProgress {
        phase: String,
        copied_bytes: u64,
        total_bytes: u64,
    },
    IndexTtsImported {
        model_dir: PathBuf,
    },
    IndexTtsImportFailed(String),
    IndexTtsInferencePhase(String),
    PreviewFinished,
    PreviewFailed(String),
    GenerationFinished {
        output_path: PathBuf,
        byte_count: usize,
    },
    SubtitleProgress {
        current: usize,
        total: usize,
    },
    SubtitleGenerationFinished {
        output_path: PathBuf,
        byte_count: usize,
        overflow_count: usize,
    },
    AsrModelProgress {
        source: String,
        downloaded_bytes: u64,
        total_bytes: u64,
    },
    AsrModelReady,
    AsrModelLoading {
        progress: f32,
    },
    TranscriptionProgress {
        progress: f32,
        remaining_seconds: u64,
    },
    TranscriptionFinished {
        cues: Vec<SubtitleCue>,
        audio_duration_ms: u64,
    },
    TranscriptionSaved(PathBuf),
    TranscriptionSaveFailed(String),
    TranscriptionFailed(String),
    GeneratedAudioLoaded {
        input_path: PathBuf,
        samples: Vec<f32>,
        sample_rate: u32,
    },
    GeneratedAudioLoadFailed {
        input_path: PathBuf,
        error: String,
    },
    GenerationFailed(String),
    WorkerFailed(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputMode {
    Text,
    Subtitles,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TtsEngine {
    Edge,
    Qwen3Local,
    IndexTts25,
}

#[derive(Clone, Copy)]
enum ModelDirectoryKind {
    Qwen3,
    IndexTts25,
    Whisper,
}

#[derive(Clone, Copy)]
enum ModelDownloadAction {
    Pause,
    Resume,
    Cancel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QwenVoiceMode {
    Preset,
    Clone,
}

#[derive(Clone, Debug)]
enum VoiceSelection {
    Edge(String),
    Qwen3Preset(QwenSelection),
    Qwen3Clone(QwenCloneSelection),
    IndexTts25(IndexTtsSelection),
}

#[derive(Clone, Copy, Debug)]
struct QwenSelection {
    version: QwenModelVersion,
    voice: QwenVoice,
}

#[derive(Clone, Debug)]
struct QwenCloneSelection {
    version: QwenModelVersion,
    reference_path: PathBuf,
    reference_text: String,
}

#[derive(Clone, Debug)]
struct IndexTtsSelection {
    reference_path: PathBuf,
}

#[derive(Debug, PartialEq, Eq)]
struct QwenClonePromptKey {
    version: QwenModelVersion,
    reference_path: PathBuf,
    reference_text: String,
    file_len: u64,
    modified_nanos: u128,
}

struct CachedQwenClonePrompt {
    key: QwenClonePromptKey,
    prompt: LocalVoiceClonePrompt,
}

struct QwenOutputSettings {
    rate_percent: i32,
    volume_percent: i32,
    qwen_settings: QwenGenerationSettings,
    output_path: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkspaceMode {
    TextToSpeech,
    AudioToSubtitles,
}

/// Vertical metrics for the left TTS card. The default macOS window leaves
/// roughly 565 px for the workspace. Qwen's offline-model row adds one more
/// line than Edge, so use a compact rhythm at that height instead of letting
/// the preview button run underneath the fixed action footer.
#[derive(Clone, Copy, Debug)]
struct VoiceCardLayout {
    engine_gap: f32,
    field_gap: f32,
    combo_summary_gap: f32,
    section_gap: f32,
    metadata_height: f32,
    settings_margin: i8,
    preview_margin: i8,
    waveform_height: f32,
    preview_button_height: f32,
}

impl VoiceCardLayout {
    fn for_height(card_height: f32) -> Self {
        if card_height < 590.0 {
            Self {
                engine_gap: 6.0,
                field_gap: 6.0,
                combo_summary_gap: 5.0,
                section_gap: 5.0,
                metadata_height: 26.0,
                settings_margin: 8,
                preview_margin: 6,
                waveform_height: 18.0,
                preview_button_height: 30.0,
            }
        } else {
            Self {
                engine_gap: 8.0,
                field_gap: 10.0,
                combo_summary_gap: 9.0,
                section_gap: 8.0,
                metadata_height: 30.0,
                settings_margin: 10,
                preview_margin: 8,
                waveform_height: 22.0,
                preview_button_height: 34.0,
            }
        }
    }
}

enum GenerationContent {
    Text(String),
    Subtitles(Vec<SubtitleCue>),
}

/// A small serializable UI model. Keeping the Edge crate's network model out of
/// the UI also lets us cache the last successful voice list with serde_json.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct VoiceChoice {
    short_name: String,
    locale: String,
    gender: String,
    friendly_name: Option<String>,
}

impl VoiceChoice {
    fn english_name(&self) -> String {
        self.friendly_name
            .as_deref()
            .and_then(|name| name.strip_prefix("Microsoft "))
            .and_then(|name| name.split(" Online").next())
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| {
                let raw_name = self
                    .short_name
                    .rsplit('-')
                    .next()
                    .unwrap_or(self.short_name.as_str());
                raw_name
                    .strip_suffix("Neural")
                    .unwrap_or(raw_name)
                    .to_owned()
            })
    }

    fn display_name(&self, language: UiLanguage) -> String {
        let english_name = self.english_name();
        if language == UiLanguage::Chinese
            && let Some(chinese_name) = chinese_voice_name(&self.short_name)
        {
            return format!("{chinese_name} {english_name}");
        }
        english_name
    }

    fn language_name(&self, language: UiLanguage) -> String {
        if language == UiLanguage::English {
            return self
                .friendly_name
                .as_deref()
                .and_then(|name| name.split(" - ").nth(1))
                .unwrap_or(self.locale.as_str())
                .to_owned();
        }

        let mut parts = self.locale.split('-');
        let language_code = parts.next().unwrap_or(self.locale.as_str());
        let region_code = parts.next().unwrap_or_default();
        let language_name = chinese_language_name(language_code);
        let region_name = chinese_region_name(region_code);

        if region_name == region_code || region_name.is_empty() {
            format!("{language_name}（{}）", self.locale)
        } else {
            format!("{language_name}（{region_name}）")
        }
    }

    fn gender_name(&self, language: UiLanguage) -> &str {
        match (language, self.gender.as_str()) {
            (UiLanguage::Chinese, "Female") => "女声",
            (UiLanguage::Chinese, "Male") => "男声",
            (_, gender) => gender,
        }
    }

    fn label(&self, language: UiLanguage) -> String {
        format!(
            "{}  ·  {}  ·  {}  ·  {}",
            self.display_name(language),
            self.language_name(language),
            self.gender_name(language),
            self.locale
        )
    }

    fn matches(&self, query: &str) -> bool {
        if query.trim().is_empty() {
            return true;
        }

        let query = query.trim().to_lowercase();
        [
            self.short_name.clone(),
            self.locale.clone(),
            self.gender.clone(),
            self.friendly_name.clone().unwrap_or_default(),
            self.label(UiLanguage::Chinese),
            self.label(UiLanguage::English),
        ]
        .into_iter()
        .any(|candidate| candidate.to_lowercase().contains(&query))
    }
}

fn chinese_voice_name(short_name: &str) -> Option<&'static str> {
    Some(match short_name {
        "zh-CN-XiaoxiaoNeural" => "晓晓",
        "zh-CN-XiaoyiNeural" => "晓伊",
        "zh-CN-YunjianNeural" => "云健",
        "zh-CN-YunxiNeural" => "云希",
        "zh-CN-YunxiaNeural" => "云夏",
        "zh-CN-YunyangNeural" => "云扬",
        "zh-CN-liaoning-XiaobeiNeural" => "晓北",
        "zh-CN-shaanxi-XiaoniNeural" => "晓妮",
        "zh-HK-HiuGaaiNeural" => "晓佳",
        "zh-HK-HiuMaanNeural" => "晓曼",
        "zh-HK-WanLungNeural" => "云龙",
        "zh-TW-HsiaoChenNeural" => "晓臻",
        "zh-TW-HsiaoYuNeural" => "晓雨",
        "zh-TW-YunJheNeural" => "云哲",
        _ => return None,
    })
}

fn chinese_language_name(code: &str) -> &str {
    match code {
        "af" => "南非荷兰语",
        "am" => "阿姆哈拉语",
        "ar" => "阿拉伯语",
        "az" => "阿塞拜疆语",
        "bg" => "保加利亚语",
        "bn" => "孟加拉语",
        "bs" => "波斯尼亚语",
        "ca" => "加泰罗尼亚语",
        "cs" => "捷克语",
        "cy" => "威尔士语",
        "da" => "丹麦语",
        "de" => "德语",
        "el" => "希腊语",
        "en" => "英语",
        "es" => "西班牙语",
        "et" => "爱沙尼亚语",
        "eu" => "巴斯克语",
        "fa" => "波斯语",
        "fi" => "芬兰语",
        "fil" => "菲律宾语",
        "fr" => "法语",
        "ga" => "爱尔兰语",
        "gl" => "加利西亚语",
        "gu" => "古吉拉特语",
        "he" => "希伯来语",
        "hi" => "印地语",
        "hr" => "克罗地亚语",
        "hu" => "匈牙利语",
        "id" => "印度尼西亚语",
        "is" => "冰岛语",
        "it" => "意大利语",
        "ja" => "日语",
        "jv" => "爪哇语",
        "ka" => "格鲁吉亚语",
        "kk" => "哈萨克语",
        "km" => "高棉语",
        "kn" => "卡纳达语",
        "ko" => "韩语",
        "lo" => "老挝语",
        "lt" => "立陶宛语",
        "lv" => "拉脱维亚语",
        "mk" => "马其顿语",
        "ml" => "马拉雅拉姆语",
        "mn" => "蒙古语",
        "mr" => "马拉地语",
        "ms" => "马来语",
        "mt" => "马耳他语",
        "my" => "缅甸语",
        "nb" => "挪威语",
        "ne" => "尼泊尔语",
        "nl" => "荷兰语",
        "pl" => "波兰语",
        "ps" => "普什图语",
        "pt" => "葡萄牙语",
        "ro" => "罗马尼亚语",
        "ru" => "俄语",
        "si" => "僧伽罗语",
        "sk" => "斯洛伐克语",
        "sl" => "斯洛文尼亚语",
        "so" => "索马里语",
        "sq" => "阿尔巴尼亚语",
        "sr" => "塞尔维亚语",
        "su" => "巽他语",
        "sv" => "瑞典语",
        "sw" => "斯瓦希里语",
        "ta" => "泰米尔语",
        "te" => "泰卢固语",
        "th" => "泰语",
        "tr" => "土耳其语",
        "uk" => "乌克兰语",
        "ur" => "乌尔都语",
        "uz" => "乌兹别克语",
        "vi" => "越南语",
        "wuu" => "吴语",
        "yue" => "粤语",
        "zh" => "中文",
        "zu" => "祖鲁语",
        other => other,
    }
}

fn chinese_region_name(code: &str) -> &str {
    match code {
        "AR" => "阿根廷",
        "AT" => "奥地利",
        "AU" => "澳大利亚",
        "BE" => "比利时",
        "BR" => "巴西",
        "CA" => "加拿大",
        "CH" => "瑞士",
        "CN" => "中国大陆",
        "DE" => "德国",
        "ES" => "西班牙",
        "FR" => "法国",
        "GB" => "英国",
        "HK" => "中国香港",
        "IE" => "爱尔兰",
        "IN" => "印度",
        "IT" => "意大利",
        "JP" => "日本",
        "KR" => "韩国",
        "MX" => "墨西哥",
        "MY" => "马来西亚",
        "NL" => "荷兰",
        "NZ" => "新西兰",
        "PT" => "葡萄牙",
        "RU" => "俄罗斯",
        "SG" => "新加坡",
        "TW" => "中国台湾",
        "US" => "美国",
        "ZA" => "南非",
        other => other,
    }
}

#[derive(Clone, Copy)]
enum StatusKind {
    Info,
    Success,
    Warning,
    Error,
}

struct StatusMessage {
    kind: StatusKind,
    chinese: String,
    english: String,
}

impl StatusMessage {
    fn new(kind: StatusKind, chinese: impl Into<String>, english: impl Into<String>) -> Self {
        Self {
            kind,
            chinese: chinese.into(),
            english: english.into(),
        }
    }

    fn text(&self, language: UiLanguage) -> &str {
        match language {
            UiLanguage::Chinese => &self.chinese,
            UiLanguage::English => &self.english,
        }
    }
}

struct TtsApp {
    command_tx: mpsc::UnboundedSender<WorkerCommand>,
    event_rx: mpsc::UnboundedReceiver<WorkerEvent>,
    voices: Vec<VoiceChoice>,
    selected_voice: Option<usize>,
    tts_engine: TtsEngine,
    selected_qwen_version: QwenModelVersion,
    selected_qwen_voice: QwenVoice,
    qwen_voice_mode: QwenVoiceMode,
    clone_reference_path: Option<PathBuf>,
    clone_reference_text: String,
    clone_authorized: bool,
    voice_filter: String,
    text: String,
    workspace_mode: WorkspaceMode,
    input_mode: InputMode,
    subtitle_track: Option<SubtitleTrack>,
    subtitle_path: Option<PathBuf>,
    subtitle_progress: Option<(usize, usize)>,
    rate_percent: i32,
    volume_percent: i32,
    qwen_max_length: usize,
    qwen_temperature: f64,
    qwen_top_k: usize,
    qwen_top_p: f64,
    qwen_repetition_penalty: f64,
    qwen_min_new_tokens: usize,
    qwen_seed: u64,
    qwen_random_seed: bool,
    indextts_duration_factor: f64,
    indextts_text_normalization: bool,
    indextts_max_text_tokens: usize,
    indextts_interval_silence_ms: usize,
    indextts_use_random: bool,
    indextts_emo_alpha: f64,
    indextts_use_emo_text: bool,
    indextts_emo_text: String,
    indextts_do_sample: bool,
    indextts_temperature: f64,
    indextts_top_k: usize,
    indextts_top_p: f64,
    indextts_repetition_penalty: f64,
    indextts_length_penalty: f64,
    indextts_num_beams: usize,
    indextts_max_mel_tokens: usize,
    ui_language: UiLanguage,
    fetching_voices: bool,
    qwen_model_preparing: bool,
    qwen_model_progress: Option<f32>,
    qwen_model_progress_label: String,
    qwen_model_ready: Option<(QwenModelVersion, QwenModelKind)>,
    qwen_device: Option<String>,
    qwen_manual_help_open: bool,
    indextts_model_preparing: bool,
    indextts_model_ready: bool,
    indextts_model_progress: Option<f32>,
    indextts_model_progress_label: String,
    indextts_manual_help_open: bool,
    previewing: bool,
    generating: bool,
    last_generated_audio: Option<PathBuf>,
    generated_audio_player: Option<GeneratedAudioPlayer>,
    generated_audio_loading: bool,
    asr_input_path: Option<PathBuf>,
    recognition_language: RecognitionLanguage,
    subtitle_export_format: SubtitleExportFormat,
    loading_asr_model: bool,
    transcribing: bool,
    saving_transcription: bool,
    transcription_progress: f32,
    transcription_remaining_seconds: u64,
    transcription_cues: Vec<SubtitleCue>,
    transcription_output_path: Option<PathBuf>,
    transcription_dirty: bool,
    status: Option<StatusMessage>,
}

impl TtsApp {
    fn new(creation_context: &eframe::CreationContext<'_>) -> Self {
        configure_egui_for_platform(&creation_context.egui_ctx);

        let (command_tx, event_rx, startup_error) = spawn_tts_worker();
        let fetching_voices = startup_error.is_none();

        let status = if let Some(error) = startup_error {
            Some(StatusMessage::new(
                StatusKind::Error,
                format!("无法启动语音服务：{error}"),
                error,
            ))
        } else if command_tx.send(WorkerCommand::FetchVoices).is_err() {
            Some(StatusMessage::new(
                StatusKind::Error,
                "语音服务尚未开始加载就已停止。",
                "The TTS worker stopped before voice loading began.",
            ))
        } else {
            Some(StatusMessage::new(
                StatusKind::Info,
                "正在加载在线音色…",
                "Loading the available online voices…",
            ))
        };

        Self {
            command_tx,
            event_rx,
            voices: Vec::new(),
            selected_voice: None,
            tts_engine: TtsEngine::Edge,
            selected_qwen_version: QwenModelVersion::DEFAULT,
            selected_qwen_voice: QwenVoice::Vivian,
            qwen_voice_mode: QwenVoiceMode::Preset,
            clone_reference_path: None,
            clone_reference_text: String::new(),
            clone_authorized: false,
            // The Chinese interface starts with the focused Chinese catalogue
            // shown in the preferred default layout. Clearing the field still
            // reveals every Edge voice.
            voice_filter: "中文".to_owned(),
            text: DEFAULT_TEXT.to_owned(),
            workspace_mode: WorkspaceMode::TextToSpeech,
            input_mode: InputMode::Text,
            subtitle_track: None,
            subtitle_path: None,
            subtitle_progress: None,
            rate_percent: 0,
            volume_percent: 0,
            qwen_max_length: QwenGenerationSettings::default().max_length,
            qwen_temperature: QwenGenerationSettings::default().temperature,
            qwen_top_k: QwenGenerationSettings::default().top_k,
            qwen_top_p: QwenGenerationSettings::default().top_p,
            qwen_repetition_penalty: QwenGenerationSettings::default().repetition_penalty,
            qwen_min_new_tokens: QwenGenerationSettings::default().min_new_tokens,
            qwen_seed: 42,
            qwen_random_seed: false,
            indextts_duration_factor: IndexTtsSettings::default().duration_factor,
            indextts_text_normalization: IndexTtsSettings::default().text_normalization,
            indextts_max_text_tokens: IndexTtsSettings::default().max_text_tokens_per_segment,
            indextts_interval_silence_ms: IndexTtsSettings::default().interval_silence_ms,
            indextts_use_random: IndexTtsSettings::default().use_random,
            indextts_emo_alpha: IndexTtsSettings::default().emo_alpha,
            indextts_use_emo_text: IndexTtsSettings::default().use_emo_text,
            indextts_emo_text: IndexTtsSettings::default().emo_text,
            indextts_do_sample: IndexTtsSettings::default().do_sample,
            indextts_temperature: IndexTtsSettings::default().temperature,
            indextts_top_k: IndexTtsSettings::default().top_k,
            indextts_top_p: IndexTtsSettings::default().top_p,
            indextts_repetition_penalty: IndexTtsSettings::default().repetition_penalty,
            indextts_length_penalty: IndexTtsSettings::default().length_penalty,
            indextts_num_beams: IndexTtsSettings::default().num_beams,
            indextts_max_mel_tokens: IndexTtsSettings::default().max_mel_tokens,
            ui_language: UiLanguage::Chinese,
            fetching_voices,
            qwen_model_preparing: false,
            qwen_model_progress: None,
            qwen_model_progress_label: String::new(),
            qwen_model_ready: None,
            qwen_device: None,
            qwen_manual_help_open: false,
            indextts_model_preparing: false,
            indextts_model_ready: false,
            indextts_model_progress: None,
            indextts_model_progress_label: String::new(),
            indextts_manual_help_open: false,
            previewing: false,
            generating: false,
            last_generated_audio: None,
            generated_audio_player: None,
            generated_audio_loading: false,
            asr_input_path: None,
            recognition_language: RecognitionLanguage::MixedChineseEnglish,
            subtitle_export_format: SubtitleExportFormat::Srt,
            loading_asr_model: false,
            transcribing: false,
            saving_transcription: false,
            transcription_progress: 0.0,
            transcription_remaining_seconds: 0,
            transcription_cues: Vec::new(),
            transcription_output_path: None,
            transcription_dirty: false,
            status,
        }
    }

    fn process_worker_events(&mut self) {
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                WorkerEvent::VoicesLoaded {
                    voices,
                    from_cache,
                    warning,
                } => {
                    let previous_voice = self
                        .selected_voice
                        .and_then(|index| self.voices.get(index))
                        .map(|voice| voice.short_name.clone());

                    self.voices = voices;
                    self.fetching_voices = false;
                    self.selected_voice = previous_voice
                        .as_deref()
                        .and_then(|name| {
                            self.voices
                                .iter()
                                .position(|voice| voice.short_name == name)
                        })
                        .or_else(|| {
                            self.voices
                                .iter()
                                .position(|voice| voice.short_name == PREFERRED_VOICE)
                        })
                        .or((!self.voices.is_empty()).then_some(0));

                    self.status = Some(if let Some(warning) = warning {
                        StatusMessage::new(
                            StatusKind::Warning,
                            format!("音色列表已加载，但遇到提示：{warning}"),
                            warning,
                        )
                    } else {
                        StatusMessage::new(
                            StatusKind::Success,
                            format!(
                                "已加载 {} 个音色{}。",
                                self.voices.len(),
                                if from_cache {
                                    "（来自本地缓存）"
                                } else {
                                    ""
                                }
                            ),
                            format!(
                                "Loaded {} voices{}.",
                                self.voices.len(),
                                if from_cache { " from cache" } else { "" }
                            ),
                        )
                    });
                }
                WorkerEvent::VoicesFailed(error) => {
                    self.fetching_voices = false;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Error,
                        format!("加载音色失败：{error}"),
                        error,
                    ));
                }
                WorkerEvent::QwenModelPreparing { version, kind } => {
                    self.qwen_model_preparing = true;
                    self.qwen_model_progress = Some(0.0);
                    self.qwen_model_progress_label =
                        format!("Qwen3 {} {}", version.short_label(), kind.short_label());
                    self.qwen_model_ready = None;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        format!(
                            "正在准备 Qwen3-TTS {} {} 本地模型，首次使用需下载约 {}…",
                            version.short_label(),
                            kind.short_label(),
                            version.download_size_label()
                        ),
                        format!(
                            "Preparing the local Qwen3-TTS {} {} model. The first run downloads about {}…",
                            version.short_label(),
                            kind.short_label(),
                            version.download_size_label()
                        ),
                    ));
                }
                WorkerEvent::QwenModelDownload {
                    version,
                    kind,
                    file,
                    downloaded_bytes,
                    total_bytes,
                } => {
                    self.qwen_model_preparing = true;
                    if file == "model-loading" {
                        self.qwen_model_progress = None;
                        self.qwen_model_progress_label = self
                            .ui_language
                            .text("正在载入 Qwen3 模型", "Loading the Qwen3 model")
                            .to_owned();
                        self.status = Some(StatusMessage::new(
                            StatusKind::Info,
                            format!(
                                "Qwen3-TTS {} {} 已下载，正在载入本地推理设备…",
                                version.short_label(),
                                kind.short_label()
                            ),
                            format!(
                                "Qwen3-TTS {} {} is downloaded and loading onto the local device…",
                                version.short_label(),
                                kind.short_label()
                            ),
                        ));
                        continue;
                    }
                    let localized_file = match file.as_str() {
                        "main-model" => ("Qwen3 主模型", "Qwen3 main model"),
                        "model-config" => ("Qwen3 配置", "Qwen3 configuration"),
                        "audio-decoder" => ("12Hz 音频解码器", "12Hz audio decoder"),
                        "text-tokenizer" => ("Qwen 文本分词器", "Qwen text tokenizer"),
                        _ => ("Qwen3 模型文件", "Qwen3 model file"),
                    };
                    let progress = format_transfer_progress(downloaded_bytes, total_bytes);
                    self.qwen_model_progress = (total_bytes > 0)
                        .then_some((downloaded_bytes as f32 / total_bytes as f32).clamp(0.0, 1.0));
                    self.qwen_model_progress_label = localized_file.0.to_owned();
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        format!(
                            "正在下载 Qwen3 {} {} {}：{progress}",
                            version.short_label(),
                            kind.short_label(),
                            localized_file.0
                        ),
                        format!(
                            "Downloading Qwen3 {} {} {}: {progress}",
                            version.short_label(),
                            kind.short_label(),
                            localized_file.1
                        ),
                    ));
                }
                WorkerEvent::QwenModelReady {
                    version,
                    kind,
                    device,
                } => {
                    self.qwen_model_preparing = false;
                    self.qwen_model_progress = None;
                    self.qwen_model_ready = Some((version, kind));
                    self.qwen_device = Some(device.clone());
                    self.status = Some(if self.generating || self.previewing {
                        StatusMessage::new(
                            StatusKind::Info,
                            format!(
                                "Qwen3-TTS {} {} 已就绪，正在使用 {device} 本地生成…",
                                version.short_label(),
                                kind.short_label()
                            ),
                            format!(
                                "Qwen3-TTS {} {} is ready. Generating locally with {device}…",
                                version.short_label(),
                                kind.short_label()
                            ),
                        )
                    } else {
                        StatusMessage::new(
                            StatusKind::Success,
                            format!(
                                "Qwen3-TTS {} {} 已下载并载入（{device}）。",
                                version.short_label(),
                                kind.short_label()
                            ),
                            format!(
                                "Qwen3-TTS {} {} is downloaded and loaded on {device}.",
                                version.short_label(),
                                kind.short_label()
                            ),
                        )
                    });
                }
                WorkerEvent::QwenModelFailed(error) => {
                    self.qwen_model_preparing = false;
                    self.qwen_model_progress = None;
                    self.qwen_model_ready = None;
                    self.qwen_device = None;
                    self.status = Some(if is_download_cancelled(&error) {
                        cancelled_download_status()
                    } else {
                        StatusMessage::new(
                            StatusKind::Error,
                            format!("Qwen3-TTS 模型准备失败：{error}"),
                            format!("Qwen3-TTS model setup failed: {error}"),
                        )
                    });
                }
                WorkerEvent::QwenModelImportProgress {
                    version,
                    kind,
                    file,
                    copied_bytes,
                    total_bytes,
                } => {
                    self.qwen_model_preparing = true;
                    let localized_file = match file.as_str() {
                        "main-model" => ("Qwen3 主模型", "Qwen3 main model"),
                        "model-config" => ("Qwen3 配置", "Qwen3 configuration"),
                        "audio-decoder" => ("12Hz 音频解码器", "12Hz audio decoder"),
                        "text-tokenizer" => ("Qwen 文本分词器", "Qwen text tokenizer"),
                        _ => ("Qwen3 模型文件", "Qwen3 model file"),
                    };
                    let progress = format_transfer_progress(copied_bytes, total_bytes);
                    self.qwen_model_progress = (total_bytes > 0)
                        .then_some((copied_bytes as f32 / total_bytes as f32).clamp(0.0, 1.0));
                    self.qwen_model_progress_label = localized_file.0.to_owned();
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        format!(
                            "正在导入 Qwen3 {} {} {}：{progress}",
                            version.short_label(),
                            kind.short_label(),
                            localized_file.0
                        ),
                        format!(
                            "Importing Qwen3 {} {} {}: {progress}",
                            version.short_label(),
                            kind.short_label(),
                            localized_file.1
                        ),
                    ));
                }
                WorkerEvent::QwenModelImported {
                    version,
                    kind,
                    device,
                    model_dir,
                } => {
                    self.qwen_model_preparing = false;
                    self.qwen_model_progress = None;
                    self.qwen_model_ready = Some((version, kind));
                    self.qwen_device = Some(device.clone());
                    self.qwen_manual_help_open = false;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Success,
                        format!(
                            "已导入并加载 Qwen3-TTS {} {}（{device}）：{}",
                            version.short_label(),
                            kind.short_label(),
                            model_dir.display()
                        ),
                        format!(
                            "Imported and loaded Qwen3-TTS {} {} on {device}: {}",
                            version.short_label(),
                            kind.short_label(),
                            model_dir.display()
                        ),
                    ));
                }
                WorkerEvent::QwenModelImportFailed {
                    version,
                    kind,
                    error,
                } => {
                    self.qwen_model_preparing = false;
                    self.qwen_model_progress = None;
                    self.qwen_model_ready = None;
                    self.qwen_device = None;
                    self.qwen_manual_help_open = true;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Error,
                        format!(
                            "Qwen3 {} {} 离线模型导入失败：{error}",
                            version.short_label(),
                            kind.short_label()
                        ),
                        format!(
                            "Could not import the Qwen3 {} {} offline model: {error}",
                            version.short_label(),
                            kind.short_label()
                        ),
                    ));
                }
                WorkerEvent::QwenModelReleased => {
                    self.qwen_model_preparing = false;
                    self.qwen_model_progress = None;
                    self.qwen_model_ready = None;
                    self.qwen_device = None;
                }
                WorkerEvent::QwenClonePromptPreparing => {
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        "正在本机分析参考音频并创建克隆提示；同一参考音频只处理一次…",
                        "Analyzing the reference locally and creating the clone prompt once…",
                    ));
                }
                WorkerEvent::QwenClonePromptReady => {
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        "克隆提示已在本机就绪，正在使用同一音色生成…",
                        "The local clone prompt is ready. Generating with the same voice…",
                    ));
                }
                WorkerEvent::QwenProgress { current, total } => {
                    self.subtitle_progress = Some((current, total));
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        format!("正在本地合成第 {current}/{total} 段…"),
                        format!("Synthesizing segment {current}/{total} locally…"),
                    ));
                }
                WorkerEvent::IndexTtsPreparing {
                    phase,
                    downloaded_bytes,
                    total_bytes,
                } => {
                    self.indextts_model_preparing = true;
                    self.indextts_model_ready = false;
                    self.indextts_model_progress = (total_bytes > 0)
                        .then_some((downloaded_bytes as f32 / total_bytes as f32).clamp(0.0, 1.0));
                    self.indextts_model_progress_label = phase.clone();
                    let transfer = (total_bytes > 0).then(|| {
                        format!(
                            " · {:.1}/{:.1} MB",
                            downloaded_bytes as f64 / 1_048_576.0,
                            total_bytes as f64 / 1_048_576.0
                        )
                    });
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        format!("{phase}{}", transfer.as_deref().unwrap_or_default()),
                        format!(
                            "IndexTTS-2.5 setup: {phase}{}",
                            transfer.as_deref().unwrap_or_default()
                        ),
                    ));
                }
                WorkerEvent::IndexTtsReady => {
                    self.indextts_model_preparing = false;
                    self.indextts_model_ready = true;
                    self.indextts_model_progress = None;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        "IndexTTS-2.5 官方 v2.5.0 模型已就绪，正在本机推理…",
                        "The official IndexTTS-2.5 v2.5.0 model is ready for local inference…",
                    ));
                }
                WorkerEvent::IndexTtsImportProgress {
                    phase,
                    copied_bytes,
                    total_bytes,
                } => {
                    self.indextts_model_preparing = true;
                    self.indextts_model_ready = false;
                    self.indextts_model_progress = (total_bytes > 0)
                        .then_some((copied_bytes as f32 / total_bytes as f32).clamp(0.0, 1.0));
                    self.indextts_model_progress_label = phase.clone();
                    let transfer = format_transfer_progress(copied_bytes, total_bytes);
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        format!("正在导入 IndexTTS-2.5 离线模型：{phase} · {transfer}"),
                        format!("Importing the offline IndexTTS-2.5 model: {phase} · {transfer}"),
                    ));
                }
                WorkerEvent::IndexTtsImported { model_dir } => {
                    self.indextts_model_preparing = false;
                    self.indextts_model_progress = None;
                    self.indextts_manual_help_open = false;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Success,
                        format!(
                            "IndexTTS-2.5 离线模型已导入：{}。请点击“准备模型”完成本机运行环境检查。",
                            model_dir.display()
                        ),
                        format!(
                            "The offline IndexTTS-2.5 model was imported to {}. Select Prepare model to finish the local runtime check.",
                            model_dir.display()
                        ),
                    ));
                }
                WorkerEvent::IndexTtsImportFailed(error) => {
                    self.indextts_model_preparing = false;
                    self.indextts_model_progress = None;
                    self.indextts_manual_help_open = true;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Error,
                        format!("IndexTTS-2.5 离线模型导入失败：{error}"),
                        format!("Could not import the offline IndexTTS-2.5 model: {error}"),
                    ));
                }
                WorkerEvent::IndexTtsFailed(error) => {
                    self.indextts_model_preparing = false;
                    self.indextts_model_progress = None;
                    self.status = Some(if is_download_cancelled(&error) {
                        StatusMessage::new(
                            StatusKind::Info,
                            "IndexTTS-2.5 下载已取消，临时文件已保留；再次点击“准备模型”即可续传。",
                            "IndexTTS-2.5 download cancelled. Partial files were kept; select Prepare model to resume.",
                        )
                    } else {
                        StatusMessage::new(
                            StatusKind::Error,
                            format!("IndexTTS-2.5 准备失败：{error}"),
                            format!("IndexTTS-2.5 setup failed: {error}"),
                        )
                    });
                }
                WorkerEvent::IndexTtsInferencePhase(phase) => {
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        phase.clone(),
                        format!("IndexTTS-2.5: {phase}"),
                    ));
                }
                WorkerEvent::PreviewFinished => {
                    self.previewing = false;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Success,
                        "音色试听播放完成。",
                        "Voice preview finished.",
                    ));
                }
                WorkerEvent::PreviewFailed(error) => {
                    let download_cancelled = is_download_cancelled(&error);
                    if !download_cancelled
                        && self.qwen_model_preparing
                        && self.tts_engine == TtsEngine::Qwen3Local
                    {
                        self.qwen_manual_help_open = true;
                    }
                    self.qwen_model_preparing = false;
                    self.qwen_model_progress = None;
                    self.indextts_model_preparing = false;
                    self.indextts_model_progress = None;
                    self.previewing = false;
                    self.status = Some(if download_cancelled {
                        cancelled_download_status()
                    } else {
                        StatusMessage::new(
                            StatusKind::Error,
                            format!("试听失败：{error}"),
                            format!("Preview failed: {error}"),
                        )
                    });
                }
                WorkerEvent::GenerationFinished {
                    output_path,
                    byte_count,
                } => {
                    self.generating = false;
                    self.subtitle_progress = None;
                    self.last_generated_audio = Some(output_path.clone());
                    self.status = Some(StatusMessage::new(
                        StatusKind::Success,
                        format!(
                            "已保存 {:.1} KB 到 {}",
                            byte_count as f64 / 1024.0,
                            output_path.display()
                        ),
                        format!(
                            "Saved {:.1} KB to {}",
                            byte_count as f64 / 1024.0,
                            output_path.display()
                        ),
                    ));
                    self.prepare_generated_audio_player(output_path);
                }
                WorkerEvent::SubtitleProgress { current, total } => {
                    self.subtitle_progress = Some((current, total));
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        format!("正在合成第 {current}/{total} 条字幕…"),
                        format!("Synthesizing subtitle {current} of {total}…"),
                    ));
                }
                WorkerEvent::SubtitleGenerationFinished {
                    output_path,
                    byte_count,
                    overflow_count,
                } => {
                    self.generating = false;
                    self.subtitle_progress = None;
                    self.last_generated_audio = Some(output_path.clone());
                    let kind = if overflow_count > 0 {
                        StatusKind::Warning
                    } else {
                        StatusKind::Success
                    };
                    self.status = Some(StatusMessage::new(
                        kind,
                        if overflow_count > 0 {
                            format!(
                                "字幕音频已保存（{:.1} KB）；保持所选语速，未自动调速或截断，{overflow_count} 条超出原时段并顺延：{}",
                                byte_count as f64 / 1024.0,
                                output_path.display()
                            )
                        } else {
                            format!(
                                "字幕音频已保存（{:.1} KB）；保持所选语速，未自动调速或截断：{}",
                                byte_count as f64 / 1024.0,
                                output_path.display()
                            )
                        },
                        if overflow_count > 0 {
                            format!(
                                "Saved subtitle audio ({:.1} KB); selected speed preserved, no auto-fit or truncation; {overflow_count} cues extended beyond their slots: {}",
                                byte_count as f64 / 1024.0,
                                output_path.display()
                            )
                        } else {
                            format!(
                                "Saved subtitle audio ({:.1} KB); selected speed preserved with no auto-fit or truncation: {}",
                                byte_count as f64 / 1024.0,
                                output_path.display()
                            )
                        },
                    ));
                    self.prepare_generated_audio_player(output_path);
                }
                WorkerEvent::AsrModelProgress {
                    source,
                    downloaded_bytes,
                    total_bytes,
                } => {
                    self.loading_asr_model = true;
                    let progress = if total_bytes > 0 {
                        downloaded_bytes as f32 / total_bytes as f32
                    } else {
                        0.0
                    };
                    self.transcription_progress = progress.clamp(0.0, 1.0);
                    let transfer = format_transfer_progress(downloaded_bytes, total_bytes);
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        format!(
                            "正在准备本地模型：{} · {transfer}",
                            compact_model_source(&source),
                        ),
                        format!(
                            "Preparing local model: {} · {transfer}",
                            compact_model_source(&source),
                        ),
                    ));
                }
                WorkerEvent::AsrModelReady => {
                    self.loading_asr_model = false;
                    self.transcription_progress = 0.0;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        "模型文件已就绪，正在进行本地语音识别…",
                        "Model files are ready. Running local speech recognition…",
                    ));
                }
                WorkerEvent::AsrModelLoading { progress } => {
                    self.loading_asr_model = true;
                    self.transcription_progress = progress.clamp(0.0, 1.0);
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        format!(
                            "正在将 Whisper 模型载入本地设备… {:.0}%",
                            self.transcription_progress * 100.0
                        ),
                        format!(
                            "Loading Whisper onto the local device… {:.0}%",
                            self.transcription_progress * 100.0
                        ),
                    ));
                }
                WorkerEvent::TranscriptionProgress {
                    progress,
                    remaining_seconds,
                } => {
                    self.loading_asr_model = false;
                    self.transcription_progress = progress.clamp(0.0, 1.0);
                    self.transcription_remaining_seconds = remaining_seconds;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        format!(
                            "正在本地识别… {:.0}% · 预计剩余 {}",
                            self.transcription_progress * 100.0,
                            format_duration_seconds(remaining_seconds)
                        ),
                        format!(
                            "Transcribing locally… {:.0}% · about {} remaining",
                            self.transcription_progress * 100.0,
                            format_duration_seconds(remaining_seconds)
                        ),
                    ));
                }
                WorkerEvent::TranscriptionFinished {
                    cues,
                    audio_duration_ms,
                } => {
                    self.loading_asr_model = false;
                    self.transcribing = false;
                    self.transcription_progress = 1.0;
                    self.transcription_remaining_seconds = 0;
                    self.transcription_cues = cues;
                    self.transcription_output_path = None;
                    self.transcription_dirty = true;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Success,
                        format!(
                            "已在本机生成 {} 条字幕（媒体时长 {}）。现在可以直接修改文字，确认后再保存。",
                            self.transcription_cues.len(),
                            format_timestamp(audio_duration_ms)
                        ),
                        format!(
                            "Generated {} subtitle cues locally from {} of media. Edit the text, then save it.",
                            self.transcription_cues.len(),
                            format_timestamp(audio_duration_ms)
                        ),
                    ));
                }
                WorkerEvent::TranscriptionSaved(output_path) => {
                    self.saving_transcription = false;
                    self.transcription_dirty = false;
                    self.transcription_output_path = Some(output_path.clone());
                    self.status = Some(StatusMessage::new(
                        StatusKind::Success,
                        format!("字幕修改已保存：{}", output_path.display()),
                        format!("Saved the edited subtitles: {}", output_path.display()),
                    ));
                }
                WorkerEvent::TranscriptionSaveFailed(error) => {
                    self.saving_transcription = false;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Error,
                        format!("保存字幕失败：{error}"),
                        format!("Could not save subtitles: {error}"),
                    ));
                }
                WorkerEvent::TranscriptionFailed(error) => {
                    self.loading_asr_model = false;
                    self.transcribing = false;
                    self.saving_transcription = false;
                    self.transcription_progress = 0.0;
                    self.transcription_remaining_seconds = 0;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Error,
                        format!("生成字幕失败：{}", localized_transcription_error(&error)),
                        format!("Subtitle generation failed: {error}"),
                    ));
                }
                WorkerEvent::GeneratedAudioLoaded {
                    input_path,
                    samples,
                    sample_rate,
                } => {
                    if self.last_generated_audio.as_ref() == Some(&input_path) {
                        self.generated_audio_loading = false;
                        match GeneratedAudioPlayer::new(input_path, samples, sample_rate) {
                            Ok(player) => self.generated_audio_player = Some(player),
                            Err(error) => {
                                self.generated_audio_player = None;
                                self.status = Some(StatusMessage::new(
                                    StatusKind::Warning,
                                    format!("音频已生成，但无法初始化预览播放器：{error}"),
                                    format!(
                                        "Audio was generated, but the preview player could not start: {error}"
                                    ),
                                ));
                            }
                        }
                    }
                }
                WorkerEvent::GeneratedAudioLoadFailed { input_path, error } => {
                    if self.last_generated_audio.as_ref() == Some(&input_path) {
                        self.generated_audio_loading = false;
                        self.generated_audio_player = None;
                        self.status = Some(StatusMessage::new(
                            StatusKind::Warning,
                            format!("音频已生成，但无法载入预览：{error}"),
                            format!("Audio was generated, but its preview could not load: {error}"),
                        ));
                    }
                }
                WorkerEvent::GenerationFailed(error) => {
                    let download_cancelled = is_download_cancelled(&error);
                    if !download_cancelled
                        && self.qwen_model_preparing
                        && self.tts_engine == TtsEngine::Qwen3Local
                    {
                        self.qwen_manual_help_open = true;
                    }
                    self.qwen_model_preparing = false;
                    self.qwen_model_progress = None;
                    self.indextts_model_preparing = false;
                    self.indextts_model_progress = None;
                    self.generating = false;
                    self.subtitle_progress = None;
                    self.status = Some(if download_cancelled {
                        cancelled_download_status()
                    } else {
                        StatusMessage::new(
                            StatusKind::Error,
                            format!("生成语音失败：{error}"),
                            error,
                        )
                    });
                }
                WorkerEvent::WorkerFailed(error) => {
                    self.fetching_voices = false;
                    self.qwen_model_preparing = false;
                    self.indextts_model_preparing = false;
                    self.indextts_model_progress = None;
                    self.previewing = false;
                    self.generating = false;
                    self.loading_asr_model = false;
                    self.transcribing = false;
                    self.saving_transcription = false;
                    self.subtitle_progress = None;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Error,
                        format!("语音服务异常：{error}"),
                        error,
                    ));
                }
            }
        }
    }

    fn reload_voices(&mut self) {
        if self.fetching_voices
            || self.previewing
            || self.generating
            || self.transcribing
            || self.qwen_model_preparing
        {
            return;
        }

        match self.command_tx.send(WorkerCommand::FetchVoices) {
            Ok(()) => {
                self.fetching_voices = true;
                self.status = Some(StatusMessage::new(
                    StatusKind::Info,
                    "正在刷新音色列表…",
                    "Refreshing voices…",
                ));
            }
            Err(_) => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Error,
                    "语音服务已停止，请重新启动应用。",
                    "The TTS worker is no longer running.",
                ));
            }
        }
    }

    fn import_selected_qwen_model(&mut self) {
        if self.tts_engine != TtsEngine::Qwen3Local || self.tts_controls_busy() {
            return;
        }
        let version = self.selected_qwen_version;
        let kind = self.selected_qwen_kind();
        let Some(source_dir) = rfd::FileDialog::new()
            .set_title(self.ui_language.text(
                "选择完整的 Qwen3-TTS 模型文件夹",
                "Choose the complete Qwen3-TTS model folder",
            ))
            .pick_folder()
        else {
            return;
        };

        match self.command_tx.send(WorkerCommand::ImportQwenModel {
            version,
            kind,
            source_dir: source_dir.clone(),
        }) {
            Ok(()) => {
                self.qwen_model_preparing = true;
                self.qwen_model_ready = None;
                self.qwen_device = None;
                self.status = Some(StatusMessage::new(
                    StatusKind::Info,
                    format!(
                        "正在检查并导入 Qwen3-TTS {} {}：{}",
                        version.short_label(),
                        kind.short_label(),
                        source_dir.display()
                    ),
                    format!(
                        "Checking and importing Qwen3-TTS {} {}: {}",
                        version.short_label(),
                        kind.short_label(),
                        source_dir.display()
                    ),
                ));
            }
            Err(_) => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Error,
                    "后台服务已停止，请重新启动应用。",
                    "The background worker stopped. Restart the app.",
                ));
            }
        }
    }

    fn prepare_indextts_model(&mut self) {
        if self.tts_engine != TtsEngine::IndexTts25 || self.tts_controls_busy() {
            return;
        }
        match self.command_tx.send(WorkerCommand::PrepareIndexTts) {
            Ok(()) => {
                self.indextts_model_preparing = true;
                self.indextts_model_ready = false;
                self.indextts_model_progress = None;
                self.indextts_model_progress_label = self
                    .ui_language
                    .text("正在检查官方运行环境", "Checking official runtime")
                    .to_owned();
                self.status = Some(StatusMessage::new(
                    StatusKind::Info,
                    "正在检查 IndexTTS-2.5 官方运行环境…",
                    "Checking the official IndexTTS-2.5 runtime…",
                ));
            }
            Err(_) => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Error,
                    "后台服务已停止，请重新启动应用。",
                    "The background worker stopped. Restart the app.",
                ));
            }
        }
    }

    fn import_indextts_model(&mut self) {
        if self.tts_engine != TtsEngine::IndexTts25 || self.tts_controls_busy() {
            return;
        }
        let Some(source_dir) = rfd::FileDialog::new()
            .set_title(self.ui_language.text(
                "选择完整的 IndexTTS-2.5 模型文件夹",
                "Choose the complete IndexTTS-2.5 model folder",
            ))
            .pick_folder()
        else {
            return;
        };
        match self.command_tx.send(WorkerCommand::ImportIndexTtsModel {
            source_dir: source_dir.clone(),
        }) {
            Ok(()) => {
                self.indextts_model_preparing = true;
                self.indextts_model_ready = false;
                self.indextts_model_progress = Some(0.0);
                self.indextts_model_progress_label = self
                    .ui_language
                    .text("正在检查离线模型文件", "Checking offline model files")
                    .to_owned();
                self.status = Some(StatusMessage::new(
                    StatusKind::Info,
                    format!("正在导入 IndexTTS-2.5 离线模型：{}", source_dir.display()),
                    format!(
                        "Importing the offline IndexTTS-2.5 model: {}",
                        source_dir.display()
                    ),
                ));
            }
            Err(_) => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Error,
                    "后台服务已停止，请重新启动应用。",
                    "The background worker stopped. Restart the app.",
                ));
            }
        }
    }

    fn open_model_directory(&mut self, kind: ModelDirectoryKind) {
        let directory = match kind {
            ModelDirectoryKind::Qwen3 => qwen_local::models_directory(),
            ModelDirectoryKind::IndexTts25 => indextts::models_directory(),
            ModelDirectoryKind::Whisper => asr::model_directory(),
        };
        let result = directory.and_then(|directory| {
            std::fs::create_dir_all(&directory)
                .map_err(|error| format!("无法创建模型目录：{error}"))?;
            open_directory_in_file_manager(&directory)?;
            Ok(directory)
        });
        match result {
            Ok(directory) => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Success,
                    format!("已打开本地模型目录：{}", directory.display()),
                    format!("Opened the local model folder: {}", directory.display()),
                ));
            }
            Err(error) => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Error,
                    format!("无法打开本地模型目录：{error}"),
                    format!("Could not open the local model folder: {error}"),
                ));
            }
        }
    }

    fn redownload_selected_qwen_model(&mut self) {
        if self.tts_engine != TtsEngine::Qwen3Local || self.tts_controls_busy() {
            return;
        }
        let version = self.selected_qwen_version;
        let kind = self.selected_qwen_kind();
        match self
            .command_tx
            .send(WorkerCommand::RedownloadQwenModel { version, kind })
        {
            Ok(()) => {
                self.qwen_model_preparing = true;
                self.qwen_model_progress = Some(0.0);
                self.qwen_model_progress_label = self
                    .ui_language
                    .text("正在清理所选模型缓存", "Clearing the selected model cache")
                    .to_owned();
                self.qwen_model_ready = None;
                self.qwen_device = None;
                self.status = Some(StatusMessage::new(
                    StatusKind::Info,
                    format!(
                        "正在重新下载 Qwen3-TTS {} {}…",
                        version.short_label(),
                        kind.short_label()
                    ),
                    format!(
                        "Downloading Qwen3-TTS {} {} again…",
                        version.short_label(),
                        kind.short_label()
                    ),
                ));
            }
            Err(_) => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Error,
                    "后台服务已停止，请重新启动应用。",
                    "The background worker stopped. Restart the app.",
                ));
            }
        }
    }

    fn prepare_selected_qwen_model(&mut self) {
        if self.tts_engine != TtsEngine::Qwen3Local || self.tts_controls_busy() {
            return;
        }
        let version = self.selected_qwen_version;
        let kind = self.selected_qwen_kind();
        match self
            .command_tx
            .send(WorkerCommand::PrepareQwenModel { version, kind })
        {
            Ok(()) => {
                self.qwen_model_preparing = true;
                self.qwen_model_progress = Some(0.0);
                self.qwen_model_progress_label = self
                    .ui_language
                    .text("正在检查断点文件", "Checking partial downloads")
                    .to_owned();
                self.qwen_model_ready = None;
                self.qwen_device = None;
                self.status = Some(StatusMessage::new(
                    StatusKind::Info,
                    format!(
                        "正在下载或续传 Qwen3-TTS {} {}…",
                        version.short_label(),
                        kind.short_label()
                    ),
                    format!(
                        "Downloading or resuming Qwen3-TTS {} {}…",
                        version.short_label(),
                        kind.short_label()
                    ),
                ));
            }
            Err(_) => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Error,
                    "后台服务已停止，请重新启动应用。",
                    "The background worker stopped. Restart the app.",
                ));
            }
        }
    }

    fn prepare_generated_audio_player(&mut self, input_path: PathBuf) {
        self.generated_audio_player = None;
        self.generated_audio_loading = true;
        if self
            .command_tx
            .send(WorkerCommand::LoadGeneratedAudio {
                input_path: input_path.clone(),
            })
            .is_err()
        {
            self.generated_audio_loading = false;
            self.status = Some(StatusMessage::new(
                StatusKind::Warning,
                format!("音频已生成，但无法载入预览：{}", input_path.display()),
                format!(
                    "Audio was generated, but its preview could not be loaded: {}",
                    input_path.display()
                ),
            ));
        }
    }

    fn apply_model_download_action(&mut self, action: ModelDownloadAction) {
        let control = model_download_control();
        let changed = match action {
            ModelDownloadAction::Pause => control.pause(),
            ModelDownloadAction::Resume => control.resume(),
            ModelDownloadAction::Cancel => control.cancel(),
        };
        if !changed {
            return;
        }
        self.status = Some(match action {
            ModelDownloadAction::Pause => StatusMessage::new(
                StatusKind::Info,
                "模型下载已暂停。点击“继续”可从当前进度恢复。",
                "Model download paused. Select Resume to continue from the current progress.",
            ),
            ModelDownloadAction::Resume => StatusMessage::new(
                StatusKind::Info,
                "正在从已有进度继续下载模型…",
                "Resuming the model download from its existing progress…",
            ),
            ModelDownloadAction::Cancel => StatusMessage::new(
                StatusKind::Info,
                "正在取消模型下载；已下载的临时文件会保留，以便下次继续。",
                "Cancelling the model download. Partial files will be kept for a later resume.",
            ),
        });
    }

    fn show_model_download_actions(&mut self, ui: &mut egui::Ui, language: UiLanguage) {
        let state = model_download_control().state();
        if matches!(state, DownloadState::Idle) {
            return;
        }
        let mut action = None;
        ui.horizontal(|ui| {
            match state {
                DownloadState::Running => {
                    if ui
                        .add(model_action_button(
                            language.text("暂停下载", "Pause download"),
                        ))
                        .clicked()
                    {
                        action = Some(ModelDownloadAction::Pause);
                    }
                }
                DownloadState::Paused => {
                    ui.label(
                        egui::RichText::new(language.text("下载已暂停", "Download paused"))
                            .size(11.0)
                            .color(TEXT_SECONDARY),
                    );
                    if ui
                        .add(model_action_button(
                            language.text("继续下载", "Resume download"),
                        ))
                        .clicked()
                    {
                        action = Some(ModelDownloadAction::Resume);
                    }
                }
                DownloadState::Cancelled => {
                    ui.spinner();
                    ui.label(
                        egui::RichText::new(language.text("正在取消…", "Cancelling…"))
                            .size(11.0)
                            .color(TEXT_SECONDARY),
                    );
                }
                DownloadState::Idle => {}
            }
            if !matches!(state, DownloadState::Cancelled)
                && ui
                    .add(model_action_button(
                        language.text("取消下载", "Cancel download"),
                    ))
                    .clicked()
            {
                action = Some(ModelDownloadAction::Cancel);
            }
        });
        if let Some(action) = action {
            self.apply_model_download_action(action);
        }
    }

    fn start_generation(&mut self) {
        if self.generating
            || self.previewing
            || self.transcribing
            || self.indextts_model_preparing
            || (self.tts_engine == TtsEngine::Edge && self.fetching_voices)
        {
            return;
        }

        let Some(voice) = self.active_voice_selection() else {
            let (chinese, english) = self.voice_selection_error();
            self.status = Some(StatusMessage::new(StatusKind::Error, chinese, english));
            return;
        };

        let (command_content, default_file_name) = match self.input_mode {
            InputMode::Text => {
                let text = self.text.trim().to_owned();
                if text.is_empty() {
                    self.status = Some(StatusMessage::new(
                        StatusKind::Error,
                        "请先输入需要转换的文字。",
                        "Enter some text before generating audio.",
                    ));
                    return;
                }
                (
                    GenerationContent::Text(text),
                    match self.tts_engine {
                        TtsEngine::Edge => {
                            self.ui_language.text("语音合成.mp3", "edge-tts-output.mp3")
                        }
                        TtsEngine::Qwen3Local => match self.selected_qwen_version {
                            QwenModelVersion::Small0_6B => self
                                .ui_language
                                .text("Qwen3-0.6B语音.mp3", "qwen3-0.6b-output.mp3"),
                            QwenModelVersion::Large1_7B => self
                                .ui_language
                                .text("Qwen3-1.7B语音.mp3", "qwen3-1.7b-output.mp3"),
                        },
                        TtsEngine::IndexTts25 => self
                            .ui_language
                            .text("IndexTTS-2.5语音.mp3", "indextts-2.5-output.mp3"),
                    },
                )
            }
            InputMode::Subtitles => {
                let Some(track) = &self.subtitle_track else {
                    self.status = Some(StatusMessage::new(
                        StatusKind::Error,
                        "请先导入 SRT/STR、WebVTT、ASS/SSA 或 LRC 字幕。",
                        "Import an SRT/STR, WebVTT, ASS/SSA, or LRC subtitle first.",
                    ));
                    return;
                };
                (
                    GenerationContent::Subtitles(track.cues.clone()),
                    self.ui_language
                        .text("字幕配音.mp3", "subtitle-voiceover.mp3"),
                )
            }
        };

        // rfd uses macOS's native NSSavePanel. It is intentionally opened on
        // the UI thread; network work and file writing remain on Tokio.
        let Some(output_path) = rfd::FileDialog::new()
            .set_title(
                self.ui_language
                    .text("保存生成的语音", "Save generated speech"),
            )
            .set_file_name(default_file_name)
            .set_can_create_directories(true)
            .add_filter(self.ui_language.text("MP3 音频", "MP3 audio"), &["mp3"])
            .save_file()
            .map(ensure_mp3_extension)
        else {
            return;
        };

        let file_name = output_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("MP3 file")
            .to_owned();

        let command = match command_content {
            GenerationContent::Text(text) => WorkerCommand::Generate {
                text,
                voice,
                rate_percent: self.rate_percent,
                volume_percent: self.volume_percent,
                qwen_settings: self.qwen_generation_settings(),
                indextts_settings: self.indextts_settings(),
                output_path,
            },
            GenerationContent::Subtitles(cues) => WorkerCommand::GenerateSubtitles {
                cues,
                voice,
                rate_percent: self.rate_percent,
                volume_percent: self.volume_percent,
                qwen_settings: self.qwen_generation_settings(),
                indextts_settings: self.indextts_settings(),
                output_path,
            },
        };

        match self.command_tx.send(command) {
            Ok(()) => {
                self.generating = true;
                self.generated_audio_player = None;
                self.generated_audio_loading = false;
                self.subtitle_progress = match &self.subtitle_track {
                    Some(track) if self.input_mode == InputMode::Subtitles => {
                        Some((0, track.cues.len()))
                    }
                    _ => None,
                };
                self.status = Some(StatusMessage::new(
                    StatusKind::Info,
                    format!("正在生成 {file_name}…"),
                    format!("Generating {file_name}…"),
                ));
            }
            Err(_) => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Error,
                    "语音服务已停止，请重新启动应用。",
                    "The TTS worker is no longer running.",
                ));
            }
        }
    }

    fn import_subtitle(&mut self) {
        if self.generating || self.previewing || self.transcribing {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .set_title(
                self.ui_language
                    .text("导入字幕文件", "Import subtitle file"),
            )
            .add_filter(
                self.ui_language.text("支持的字幕", "Supported subtitles"),
                &["srt", "str", "vtt", "ass", "ssa", "lrc"],
            )
            .pick_file()
        else {
            return;
        };

        let result = std::fs::read(&path)
            .map_err(|error| format!("Could not read the subtitle file: {error}"))
            .and_then(|bytes| parse_subtitle(&path, &bytes));
        match result {
            Ok(track) if track.cues.len() > 10_000 => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Error,
                    "字幕超过 10,000 条，为避免误操作未导入。",
                    "The subtitle has more than 10,000 cues and was not imported.",
                ));
            }
            Ok(track) if track.duration_ms() > 24 * 60 * 60 * 1_000 => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Error,
                    "字幕时间轴超过 24 小时，为避免异常文件未导入。",
                    "The subtitle timeline exceeds 24 hours and was not imported.",
                ));
            }
            Ok(track) => {
                let cue_count = track.cues.len();
                let format = track.format.label();
                self.subtitle_track = Some(track);
                self.subtitle_path = Some(path);
                self.input_mode = InputMode::Subtitles;
                self.status = Some(StatusMessage::new(
                    StatusKind::Success,
                    format!("已导入 {format} 字幕，共 {cue_count} 条。"),
                    format!("Imported {cue_count} {format} subtitle cues."),
                ));
            }
            Err(error) => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Error,
                    format!("导入字幕失败：{error}"),
                    format!("Could not import subtitles: {error}"),
                ));
            }
        }
    }

    fn start_preview(&mut self) {
        if self.previewing
            || self.generating
            || self.transcribing
            || self.indextts_model_preparing
            || (self.tts_engine == TtsEngine::Edge && self.fetching_voices)
        {
            return;
        }

        let Some(voice) = self.active_voice_selection() else {
            let (chinese, english) = self.voice_selection_error();
            self.status = Some(StatusMessage::new(StatusKind::Error, chinese, english));
            return;
        };

        let source_text = if self.input_mode == InputMode::Subtitles {
            self.subtitle_track
                .as_ref()
                .and_then(|track| track.cues.first())
                .map(|cue| cue.text.as_str())
                .unwrap_or_default()
        } else {
            &self.text
        };
        let text = preview_text(source_text, self.ui_language);
        match self.command_tx.send(WorkerCommand::Preview {
            text,
            voice,
            rate_percent: self.rate_percent,
            volume_percent: self.volume_percent,
            qwen_settings: self.qwen_generation_settings(),
            indextts_settings: self.indextts_settings(),
        }) {
            Ok(()) => {
                self.previewing = true;
                self.status = Some(StatusMessage::new(
                    StatusKind::Info,
                    "正在生成并播放试听音频…",
                    "Generating and playing the voice preview…",
                ));
            }
            Err(_) => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Error,
                    "语音服务已停止，请重新启动应用。",
                    "The TTS worker is no longer running.",
                ));
            }
        }
    }

    fn choose_media_for_transcription(&mut self) {
        if self.transcribing {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .set_title(self.ui_language.text(
                "选择需要生成字幕的音频或视频",
                "Choose audio or video to transcribe",
            ))
            .add_filter(
                self.ui_language.text("音频与视频", "Audio and video"),
                &[
                    "mp3", "wav", "m4a", "flac", "ogg", "mp4", "mov", "m4v", "mkv", "webm",
                ],
            )
            .add_filter(
                self.ui_language.text("常见视频", "Common video"),
                &["mp4", "mov", "m4v", "mkv", "webm"],
            )
            .add_filter(
                self.ui_language.text("常见音频", "Common audio"),
                &["mp3", "wav", "m4a", "flac", "ogg"],
            )
            .pick_file()
        else {
            return;
        };
        self.asr_input_path = Some(path);
        self.transcription_cues.clear();
        self.transcription_output_path = None;
        self.transcription_dirty = false;
        self.status = Some(StatusMessage::new(
            StatusKind::Info,
            "媒体已选择。视频会直接读取其中的音轨；模型首次使用会下载到本机。",
            "Media selected. Video audio is read directly; the model is downloaded on first use.",
        ));
    }

    fn use_last_generated_audio(&mut self) {
        let Some(path) = self
            .last_generated_audio
            .as_ref()
            .filter(|path| path.is_file())
            .cloned()
        else {
            self.status = Some(StatusMessage::new(
                StatusKind::Warning,
                "还没有可用的已生成 MP3，请先生成语音或手动选择文件。",
                "There is no generated MP3 yet. Generate speech or choose a file first.",
            ));
            return;
        };
        self.asr_input_path = Some(path);
        self.transcription_cues.clear();
        self.transcription_output_path = None;
        self.transcription_dirty = false;
    }

    fn start_transcription(&mut self) {
        if self.transcribing || self.generating || self.previewing {
            return;
        }
        let Some(input_path) = self.asr_input_path.clone() else {
            self.status = Some(StatusMessage::new(
                StatusKind::Error,
                "请先选择需要识别的音频或视频。",
                "Choose an audio or video file before transcribing.",
            ));
            return;
        };
        if !input_path.is_file() {
            self.status = Some(StatusMessage::new(
                StatusKind::Error,
                "所选音频已不存在，请重新选择。",
                "The selected audio no longer exists. Choose it again.",
            ));
            return;
        }

        let command = WorkerCommand::TranscribeMedia {
            input_path,
            language: self.recognition_language,
        };
        match self.command_tx.send(command) {
            Ok(()) => {
                self.transcribing = true;
                self.loading_asr_model = true;
                self.transcription_progress = 0.0;
                self.transcription_remaining_seconds = 0;
                self.transcription_cues.clear();
                self.transcription_output_path = None;
                self.transcription_dirty = false;
                self.status = Some(StatusMessage::new(
                    StatusKind::Info,
                    "正在检查并加载本地 Whisper 模型…",
                    "Checking and loading the local Whisper model…",
                ));
            }
            Err(_) => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Error,
                    "后台服务已停止，请重新启动应用。",
                    "The background worker has stopped. Restart the app.",
                ));
            }
        }
    }

    fn save_transcription(&mut self) {
        if self.transcribing || self.saving_transcription || self.transcription_cues.is_empty() {
            return;
        }
        let mut cues = self.transcription_cues.clone();
        cues.retain(|cue| !cue.text.trim().is_empty());
        if cues.is_empty() {
            self.status = Some(StatusMessage::new(
                StatusKind::Error,
                "所有字幕文字都为空，无法保存。请至少保留一条字幕。",
                "All subtitle cues are empty. Keep at least one cue before saving.",
            ));
            return;
        }
        let stem = self
            .asr_input_path
            .as_ref()
            .and_then(|path| path.file_stem())
            .and_then(|name| name.to_str())
            .unwrap_or("transcript");
        let default_name = format!("{stem}.{}", self.subtitle_export_format.extension());
        let Some(output_path) = rfd::FileDialog::new()
            .set_title(
                self.ui_language
                    .text("保存已编辑字幕", "Save edited subtitles"),
            )
            .set_file_name(default_name)
            .set_can_create_directories(true)
            .add_filter(
                self.subtitle_export_format.label(),
                &[self.subtitle_export_format.extension()],
            )
            .save_file()
            .map(|path| ensure_subtitle_extension(path, self.subtitle_export_format))
        else {
            return;
        };
        let command = WorkerCommand::SaveTranscription {
            output_path,
            cues,
            format: self.subtitle_export_format,
        };
        match self.command_tx.send(command) {
            Ok(()) => {
                self.saving_transcription = true;
                self.status = Some(StatusMessage::new(
                    StatusKind::Info,
                    "正在保存已编辑字幕…",
                    "Saving the edited subtitles…",
                ));
            }
            Err(_) => {
                self.status = Some(StatusMessage::new(
                    StatusKind::Error,
                    "后台服务已停止，请重新启动应用。",
                    "The background worker has stopped. Restart the app.",
                ));
            }
        }
    }

    fn selected_voice_label(&self) -> String {
        match self.tts_engine {
            TtsEngine::Edge => self
                .selected_voice
                .and_then(|index| self.voices.get(index))
                .map(|voice| voice.label(self.ui_language))
                .unwrap_or_else(|| {
                    self.ui_language
                        .text("尚未选择音色", "No voice selected")
                        .to_owned()
                }),
            TtsEngine::Qwen3Local if self.qwen_voice_mode == QwenVoiceMode::Clone => self
                .clone_reference_path
                .as_ref()
                .and_then(|path| path.file_name())
                .and_then(|name| name.to_str())
                .unwrap_or_else(|| {
                    self.ui_language
                        .text("尚未选择参考音频", "No reference audio")
                })
                .to_owned(),
            TtsEngine::Qwen3Local => match self.ui_language {
                UiLanguage::Chinese => self.selected_qwen_voice.chinese_label().to_owned(),
                UiLanguage::English => self.selected_qwen_voice.english_label().to_owned(),
            },
            TtsEngine::IndexTts25 => self
                .clone_reference_path
                .as_ref()
                .and_then(|path| path.file_name())
                .and_then(|name| name.to_str())
                .unwrap_or_else(|| {
                    self.ui_language
                        .text("尚未选择参考音频", "No reference audio")
                })
                .to_owned(),
        }
    }

    fn matching_voice_count(&self) -> usize {
        match self.tts_engine {
            TtsEngine::Edge => self
                .voices
                .iter()
                .filter(|voice| voice.matches(&self.voice_filter))
                .count(),
            TtsEngine::Qwen3Local => QwenVoice::ALL
                .iter()
                .filter(|voice| voice.matches(&self.voice_filter))
                .count(),
            TtsEngine::IndexTts25 => usize::from(self.clone_reference_path.is_some()),
        }
    }

    fn active_voice_selection(&self) -> Option<VoiceSelection> {
        match self.tts_engine {
            TtsEngine::Edge => self
                .selected_voice
                .and_then(|index| self.voices.get(index))
                .map(|voice| VoiceSelection::Edge(voice.short_name.clone())),
            TtsEngine::Qwen3Local if self.qwen_voice_mode == QwenVoiceMode::Clone => {
                let reference_path = self.clone_reference_path.as_ref()?;
                (self.clone_authorized && reference_path.is_file()).then(|| {
                    VoiceSelection::Qwen3Clone(QwenCloneSelection {
                        version: self.selected_qwen_version,
                        reference_path: reference_path.clone(),
                        reference_text: self.clone_reference_text.trim().to_owned(),
                    })
                })
            }
            TtsEngine::Qwen3Local => Some(VoiceSelection::Qwen3Preset(QwenSelection {
                version: self.selected_qwen_version,
                voice: self.selected_qwen_voice,
            })),
            TtsEngine::IndexTts25 => {
                let reference_path = self.clone_reference_path.as_ref()?;
                (self.clone_authorized && reference_path.is_file()).then(|| {
                    VoiceSelection::IndexTts25(IndexTtsSelection {
                        reference_path: reference_path.clone(),
                    })
                })
            }
        }
    }

    fn selected_qwen_kind(&self) -> QwenModelKind {
        match self.qwen_voice_mode {
            QwenVoiceMode::Preset => QwenModelKind::CustomVoice,
            QwenVoiceMode::Clone => QwenModelKind::VoiceClone,
        }
    }

    fn qwen_generation_settings(&self) -> QwenGenerationSettings {
        QwenGenerationSettings {
            max_length: self.qwen_max_length,
            temperature: self.qwen_temperature,
            top_k: self.qwen_top_k,
            top_p: self.qwen_top_p,
            repetition_penalty: self.qwen_repetition_penalty,
            min_new_tokens: self.qwen_min_new_tokens,
            seed: (!self.qwen_random_seed).then_some(self.qwen_seed),
        }
        .validated()
    }

    fn indextts_settings(&self) -> IndexTtsSettings {
        let rate_factor = (1.0 / (1.0 + self.rate_percent as f64 / 100.0)).clamp(0.5, 2.0);
        IndexTtsSettings {
            duration_factor: rate_factor * self.indextts_duration_factor,
            text_normalization: self.indextts_text_normalization,
            max_text_tokens_per_segment: self.indextts_max_text_tokens,
            interval_silence_ms: self.indextts_interval_silence_ms,
            use_random: self.indextts_use_random,
            emo_alpha: self.indextts_emo_alpha,
            use_emo_text: self.indextts_use_emo_text,
            emo_text: self.indextts_emo_text.clone(),
            do_sample: self.indextts_do_sample,
            temperature: self.indextts_temperature,
            top_k: self.indextts_top_k,
            top_p: self.indextts_top_p,
            repetition_penalty: self.indextts_repetition_penalty,
            length_penalty: self.indextts_length_penalty,
            num_beams: self.indextts_num_beams,
            max_mel_tokens: self.indextts_max_mel_tokens,
        }
        .validated()
    }

    fn reset_generation_settings(&mut self) {
        self.rate_percent = 0;
        self.volume_percent = 0;
        let qwen = QwenGenerationSettings::default();
        self.qwen_max_length = qwen.max_length;
        self.qwen_temperature = qwen.temperature;
        self.qwen_top_k = qwen.top_k;
        self.qwen_top_p = qwen.top_p;
        self.qwen_repetition_penalty = qwen.repetition_penalty;
        self.qwen_min_new_tokens = qwen.min_new_tokens;
        self.qwen_seed = 42;
        self.qwen_random_seed = false;
        let index = IndexTtsSettings::default();
        self.indextts_duration_factor = index.duration_factor;
        self.indextts_text_normalization = index.text_normalization;
        self.indextts_max_text_tokens = index.max_text_tokens_per_segment;
        self.indextts_interval_silence_ms = index.interval_silence_ms;
        self.indextts_use_random = index.use_random;
        self.indextts_emo_alpha = index.emo_alpha;
        self.indextts_use_emo_text = index.use_emo_text;
        self.indextts_emo_text = index.emo_text;
        self.indextts_do_sample = index.do_sample;
        self.indextts_temperature = index.temperature;
        self.indextts_top_k = index.top_k;
        self.indextts_top_p = index.top_p;
        self.indextts_repetition_penalty = index.repetition_penalty;
        self.indextts_length_penalty = index.length_penalty;
        self.indextts_num_beams = index.num_beams;
        self.indextts_max_mel_tokens = index.max_mel_tokens;
    }

    fn voice_selection_error(&self) -> (&'static str, &'static str) {
        if (self.tts_engine == TtsEngine::Qwen3Local
            && self.qwen_voice_mode == QwenVoiceMode::Clone)
            || self.tts_engine == TtsEngine::IndexTts25
        {
            if self.clone_reference_path.is_none() {
                return (
                    "请先选择一段 WAV/MP3 参考音频。",
                    "Choose a WAV/MP3 reference recording first.",
                );
            }
            if !self.clone_authorized {
                return (
                    "请先确认已获得该声音所有者授权。",
                    "Confirm that you have the voice owner's permission first.",
                );
            }
        }
        ("请先选择一个音色。", "Choose a voice first.")
    }

    fn choose_clone_reference(&mut self) {
        if self.tts_controls_busy() {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .set_title(self.ui_language.text(
                "选择干净的单人参考音频",
                "Choose a clean single-speaker reference",
            ))
            .add_filter(
                self.ui_language.text("WAV / MP3 音频", "WAV / MP3 audio"),
                &["wav", "mp3"],
            )
            .pick_file()
        else {
            return;
        };
        self.clone_reference_path = Some(path.clone());
        self.clone_authorized = false;
        self.status = Some(StatusMessage::new(
            StatusKind::Info,
            format!("已选择本地参考音频：{}", path.display()),
            format!("Selected local reference audio: {}", path.display()),
        ));
    }

    fn tts_controls_busy(&self) -> bool {
        self.previewing
            || self.generating
            || self.transcribing
            || self.saving_transcription
            || self.qwen_model_preparing
            || self.indextts_model_preparing
            || (self.tts_engine == TtsEngine::Edge && self.fetching_voices)
    }
}

impl eframe::App for TtsApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.process_worker_events();

        // Poll only while work is active. This keeps the idle app at near-zero
        // repaint CPU while still noticing worker results promptly.
        if self.fetching_voices
            || self.previewing
            || self.generating
            || self.transcribing
            || self.saving_transcription
            || self.qwen_model_preparing
            || self.indextts_model_preparing
            || self.generated_audio_loading
            || self
                .generated_audio_player
                .as_ref()
                .is_some_and(GeneratedAudioPlayer::is_playing)
        {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }

        let language = self.ui_language;
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(BACKGROUND))
            .show(ui, |ui| {
                // The main page deliberately has no ScrollArea. Its two cards
                // share the available height; only the editor and voice popup
                // scroll their own content.
                egui::Frame::new()
                    .inner_margin(egui::Margin::symmetric(24, 18))
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        self.show_header(ui);
                        ui.add_space(14.0);
                        self.show_workspace_switcher(ui, language);
                        ui.add_space(14.0);

                        // Reserve the complete action/status/privacy area before
                        // sizing the cards. The previous fixed 88 px estimate
                        // was too small once a success message wrapped, pushing
                        // the action row below the macOS window edge.
                        let footer_height = match self.workspace_mode {
                            WorkspaceMode::TextToSpeech
                                if self.generating
                                    || self.qwen_model_preparing
                                    || self.indextts_model_preparing =>
                            {
                                168.0
                            }
                            WorkspaceMode::TextToSpeech
                                if self.generated_audio_loading
                                    || self.generated_audio_player.is_some() =>
                            {
                                154.0
                            }
                            WorkspaceMode::TextToSpeech
                                if self.previewing
                                    || (self.tts_engine == TtsEngine::Edge
                                        && self.fetching_voices) =>
                            {
                                134.0
                            }
                            WorkspaceMode::TextToSpeech => 98.0,
                            WorkspaceMode::AudioToSubtitles
                                if self.transcribing || self.saving_transcription =>
                            {
                                148.0
                            }
                            WorkspaceMode::AudioToSubtitles => 118.0,
                        };
                        let visible_height = (ui.clip_rect().bottom() - ui.cursor().min.y).max(0.0);
                        let workspace_height = (visible_height - footer_height).max(280.0);
                        let gap = 14.0;
                        let visible_width = (ui.clip_rect().right() - ui.cursor().min.x).max(0.0);
                        let total_width = ui.available_width().min(visible_width);
                        let (workspace_rect, _) = ui.allocate_exact_size(
                            egui::vec2(total_width, workspace_height),
                            egui::Sense::hover(),
                        );
                        let mut workspace_ui = ui.new_child(
                            egui::UiBuilder::new()
                                .id_salt("fixed-workspace")
                                .max_rect(workspace_rect)
                                .layout(egui::Layout::top_down(egui::Align::Min)),
                        );
                        workspace_ui.set_clip_rect(workspace_rect.intersect(ui.clip_rect()));
                        match self.workspace_mode {
                            WorkspaceMode::TextToSpeech => {
                                let voice_width = (total_width * 0.39).clamp(360.0, 420.0);
                                let text_width = (total_width - voice_width - gap).max(420.0);
                                workspace_ui.horizontal_top(|ui| {
                                    ui.spacing_mut().item_spacing.x = gap;
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(voice_width, workspace_height),
                                        egui::Layout::top_down(egui::Align::Min),
                                        |ui| self.show_voice_card(ui, language, workspace_height),
                                    );
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(text_width, workspace_height),
                                        egui::Layout::top_down(egui::Align::Min),
                                        |ui| self.show_text_card(ui, language, workspace_height),
                                    );
                                });
                            }
                            WorkspaceMode::AudioToSubtitles => {
                                self.show_asr_workspace(
                                    &mut workspace_ui,
                                    language,
                                    workspace_height,
                                    gap,
                                );
                            }
                        }
                        ui.add_space(14.0);
                        match self.workspace_mode {
                            WorkspaceMode::TextToSpeech => self.show_generate_area(ui, language),
                            WorkspaceMode::AudioToSubtitles => {
                                self.show_transcription_area(ui, language)
                            }
                        }
                    });
            });
        if self.qwen_manual_help_open {
            let context = ui.ctx().clone();
            self.show_qwen_manual_help(&context, language);
        }
        if self.indextts_manual_help_open {
            let context = ui.ctx().clone();
            self.show_indextts_manual_help(&context, language);
        }
    }
}

impl TtsApp {
    fn show_indextts_manual_help(&mut self, context: &egui::Context, language: UiLanguage) {
        let mut open = self.indextts_manual_help_open;
        let mut import_clicked = false;
        egui::Window::new(language.text(
            "IndexTTS-2.5 离线模型导入",
            "IndexTTS-2.5 offline model import",
        ))
        .id(egui::Id::new("indextts-offline-model-help"))
        .open(&mut open)
        .collapsible(false)
        .resizable(true)
        .default_width(600.0)
        .show(context, |ui| {
            ui.label(
                egui::RichText::new(language.text(
                    "下载不可用时，可在另一台机器准备完整的官方 IndexTTS-2.5 v2.5.0 模型目录，再选择该目录导入。",
                    "When downloading is unavailable, prepare the complete official IndexTTS-2.5 v2.5.0 model directory on another machine and import it here.",
                ))
                .size(13.0)
                .color(TEXT_PRIMARY),
            );
            ui.add_space(8.0);
            ui.hyperlink_to(
                "IndexTeam/IndexTTS-2.5",
                "https://huggingface.co/IndexTeam/IndexTTS-2.5",
            );
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(language.text(
                    "请选择包含 config.yaml 的完整 models 文件夹，不能只选择单个权重文件。程序会校验 15 个必需文件，并优先使用硬链接；跨磁盘时自动复制。",
                    "Select the complete models folder containing config.yaml, not a single weight file. The app validates all 15 required files and uses hard links when possible, copying across volumes.",
                ))
                .size(12.0)
                .color(TEXT_SECONDARY),
            );
            ui.add_space(12.0);
            if ui
                .add_enabled(
                    !self.tts_controls_busy(),
                    egui::Button::new(
                        egui::RichText::new(language.text(
                            "选择模型文件夹并导入",
                            "Choose model folder and import",
                        ))
                        .strong()
                        .color(egui::Color32::WHITE),
                    )
                    .fill(PRIMARY)
                    .stroke(egui::Stroke::NONE)
                    .corner_radius(8)
                    .min_size(egui::vec2(190.0, 36.0)),
                )
                .clicked()
            {
                import_clicked = true;
            }
        });
        self.indextts_manual_help_open = open && !import_clicked;
        if import_clicked {
            self.import_indextts_model();
        }
    }

    fn show_qwen_manual_help(&mut self, context: &egui::Context, language: UiLanguage) {
        let version = self.selected_qwen_version;
        let kind = self.selected_qwen_kind();
        let mut open = self.qwen_manual_help_open;
        let mut import_clicked = false;
        let hf_command = format!(
            "huggingface-cli download {} --local-dir {}",
            version.model_id(kind),
            version.model_folder_name(kind)
        );
        let modelscope_command = format!(
            "modelscope download --model {} --local_dir {}",
            version.model_id(kind),
            version.model_folder_name(kind)
        );

        egui::Window::new(language.text("离线模型下载与导入", "Offline model download and import"))
            .id(egui::Id::new("qwen-offline-model-help"))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(620.0)
            .show(context, |ui| {
                ui.label(
                    egui::RichText::new(match language {
                        UiLanguage::Chinese => format!(
                            "自动下载失败时，请从下面任一官方模型页面下载完整的 Qwen3-TTS {} {} 文件夹。",
                            version.short_label(),
                            kind.short_label()
                        ),
                        UiLanguage::English => format!(
                            "If automatic download fails, download the complete Qwen3-TTS {} {} folder from either official model page below.",
                            version.short_label(),
                            kind.short_label()
                        ),
                    })
                    .size(13.0)
                    .color(TEXT_PRIMARY),
                );
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    ui.hyperlink_to("Hugging Face", version.hugging_face_url(kind));
                    ui.label("·");
                    ui.hyperlink_to("ModelScope（中国大陆）", version.model_scope_url(kind));
                });
                ui.add_space(10.0);
                ui.label(
                    egui::RichText::new(language.text(
                        "推荐命令（二选一）：",
                        "Recommended commands (choose one):",
                    ))
                    .strong()
                    .color(TEXT_PRIMARY),
                );
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(&hf_command)
                        .monospace()
                        .size(11.0)
                        .color(TEXT_SECONDARY),
                );
                ui.label(
                    egui::RichText::new(&modelscope_command)
                        .monospace()
                        .size(11.0)
                        .color(TEXT_SECONDARY),
                );
                ui.add_space(10.0);
                ui.label(
                    egui::RichText::new(language.text(
                        "所选文件夹至少需要：",
                        "The selected folder must contain:",
                    ))
                    .strong()
                    .color(TEXT_PRIMARY),
                );
                ui.label(
                    egui::RichText::new(
                        "model.safetensors\nconfig.json\nspeech_tokenizer/model.safetensors\ntokenizer.json  或  vocab.json + merges.txt",
                    )
                    .monospace()
                    .size(11.0)
                    .color(TEXT_SECONDARY),
                );
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(language.text(
                        "请选择完整模型文件夹，不要只选择或只下载 model.safetensors。程序会校验 0.6B/1.7B 是否与当前选择一致，并优先使用硬链接安装；跨磁盘时会在后台复制。",
                        "Choose the complete model folder, not only model.safetensors. The app validates that 0.6B/1.7B matches the current selection and installs with hard links when possible; cross-volume files are copied in the background.",
                    ))
                    .size(12.0)
                    .color(TEXT_SECONDARY),
                );
                ui.add_space(12.0);
                if ui
                    .add_enabled(
                        !self.tts_controls_busy(),
                        egui::Button::new(
                            egui::RichText::new(language.text(
                                "选择模型文件夹并导入",
                                "Choose model folder and import",
                            ))
                            .strong()
                            .color(egui::Color32::WHITE),
                        )
                        .fill(PRIMARY)
                        .stroke(egui::Stroke::NONE)
                        .corner_radius(8)
                        .min_size(egui::vec2(190.0, 36.0)),
                    )
                    .clicked()
                {
                    import_clicked = true;
                }
            });
        self.qwen_manual_help_open = open && !import_clicked;
        if import_clicked {
            self.import_selected_qwen_model();
        }
    }

    fn show_header(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let (icon_rect, _) =
                ui.allocate_exact_size(egui::vec2(48.0, 48.0), egui::Sense::hover());
            ui.painter()
                .rect_filled(icon_rect, egui::CornerRadius::same(13), PRIMARY);

            let center = icon_rect.center();
            let bar_heights = [12.0, 22.0, 30.0, 20.0, 10.0];
            for (index, height) in bar_heights.into_iter().enumerate() {
                let x = center.x - 14.0 + index as f32 * 7.0;
                ui.painter().line_segment(
                    [
                        egui::pos2(x, center.y - height / 2.0),
                        egui::pos2(x, center.y + height / 2.0),
                    ],
                    egui::Stroke::new(3.0, egui::Color32::WHITE),
                );
            }

            ui.add_space(10.0);
            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new(
                        self.ui_language
                            .text("Edge TTS 语音工作室", "Edge TTS Studio"),
                    )
                    .size(24.0)
                    .strong()
                    .color(TEXT_PRIMARY),
                );
                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new(match self.workspace_mode {
                        WorkspaceMode::TextToSpeech => self.ui_language.text(
                            "把文字快速转换为自然流畅的 MP3 语音",
                            "Turn text into natural-sounding MP3 speech",
                        ),
                        WorkspaceMode::AudioToSubtitles => self.ui_language.text(
                            "使用本地 Whisper 从音频或视频生成可编辑字幕",
                            "Create editable subtitles from audio or video with local Whisper",
                        ),
                    })
                    .size(14.0)
                    .color(TEXT_SECONDARY),
                );
            });

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let english = language_button(ui, "EN", self.ui_language == UiLanguage::English);
                let chinese = language_button(ui, "中文", self.ui_language == UiLanguage::Chinese);
                if english.clicked() {
                    self.ui_language = UiLanguage::English;
                }
                if chinese.clicked() {
                    self.ui_language = UiLanguage::Chinese;
                }
            });
        });
    }

    fn show_workspace_switcher(&mut self, ui: &mut egui::Ui, language: UiLanguage) {
        egui::Frame::new()
            .fill(PRIMARY_SOFT)
            .corner_radius(11)
            .inner_margin(egui::Margin::same(4))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let tts = mode_button(
                        ui,
                        language.text("文字 / 字幕转语音", "Text / subtitles to speech"),
                        self.workspace_mode == WorkspaceMode::TextToSpeech,
                        178.0,
                    );
                    let asr = mode_button(
                        ui,
                        language.text("音频转字幕", "Audio to subtitles"),
                        self.workspace_mode == WorkspaceMode::AudioToSubtitles,
                        150.0,
                    );
                    if tts.clicked() {
                        self.workspace_mode = WorkspaceMode::TextToSpeech;
                    }
                    if asr.clicked() {
                        self.workspace_mode = WorkspaceMode::AudioToSubtitles;
                        if self.asr_input_path.is_none()
                            && let Some(path) = self
                                .last_generated_audio
                                .as_ref()
                                .filter(|path| path.is_file())
                                .cloned()
                        {
                            self.asr_input_path = Some(path);
                        }
                    }
                });
            });
    }

    fn show_voice_card(&mut self, ui: &mut egui::Ui, language: UiLanguage, card_height: f32) {
        let layout = VoiceCardLayout::for_height(card_height);
        let clone_mode = (self.tts_engine == TtsEngine::Qwen3Local
            && self.qwen_voice_mode == QwenVoiceMode::Clone)
            || self.tts_engine == TtsEngine::IndexTts25;
        card_frame().show(ui, |ui| {
            let scroll_height = (card_height - 36.0).max(0.0);
            ui.set_height(scroll_height);
            egui::ScrollArea::vertical()
                .id_salt("voice-card-scroll")
                .max_height(scroll_height)
                .min_scrolled_height(scroll_height)
                .auto_shrink([false, false])
                .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
                .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            let operation_busy = self.previewing
                || self.generating
                || self.transcribing
                || self.qwen_model_preparing
                || self.indextts_model_preparing;
            let busy = self.tts_controls_busy();

            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(language.text("语音引擎与音色", "Engine and voice"))
                        .size(17.0)
                        .strong()
                        .color(TEXT_PRIMARY),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.tts_engine == TtsEngine::Edge {
                        let refresh = ui
                            .add_enabled(
                                !busy,
                                egui::Button::new(
                                    egui::RichText::new("↻").size(16.0).color(PRIMARY),
                                )
                                .fill(PRIMARY_SOFT)
                                .stroke(egui::Stroke::NONE)
                                .corner_radius(8)
                                .min_size(egui::vec2(30.0, 30.0)),
                            )
                            .on_hover_text(language.text("刷新 Edge 音色", "Refresh Edge voices"));
                        if refresh.clicked() {
                            self.reload_voices();
                        }
                    }
                });
            });

            let engine_description = match (self.tts_engine, language) {
                (TtsEngine::Edge, UiLanguage::Chinese) => {
                    "Edge 在线音色，支持多语言搜索".to_owned()
                }
                (TtsEngine::Edge, UiLanguage::English) => {
                    "Online Edge voices with multilingual search".to_owned()
                }
                (TtsEngine::Qwen3Local, UiLanguage::Chinese)
                    if self.qwen_voice_mode == QwenVoiceMode::Clone =>
                {
                    format!(
                        "Qwen3 {} Base · 3–15 秒干净单人录音 · 首次约 {}",
                        self.selected_qwen_version.short_label(),
                        self.selected_qwen_version.download_size_label()
                    )
                }
                (TtsEngine::Qwen3Local, UiLanguage::English)
                    if self.qwen_voice_mode == QwenVoiceMode::Clone =>
                {
                    format!(
                        "Qwen3 {} Base · clean 3–15s voice · about {} first use",
                        self.selected_qwen_version.short_label(),
                        self.selected_qwen_version.download_size_label()
                    )
                }
                (TtsEngine::Qwen3Local, UiLanguage::Chinese) => format!(
                    "Qwen3 {} {} 本地生成 · 首次约 {}",
                    self.selected_qwen_version.short_label(),
                    self.selected_qwen_kind().short_label(),
                    self.selected_qwen_version.download_size_label()
                ),
                (TtsEngine::Qwen3Local, UiLanguage::English) => format!(
                    "Local Qwen3 {} {} · about {} on first use",
                    self.selected_qwen_version.short_label(),
                    self.selected_qwen_kind().short_label(),
                    self.selected_qwen_version.download_size_label()
                ),
                (TtsEngine::IndexTts25, UiLanguage::Chinese) =>
                    "IndexTTS-2.5 官方 v2.5.0 · 3–15 秒参考音频 · 本机推理".to_owned(),
                (TtsEngine::IndexTts25, UiLanguage::English) =>
                    "Official IndexTTS-2.5 v2.5.0 · clean 3–15s reference · local inference"
                        .to_owned(),
            };
            ui.label(
                egui::RichText::new(engine_description)
                    .size(12.0)
                    .color(TEXT_SECONDARY),
            );

            ui.add_space(layout.engine_gap);
            let previous_engine = self.tts_engine;
            let mut chosen_engine = None;
            egui::Frame::new()
                .fill(EDITOR_BACKGROUND)
                .stroke(egui::Stroke::new(1.0, BORDER))
                .corner_radius(10)
                .inner_margin(egui::Margin::same(3))
                .show(ui, |ui| {
                    ui.add_enabled_ui(!operation_busy, |ui| {
                        ui.horizontal(|ui| {
                            let gap = ui.spacing().item_spacing.x;
                            let segment_width =
                                ((ui.available_width() - gap * 3.0) / 4.0).max(66.0);
                            let edge = engine_button(
                                ui,
                                language.text("Edge 在线", "Edge online"),
                                self.tts_engine == TtsEngine::Edge,
                                segment_width,
                            );
                            let qwen_small = engine_button(
                                ui,
                                if self.qwen_voice_mode == QwenVoiceMode::Clone {
                                    "0.6B Base"
                                } else {
                                    "Qwen 0.6B"
                                },
                                self.tts_engine == TtsEngine::Qwen3Local
                                    && self.selected_qwen_version == QwenModelVersion::Small0_6B,
                                segment_width,
                            )
                            .on_hover_text(language.text(
                                "轻量版本，首次约下载 2.4 GB",
                                "Lightweight model; about 2.4 GB on first use",
                            ));
                            let qwen_large = engine_button(
                                ui,
                                if self.qwen_voice_mode == QwenVoiceMode::Clone {
                                    "1.7B Base"
                                } else {
                                    "Qwen 1.7B"
                                },
                                self.tts_engine == TtsEngine::Qwen3Local
                                    && self.selected_qwen_version == QwenModelVersion::Large1_7B,
                                segment_width,
                            )
                            .on_hover_text(language.text(
                                "高质量版本，首次约下载 4.5 GB，内存占用更高",
                                "Higher-quality model; about 4.5 GB on first use and uses more memory",
                            ));
                            let index_tts = engine_button(
                                ui,
                                "IndexTTS 2.5",
                                self.tts_engine == TtsEngine::IndexTts25,
                                segment_width,
                            )
                            .on_hover_text(language.text(
                                "官方 v2.5.0，支持 macOS MPS / NVIDIA CUDA / CPU",
                                "Official v2.5.0 with macOS MPS, NVIDIA CUDA, and CPU support",
                            ));
                            if edge.clicked() {
                                chosen_engine = Some((TtsEngine::Edge, self.selected_qwen_version));
                            } else if qwen_small.clicked() {
                                chosen_engine =
                                    Some((TtsEngine::Qwen3Local, QwenModelVersion::Small0_6B));
                            } else if qwen_large.clicked() {
                                chosen_engine =
                                    Some((TtsEngine::Qwen3Local, QwenModelVersion::Large1_7B));
                            } else if index_tts.clicked() {
                                chosen_engine =
                                    Some((TtsEngine::IndexTts25, self.selected_qwen_version));
                            }
                        });
                    });
                });
            if let Some((engine, version)) = chosen_engine {
                self.tts_engine = engine;
                self.selected_qwen_version = version;
                if engine == TtsEngine::Edge && self.voice_filter.trim().is_empty() {
                    self.voice_filter = language.text("中文", "English").to_owned();
                } else if engine != TtsEngine::Edge && previous_engine == TtsEngine::Edge {
                    self.voice_filter.clear();
                }
            }

            if self.tts_engine == TtsEngine::Qwen3Local {
                ui.add_space(layout.field_gap);
                egui::Frame::new()
                    .fill(PRIMARY_SOFT)
                    .corner_radius(9)
                    .inner_margin(egui::Margin::same(3))
                    .show(ui, |ui| {
                        ui.add_enabled_ui(!operation_busy, |ui| {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 4.0;
                                let width = ((ui.available_width() - 4.0) / 2.0).max(90.0);
                                let preset = mode_button(
                                    ui,
                                    language.text("预置音色", "Preset voices"),
                                    self.qwen_voice_mode == QwenVoiceMode::Preset,
                                    width,
                                );
                                let clone = mode_button(
                                    ui,
                                    language.text("音色克隆", "Voice clone"),
                                    self.qwen_voice_mode == QwenVoiceMode::Clone,
                                    width,
                                );
                                if preset.clicked() {
                                    self.qwen_voice_mode = QwenVoiceMode::Preset;
                                }
                                if clone.clicked() {
                                    self.qwen_voice_mode = QwenVoiceMode::Clone;
                                }
                            });
                        });
                    });
            }

            if clone_mode {
                self.show_clone_setup(ui, language, layout, busy);
            } else {
            ui.add_space(layout.field_gap);
            ui.add_enabled_ui(!busy, |ui| {
                input_frame().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let row_width = ui.available_width();
                        let has_clear = !self.voice_filter.is_empty();
                        let clear_slot = if has_clear { 54.0 } else { 0.0 };
                        let (search_rect, _) =
                            ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
                        let painter = ui.painter();
                        let stroke = egui::Stroke::new(1.6, TEXT_SECONDARY);
                        let lens_center = search_rect.center() - egui::vec2(1.5, 1.5);
                        painter.circle_stroke(lens_center, 5.2, stroke);
                        painter.line_segment(
                            [
                                lens_center + egui::vec2(3.8, 3.8),
                                lens_center + egui::vec2(7.2, 7.2),
                            ],
                            stroke,
                        );
                        ui.add(
                            egui::TextEdit::singleline(&mut self.voice_filter)
                                .hint_text(match self.tts_engine {
                                    TtsEngine::Edge => language.text(
                                        "搜索音色，如：晓晓、英语、en-US",
                                        "Search voices, e.g. Xiaoxiao, English, en-US",
                                    ),
                                    TtsEngine::Qwen3Local => language.text(
                                        "搜索本地音色，如：Vivian、中文、英语",
                                        "Search local voices, e.g. Vivian, Chinese",
                                    ),
                                    TtsEngine::IndexTts25 => language.text(
                                        "IndexTTS 使用上方参考音频",
                                        "IndexTTS uses the reference audio above",
                                    ),
                                })
                                .desired_width((row_width - 34.0 - clear_slot).max(80.0))
                                .frame(egui::Frame::NONE)
                                .text_color(TEXT_PRIMARY),
                        );
                        if has_clear
                            && ui
                                .add(
                                    egui::Button::new(
                                        egui::RichText::new(language.text("清除", "Clear"))
                                            .size(12.0)
                                            .color(TEXT_SECONDARY),
                                    )
                                    .fill(egui::Color32::TRANSPARENT)
                                    .stroke(egui::Stroke::NONE)
                                    .corner_radius(7)
                                    .min_size(egui::vec2(46.0, 26.0)),
                                )
                                .on_hover_text(language.text("清除搜索", "Clear search"))
                                .clicked()
                        {
                            self.voice_filter.clear();
                        }
                    });
                });

                ui.add_space(layout.field_gap);
                let selected_text = self.selected_voice_label();
                let mut edge_selection = self.selected_voice;
                let mut qwen_selection = self.selected_qwen_voice;
                ui.scope(|ui| {
                    configure_voice_combo_style(ui);
                    egui::ComboBox::from_id_salt(match self.tts_engine {
                        TtsEngine::Edge => "edge-voice-combo",
                        TtsEngine::Qwen3Local => "qwen-voice-combo",
                        TtsEngine::IndexTts25 => "indextts-reference",
                    })
                    .width(ui.available_width())
                    .height(280.0)
                    .truncate()
                    .selected_text(egui::RichText::new(selected_text).color(TEXT_PRIMARY))
                    .popup_style(voice_popup_style())
                    .show_ui(ui, |ui| {
                        let mut match_count = 0;
                        match self.tts_engine {
                            TtsEngine::Edge => {
                                for (index, voice) in self.voices.iter().enumerate() {
                                    if voice.matches(&self.voice_filter) {
                                        match_count += 1;
                                        ui.selectable_value(
                                            &mut edge_selection,
                                            Some(index),
                                            voice.label(language),
                                        );
                                    }
                                }
                            }
                            TtsEngine::Qwen3Local => {
                                for voice in QwenVoice::ALL {
                                    if voice.matches(&self.voice_filter) {
                                        match_count += 1;
                                        let label = match language {
                                            UiLanguage::Chinese => voice.chinese_label(),
                                            UiLanguage::English => voice.english_label(),
                                        };
                                        ui.selectable_value(&mut qwen_selection, voice, label);
                                    }
                                }
                            }
                            TtsEngine::IndexTts25 => {}
                        }

                        if match_count == 0 {
                            ui.label(
                                egui::RichText::new(
                                    language.text("没有找到匹配的音色", "No matching voices"),
                                )
                                .color(TEXT_SECONDARY),
                            );
                        }
                    });
                });
                self.selected_voice = edge_selection;
                self.selected_qwen_voice = qwen_selection;
            });

            ui.add_space(layout.combo_summary_gap);
            let matching = self.matching_voice_count();
            let summary = match (self.tts_engine, language) {
                (TtsEngine::Edge, UiLanguage::Chinese) => {
                    format!(
                        "Edge 共 {} 个音色，当前显示 {} 个",
                        self.voices.len(),
                        matching
                    )
                }
                (TtsEngine::Edge, UiLanguage::English) => {
                    format!("{} Edge voices · {} shown", self.voices.len(), matching)
                }
                (TtsEngine::Qwen3Local, UiLanguage::Chinese) => {
                    let state = if self.qwen_model_ready
                        == Some((self.selected_qwen_version, QwenModelKind::CustomVoice))
                    {
                        self.qwen_device.as_deref().unwrap_or("本地设备")
                    } else if self.qwen_model_preparing {
                        "准备中"
                    } else {
                        "待加载"
                    };
                    format!(
                        "Qwen3 {} · {matching}/9 音色 · {state}",
                        self.selected_qwen_version.short_label()
                    )
                }
                (TtsEngine::Qwen3Local, UiLanguage::English) => {
                    let state = if self.qwen_model_ready
                        == Some((self.selected_qwen_version, QwenModelKind::CustomVoice))
                    {
                        self.qwen_device.as_deref().unwrap_or("local device")
                    } else if self.qwen_model_preparing {
                        "preparing"
                    } else {
                        "not loaded"
                    };
                    format!(
                        "Qwen3 {} · {matching}/9 voices · {state}",
                        self.selected_qwen_version.short_label()
                    )
                }
                (TtsEngine::IndexTts25, UiLanguage::Chinese) =>
                    "IndexTTS-2.5 使用参考音频克隆音色".to_owned(),
                (TtsEngine::IndexTts25, UiLanguage::English) =>
                    "IndexTTS-2.5 clones the selected reference voice".to_owned(),
            };
            if self.tts_engine == TtsEngine::Qwen3Local {
                ui.horizontal(|ui| {
                    let gap = ui.spacing().item_spacing.x;
                    let import_width = match language {
                        UiLanguage::Chinese => 96.0,
                        UiLanguage::English => 104.0,
                    };
                    let directory_width = if language == UiLanguage::Chinese {
                        76.0
                    } else {
                        82.0
                    };
                    let download_width = if language == UiLanguage::Chinese {
                        72.0
                    } else {
                        78.0
                    };
                    let selected_model_ready = self.qwen_model_ready
                        == Some((self.selected_qwen_version, QwenModelKind::CustomVoice));
                    let summary_width = (ui.available_width()
                        - import_width
                        - directory_width
                        - download_width
                        - 30.0
                        - gap * 4.0)
                        .max(8.0);
                    ui.add_sized(
                        egui::vec2(summary_width, layout.metadata_height),
                        egui::Label::new(
                            egui::RichText::new(&summary)
                                .size(12.0)
                                .color(TEXT_SECONDARY),
                        )
                        .truncate(),
                    )
                    .on_hover_text(&summary);
                    if ui
                        .add(model_directory_button(language, directory_width))
                        .clicked()
                    {
                        self.open_model_directory(ModelDirectoryKind::Qwen3);
                    }
                    if ui
                        .add_enabled(
                            !busy,
                            model_download_entry_button(
                                language,
                                selected_model_ready,
                                download_width,
                                layout.metadata_height,
                            ),
                        )
                        .on_hover_text(if selected_model_ready {
                            language.text(
                                "删除所选模型缓存并重新下载",
                                "Delete and download the selected model again",
                            )
                        } else {
                            language.text(
                                "下载模型；存在断点文件时从已有进度继续",
                                "Download the model, resuming any partial files",
                            )
                        })
                        .clicked()
                    {
                        if selected_model_ready {
                            self.redownload_selected_qwen_model();
                        } else {
                            self.prepare_selected_qwen_model();
                        }
                    }
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("?").size(12.0).strong().color(PRIMARY),
                            )
                            .fill(EDITOR_BACKGROUND)
                            .stroke(egui::Stroke::new(1.0, BORDER))
                            .corner_radius(8)
                            .min_size(egui::vec2(30.0, layout.metadata_height)),
                        )
                        .on_hover_text(language.text(
                            "查看离线模型下载说明",
                            "Show offline model download instructions",
                        ))
                        .clicked()
                    {
                        self.qwen_manual_help_open = true;
                    }
                    let import = egui::Button::new(
                        egui::RichText::new(language.text("＋ 离线模型", "+ Offline model"))
                            .size(11.0)
                            .strong()
                            .color(PRIMARY),
                    )
                    .fill(PRIMARY_SOFT)
                    .stroke(egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgb(205, 214, 255),
                    ))
                    .corner_radius(8)
                    .min_size(egui::vec2(import_width, layout.metadata_height));
                    if ui.add_enabled(!busy, import).clicked() {
                        self.import_selected_qwen_model();
                    }
                });
                if self.qwen_model_preparing
                    && let Some(progress) = self.qwen_model_progress
                {
                    ui.add(model_download_progress_bar(
                        progress,
                        format!(
                            "{} · {}",
                            language.text("本地模型", "Local model"),
                            self.qwen_model_progress_label
                        ),
                    ));
                }
            } else {
                ui.label(
                    egui::RichText::new(summary)
                        .size(12.0)
                        .color(TEXT_SECONDARY),
                );
            }
            }

            ui.add_space(layout.section_gap);
            egui::Frame::new()
                .fill(EDITOR_BACKGROUND)
                .stroke(egui::Stroke::new(1.0, BORDER))
                .corner_radius(10)
                .inner_margin(egui::Margin::same(if clone_mode {
                    6
                } else {
                    layout.settings_margin
                }))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 3.0;
                    ui.spacing_mut().interact_size.y = 26.0;
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(language.text("声音设置", "Speech settings"))
                                .size(13.0)
                                .strong()
                                .color(TEXT_PRIMARY),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let reset = egui::Button::new(
                                egui::RichText::new(language.text("恢复默认", "Reset"))
                                    .size(11.0)
                                    .color(PRIMARY),
                            )
                            .fill(CARD_BACKGROUND)
                            .stroke(egui::Stroke::new(1.0, BORDER))
                            .corner_radius(7)
                            .min_size(egui::vec2(70.0, 24.0));
                            if ui.add(reset).clicked() {
                                self.reset_generation_settings();
                            }
                        });
                    });
                    adjustment_row(
                        ui,
                        language.text("语速", "Speed"),
                        &mut self.rate_percent,
                        -50..=100,
                    );
                    adjustment_row(
                        ui,
                        language.text("音量", "Volume"),
                        &mut self.volume_percent,
                        -100..=100,
                    );
                    if self.tts_engine == TtsEngine::Qwen3Local {
                        ui.add_space(2.0);
                        ui.add_enabled_ui(!busy, |ui| {
                            ui.collapsing(
                                language.text("Qwen 高级生成参数", "Qwen advanced generation"),
                                |ui| {
                                    model_usize_row(ui, language.text("最大生成帧", "Max frames"), &mut self.qwen_max_length, 128..=4_096, "");
                                    model_f64_row(ui, "Temperature", &mut self.qwen_temperature, 0.0..=2.0, "{:.2}");
                                    model_usize_row(ui, "Top-K", &mut self.qwen_top_k, 1..=200, "");
                                    model_f64_row(ui, "Top-P", &mut self.qwen_top_p, 0.01..=1.0, "{:.2}");
                                    model_f64_row(ui, language.text("重复惩罚", "Repetition penalty"), &mut self.qwen_repetition_penalty, 0.5..=3.0, "{:.2}");
                                    model_usize_row(ui, language.text("最少生成 token", "Min new tokens"), &mut self.qwen_min_new_tokens, 0..=64, "");
                                    ui.horizontal(|ui| {
                                        ui.label(language.text("随机种子", "Seed"));
                                        ui.checkbox(&mut self.qwen_random_seed, language.text("每次随机", "Random each run"));
                                        if !self.qwen_random_seed {
                                            ui.add(egui::DragValue::new(&mut self.qwen_seed).speed(1).range(0..=u64::MAX));
                                        }
                                    });
                                },
                            );
                        });
                    } else if self.tts_engine == TtsEngine::IndexTts25 {
                        ui.add_space(2.0);
                        ui.add_enabled_ui(!busy, |ui| {
                            ui.collapsing(
                                language.text("IndexTTS-2.5 高级生成参数", "IndexTTS-2.5 advanced generation"),
                                |ui| {
                                    model_f64_row(ui, language.text("时长倍率", "Duration factor"), &mut self.indextts_duration_factor, 0.5..=2.0, "{:.2}x");
                                    ui.checkbox(&mut self.indextts_text_normalization, language.text("文本规范化", "Text normalization"));
                                    model_usize_row(ui, language.text("每段最大文本 token", "Max text tokens/segment"), &mut self.indextts_max_text_tokens, 16..=512, "");
                                    model_usize_row(ui, language.text("段间静音", "Silence between segments"), &mut self.indextts_interval_silence_ms, 0..=5_000, " ms");
                                    ui.checkbox(&mut self.indextts_use_random, language.text("随机情绪参考", "Random emotion reference"));
                                    model_f64_row(ui, language.text("情绪强度", "Emotion strength"), &mut self.indextts_emo_alpha, 0.0..=1.0, "{:.2}");
                                    ui.checkbox(&mut self.indextts_use_emo_text, language.text("使用情绪文字", "Use emotion text"));
                                    if self.indextts_use_emo_text {
                                        ui.add(egui::TextEdit::singleline(&mut self.indextts_emo_text).hint_text(language.text("例如：开心、温柔地说", "e.g. speak happily and warmly")).desired_width(ui.available_width()));
                                    }
                                    ui.checkbox(&mut self.indextts_do_sample, language.text("启用采样", "Enable sampling"));
                                    model_f64_row(ui, "Temperature", &mut self.indextts_temperature, 0.0..=2.0, "{:.2}");
                                    model_usize_row(ui, "Top-K", &mut self.indextts_top_k, 1..=200, "");
                                    model_f64_row(ui, "Top-P", &mut self.indextts_top_p, 0.01..=1.0, "{:.2}");
                                    model_f64_row(ui, language.text("重复惩罚", "Repetition penalty"), &mut self.indextts_repetition_penalty, 0.1..=20.0, "{:.2}");
                                    model_f64_row(ui, language.text("长度惩罚", "Length penalty"), &mut self.indextts_length_penalty, -2.0..=2.0, "{:.2}");
                                    model_usize_row(ui, language.text("Beam 数", "Beam count"), &mut self.indextts_num_beams, 1..=8, "");
                                    model_usize_row(ui, language.text("最大梅尔 token", "Max mel tokens"), &mut self.indextts_max_mel_tokens, 128..=4_000, "");
                                },
                            );
                        });
                    }
                });

            ui.add_space(layout.section_gap);
            egui::Frame::new()
                .fill(PRIMARY_SOFT)
                .stroke(egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgb(213, 220, 255),
                ))
                .corner_radius(11)
                .inner_margin(egui::Margin::same(if clone_mode {
                    4
                } else {
                    layout.preview_margin
                }))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(language.text("试听音色", "Voice preview"))
                                .size(13.0)
                                .strong()
                                .color(TEXT_PRIMARY),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                egui::RichText::new(
                                    language.text("取正文前 80 字符", "First 80 characters"),
                                )
                                .size(11.0)
                                .color(TEXT_SECONDARY),
                            );
                        });
                    });
                    // Clone mode has several required setup controls above this
                    // panel. Keep its actionable preview button inside the fixed
                    // workspace by omitting the decorative waveform there.
                    if !clone_mode {
                        ui.add_space(2.0);
                        let (waveform_rect, _) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width(), layout.waveform_height),
                            egui::Sense::hover(),
                        );
                        paint_preview_waveform(ui, waveform_rect, self.previewing);
                        ui.add_space(2.0);
                    } else {
                        ui.add_space(3.0);
                    }

                    let can_preview = !busy && self.active_voice_selection().is_some();
                    let preview_label = if self.previewing {
                        language.text("正在试听…", "Playing…")
                    } else {
                        language.text("▶  播放试听", "▶  Play preview")
                    };
                    let (button_fill, button_text) = if self.previewing {
                        (CARD_BACKGROUND, PRIMARY)
                    } else if can_preview {
                        (PRIMARY, egui::Color32::WHITE)
                    } else {
                        (CARD_BACKGROUND, TEXT_SECONDARY)
                    };
                    let preview_button = egui::Button::new(
                        egui::RichText::new(preview_label)
                            .size(13.0)
                            .strong()
                            .color(button_text),
                    )
                    .fill(button_fill)
                    .stroke(if can_preview {
                        egui::Stroke::NONE
                    } else {
                        egui::Stroke::new(1.0, BORDER)
                    })
                    .corner_radius(9)
                    .min_size(egui::vec2(ui.available_width(), layout.preview_button_height));

                    if ui.add(preview_button).clicked() && can_preview {
                        self.start_preview();
                    }
                });
                });
        });
    }

    fn show_clone_setup(
        &mut self,
        ui: &mut egui::Ui,
        language: UiLanguage,
        layout: VoiceCardLayout,
        busy: bool,
    ) {
        ui.add_space(layout.field_gap);
        let selected_name = self
            .clone_reference_path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or_else(|| language.text("尚未选择 WAV/MP3", "No WAV/MP3 selected"))
            .to_owned();
        input_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                let button_width = if language == UiLanguage::Chinese {
                    92.0
                } else {
                    108.0
                };
                let label_width = (ui.available_width() - button_width - 8.0).max(80.0);
                ui.add_sized(
                    egui::vec2(label_width, 28.0),
                    egui::Label::new(egui::RichText::new(&selected_name).size(12.0).color(
                        if self.clone_reference_path.is_some() {
                            TEXT_PRIMARY
                        } else {
                            TEXT_SECONDARY
                        },
                    ))
                    .truncate(),
                )
                .on_hover_text(&selected_name);
                let choose = egui::Button::new(
                    egui::RichText::new(language.text("选择音频", "Choose audio"))
                        .size(11.0)
                        .strong()
                        .color(PRIMARY),
                )
                .fill(PRIMARY_SOFT)
                .stroke(egui::Stroke::NONE)
                .corner_radius(7)
                .min_size(egui::vec2(button_width, 28.0));
                if ui.add_enabled(!busy, choose).clicked() {
                    self.choose_clone_reference();
                }
            });
        });

        if self.tts_engine == TtsEngine::Qwen3Local {
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(language.text(
                    "参考音频原文（准确填写可提高克隆精度）",
                    "Reference transcript (exact text improves fidelity)",
                ))
                .size(11.0)
                .color(TEXT_SECONDARY),
            );
            input_frame().show(ui, |ui| {
                ui.add_enabled(
                    !busy,
                    egui::TextEdit::singleline(&mut self.clone_reference_text)
                        .desired_width(ui.available_width())
                        .frame(egui::Frame::NONE)
                        .text_color(TEXT_PRIMARY)
                        .hint_text(language.text(
                            "输入录音中实际说出的完整原文…",
                            "Type exactly what is spoken…",
                        )),
                );
            });
        } else {
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(language.text(
                    "IndexTTS-2.5 从参考音频自动提取音色，无需填写原文",
                    "IndexTTS-2.5 extracts the voice automatically; no transcript needed",
                ))
                .size(11.0)
                .color(TEXT_SECONDARY),
            );
        }

        ui.add_space(3.0);
        ui.add_enabled_ui(!busy, |ui| {
            ui.checkbox(
                &mut self.clone_authorized,
                egui::RichText::new(language.text(
                    "我确认已获得该声音所有者授权",
                    "I confirm I have the voice owner's permission",
                ))
                .size(11.0)
                .color(TEXT_PRIMARY),
            );
        });

        if self.tts_engine == TtsEngine::IndexTts25 {
            let state = if self.indextts_model_ready {
                language.text("已就绪", "ready")
            } else if self.indextts_model_preparing {
                language.text("准备中", "preparing")
            } else {
                language.text("待准备", "not prepared")
            };
            ui.horizontal(|ui| {
                let gap = ui.spacing().item_spacing.x;
                let button_width = if language == UiLanguage::Chinese {
                    92.0
                } else {
                    104.0
                };
                let directory_width = if language == UiLanguage::Chinese {
                    76.0
                } else {
                    94.0
                };
                let import_width = if language == UiLanguage::Chinese {
                    84.0
                } else {
                    102.0
                };
                let summary = format!("IndexTTS-2.5 · v2.5.0 · {state}");
                ui.add_sized(
                    egui::vec2(
                        (ui.available_width()
                            - button_width
                            - directory_width
                            - import_width
                            - gap * 3.0)
                            .max(80.0),
                        layout.metadata_height,
                    ),
                    egui::Label::new(
                        egui::RichText::new(summary)
                            .size(11.0)
                            .color(TEXT_SECONDARY),
                    )
                    .truncate(),
                )
                .on_hover_text(language.text(
                    "固定使用官方 v2.5.0；首次准备需要 Git、uv 和网络",
                    "Pinned to official v2.5.0; first setup needs Git, uv, and internet",
                ));
                if ui
                    .add(model_directory_button(language, directory_width))
                    .clicked()
                {
                    self.open_model_directory(ModelDirectoryKind::IndexTts25);
                }
                let prepare = egui::Button::new(
                    egui::RichText::new(if self.indextts_model_ready {
                        language.text("检查模型", "Check model")
                    } else {
                        language.text("准备模型", "Prepare model")
                    })
                    .size(11.0)
                    .strong()
                    .color(PRIMARY),
                )
                .fill(PRIMARY_SOFT)
                .stroke(egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgb(205, 214, 255),
                ))
                .corner_radius(5)
                .min_size(egui::vec2(button_width, layout.metadata_height));
                if ui.add_enabled(!busy, prepare).clicked() {
                    self.prepare_indextts_model();
                }
                let import = egui::Button::new(
                    egui::RichText::new(language.text("离线导入", "Offline import"))
                        .size(11.0)
                        .strong()
                        .color(PRIMARY),
                )
                .fill(EDITOR_BACKGROUND)
                .stroke(egui::Stroke::new(1.0, BORDER))
                .corner_radius(5)
                .min_size(egui::vec2(import_width, layout.metadata_height));
                if ui.add_enabled(!busy, import).clicked() {
                    self.import_indextts_model();
                }
            });
            if self.indextts_model_preparing {
                if let Some(progress) = self.indextts_model_progress {
                    ui.add(model_download_progress_bar(
                        progress,
                        self.indextts_model_progress_label.clone(),
                    ));
                } else {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(
                            egui::RichText::new(&self.indextts_model_progress_label)
                                .size(11.0)
                                .color(TEXT_SECONDARY),
                        );
                    });
                }
            }
        }

        if self.tts_engine != TtsEngine::IndexTts25 {
            let kind = QwenModelKind::VoiceClone;
            let state = if self.qwen_model_ready == Some((self.selected_qwen_version, kind)) {
                self.qwen_device.as_deref().unwrap_or("local")
            } else if self.qwen_model_preparing {
                language.text("准备中", "preparing")
            } else {
                language.text("待加载", "not loaded")
            };
            let summary = match language {
                UiLanguage::Chinese => format!(
                    "{} Base · {state}{}",
                    self.selected_qwen_version.short_label(),
                    if self.selected_qwen_version == QwenModelVersion::Large1_7B {
                        " · 需更大统一内存"
                    } else {
                        " · 推荐"
                    }
                ),
                UiLanguage::English => format!(
                    "{} Base · {state}{}",
                    self.selected_qwen_version.short_label(),
                    if self.selected_qwen_version == QwenModelVersion::Large1_7B {
                        " · more unified memory"
                    } else {
                        " · recommended"
                    }
                ),
            };
            ui.horizontal(|ui| {
                let gap = ui.spacing().item_spacing.x;
                let import_width = if language == UiLanguage::Chinese {
                    96.0
                } else {
                    104.0
                };
                let directory_width = if language == UiLanguage::Chinese {
                    76.0
                } else {
                    82.0
                };
                let download_width = if language == UiLanguage::Chinese {
                    72.0
                } else {
                    78.0
                };
                let selected_model_ready = self.qwen_model_ready
                    == Some((self.selected_qwen_version, QwenModelKind::VoiceClone));
                let summary_width = (ui.available_width()
                    - import_width
                    - directory_width
                    - download_width
                    - 30.0
                    - gap * 4.0)
                    .max(8.0);
                ui.add_sized(
                    egui::vec2(summary_width, layout.metadata_height),
                    egui::Label::new(
                        egui::RichText::new(&summary)
                            .size(11.0)
                            .color(TEXT_SECONDARY),
                    )
                    .truncate(),
                )
                .on_hover_text(language.text(
                    "参考音频、原文、克隆提示和模型均只保留在本机",
                    "Reference audio, text, clone prompt, and model stay on this device",
                ));
                if ui
                    .add(model_directory_button(language, directory_width))
                    .clicked()
                {
                    self.open_model_directory(ModelDirectoryKind::Qwen3);
                }
                if ui
                    .add_enabled(
                        !busy,
                        model_download_entry_button(
                            language,
                            selected_model_ready,
                            download_width,
                            layout.metadata_height,
                        ),
                    )
                    .on_hover_text(if selected_model_ready {
                        language.text(
                            "删除所选模型缓存并重新下载",
                            "Delete and download the selected model again",
                        )
                    } else {
                        language.text(
                            "下载模型；存在断点文件时从已有进度继续",
                            "Download the model, resuming any partial files",
                        )
                    })
                    .clicked()
                {
                    if selected_model_ready {
                        self.redownload_selected_qwen_model();
                    } else {
                        self.prepare_selected_qwen_model();
                    }
                }
                if ui
                    .add(
                        egui::Button::new(
                            egui::RichText::new("?").size(12.0).strong().color(PRIMARY),
                        )
                        .fill(EDITOR_BACKGROUND)
                        .stroke(egui::Stroke::new(1.0, BORDER))
                        .corner_radius(8)
                        .min_size(egui::vec2(30.0, layout.metadata_height)),
                    )
                    .on_hover_text(language.text(
                        "查看 Base 离线模型下载说明",
                        "Show Base model download instructions",
                    ))
                    .clicked()
                {
                    self.qwen_manual_help_open = true;
                }
                let import = egui::Button::new(
                    egui::RichText::new(language.text("＋ 离线模型", "+ Offline model"))
                        .size(11.0)
                        .strong()
                        .color(PRIMARY),
                )
                .fill(PRIMARY_SOFT)
                .stroke(egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgb(205, 214, 255),
                ))
                .corner_radius(8)
                .min_size(egui::vec2(import_width, layout.metadata_height));
                if ui.add_enabled(!busy, import).clicked() {
                    self.import_selected_qwen_model();
                }
            });
            if self.qwen_model_preparing
                && let Some(progress) = self.qwen_model_progress
            {
                ui.add(model_download_progress_bar(
                    progress,
                    format!(
                        "{} · {}",
                        language.text("本地模型", "Local model"),
                        self.qwen_model_progress_label
                    ),
                ));
            }
        }
    }

    fn show_text_card(&mut self, ui: &mut egui::Ui, language: UiLanguage, card_height: f32) {
        card_frame().show(ui, |ui| {
            ui.set_min_height((card_height - 36.0).max(0.0));
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new(language.text("配音内容", "Voiceover content"))
                            .size(17.0)
                            .strong()
                            .color(TEXT_PRIMARY),
                    );
                    ui.label(
                        egui::RichText::new(language.text(
                            "输入文字，或导入字幕按时间轴生成音频",
                            "Enter text, or import subtitles for timed audio",
                        ))
                        .size(13.0)
                        .color(TEXT_SECONDARY),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let import = egui::Button::new(
                        egui::RichText::new(language.text("＋ 导入字幕", "+ Import subtitles"))
                            .size(12.0)
                            .strong()
                            .color(PRIMARY),
                    )
                    .fill(PRIMARY_SOFT)
                    .stroke(egui::Stroke::NONE)
                    .corner_radius(8)
                    .min_size(egui::vec2(108.0, 32.0));
                    if ui.add_enabled(!self.generating, import).clicked() {
                        self.import_subtitle();
                    }
                });
            });

            ui.add_space(11.0);
            egui::Frame::new()
                .fill(EDITOR_BACKGROUND)
                .stroke(egui::Stroke::new(1.0, BORDER))
                .corner_radius(10)
                .inner_margin(egui::Margin::same(4))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        let mode_width = ((ui.available_width() - 4.0) / 2.0).max(100.0);
                        let text_button = mode_button(
                            ui,
                            language.text("普通文本", "Plain text"),
                            self.input_mode == InputMode::Text,
                            mode_width,
                        );
                        let subtitle_button = mode_button(
                            ui,
                            language.text("字幕时间轴", "Subtitle timeline"),
                            self.input_mode == InputMode::Subtitles,
                            mode_width,
                        );
                        if text_button.clicked() && !self.generating {
                            self.input_mode = InputMode::Text;
                        }
                        if subtitle_button.clicked() && !self.generating {
                            self.input_mode = InputMode::Subtitles;
                        }
                    });
                });

            ui.add_space(10.0);
            match self.input_mode {
                InputMode::Text => self.show_plain_text_editor(ui, language),
                InputMode::Subtitles => self.show_subtitle_editor(ui, language),
            }
        });
    }

    fn show_plain_text_editor(&mut self, ui: &mut egui::Ui, language: UiLanguage) {
        ui.horizontal(|ui| {
            let count = self.text.chars().count();
            ui.label(
                egui::RichText::new(if language == UiLanguage::Chinese {
                    format!("{count} 字符")
                } else {
                    format!("{count} characters")
                })
                .size(12.0)
                .color(TEXT_SECONDARY),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let clear = egui::Button::new(
                    egui::RichText::new(language.text("清空文本", "Clear text"))
                        .size(12.0)
                        .color(TEXT_SECONDARY),
                )
                .fill(egui::Color32::from_rgb(247, 249, 253))
                .stroke(egui::Stroke::new(1.0, BORDER))
                .corner_radius(8)
                .min_size(egui::vec2(84.0, 30.0));
                if ui
                    .add_enabled(!self.text.is_empty() && !self.generating, clear)
                    .on_hover_text(language.text("清空全部文本", "Clear all text"))
                    .clicked()
                {
                    self.text.clear();
                }
            });
        });

        ui.add_space(7.0);
        let editor_height = (ui.available_height() - 16.0).max(96.0);
        egui::Frame::new()
            .fill(EDITOR_BACKGROUND)
            .stroke(egui::Stroke::new(1.0, BORDER))
            .corner_radius(9)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                // Long documents scroll only inside the editor; the app itself
                // stays fixed and never gains an outer scrollbar.
                egui::ScrollArea::vertical()
                    .id_salt("text-editor-scroll")
                    .max_height(editor_height)
                    .min_scrolled_height(editor_height)
                    .auto_shrink([false, false])
                    .scroll_bar_visibility(
                        egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded,
                    )
                    .show(ui, |ui| {
                        // Keep the first CJK glyph row clear of the scroll area's
                        // top clip edge. macOS font ascenders can otherwise lose
                        // a pixel or two when the row starts exactly at y = 0.
                        ui.add_space(5.0);
                        ui.add_enabled(
                            !self.generating,
                            egui::TextEdit::multiline(&mut self.text)
                                .desired_width(f32::INFINITY)
                                .desired_rows(10)
                                .cursor_at_end(false)
                                .frame(egui::Frame::NONE)
                                .margin(egui::Margin::symmetric(5, 7))
                                .text_color(TEXT_PRIMARY)
                                .hint_text(language.text(
                                    "在这里输入或粘贴需要转换的文字…",
                                    "Type or paste the text to synthesize…",
                                )),
                        );
                    });
            });
    }

    fn show_subtitle_editor(&mut self, ui: &mut egui::Ui, language: UiLanguage) {
        let Some(track) = self.subtitle_track.as_ref() else {
            let available = ui.available_size();
            egui::Frame::new()
                .fill(EDITOR_BACKGROUND)
                .stroke(egui::Stroke::new(1.0, BORDER))
                .corner_radius(10)
                .inner_margin(egui::Margin::same(18))
                .show(ui, |ui| {
                    ui.set_min_size(available - egui::vec2(36.0, 36.0));
                    ui.with_layout(
                        egui::Layout::top_down(egui::Align::Center)
                            .with_main_align(egui::Align::Center),
                        |ui| {
                            ui.label(egui::RichText::new("CC").size(28.0).strong().color(PRIMARY));
                            ui.add_space(6.0);
                            ui.label(
                                egui::RichText::new(language.text(
                                    "导入字幕后按每条时间间隔生成完整音轨",
                                    "Import subtitles to build a fully timed audio track",
                                ))
                                .size(14.0)
                                .strong()
                                .color(TEXT_PRIMARY),
                            );
                            ui.label(
                                egui::RichText::new(language.text(
                                    "支持 SRT/STR、WebVTT、ASS/SSA、LRC · UTF-8 / UTF-16 / GBK",
                                    "SRT/STR, WebVTT, ASS/SSA, LRC · UTF-8 / UTF-16 / GBK",
                                ))
                                .size(12.0)
                                .color(TEXT_SECONDARY),
                            );
                            ui.add_space(10.0);
                            let import = egui::Button::new(
                                egui::RichText::new(
                                    language.text("选择字幕文件", "Choose subtitle file"),
                                )
                                .strong()
                                .color(egui::Color32::WHITE),
                            )
                            .fill(PRIMARY)
                            .stroke(egui::Stroke::NONE)
                            .corner_radius(9)
                            .min_size(egui::vec2(142.0, 38.0));
                            if ui.add(import).clicked() {
                                self.import_subtitle();
                            }
                        },
                    );
                });
            return;
        };

        let file_name = self
            .subtitle_path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("subtitle")
            .to_owned();
        let format = track.format.label();
        let cue_count = track.cues.len();
        let duration = format_timestamp(track.duration_ms());
        let mut remove_subtitle = false;

        egui::Frame::new()
            .fill(PRIMARY_SOFT)
            .stroke(egui::Stroke::new(
                1.0,
                egui::Color32::from_rgb(213, 220, 255),
            ))
            .corner_radius(9)
            .inner_margin(egui::Margin::symmetric(11, 8))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.label(
                            egui::RichText::new(file_name)
                                .size(13.0)
                                .strong()
                                .color(TEXT_PRIMARY),
                        );
                        ui.label(
                            egui::RichText::new(if language == UiLanguage::Chinese {
                                format!("{format} · {cue_count} 条 · 总时长 {duration}")
                            } else {
                                format!("{format} · {cue_count} cues · {duration} total")
                            })
                            .size(11.0)
                            .color(TEXT_SECONDARY),
                        );
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let remove = egui::Button::new(
                            egui::RichText::new(language.text("移除", "Remove"))
                                .size(11.0)
                                .color(TEXT_SECONDARY),
                        )
                        .fill(CARD_BACKGROUND)
                        .stroke(egui::Stroke::new(1.0, BORDER))
                        .corner_radius(7)
                        .min_size(egui::vec2(62.0, 28.0));
                        remove_subtitle = ui.add_enabled(!self.generating, remove).clicked();
                    });
                });
            });

        if remove_subtitle {
            self.subtitle_track = None;
            self.subtitle_path = None;
            return;
        }

        ui.add_space(8.0);
        let list_height = (ui.available_height() - 14.0).max(96.0);
        let track = self
            .subtitle_track
            .as_ref()
            .expect("subtitle track exists after the remove check");
        egui::Frame::new()
            .fill(EDITOR_BACKGROUND)
            .stroke(egui::Stroke::new(1.0, BORDER))
            .corner_radius(9)
            .inner_margin(egui::Margin::symmetric(9, 7))
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("subtitle-cue-scroll")
                    .max_height(list_height)
                    .min_scrolled_height(list_height)
                    .auto_shrink([false, false])
                    .scroll_bar_visibility(
                        egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded,
                    )
                    .show(ui, |ui| {
                        // Separate the first row from the ScrollArea clip edge so
                        // Chinese glyph ascenders are never cut off on macOS.
                        ui.add_space(5.0);
                        for (index, cue) in track.cues.iter().enumerate() {
                            ui.horizontal_top(|ui| {
                                ui.add_sized(
                                    [34.0, 26.0],
                                    egui::Label::new(
                                        egui::RichText::new(format!("{}", index + 1))
                                            .size(11.0)
                                            .color(TEXT_SECONDARY),
                                    ),
                                );
                                ui.add_sized(
                                    [150.0, 26.0],
                                    egui::Label::new(
                                        egui::RichText::new(format!(
                                            "{} – {}",
                                            format_timestamp(cue.start_ms),
                                            format_timestamp(cue.end_ms)
                                        ))
                                        .monospace()
                                        .size(10.0)
                                        .color(PRIMARY),
                                    ),
                                );
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(&cue.text)
                                            .size(12.0)
                                            .color(TEXT_PRIMARY),
                                    )
                                    .wrap(),
                                );
                            });
                            if index + 1 < track.cues.len() {
                                ui.separator();
                            }
                        }
                    });
            });
    }

    fn show_asr_workspace(
        &mut self,
        ui: &mut egui::Ui,
        language: UiLanguage,
        workspace_height: f32,
        gap: f32,
    ) {
        let total_width = ui.available_width();
        let settings_width = (total_width * 0.38).clamp(350.0, 410.0);
        let preview_width = (total_width - settings_width - gap).max(430.0);
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            ui.allocate_ui_with_layout(
                egui::vec2(settings_width, workspace_height),
                egui::Layout::top_down(egui::Align::Min),
                |ui| self.show_asr_settings_card(ui, language, workspace_height),
            );
            ui.allocate_ui_with_layout(
                egui::vec2(preview_width, workspace_height),
                egui::Layout::top_down(egui::Align::Min),
                |ui| self.show_asr_result_card(ui, language, workspace_height),
            );
        });
    }

    fn show_asr_settings_card(
        &mut self,
        ui: &mut egui::Ui,
        language: UiLanguage,
        card_height: f32,
    ) {
        card_frame().show(ui, |ui| {
            ui.set_min_height((card_height - 36.0).max(0.0));
            ui.label(
                egui::RichText::new(language.text("音频与识别设置", "Audio & recognition"))
                    .size(17.0)
                    .strong()
                    .color(TEXT_PRIMARY),
            );
            ui.label(
                egui::RichText::new(language.text(
                    "音频只在本机处理，不会上传",
                    "Audio stays on this Mac and is never uploaded",
                ))
                .size(13.0)
                .color(TEXT_SECONDARY),
            );
            ui.add_space(16.0);

            let selected_name = self
                .asr_input_path
                .as_ref()
                .and_then(|path| path.file_name())
                .and_then(|name| name.to_str())
                .map(str::to_owned);
            egui::Frame::new()
                .fill(EDITOR_BACKGROUND)
                .stroke(egui::Stroke::new(1.0, BORDER))
                .corner_radius(11)
                .inner_margin(egui::Margin::same(12))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("♫").size(24.0).color(PRIMARY));
                        ui.vertical(|ui| {
                            ui.label(
                                egui::RichText::new(selected_name.as_deref().unwrap_or_else(|| {
                                    language.text("尚未选择音频或视频", "No audio or video selected")
                                }))
                                .size(13.0)
                                .strong()
                                .color(TEXT_PRIMARY),
                            );
                            ui.label(
                                egui::RichText::new(language.text(
                                    "支持 MP3/WAV/M4A 与 MP4/MOV/MKV 等媒体",
                                    "Supports MP3/WAV/M4A and MP4/MOV/MKV media",
                                ))
                                .size(11.0)
                                .color(TEXT_SECONDARY),
                            );
                        });
                    });
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        let choose = secondary_action_button(
                            language.text("导入音频/视频", "Import media"),
                            108.0,
                        );
                        if ui.add_enabled(!self.transcribing, choose).clicked() {
                            self.choose_media_for_transcription();
                        }
                        let use_latest = secondary_action_button(
                            language.text("使用刚生成的音频", "Use latest audio"),
                            146.0,
                        );
                        let latest_available = self
                            .last_generated_audio
                            .as_ref()
                            .is_some_and(|path| path.is_file());
                        if ui
                            .add_enabled(!self.transcribing && latest_available, use_latest)
                            .clicked()
                        {
                            self.use_last_generated_audio();
                        }
                    });
                });

            ui.add_space(18.0);
            ui.label(
                egui::RichText::new(language.text("识别语言", "Recognition language"))
                    .size(13.0)
                    .strong()
                    .color(TEXT_PRIMARY),
            );
            ui.label(
                egui::RichText::new(language.text(
                    "逐句中英文交替请选择“中英混合”",
                    "Use CN + EN when the spoken language alternates",
                ))
                .size(11.0)
                .color(TEXT_SECONDARY),
            );
            ui.add_space(7.0);
            egui::Frame::new()
                .fill(PRIMARY_SOFT)
                .corner_radius(9)
                .inner_margin(egui::Margin::same(3))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for (value, chinese, english, width) in [
                            (RecognitionLanguage::Chinese, "纯中文", "Chinese", 84.0),
                            (
                                RecognitionLanguage::MixedChineseEnglish,
                                "中英混合（推荐）",
                                "CN + EN",
                                132.0,
                            ),
                            (RecognitionLanguage::English, "纯英文", "English", 84.0),
                        ] {
                            let response = mode_button(
                                ui,
                                language.text(chinese, english),
                                self.recognition_language == value,
                                width,
                            );
                            if response.clicked() && !self.transcribing {
                                self.recognition_language = value;
                            }
                        }
                    });
                });

            ui.add_space(18.0);
            ui.label(
                egui::RichText::new(language.text("字幕格式", "Subtitle format"))
                    .size(13.0)
                    .strong()
                    .color(TEXT_PRIMARY),
            );
            ui.add_space(7.0);
            ui.horizontal(|ui| {
                for format in [SubtitleExportFormat::Srt, SubtitleExportFormat::WebVtt] {
                    let response = mode_button(
                        ui,
                        format.label(),
                        self.subtitle_export_format == format,
                        106.0,
                    );
                    if response.clicked() && !self.transcribing {
                        self.subtitle_export_format = format;
                    }
                }
            });

            ui.add_space(18.0);
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(242, 247, 255))
                .stroke(egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgb(215, 226, 247),
                ))
                .corner_radius(10)
                .inner_margin(egui::Margin::same(11))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Whisper Large-v3 Turbo · Q8")
                                .size(12.0)
                                .strong()
                                .color(TEXT_PRIMARY),
                        );
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if ui
                                    .add(model_directory_button(language, 82.0))
                                    .clicked()
                                {
                                    self.open_model_directory(ModelDirectoryKind::Whisper);
                                }
                            },
                        );
                    });
                    ui.label(
                        egui::RichText::new(language.text(
                            "纯 Rust + Candle · Metal 加速 · 首次约需下载 478 MB",
                            "Pure Rust + Candle · Metal · about 478 MB on first use",
                        ))
                        .size(11.0)
                        .color(TEXT_SECONDARY),
                    );
                    ui.add_space(5.0);
                    ui.label(
                        egui::RichText::new(language.text(
                            "首次准备模型需要联网；模型缓存后，识别过程完全离线。",
                            "Internet is needed once; transcription is fully offline after caching.",
                        ))
                        .size(11.0)
                        .color(PRIMARY),
                    );
                });
        });
    }

    fn show_asr_result_card(&mut self, ui: &mut egui::Ui, language: UiLanguage, card_height: f32) {
        card_frame().show(ui, |ui| {
            ui.set_min_height((card_height - 36.0).max(0.0));
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new(language.text("字幕预览", "Subtitle preview"))
                            .size(17.0)
                            .strong()
                            .color(TEXT_PRIMARY),
                    );
                    ui.label(
                        egui::RichText::new(language.text(
                            "识别完成后可直接点击每条文字修改，再保存字幕",
                            "Edit each recognized cue directly, then save the subtitles",
                        ))
                        .size(13.0)
                        .color(TEXT_SECONDARY),
                    );
                });
                if !self.transcription_cues.is_empty() {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(if language == UiLanguage::Chinese {
                                format!("{} 条", self.transcription_cues.len())
                            } else {
                                format!("{} cues", self.transcription_cues.len())
                            })
                            .size(12.0)
                            .color(PRIMARY),
                        );
                    });
                }
            });
            ui.add_space(14.0);

            // Stay inside the height allocated by the parent card. Forcing a
            // 250 px minimum here could overflow the card on smaller windows
            // and cover the action bar below it.
            let content_height = ui.available_height().max(120.0);
            egui::Frame::new()
                .fill(EDITOR_BACKGROUND)
                .stroke(egui::Stroke::new(1.0, BORDER))
                .corner_radius(10)
                .inner_margin(egui::Margin::symmetric(10, 8))
                .show(ui, |ui| {
                    ui.set_min_height((content_height - 16.0).max(0.0));
                    if self.transcribing && self.transcription_cues.is_empty() {
                        ui.with_layout(
                            egui::Layout::top_down(egui::Align::Center)
                                .with_main_align(egui::Align::Center),
                            |ui| {
                                ui.spinner();
                                ui.add_space(8.0);
                                ui.label(
                                    egui::RichText::new(if self.loading_asr_model {
                                        language.text("正在准备本地模型…", "Preparing local model…")
                                    } else {
                                        language.text("正在识别语音…", "Transcribing audio…")
                                    })
                                    .size(14.0)
                                    .strong()
                                    .color(TEXT_PRIMARY),
                                );
                                ui.add_space(8.0);
                                ui.add(
                                    egui::ProgressBar::new(self.transcription_progress)
                                        .desired_width(280.0)
                                        .desired_height(20.0)
                                        .corner_radius(4)
                                        .text(format!(
                                            "{:.1}%",
                                            self.transcription_progress * 100.0
                                        )),
                                );
                            },
                        );
                    } else if self.transcription_cues.is_empty() {
                        ui.with_layout(
                            egui::Layout::top_down(egui::Align::Center)
                                .with_main_align(egui::Align::Center),
                            |ui| {
                                ui.label(
                                    egui::RichText::new("CC").size(30.0).strong().color(PRIMARY),
                                );
                                ui.add_space(6.0);
                                ui.label(
                                    egui::RichText::new(language.text(
                                        "导入音频或视频后开始本地识别",
                                        "Import audio or video to start local transcription",
                                    ))
                                    .size(14.0)
                                    .color(TEXT_SECONDARY),
                                );
                            },
                        );
                    } else {
                        egui::ScrollArea::vertical()
                            .id_salt("asr-result-scroll")
                            .max_height(content_height)
                            .min_scrolled_height(content_height)
                            .auto_shrink([false, false])
                            .scroll_bar_visibility(
                                egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded,
                            )
                            .show(ui, |ui| {
                                // The scroll clip starts exactly at the first row;
                                // add a small safe area for CJK font ascenders.
                                ui.add_space(5.0);
                                let mut edited = false;
                                let cue_count = self.transcription_cues.len();
                                for (index, cue) in self.transcription_cues.iter_mut().enumerate() {
                                    ui.horizontal_top(|ui| {
                                        ui.add_sized(
                                            [30.0, 26.0],
                                            egui::Label::new(
                                                egui::RichText::new(format!("{}", index + 1))
                                                    .size(11.0)
                                                    .color(TEXT_SECONDARY),
                                            ),
                                        );
                                        ui.add_sized(
                                            [142.0, 26.0],
                                            egui::Label::new(
                                                egui::RichText::new(format!(
                                                    "{} – {}",
                                                    format_timestamp(cue.start_ms),
                                                    format_timestamp(cue.end_ms)
                                                ))
                                                .monospace()
                                                .size(10.0)
                                                .color(PRIMARY),
                                            ),
                                        );
                                        let response = ui.add(
                                            egui::TextEdit::multiline(&mut cue.text)
                                                .desired_rows(2)
                                                .desired_width(f32::INFINITY)
                                                .font(egui::TextStyle::Body)
                                                .text_color(TEXT_PRIMARY)
                                                .hint_text(language.text(
                                                    "点击修改字幕文字",
                                                    "Click to edit subtitle text",
                                                )),
                                        );
                                        edited |= response.changed();
                                    });
                                    if index + 1 < cue_count {
                                        ui.separator();
                                    }
                                }
                                if edited {
                                    self.transcription_dirty = true;
                                }
                            });
                    }
                });
        });
    }

    fn show_transcription_area(&mut self, ui: &mut egui::Ui, language: UiLanguage) {
        let can_transcribe = !self.transcribing
            && !self.saving_transcription
            && !self.generating
            && !self.previewing
            && self
                .asr_input_path
                .as_ref()
                .is_some_and(|path| path.is_file());
        ui.horizontal(|ui| {
            let label = if self.transcribing {
                language.text("正在生成字幕…", "Generating subtitles…")
            } else if self.transcription_cues.is_empty() {
                language.text("开始识别", "Start transcription")
            } else {
                language.text("重新识别", "Transcribe again")
            };
            let button = egui::Button::new(
                egui::RichText::new(label)
                    .size(15.0)
                    .strong()
                    .color(egui::Color32::WHITE),
            )
            .fill(PRIMARY)
            .stroke(egui::Stroke::NONE)
            .corner_radius(10)
            .min_size(egui::vec2(160.0, 44.0));
            if ui.add_enabled(can_transcribe, button).clicked() {
                self.start_transcription();
            }

            if !self.transcription_cues.is_empty() {
                let save_label = if self.saving_transcription {
                    language.text("正在保存…", "Saving…")
                } else if self.transcription_dirty {
                    language.text("保存修改", "Save edits")
                } else {
                    language.text("另存字幕", "Save as")
                };
                let save = egui::Button::new(
                    egui::RichText::new(save_label)
                        .size(14.0)
                        .strong()
                        .color(PRIMARY),
                )
                .fill(PRIMARY_SOFT)
                .stroke(egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgb(205, 214, 255),
                ))
                .corner_radius(10)
                .min_size(egui::vec2(132.0, 44.0));
                if ui
                    .add_enabled(!self.transcribing && !self.saving_transcription, save)
                    .clicked()
                {
                    self.save_transcription();
                }
            }

            if self.transcribing || self.saving_transcription {
                ui.spinner();
            }
            self.show_inline_status(ui, language);
        });
        if self.transcribing {
            ui.add(model_download_progress_bar(
                self.transcription_progress,
                if self.loading_asr_model {
                    language.text("Whisper 模型准备进度", "Whisper model progress")
                } else {
                    language.text("字幕识别进度", "Transcription progress")
                },
            ));
        } else if self.saving_transcription {
            ui.add(
                egui::ProgressBar::new(0.5)
                    .desired_height(20.0)
                    .animate(true)
                    .text(language.text("正在保存字幕…", "Saving subtitles…")),
            );
        }
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(language.text(
                "仅供个人学习与非商业研究，禁止商业使用；识别时音频和字幕均保留在本机。",
                "Personal learning and noncommercial research only; transcription stays local.",
            ))
            .size(11.0)
            .color(TEXT_SECONDARY),
        );
    }

    fn show_inline_status(&self, ui: &mut egui::Ui, language: UiLanguage) {
        if let Some(status) = &self.status {
            let (fill, stroke, text_color) = status_colors(status.kind);
            let full_status = status.text(language).to_owned();
            let response = egui::Frame::new()
                .fill(fill)
                .stroke(egui::Stroke::new(1.0, stroke))
                .corner_radius(9)
                .inner_margin(egui::Margin::symmetric(12, 8))
                .show(ui, |ui| {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(&full_status)
                                .size(12.0)
                                .color(text_color),
                        )
                        .truncate(),
                    )
                })
                .inner;
            response.on_hover_text(full_status);
        }
    }

    fn show_generated_audio_player(&mut self, ui: &mut egui::Ui, language: UiLanguage) {
        if self.generated_audio_loading {
            egui::Frame::new()
                .fill(EDITOR_BACKGROUND)
                .stroke(egui::Stroke::new(1.0, BORDER))
                .corner_radius(6)
                .inner_margin(egui::Margin::symmetric(10, 7))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(
                            egui::RichText::new(language.text(
                                "正在载入生成音频的预览播放器…",
                                "Loading the generated-audio preview…",
                            ))
                            .size(11.0)
                            .color(TEXT_SECONDARY),
                        );
                    });
                    ui.add_space(4.0);
                    ui.add(
                        egui::ProgressBar::new(0.35)
                            .desired_height(18.0)
                            .animate(true)
                            .text(language.text("正在解码音频…", "Decoding audio…")),
                    );
                });
            return;
        }

        let Some(player) = self.generated_audio_player.as_mut() else {
            return;
        };
        let file_name = player
            .path()
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("MP3")
            .to_owned();
        let duration = player.duration_seconds().max(0.001);
        let mut position = player.position_seconds();
        let playing = player.is_playing();
        let mut playback_error = None;

        egui::Frame::new()
            .fill(EDITOR_BACKGROUND)
            .stroke(egui::Stroke::new(1.0, BORDER))
            .corner_radius(6)
            .inner_margin(egui::Margin::symmetric(9, 6))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new(if playing {
                                    language.text("⏸ 暂停", "⏸ Pause")
                                } else {
                                    language.text("▶ 播放", "▶ Play")
                                })
                                .size(11.0)
                                .strong()
                                .color(PRIMARY),
                            )
                            .fill(PRIMARY_SOFT)
                            .stroke(egui::Stroke::new(
                                1.0,
                                egui::Color32::from_rgb(205, 214, 255),
                            ))
                            .corner_radius(5)
                            .min_size(egui::vec2(76.0, 30.0)),
                        )
                        .clicked()
                    {
                        player.toggle();
                    }
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("↺")
                                    .size(14.0)
                                    .strong()
                                    .color(TEXT_SECONDARY),
                            )
                            .fill(CARD_BACKGROUND)
                            .stroke(egui::Stroke::new(1.0, BORDER))
                            .corner_radius(5)
                            .min_size(egui::vec2(32.0, 30.0)),
                        )
                        .on_hover_text(language.text("从头播放", "Play from start"))
                        .clicked()
                    {
                        player.restart();
                        position = 0.0;
                    }
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(&file_name)
                                .size(11.0)
                                .color(TEXT_SECONDARY),
                        )
                        .truncate(),
                    )
                    .on_hover_text(&file_name);
                    ui.style_mut().visuals.handle_shape = egui::style::HandleShape::Circle;
                    let slider_width = (ui.available_width() - 116.0).max(120.0);
                    if ui
                        .add_sized(
                            [slider_width, 30.0],
                            egui::Slider::new(&mut position, 0.0..=duration)
                                .show_value(false)
                                .trailing_fill(true),
                        )
                        .changed()
                        && let Err(error) = player.seek(position)
                    {
                        playback_error = Some(error);
                    }
                    ui.label(
                        egui::RichText::new(format!(
                            "{} / {}",
                            format_playback_seconds(position),
                            format_playback_seconds(duration)
                        ))
                        .monospace()
                        .size(10.0)
                        .color(TEXT_PRIMARY),
                    );
                });
            });

        if let Some(error) = playback_error {
            self.status = Some(StatusMessage::new(
                StatusKind::Warning,
                error.clone(),
                error,
            ));
        }
    }

    fn show_generate_area(&mut self, ui: &mut egui::Ui, language: UiLanguage) {
        let has_content = match self.input_mode {
            InputMode::Text => !self.text.trim().is_empty(),
            InputMode::Subtitles => self
                .subtitle_track
                .as_ref()
                .is_some_and(|track| !track.cues.is_empty()),
        };
        let can_generate = !self.generating
            && !self.previewing
            && !self.transcribing
            && !self.qwen_model_preparing
            && !self.indextts_model_preparing
            && (self.tts_engine != TtsEngine::Edge || !self.fetching_voices)
            && self.active_voice_selection().is_some()
            && has_content;

        ui.horizontal(|ui| {
            let generate_label = match (self.generating, self.input_mode) {
                (true, InputMode::Subtitles) => {
                    language.text("正在生成字幕音频…", "Generating timed audio…")
                }
                (false, InputMode::Subtitles) => {
                    language.text("生成字幕 MP3", "Generate subtitle MP3")
                }
                (true, InputMode::Text) => language.text("正在生成…", "Generating…"),
                (false, InputMode::Text) => language.text("生成 MP3", "Generate MP3"),
            };
            let button = egui::Button::new(
                egui::RichText::new(generate_label)
                    .size(15.0)
                    .strong()
                    .color(egui::Color32::WHITE),
            )
            .fill(PRIMARY)
            .stroke(egui::Stroke::NONE)
            .corner_radius(10)
            .min_size(egui::vec2(150.0, 44.0));
            if ui.add_enabled(can_generate, button).clicked() {
                self.start_generation();
            }

            if self.generating {
                ui.spinner();
                ui.label(
                    egui::RichText::new(if let Some((current, total)) = self.subtitle_progress {
                        match (self.input_mode, language) {
                            (InputMode::Subtitles, UiLanguage::Chinese) => {
                                format!("正在合成第 {current}/{total} 条字幕")
                            }
                            (InputMode::Subtitles, UiLanguage::English) => {
                                format!("Synthesizing subtitle {current}/{total}")
                            }
                            (InputMode::Text, UiLanguage::Chinese) => {
                                format!("正在本地合成第 {current}/{total} 段")
                            }
                            (InputMode::Text, UiLanguage::English) => {
                                format!("Synthesizing local segment {current}/{total}")
                            }
                        }
                    } else {
                        language
                            .text("正在合成并保存…", "Synthesizing and saving…")
                            .to_owned()
                    })
                    .color(TEXT_SECONDARY),
                );
            }

            if let Some(status) = &self.status {
                let (fill, stroke, text_color) = status_colors(status.kind);
                let full_status = status.text(language).to_owned();
                let response = egui::Frame::new()
                    .fill(fill)
                    .stroke(egui::Stroke::new(1.0, stroke))
                    .corner_radius(9)
                    .inner_margin(egui::Margin::symmetric(12, 8))
                    .show(ui, |ui| {
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(&full_status)
                                    .size(12.0)
                                    .color(text_color),
                            )
                            .truncate(),
                        )
                    })
                    .inner;
                response.on_hover_text(full_status);
            }
        });

        if self.generating {
            let segment_progress = self.subtitle_progress.and_then(|(current, total)| {
                (total > 0).then_some(current.saturating_sub(1) as f32 / total as f32)
            });
            let exact_progress = segment_progress
                .or(self.qwen_model_progress)
                .or(self.indextts_model_progress);
            if let Some(progress) = exact_progress {
                ui.add(model_download_progress_bar(
                    progress,
                    language.text("当前任务进度", "Current task progress"),
                ));
            } else {
                ui.add(
                    egui::ProgressBar::new(0.35)
                        .desired_height(20.0)
                        .animate(true)
                        .text(language.text("正在处理当前任务…", "Processing the current task…")),
                );
            }
        }

        if !self.generating && (self.qwen_model_preparing || self.indextts_model_preparing) {
            let progress = if self.qwen_model_preparing {
                self.qwen_model_progress
            } else {
                self.indextts_model_progress
            };
            let label = if self.qwen_model_preparing {
                self.qwen_model_progress_label.as_str()
            } else {
                self.indextts_model_progress_label.as_str()
            };
            if let Some(progress) = progress {
                ui.add(model_download_progress_bar(progress, label));
            } else {
                ui.add(
                    egui::ProgressBar::new(0.35)
                        .desired_height(20.0)
                        .animate(true)
                        .text(label),
                );
            }
        }

        if !self.generating
            && !self.qwen_model_preparing
            && !self.indextts_model_preparing
            && self.previewing
        {
            ui.add(
                egui::ProgressBar::new(0.35)
                    .desired_height(20.0)
                    .animate(true)
                    .text(language.text("正在生成试听音频…", "Generating the preview…")),
            );
        } else if !self.generating
            && !self.previewing
            && self.tts_engine == TtsEngine::Edge
            && self.fetching_voices
        {
            ui.add(
                egui::ProgressBar::new(0.35)
                    .desired_height(20.0)
                    .animate(true)
                    .text(language.text("正在加载在线音色…", "Loading online voices…")),
            );
        }

        if (self.qwen_model_preparing || self.indextts_model_preparing)
            && model_download_control().state() != DownloadState::Idle
        {
            self.show_model_download_actions(ui, language);
        }

        if !self.generating {
            self.show_generated_audio_player(ui, language);
        }

        ui.add_space(6.0);
        let privacy_note = match self.tts_engine {
            TtsEngine::Edge => language.text(
                "仅供个人学习与非商业研究，禁止商业使用；文本会发送至 Microsoft Edge 朗读服务。",
                "Personal learning and noncommercial research only; text is sent to Microsoft Edge Read Aloud.",
            ),
            TtsEngine::Qwen3Local if self.qwen_voice_mode == QwenVoiceMode::Clone => language.text(
                "仅供个人学习与非商业研究，禁止商业使用；参考音频、原文与克隆提示均保留在本机，并须取得声音所有者授权。",
                "Personal learning and noncommercial research only; reference audio, transcript, and clone prompt stay local, and voice-owner permission is required.",
            ),
            TtsEngine::Qwen3Local => language.text(
                "仅供个人学习与非商业研究，禁止商业使用；Qwen3 合成在本机完成，首次使用需下载模型。",
                "Personal learning and noncommercial research only; Qwen3 synthesis stays local after its first model download.",
            ),
            TtsEngine::IndexTts25 => language.text(
                "仅供个人学习与非商业研究；IndexTTS-2.5 在本机运行，克隆声音前必须取得声音所有者授权，并受官方模型许可约束。",
                "Personal learning and noncommercial research only; IndexTTS-2.5 runs locally, requires voice-owner permission, and remains subject to its official model license.",
            ),
        };
        ui.label(
            egui::RichText::new(privacy_note)
                .size(11.0)
                .color(TEXT_SECONDARY),
        );
    }
}

fn card_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(CARD_BACKGROUND)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .corner_radius(14)
        .inner_margin(egui::Margin::same(18))
}

fn input_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(EDITOR_BACKGROUND)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .corner_radius(9)
        .inner_margin(egui::Margin::symmetric(11, 6))
}

fn secondary_action_button(label: &str, width: f32) -> egui::Button<'_> {
    egui::Button::new(
        egui::RichText::new(label)
            .size(12.0)
            .strong()
            .color(PRIMARY),
    )
    .fill(CARD_BACKGROUND)
    .stroke(egui::Stroke::new(
        1.0,
        egui::Color32::from_rgb(207, 216, 248),
    ))
    .corner_radius(8)
    .min_size(egui::vec2(width, 34.0))
}

fn model_directory_button(language: UiLanguage, width: f32) -> egui::Button<'static> {
    let label = match language {
        UiLanguage::Chinese => "打开目录",
        UiLanguage::English => "Open folder",
    };
    egui::Button::new(
        egui::RichText::new(label)
            .size(11.0)
            .strong()
            .color(TEXT_PRIMARY),
    )
    .fill(EDITOR_BACKGROUND)
    .stroke(egui::Stroke::new(1.0, BORDER))
    .corner_radius(5)
    .min_size(egui::vec2(width, 28.0))
}

fn model_download_entry_button(
    language: UiLanguage,
    ready: bool,
    width: f32,
    height: f32,
) -> egui::Button<'static> {
    let label = match (language, ready) {
        (UiLanguage::Chinese, true) => "重新下载",
        (UiLanguage::Chinese, false) => "下载/续传",
        (UiLanguage::English, true) => "Reload",
        (UiLanguage::English, false) => "Download",
    };
    egui::Button::new(
        egui::RichText::new(label)
            .size(11.0)
            .strong()
            .color(PRIMARY),
    )
    .fill(EDITOR_BACKGROUND)
    .stroke(egui::Stroke::new(1.0, BORDER))
    .corner_radius(5)
    .min_size(egui::vec2(width, height))
}

fn model_action_button(label: &str) -> egui::Button<'_> {
    egui::Button::new(
        egui::RichText::new(label)
            .size(11.0)
            .strong()
            .color(PRIMARY),
    )
    .fill(PRIMARY_SOFT)
    .stroke(egui::Stroke::new(
        1.0,
        egui::Color32::from_rgb(205, 214, 255),
    ))
    .corner_radius(5)
    .min_size(egui::vec2(72.0, 26.0))
}

fn model_download_progress_bar(progress: f32, label: impl Into<String>) -> egui::ProgressBar {
    let progress = progress.clamp(0.0, 1.0);
    egui::ProgressBar::new(progress)
        .desired_height(20.0)
        .corner_radius(4)
        .text(format!("{} · {:.1}%", label.into(), progress * 100.0))
}

fn cancelled_download_status() -> StatusMessage {
    StatusMessage::new(
        StatusKind::Info,
        "模型下载已取消，临时文件已保留；再次开始时会从已有进度继续。",
        "Model download cancelled. Partial files were kept and will be resumed next time.",
    )
}

fn open_directory_in_file_manager(directory: &Path) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    let result = Command::new("/usr/bin/open").arg(directory).spawn();
    #[cfg(target_os = "windows")]
    let result = Command::new("explorer").arg(directory).spawn();
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let result = Command::new("xdg-open").arg(directory).spawn();

    result
        .map(|_| ())
        .map_err(|error| format!("{}（{}）", directory.display(), error))
}

fn configure_voice_combo_style(ui: &mut egui::Ui) {
    let style = ui.style_mut();
    style.spacing.button_padding = egui::vec2(12.0, 9.0);
    style.spacing.interact_size.y = 42.0;

    let widgets = &mut style.visuals.widgets;
    widgets.inactive.weak_bg_fill = EDITOR_BACKGROUND;
    widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    widgets.inactive.corner_radius = egui::CornerRadius::same(9);
    widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);

    widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(242, 245, 255);
    widgets.hovered.bg_stroke = egui::Stroke::new(1.0, PRIMARY);
    widgets.hovered.corner_radius = egui::CornerRadius::same(9);
    widgets.hovered.fg_stroke = egui::Stroke::new(1.0, PRIMARY);

    widgets.active.weak_bg_fill = PRIMARY_SOFT;
    widgets.active.bg_stroke = egui::Stroke::new(1.0, PRIMARY);
    widgets.active.corner_radius = egui::CornerRadius::same(9);
    widgets.active.fg_stroke = egui::Stroke::new(1.0, PRIMARY);

    widgets.open.weak_bg_fill = PRIMARY_SOFT;
    widgets.open.bg_stroke = egui::Stroke::new(1.0, PRIMARY);
    widgets.open.corner_radius = egui::CornerRadius::same(9);
    widgets.open.fg_stroke = egui::Stroke::new(1.0, PRIMARY);
}

fn voice_popup_style() -> egui::style::StyleModifier {
    egui::style::StyleModifier::new(|style| {
        style.spacing.item_spacing = egui::vec2(4.0, 4.0);
        style.spacing.button_padding = egui::vec2(10.0, 8.0);
        style.visuals.window_fill = CARD_BACKGROUND;
        style.visuals.panel_fill = CARD_BACKGROUND;
        style.visuals.window_stroke = egui::Stroke::new(1.0, BORDER);
        style.visuals.selection.bg_fill = PRIMARY_SOFT;
        style.visuals.selection.stroke = egui::Stroke::new(1.0, PRIMARY);

        let widgets = &mut style.visuals.widgets;
        widgets.inactive.weak_bg_fill = egui::Color32::TRANSPARENT;
        widgets.inactive.bg_stroke = egui::Stroke::NONE;
        widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);
        widgets.hovered.weak_bg_fill = PRIMARY_SOFT;
        widgets.hovered.bg_stroke = egui::Stroke::NONE;
        widgets.hovered.corner_radius = egui::CornerRadius::same(7);
        widgets.hovered.fg_stroke = egui::Stroke::new(1.0, PRIMARY);
        widgets.active.weak_bg_fill = PRIMARY_SOFT;
        widgets.active.bg_stroke = egui::Stroke::NONE;
        widgets.active.corner_radius = egui::CornerRadius::same(7);
        widgets.active.fg_stroke = egui::Stroke::new(1.0, PRIMARY);
    })
}

fn status_colors(kind: StatusKind) -> (egui::Color32, egui::Color32, egui::Color32) {
    match kind {
        StatusKind::Info => (
            egui::Color32::from_rgb(238, 242, 255),
            egui::Color32::from_rgb(205, 214, 255),
            egui::Color32::from_rgb(60, 81, 196),
        ),
        StatusKind::Success => (
            egui::Color32::from_rgb(236, 249, 241),
            egui::Color32::from_rgb(193, 230, 207),
            egui::Color32::from_rgb(34, 125, 71),
        ),
        StatusKind::Warning => (
            egui::Color32::from_rgb(255, 247, 229),
            egui::Color32::from_rgb(242, 215, 158),
            egui::Color32::from_rgb(162, 101, 14),
        ),
        StatusKind::Error => (
            egui::Color32::from_rgb(255, 239, 240),
            egui::Color32::from_rgb(246, 195, 198),
            egui::Color32::from_rgb(184, 54, 60),
        ),
    }
}

fn preview_text(text: &str, language: UiLanguage) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return language
            .text(
                "你好，这是当前音色的试听效果。",
                "Hello, this is a preview of the selected voice.",
            )
            .to_owned();
    }

    let mut preview: String = trimmed.chars().take(80).collect();
    if trimmed.chars().count() > 80 {
        preview.push('…');
    }
    preview
}

fn signed_percent(value: i32) -> String {
    format!("{value:+}%")
}

fn compact_model_source(source: &str) -> &str {
    if source.starts_with("Model") {
        "Whisper Large-v3 Turbo"
    } else if source.starts_with("Tokenizer") {
        "Tokenizer"
    } else if source.starts_with("Config") {
        "Config"
    } else {
        source
    }
}

fn format_duration_seconds(seconds: u64) -> String {
    if seconds >= 60 {
        format!("{}:{:02}", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

fn format_playback_seconds(seconds: f32) -> String {
    let seconds = seconds.max(0.0).round() as u64;
    if seconds >= 3_600 {
        format!(
            "{}:{:02}:{:02}",
            seconds / 3_600,
            (seconds / 60) % 60,
            seconds % 60
        )
    } else {
        format!("{}:{:02}", seconds / 60, seconds % 60)
    }
}

fn format_transfer_progress(current_bytes: u64, total_bytes: u64) -> String {
    if total_bytes == 0 {
        return format!("{:.0} MB", current_bytes as f64 / 1_048_576.0);
    }
    if total_bytes < 1_048_576 {
        format!(
            "{:.0}% · {:.0}/{:.0} KB",
            current_bytes as f64 / total_bytes as f64 * 100.0,
            current_bytes as f64 / 1_024.0,
            total_bytes as f64 / 1_024.0
        )
    } else {
        format!(
            "{:.0}% · {:.0}/{:.0} MB",
            current_bytes as f64 / total_bytes as f64 * 100.0,
            current_bytes as f64 / 1_048_576.0,
            total_bytes as f64 / 1_048_576.0
        )
    }
}

fn adjustment_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut i32,
    range: std::ops::RangeInclusive<i32>,
) {
    ui.horizontal(|ui| {
        ui.style_mut().visuals.handle_shape = egui::style::HandleShape::Circle;
        ui.spacing_mut().slider_rail_height = 4.0;
        ui.visuals_mut().widgets.inactive.bg_fill = CARD_BACKGROUND;
        ui.visuals_mut().widgets.inactive.fg_stroke = egui::Stroke::new(1.5, PRIMARY);
        ui.visuals_mut().widgets.hovered.bg_fill = CARD_BACKGROUND;
        ui.visuals_mut().widgets.hovered.fg_stroke = egui::Stroke::new(1.8, PRIMARY);
        ui.visuals_mut().widgets.active.bg_fill = PRIMARY_SOFT;
        ui.visuals_mut().widgets.active.fg_stroke = egui::Stroke::new(1.8, PRIMARY);

        ui.add_sized(
            [46.0, 24.0],
            egui::Label::new(egui::RichText::new(label).size(12.0).color(TEXT_SECONDARY)),
        );
        let slider_width = (ui.available_width() - 56.0).max(64.0);
        ui.add_sized(
            [slider_width, 24.0],
            egui::Slider::new(value, range)
                .show_value(false)
                .trailing_fill(true),
        );
        ui.add_sized(
            [48.0, 24.0],
            egui::Label::new(
                egui::RichText::new(signed_percent(*value))
                    .size(12.0)
                    .strong()
                    .color(TEXT_PRIMARY),
            ),
        );
    });
}

fn model_f64_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    suffix_format: &str,
) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [132.0, 22.0],
            egui::Label::new(egui::RichText::new(label).size(11.0).color(TEXT_SECONDARY)),
        );
        let slider_width = (ui.available_width() - 66.0).max(64.0);
        ui.add_sized(
            [slider_width, 22.0],
            egui::Slider::new(value, range)
                .show_value(false)
                .trailing_fill(true),
        );
        let display = match suffix_format {
            "{:.2}x" => format!("{value:.2}x"),
            _ => format!("{value:.2}"),
        };
        ui.add_sized(
            [58.0, 22.0],
            egui::Label::new(egui::RichText::new(display).size(11.0).color(TEXT_PRIMARY)),
        );
    });
}

fn model_usize_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut usize,
    range: std::ops::RangeInclusive<usize>,
    suffix: &str,
) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [132.0, 22.0],
            egui::Label::new(egui::RichText::new(label).size(11.0).color(TEXT_SECONDARY)),
        );
        let slider_width = (ui.available_width() - 66.0).max(64.0);
        ui.add_sized(
            [slider_width, 22.0],
            egui::Slider::new(value, range)
                .show_value(false)
                .trailing_fill(true),
        );
        ui.add_sized(
            [58.0, 22.0],
            egui::Label::new(
                egui::RichText::new(format!("{}{suffix}", *value))
                    .size(11.0)
                    .color(TEXT_PRIMARY),
            ),
        );
    });
}

fn paint_preview_waveform(ui: &egui::Ui, rect: egui::Rect, active: bool) {
    const LEVELS: &[f32] = &[
        0.24, 0.42, 0.68, 0.48, 0.82, 0.58, 1.0, 0.72, 0.44, 0.76, 0.54, 0.88, 0.6, 0.38, 0.22,
    ];

    let total_width = rect.width().min(220.0);
    let gap = 5.0;
    let bar_width =
        ((total_width - gap * (LEVELS.len() - 1) as f32) / LEVELS.len() as f32).clamp(2.0, 6.0);
    let used_width = bar_width * LEVELS.len() as f32 + gap * (LEVELS.len() - 1) as f32;
    let start_x = rect.center().x - used_width / 2.0;
    let time = ui.input(|input| input.time) as f32;
    let color = if active {
        PRIMARY
    } else {
        egui::Color32::from_rgb(165, 176, 245)
    };

    for (index, level) in LEVELS.iter().enumerate() {
        let pulse = if active {
            0.68 + 0.32 * (time * 5.0 + index as f32 * 0.72).sin().abs()
        } else {
            1.0
        };
        let height = (rect.height() * level * pulse).max(4.0);
        let center = egui::pos2(
            start_x + index as f32 * (bar_width + gap) + bar_width / 2.0,
            rect.center().y,
        );
        ui.painter().rect_filled(
            egui::Rect::from_center_size(center, egui::vec2(bar_width, height)),
            egui::CornerRadius::same(2),
            color,
        );
    }
}

fn language_button(ui: &mut egui::Ui, label: &str, selected: bool) -> egui::Response {
    ui.add(
        egui::Button::new(
            egui::RichText::new(label)
                .size(13.0)
                .strong()
                .color(if selected { PRIMARY } else { TEXT_SECONDARY }),
        )
        .fill(if selected {
            PRIMARY_SOFT
        } else {
            egui::Color32::TRANSPARENT
        })
        .stroke(if selected {
            egui::Stroke::new(1.0, egui::Color32::from_rgb(206, 215, 255))
        } else {
            egui::Stroke::NONE
        })
        .corner_radius(8)
        .min_size(egui::vec2(54.0, 32.0)),
    )
}

fn mode_button(ui: &mut egui::Ui, label: &str, selected: bool, width: f32) -> egui::Response {
    ui.add(
        egui::Button::new(
            egui::RichText::new(label)
                .size(12.0)
                .strong()
                .color(if selected { PRIMARY } else { TEXT_SECONDARY }),
        )
        .fill(if selected {
            CARD_BACKGROUND
        } else {
            egui::Color32::TRANSPARENT
        })
        .stroke(if selected {
            egui::Stroke::new(1.0, egui::Color32::from_rgb(216, 223, 245))
        } else {
            egui::Stroke::NONE
        })
        .corner_radius(8)
        .min_size(egui::vec2(width, 32.0)),
    )
}

fn engine_button(ui: &mut egui::Ui, label: &str, selected: bool, width: f32) -> egui::Response {
    ui.add_sized(
        egui::vec2(width, 32.0),
        egui::Button::new(
            egui::RichText::new(label)
                .size(11.0)
                .strong()
                .color(if selected { PRIMARY } else { TEXT_SECONDARY }),
        )
        .fill(if selected {
            PRIMARY_SOFT
        } else {
            EDITOR_BACKGROUND
        })
        .stroke(egui::Stroke::new(
            1.0,
            if selected {
                egui::Color32::from_rgb(205, 214, 255)
            } else {
                BORDER
            },
        ))
        .corner_radius(8),
    )
}

fn spawn_tts_worker() -> (
    mpsc::UnboundedSender<WorkerCommand>,
    mpsc::UnboundedReceiver<WorkerEvent>,
    Option<String>,
) {
    let (command_tx, mut command_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let thread_event_tx = event_tx.clone();

    let spawn_result = thread::Builder::new()
        .name("tts-background-worker".to_owned())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = thread_event_tx.send(WorkerEvent::WorkerFailed(format!(
                        "Could not start the async runtime: {error}"
                    )));
                    return;
                }
            };

            runtime.block_on(async move {
                let client = match EdgeTtsClient::new() {
                    Ok(client) => client,
                    Err(error) => {
                        let _ = thread_event_tx.send(WorkerEvent::WorkerFailed(format!(
                            "Could not initialize Edge TTS: {error}"
                        )));
                        return;
                    }
                };

                let cache_path = voice_cache_path();
                let mut whisper_model: Option<(RecognitionLanguage, Whisper)> = None;
                let mut qwen_model: Option<LocalQwenModel> = None;
                let mut qwen_clone_prompt: Option<CachedQwenClonePrompt> = None;
                let indextts_runtime = IndexTtsRuntime::new();

                while let Some(command) = command_rx.recv().await {
                    match command {
                        WorkerCommand::FetchVoices => {
                            fetch_voices(&client, &thread_event_tx, &cache_path).await;
                        }
                        WorkerCommand::Preview {
                            text,
                            voice,
                            rate_percent,
                            volume_percent,
                            qwen_settings,
                            indextts_settings,
                        } => match voice {
                            VoiceSelection::Edge(voice) => {
                                preview_voice(
                                    &client,
                                    &thread_event_tx,
                                    text,
                                    voice,
                                    rate_percent,
                                    volume_percent,
                                )
                                .await;
                            }
                            VoiceSelection::Qwen3Preset(selection) => {
                                whisper_model = None;
                                qwen_clone_prompt = None;
                                preview_qwen_voice(
                                    &mut qwen_model,
                                    &thread_event_tx,
                                    text,
                                    selection,
                                    rate_percent,
                                    volume_percent,
                                    qwen_settings,
                                )
                                .await;
                            }
                            VoiceSelection::Qwen3Clone(selection) => {
                                whisper_model = None;
                                preview_qwen_clone(
                                    &mut qwen_model,
                                    &mut qwen_clone_prompt,
                                    &thread_event_tx,
                                    text,
                                    selection,
                                    rate_percent,
                                    volume_percent,
                                    qwen_settings,
                                )
                                .await;
                            }
                            VoiceSelection::IndexTts25(selection) => {
                                whisper_model = None;
                                qwen_clone_prompt = None;
                                if qwen_model.take().is_some() {
                                    let _ = thread_event_tx.send(WorkerEvent::QwenModelReleased);
                                }
                                preview_indextts_voice(
                                    &indextts_runtime,
                                    &thread_event_tx,
                                    text,
                                    selection,
                                    rate_percent,
                                    volume_percent,
                                    indextts_settings,
                                )
                                .await;
                            }
                        },
                        WorkerCommand::Generate {
                            text,
                            voice,
                            rate_percent,
                            volume_percent,
                            qwen_settings,
                            indextts_settings,
                            output_path,
                        } => match voice {
                            VoiceSelection::Edge(voice) => {
                                generate_mp3(
                                    &client,
                                    &thread_event_tx,
                                    text,
                                    voice,
                                    rate_percent,
                                    volume_percent,
                                    output_path,
                                )
                                .await;
                            }
                            VoiceSelection::Qwen3Preset(selection) => {
                                whisper_model = None;
                                qwen_clone_prompt = None;
                                generate_qwen_mp3(
                                    &mut qwen_model,
                                    &thread_event_tx,
                                    text,
                                    selection,
                                    rate_percent,
                                    volume_percent,
                                    qwen_settings,
                                    output_path,
                                )
                                .await;
                            }
                            VoiceSelection::Qwen3Clone(selection) => {
                                whisper_model = None;
                                generate_qwen_clone_mp3(
                                    &mut qwen_model,
                                    &mut qwen_clone_prompt,
                                    &thread_event_tx,
                                    text,
                                    selection,
                                    QwenOutputSettings {
                                        rate_percent,
                                        volume_percent,
                                        qwen_settings,
                                        output_path,
                                    },
                                )
                                .await;
                            }
                            VoiceSelection::IndexTts25(selection) => {
                                whisper_model = None;
                                qwen_clone_prompt = None;
                                if qwen_model.take().is_some() {
                                    let _ = thread_event_tx.send(WorkerEvent::QwenModelReleased);
                                }
                                generate_indextts_mp3(
                                    &indextts_runtime,
                                    &thread_event_tx,
                                    text,
                                    selection,
                                    rate_percent,
                                    volume_percent,
                                    indextts_settings,
                                    output_path,
                                )
                                .await;
                            }
                        },
                        WorkerCommand::GenerateSubtitles {
                            cues,
                            voice,
                            rate_percent,
                            volume_percent,
                            qwen_settings,
                            indextts_settings,
                            output_path,
                        } => match voice {
                            VoiceSelection::Edge(voice) => {
                                generate_subtitle_mp3(
                                    &client,
                                    &thread_event_tx,
                                    cues,
                                    voice,
                                    rate_percent,
                                    volume_percent,
                                    output_path,
                                )
                                .await;
                            }
                            VoiceSelection::Qwen3Preset(selection) => {
                                whisper_model = None;
                                qwen_clone_prompt = None;
                                generate_qwen_subtitle_mp3(
                                    &mut qwen_model,
                                    &thread_event_tx,
                                    cues,
                                    selection,
                                    rate_percent,
                                    volume_percent,
                                    qwen_settings,
                                    output_path,
                                )
                                .await;
                            }
                            VoiceSelection::Qwen3Clone(selection) => {
                                whisper_model = None;
                                generate_qwen_clone_subtitle_mp3(
                                    &mut qwen_model,
                                    &mut qwen_clone_prompt,
                                    &thread_event_tx,
                                    cues,
                                    selection,
                                    QwenOutputSettings {
                                        rate_percent,
                                        volume_percent,
                                        qwen_settings,
                                        output_path,
                                    },
                                )
                                .await;
                            }
                            VoiceSelection::IndexTts25(selection) => {
                                whisper_model = None;
                                qwen_clone_prompt = None;
                                if qwen_model.take().is_some() {
                                    let _ = thread_event_tx.send(WorkerEvent::QwenModelReleased);
                                }
                                generate_indextts_subtitle_mp3(
                                    &indextts_runtime,
                                    &thread_event_tx,
                                    cues,
                                    selection,
                                    rate_percent,
                                    volume_percent,
                                    indextts_settings,
                                    output_path,
                                )
                                .await;
                            }
                        },
                        WorkerCommand::ImportQwenModel {
                            version,
                            kind,
                            source_dir,
                        } => {
                            whisper_model = None;
                            qwen_clone_prompt = None;
                            if qwen_model.take().is_some() {
                                let _ = thread_event_tx.send(WorkerEvent::QwenModelReleased);
                            }
                            import_qwen_model_locally(
                                &mut qwen_model,
                                &thread_event_tx,
                                version,
                                kind,
                                &source_dir,
                            );
                        }
                        WorkerCommand::PrepareQwenModel { version, kind } => {
                            whisper_model = None;
                            qwen_clone_prompt = None;
                            prepare_qwen_model_locally(
                                &mut qwen_model,
                                &thread_event_tx,
                                version,
                                kind,
                            );
                        }
                        WorkerCommand::RedownloadQwenModel { version, kind } => {
                            whisper_model = None;
                            qwen_clone_prompt = None;
                            qwen_model = None;
                            let _ = thread_event_tx.send(WorkerEvent::QwenModelReleased);
                            redownload_qwen_model_locally(
                                &mut qwen_model,
                                &thread_event_tx,
                                version,
                                kind,
                            );
                        }
                        WorkerCommand::ImportIndexTtsModel { source_dir } => {
                            whisper_model = None;
                            qwen_clone_prompt = None;
                            if qwen_model.take().is_some() {
                                let _ = thread_event_tx.send(WorkerEvent::QwenModelReleased);
                            }
                            import_indextts_model_locally(
                                &indextts_runtime,
                                &thread_event_tx,
                                &source_dir,
                            );
                        }
                        WorkerCommand::PrepareIndexTts => {
                            whisper_model = None;
                            qwen_clone_prompt = None;
                            if qwen_model.take().is_some() {
                                let _ = thread_event_tx.send(WorkerEvent::QwenModelReleased);
                            }
                            prepare_indextts_runtime(&indextts_runtime, &thread_event_tx);
                        }
                        WorkerCommand::TranscribeMedia {
                            input_path,
                            language,
                        } => {
                            qwen_clone_prompt = None;
                            if qwen_model.take().is_some() {
                                let _ = thread_event_tx.send(WorkerEvent::QwenModelReleased);
                            }
                            transcribe_media_locally(
                                &thread_event_tx,
                                &mut whisper_model,
                                input_path,
                                language,
                            )
                            .await;
                        }
                        WorkerCommand::SaveTranscription {
                            output_path,
                            cues,
                            format,
                        } => {
                            save_transcription_locally(&thread_event_tx, output_path, cues, format)
                                .await;
                        }
                        WorkerCommand::LoadGeneratedAudio { input_path } => {
                            load_generated_audio(&thread_event_tx, input_path).await;
                        }
                    }
                }
            });
        });

    let startup_error = spawn_result
        .err()
        .map(|error| format!("Could not start the TTS worker thread: {error}"));

    (command_tx, event_rx, startup_error)
}

fn import_qwen_model_locally(
    model_cache: &mut Option<LocalQwenModel>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    version: QwenModelVersion,
    kind: QwenModelKind,
    source_dir: &Path,
) {
    let progress_tx = event_tx.clone();
    let model_dir =
        match qwen_local::import_offline_model(version, kind, source_dir, move |progress| {
            let _ = progress_tx.send(WorkerEvent::QwenModelImportProgress {
                version,
                kind,
                file: progress.file.to_owned(),
                copied_bytes: progress.downloaded_bytes,
                total_bytes: progress.total_bytes,
            });
        }) {
            Ok(model_dir) => model_dir,
            Err(error) => {
                let _ = event_tx.send(WorkerEvent::QwenModelImportFailed {
                    version,
                    kind,
                    error,
                });
                return;
            }
        };

    match LocalQwenModel::load(version, kind, |_| {}) {
        Ok(model) => {
            let device = model.device_label().to_owned();
            *model_cache = Some(model);
            let _ = event_tx.send(WorkerEvent::QwenModelImported {
                version,
                kind,
                device,
                model_dir,
            });
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::QwenModelImportFailed {
                version,
                kind,
                error,
            });
        }
    }
}

fn redownload_qwen_model_locally(
    model_cache: &mut Option<LocalQwenModel>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    version: QwenModelVersion,
    kind: QwenModelKind,
) {
    let _ = event_tx.send(WorkerEvent::QwenModelPreparing { version, kind });
    let progress_tx = event_tx.clone();
    match LocalQwenModel::redownload(version, kind, move |progress| {
        let _ = progress_tx.send(WorkerEvent::QwenModelDownload {
            version,
            kind,
            file: progress.file.to_owned(),
            downloaded_bytes: progress.downloaded_bytes,
            total_bytes: progress.total_bytes,
        });
    }) {
        Ok(model) => {
            let device = model.device_label().to_owned();
            *model_cache = Some(model);
            let _ = event_tx.send(WorkerEvent::QwenModelReady {
                version,
                kind,
                device,
            });
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::QwenModelFailed(error));
        }
    }
}

fn prepare_qwen_model_locally(
    model_cache: &mut Option<LocalQwenModel>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    version: QwenModelVersion,
    kind: QwenModelKind,
) {
    if let Err(error) = ensure_qwen_model(model_cache, event_tx, version, kind) {
        let _ = event_tx.send(WorkerEvent::QwenModelFailed(error));
    }
}

fn prepare_indextts_runtime(
    runtime: &Result<IndexTtsRuntime, String>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
) {
    let runtime = match runtime.as_ref() {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::IndexTtsFailed(error.clone()));
            return;
        }
    };
    let progress_tx = event_tx.clone();
    match runtime.ensure_ready(move |progress| forward_indextts_progress(&progress_tx, progress)) {
        Ok(()) => {
            let _ = event_tx.send(WorkerEvent::IndexTtsReady);
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::IndexTtsFailed(error));
        }
    }
}

fn import_indextts_model_locally(
    runtime: &Result<IndexTtsRuntime, String>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    source_dir: &Path,
) {
    let runtime = match runtime.as_ref() {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::IndexTtsImportFailed(error.clone()));
            return;
        }
    };
    let progress_tx = event_tx.clone();
    match runtime.import_offline_model(source_dir, move |progress| {
        if let IndexTtsProgress::ModelDownload {
            phase,
            downloaded_bytes,
            total_bytes,
        } = progress
        {
            let _ = progress_tx.send(WorkerEvent::IndexTtsImportProgress {
                phase,
                copied_bytes: downloaded_bytes,
                total_bytes,
            });
        }
    }) {
        Ok(model_dir) => {
            let _ = event_tx.send(WorkerEvent::IndexTtsImported { model_dir });
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::IndexTtsImportFailed(error));
        }
    }
}

fn ensure_indextts_ready<'a>(
    runtime: &'a Result<IndexTtsRuntime, String>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
) -> Result<&'a IndexTtsRuntime, String> {
    let runtime = runtime.as_ref().map_err(Clone::clone)?;
    let progress_tx = event_tx.clone();
    runtime.ensure_ready(move |progress| forward_indextts_progress(&progress_tx, progress))?;
    let _ = event_tx.send(WorkerEvent::IndexTtsReady);
    Ok(runtime)
}

fn forward_indextts_progress(
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    progress: IndexTtsProgress,
) {
    match progress {
        IndexTtsProgress::Phase(phase) => {
            let _ = event_tx.send(WorkerEvent::IndexTtsPreparing {
                phase,
                downloaded_bytes: 0,
                total_bytes: 0,
            });
        }
        IndexTtsProgress::ModelDownload {
            phase,
            downloaded_bytes,
            total_bytes,
        } => {
            let _ = event_tx.send(WorkerEvent::IndexTtsPreparing {
                phase,
                downloaded_bytes,
                total_bytes,
            });
        }
        IndexTtsProgress::Inference { .. } => {}
    }
}

fn synthesize_indextts_batch(
    runtime: &Result<IndexTtsRuntime, String>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    texts: &[String],
    selection: &IndexTtsSelection,
    _rate_percent: i32,
    settings: IndexTtsSettings,
    subtitle_progress: bool,
) -> Result<Vec<Vec<f32>>, String> {
    let runtime = ensure_indextts_ready(runtime, event_tx)?;
    let progress_tx = event_tx.clone();
    runtime.synthesize_batch_with_settings(
        texts,
        &selection.reference_path,
        &settings,
        move |progress| match progress {
            IndexTtsProgress::Inference { current, total } if subtitle_progress => {
                let _ = progress_tx.send(WorkerEvent::SubtitleProgress { current, total });
            }
            IndexTtsProgress::Inference { current, total } => {
                let _ = progress_tx.send(WorkerEvent::QwenProgress { current, total });
            }
            IndexTtsProgress::Phase(phase) => {
                let _ = progress_tx.send(WorkerEvent::IndexTtsInferencePhase(phase));
            }
            other => forward_indextts_progress(&progress_tx, other),
        },
    )
}

async fn preview_indextts_voice(
    runtime: &Result<IndexTtsRuntime, String>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    text: String,
    selection: IndexTtsSelection,
    rate_percent: i32,
    volume_percent: i32,
    settings: IndexTtsSettings,
) {
    let mut batches = match synthesize_indextts_batch(
        runtime,
        event_tx,
        &[text],
        &selection,
        rate_percent,
        settings,
        false,
    ) {
        Ok(batches) => batches,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::PreviewFailed(error));
            return;
        }
    };
    let Some(mut samples) = batches.pop() else {
        let _ = event_tx.send(WorkerEvent::PreviewFailed(
            "IndexTTS-2.5 没有返回试听音频。".to_owned(),
        ));
        return;
    };
    timeline_audio::apply_volume(&mut samples, volume_percent);
    match timeline_audio::encode_mono_mp3(&samples) {
        Ok(mp3) => play_preview_mp3(event_tx, mp3).await,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::PreviewFailed(error));
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn generate_indextts_mp3(
    runtime: &Result<IndexTtsRuntime, String>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    text: String,
    selection: IndexTtsSelection,
    rate_percent: i32,
    volume_percent: i32,
    settings: IndexTtsSettings,
    output_path: PathBuf,
) {
    let mut batches = match synthesize_indextts_batch(
        runtime,
        event_tx,
        &[text],
        &selection,
        rate_percent,
        settings,
        false,
    ) {
        Ok(batches) => batches,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    };
    let Some(mut samples) = batches.pop() else {
        let _ = event_tx.send(WorkerEvent::GenerationFailed(
            "IndexTTS-2.5 没有返回可保存的音频。".to_owned(),
        ));
        return;
    };
    timeline_audio::apply_volume(&mut samples, volume_percent);
    let mp3 = match timeline_audio::encode_mono_mp3(&samples) {
        Ok(mp3) => mp3,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    };
    let byte_count = mp3.len();
    match tokio::fs::write(&output_path, mp3).await {
        Ok(()) => {
            let _ = event_tx.send(WorkerEvent::GenerationFinished {
                output_path,
                byte_count,
            });
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(format!(
                "IndexTTS-2.5 已完成合成，但无法保存 MP3：{error}"
            )));
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn generate_indextts_subtitle_mp3(
    runtime: &Result<IndexTtsRuntime, String>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    cues: Vec<SubtitleCue>,
    selection: IndexTtsSelection,
    rate_percent: i32,
    volume_percent: i32,
    settings: IndexTtsSettings,
    output_path: PathBuf,
) {
    if cues.is_empty() {
        let _ = event_tx.send(WorkerEvent::GenerationFailed(
            "导入的字幕没有可合成的时间轴文本。".to_owned(),
        ));
        return;
    }
    let texts: Vec<_> = cues.iter().map(|cue| cue.text.clone()).collect();
    let mut clips = match synthesize_indextts_batch(
        runtime,
        event_tx,
        &texts,
        &selection,
        rate_percent,
        settings,
        true,
    ) {
        Ok(clips) => clips,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    };
    if clips.len() != cues.len() {
        let _ = event_tx.send(WorkerEvent::GenerationFailed(
            "IndexTTS-2.5 返回的音频段数量与字幕条数不一致。".to_owned(),
        ));
        return;
    }
    let timeline_end_ms = cues.iter().map(|cue| cue.end_ms).max().unwrap_or(0);
    let mut encoder = timeline_audio::TimelineMp3Encoder::new();
    let mut overflow_count = 0;
    for (index, (cue, samples)) in cues.iter().zip(clips.iter_mut()).enumerate() {
        timeline_audio::apply_volume(samples, volume_percent);
        if let Err(error) =
            encoder.write_silence_until(timeline_audio::milliseconds_to_samples(cue.start_ms))
        {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
        let slot_end_ms = cues
            .get(index + 1)
            .map(|next| cue.end_ms.min(next.start_ms))
            .unwrap_or(cue.end_ms)
            .max(cue.start_ms + 1);
        let available =
            timeline_audio::milliseconds_to_samples(slot_end_ms - cue.start_ms) as usize;
        if samples.len() > available {
            overflow_count += 1;
        }
        if let Err(error) = encoder.write_clip(samples) {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    }
    if let Err(error) =
        encoder.write_silence_until(timeline_audio::milliseconds_to_samples(timeline_end_ms))
    {
        let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
        return;
    }
    let mp3 = match encoder.finish() {
        Ok(mp3) => mp3,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    };
    let byte_count = mp3.len();
    match tokio::fs::write(&output_path, mp3).await {
        Ok(()) => {
            let _ = event_tx.send(WorkerEvent::SubtitleGenerationFinished {
                output_path,
                byte_count,
                overflow_count,
            });
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(format!(
                "IndexTTS-2.5 字幕音频已完成，但无法保存 MP3：{error}"
            )));
        }
    }
}

async fn preview_voice(
    client: &EdgeTtsClient,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    text: String,
    voice: String,
    rate_percent: i32,
    volume_percent: i32,
) {
    let options = SpeakOptions {
        voice,
        rate: signed_percent(rate_percent),
        volume: signed_percent(volume_percent),
        ..SpeakOptions::default()
    };

    let result = match client.synthesize(text, options).await {
        Ok(result) => result,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::PreviewFailed(format!(
                "Could not synthesize the preview: {error}"
            )));
            return;
        }
    };

    play_preview_mp3(event_tx, result.audio).await;
}

async fn play_preview_mp3(event_tx: &mpsc::UnboundedSender<WorkerEvent>, audio: Vec<u8>) {
    #[cfg(target_os = "macos")]
    {
        let preview_path = std::env::temp_dir().join(format!(
            "edge-tts-studio-preview-{}.mp3",
            std::process::id()
        ));
        if let Err(error) = tokio::fs::write(&preview_path, &audio).await {
            let _ = event_tx.send(WorkerEvent::PreviewFailed(format!(
                "Could not prepare the preview audio: {error}"
            )));
            return;
        }

        let playback_path = preview_path.clone();
        let playback_result = tokio::task::spawn_blocking(move || {
            std::process::Command::new("/usr/bin/afplay")
                .arg(playback_path)
                .status()
        })
        .await;
        let _ = tokio::fs::remove_file(&preview_path).await;

        match playback_result {
            Ok(Ok(status)) if status.success() => {
                let _ = event_tx.send(WorkerEvent::PreviewFinished);
            }
            Ok(Ok(status)) => {
                let _ = event_tx.send(WorkerEvent::PreviewFailed(format!(
                    "macOS audio playback exited with status {status}"
                )));
            }
            Ok(Err(error)) => {
                let _ = event_tx.send(WorkerEvent::PreviewFailed(format!(
                    "Could not start macOS audio playback: {error}"
                )));
            }
            Err(error) => {
                let _ = event_tx.send(WorkerEvent::PreviewFailed(format!(
                    "The audio playback task failed: {error}"
                )));
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        let playback_result = tokio::task::spawn_blocking(move || -> Result<(), String> {
            let samples = timeline_audio::decode_mp3_mono_preserving_silence(&audio)
                .map_err(|error| format!("Could not decode the preview audio: {error}"))?;
            if samples.is_empty() {
                return Err("The preview audio did not contain any samples.".to_owned());
            }

            let (_stream, stream_handle) = rodio::OutputStream::try_default()
                .map_err(|error| format!("Could not open the Windows audio device: {error}"))?;
            let sink = rodio::Sink::try_new(&stream_handle)
                .map_err(|error| format!("Could not create the Windows audio player: {error}"))?;
            sink.append(rodio::buffer::SamplesBuffer::new(
                1,
                timeline_audio::TIMELINE_SAMPLE_RATE,
                samples,
            ));
            sink.sleep_until_end();
            Ok(())
        })
        .await;

        match playback_result {
            Ok(Ok(())) => {
                let _ = event_tx.send(WorkerEvent::PreviewFinished);
            }
            Ok(Err(error)) => {
                let _ = event_tx.send(WorkerEvent::PreviewFailed(error));
            }
            Err(error) => {
                let _ = event_tx.send(WorkerEvent::PreviewFailed(format!(
                    "The Windows audio playback task failed: {error}"
                )));
            }
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = audio;
        let _ = event_tx.send(WorkerEvent::PreviewFailed(
            "Voice preview playback is currently available on macOS and Windows.".to_owned(),
        ));
    }
}

fn ensure_qwen_model<'a>(
    model_cache: &'a mut Option<LocalQwenModel>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    version: QwenModelVersion,
    kind: QwenModelKind,
) -> Result<&'a LocalQwenModel, String> {
    if model_cache
        .as_ref()
        .is_some_and(|model| model.version() != version || model.kind() != kind)
    {
        *model_cache = None;
        let _ = event_tx.send(WorkerEvent::QwenModelReleased);
    }
    if model_cache.is_none() {
        let _ = event_tx.send(WorkerEvent::QwenModelPreparing { version, kind });
        let progress_tx = event_tx.clone();
        let model = LocalQwenModel::load(version, kind, move |progress| {
            let _ = progress_tx.send(WorkerEvent::QwenModelDownload {
                version,
                kind,
                file: progress.file.to_owned(),
                downloaded_bytes: progress.downloaded_bytes,
                total_bytes: progress.total_bytes,
            });
        })?;
        let device = model.device_label().to_owned();
        *model_cache = Some(model);
        let _ = event_tx.send(WorkerEvent::QwenModelReady {
            version,
            kind,
            device,
        });
    }
    model_cache
        .as_ref()
        .ok_or_else(|| "Qwen3-TTS 模型未能完成初始化。".to_owned())
}

fn synthesize_qwen_pcm(
    model: &LocalQwenModel,
    text: &str,
    voice: QwenVoice,
    language: QwenSynthesisLanguage,
    rate_percent: i32,
    volume_percent: i32,
    settings: QwenGenerationSettings,
) -> Result<Vec<f32>, String> {
    let audio = model.synthesize_with_settings(text, voice, language, settings)?;
    if audio.samples.is_empty() {
        return Err("Qwen3-TTS 没有生成可用的音频采样。".to_owned());
    }
    let samples = if audio.sample_rate == timeline_audio::TIMELINE_SAMPLE_RATE {
        audio.samples
    } else {
        timeline_audio::resample_linear(
            &audio.samples,
            audio.sample_rate,
            timeline_audio::TIMELINE_SAMPLE_RATE,
        )
    };
    let mut samples = timeline_audio::adjust_speed(&samples, rate_percent);
    timeline_audio::apply_volume(&mut samples, volume_percent);
    Ok(samples)
}

fn clone_prompt_key(selection: &QwenCloneSelection) -> Result<QwenClonePromptKey, String> {
    let metadata = std::fs::metadata(&selection.reference_path).map_err(|error| {
        format!(
            "无法读取参考音频 {}：{error}",
            selection.reference_path.display()
        )
    })?;
    let modified_nanos = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    Ok(QwenClonePromptKey {
        version: selection.version,
        reference_path: selection.reference_path.clone(),
        reference_text: selection.reference_text.clone(),
        file_len: metadata.len(),
        modified_nanos,
    })
}

fn ensure_qwen_clone_context<'a>(
    model_cache: &'a mut Option<LocalQwenModel>,
    prompt_cache: &'a mut Option<CachedQwenClonePrompt>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    selection: &QwenCloneSelection,
) -> Result<(&'a LocalQwenModel, &'a LocalVoiceClonePrompt), String> {
    let key = clone_prompt_key(selection)?;
    let model = ensure_qwen_model(
        model_cache,
        event_tx,
        selection.version,
        QwenModelKind::VoiceClone,
    )?;
    if prompt_cache.as_ref().is_none_or(|cached| cached.key != key) {
        let _ = event_tx.send(WorkerEvent::QwenClonePromptPreparing);
        let reference_audio = qwen_local::load_reference_audio(&selection.reference_path)?;
        let reference_text = (!selection.reference_text.trim().is_empty())
            .then_some(selection.reference_text.trim());
        let prompt = model.create_voice_clone_prompt(&reference_audio, reference_text)?;
        *prompt_cache = Some(CachedQwenClonePrompt { key, prompt });
        let _ = event_tx.send(WorkerEvent::QwenClonePromptReady);
    }
    let prompt = &prompt_cache
        .as_ref()
        .ok_or_else(|| "音色克隆提示未能完成初始化。".to_owned())?
        .prompt;
    Ok((model, prompt))
}

fn synthesize_qwen_clone_pcm(
    model: &LocalQwenModel,
    prompt: &LocalVoiceClonePrompt,
    text: &str,
    language: QwenSynthesisLanguage,
    rate_percent: i32,
    volume_percent: i32,
    settings: QwenGenerationSettings,
) -> Result<Vec<f32>, String> {
    let audio = model.synthesize_voice_clone_with_settings(text, prompt, language, settings)?;
    if audio.samples.is_empty() {
        return Err("Qwen3-TTS 音色克隆没有生成可用的音频采样。".to_owned());
    }
    let samples = if audio.sample_rate == timeline_audio::TIMELINE_SAMPLE_RATE {
        audio.samples
    } else {
        timeline_audio::resample_linear(
            &audio.samples,
            audio.sample_rate,
            timeline_audio::TIMELINE_SAMPLE_RATE,
        )
    };
    let mut samples = timeline_audio::adjust_speed(&samples, rate_percent);
    timeline_audio::apply_volume(&mut samples, volume_percent);
    Ok(samples)
}

#[allow(clippy::too_many_arguments)]
async fn preview_qwen_clone(
    model_cache: &mut Option<LocalQwenModel>,
    prompt_cache: &mut Option<CachedQwenClonePrompt>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    text: String,
    selection: QwenCloneSelection,
    rate_percent: i32,
    volume_percent: i32,
    settings: QwenGenerationSettings,
) {
    let (model, prompt) =
        match ensure_qwen_clone_context(model_cache, prompt_cache, event_tx, &selection) {
            Ok(context) => context,
            Err(error) => {
                let _ = event_tx.send(WorkerEvent::PreviewFailed(error));
                return;
            }
        };
    let language = qwen_local::synthesis_language_for_clone([text.as_str()]);
    let samples = match synthesize_qwen_clone_pcm(
        model,
        prompt,
        &text,
        language,
        rate_percent,
        volume_percent,
        settings,
    ) {
        Ok(samples) => samples,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::PreviewFailed(error));
            return;
        }
    };
    match timeline_audio::encode_mono_mp3(&samples) {
        Ok(mp3) => play_preview_mp3(event_tx, mp3).await,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::PreviewFailed(format!(
                "Qwen3-TTS 克隆试听编码失败：{error}"
            )));
        }
    }
}

async fn preview_qwen_voice(
    model_cache: &mut Option<LocalQwenModel>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    text: String,
    selection: QwenSelection,
    rate_percent: i32,
    volume_percent: i32,
    settings: QwenGenerationSettings,
) {
    let model = match ensure_qwen_model(
        model_cache,
        event_tx,
        selection.version,
        QwenModelKind::CustomVoice,
    ) {
        Ok(model) => model,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::PreviewFailed(error));
            return;
        }
    };
    let language = qwen_local::synthesis_language([text.as_str()], selection.voice);
    let samples = match synthesize_qwen_pcm(
        model,
        &text,
        selection.voice,
        language,
        rate_percent,
        volume_percent,
        settings,
    ) {
        Ok(samples) => samples,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::PreviewFailed(error));
            return;
        }
    };
    match timeline_audio::encode_mono_mp3(&samples) {
        Ok(mp3) => play_preview_mp3(event_tx, mp3).await,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::PreviewFailed(format!(
                "Qwen3-TTS 试听编码失败：{error}"
            )));
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn generate_qwen_mp3(
    model_cache: &mut Option<LocalQwenModel>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    text: String,
    selection: QwenSelection,
    rate_percent: i32,
    volume_percent: i32,
    settings: QwenGenerationSettings,
    output_path: PathBuf,
) {
    let model = match ensure_qwen_model(
        model_cache,
        event_tx,
        selection.version,
        QwenModelKind::CustomVoice,
    ) {
        Ok(model) => model,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    };
    let chunks = qwen_local::split_for_synthesis(&text);
    if chunks.is_empty() {
        let _ = event_tx.send(WorkerEvent::GenerationFailed(
            "没有可供 Qwen3-TTS 合成的文字。".to_owned(),
        ));
        return;
    }

    let total = chunks.len();
    let language = qwen_local::synthesis_language([text.as_str()], selection.voice);
    let gap = vec![0.0_f32; timeline_audio::TIMELINE_SAMPLE_RATE as usize * 90 / 1_000];
    let mut encoder = timeline_audio::TimelineMp3Encoder::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let current = index + 1;
        let _ = event_tx.send(WorkerEvent::QwenProgress { current, total });
        let samples = match synthesize_qwen_pcm(
            model,
            chunk,
            selection.voice,
            language,
            rate_percent,
            volume_percent,
            settings,
        ) {
            Ok(samples) => samples,
            Err(error) => {
                let _ = event_tx.send(WorkerEvent::GenerationFailed(format!(
                    "第 {current}/{total} 段：{error}"
                )));
                return;
            }
        };
        if let Err(error) = encoder.write_clip(&samples) {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
        if current < total
            && let Err(error) = encoder.write_clip(&gap)
        {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    }

    let mp3 = match encoder.finish() {
        Ok(mp3) => mp3,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    };
    let byte_count = mp3.len();
    match tokio::fs::write(&output_path, mp3).await {
        Ok(()) => {
            let _ = event_tx.send(WorkerEvent::GenerationFinished {
                output_path,
                byte_count,
            });
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(format!(
                "Qwen3-TTS 已完成合成，但无法保存 MP3：{error}"
            )));
        }
    }
}

async fn generate_qwen_clone_mp3(
    model_cache: &mut Option<LocalQwenModel>,
    prompt_cache: &mut Option<CachedQwenClonePrompt>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    text: String,
    selection: QwenCloneSelection,
    settings: QwenOutputSettings,
) {
    let (model, prompt) =
        match ensure_qwen_clone_context(model_cache, prompt_cache, event_tx, &selection) {
            Ok(context) => context,
            Err(error) => {
                let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
                return;
            }
        };
    let chunks = qwen_local::split_for_synthesis(&text);
    if chunks.is_empty() {
        let _ = event_tx.send(WorkerEvent::GenerationFailed(
            "没有可供 Qwen3-TTS 克隆音色合成的文字。".to_owned(),
        ));
        return;
    }

    let total = chunks.len();
    // The language token and clone prompt are selected once for the complete
    // article, then reused unchanged for every chunk.
    let language = qwen_local::synthesis_language_for_clone([text.as_str()]);
    let gap = vec![0.0_f32; timeline_audio::TIMELINE_SAMPLE_RATE as usize * 90 / 1_000];
    let mut encoder = timeline_audio::TimelineMp3Encoder::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let current = index + 1;
        let _ = event_tx.send(WorkerEvent::QwenProgress { current, total });
        let samples = match synthesize_qwen_clone_pcm(
            model,
            prompt,
            chunk,
            language,
            settings.rate_percent,
            settings.volume_percent,
            settings.qwen_settings,
        ) {
            Ok(samples) => samples,
            Err(error) => {
                let _ = event_tx.send(WorkerEvent::GenerationFailed(format!(
                    "第 {current}/{total} 段：{error}"
                )));
                return;
            }
        };
        if let Err(error) = encoder.write_clip(&samples) {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
        if current < total
            && let Err(error) = encoder.write_clip(&gap)
        {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    }

    let mp3 = match encoder.finish() {
        Ok(mp3) => mp3,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    };
    let byte_count = mp3.len();
    match tokio::fs::write(&settings.output_path, mp3).await {
        Ok(()) => {
            let _ = event_tx.send(WorkerEvent::GenerationFinished {
                output_path: settings.output_path,
                byte_count,
            });
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(format!(
                "克隆音色已完成合成，但无法保存 MP3：{error}"
            )));
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn generate_qwen_subtitle_mp3(
    model_cache: &mut Option<LocalQwenModel>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    cues: Vec<SubtitleCue>,
    selection: QwenSelection,
    rate_percent: i32,
    volume_percent: i32,
    settings: QwenGenerationSettings,
    output_path: PathBuf,
) {
    if cues.is_empty() {
        let _ = event_tx.send(WorkerEvent::GenerationFailed(
            "导入的字幕没有可合成的时间轴文本。".to_owned(),
        ));
        return;
    }
    let model = match ensure_qwen_model(
        model_cache,
        event_tx,
        selection.version,
        QwenModelKind::CustomVoice,
    ) {
        Ok(model) => model,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    };

    let total = cues.len();
    let language =
        qwen_local::synthesis_language(cues.iter().map(|cue| cue.text.as_str()), selection.voice);
    let timeline_end_ms = cues.iter().map(|cue| cue.end_ms).max().unwrap_or(0);
    let mut encoder = timeline_audio::TimelineMp3Encoder::new();
    let mut overflow_count = 0;

    for (index, cue) in cues.iter().enumerate() {
        let current = index + 1;
        let _ = event_tx.send(WorkerEvent::SubtitleProgress { current, total });
        let cue_start_sample = timeline_audio::milliseconds_to_samples(cue.start_ms);
        if let Err(error) = encoder.write_silence_until(cue_start_sample) {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }

        let cue_slot_end_ms = cues
            .get(index + 1)
            .map(|next| cue.end_ms.min(next.start_ms))
            .unwrap_or(cue.end_ms)
            .max(cue.start_ms + 1);
        let available_samples =
            timeline_audio::milliseconds_to_samples(cue_slot_end_ms.saturating_sub(cue.start_ms))
                as usize;
        let clip = match synthesize_qwen_pcm(
            model,
            &cue.text,
            selection.voice,
            language,
            rate_percent,
            volume_percent,
            settings,
        ) {
            Ok(clip) => clip,
            Err(error) => {
                let _ = event_tx.send(WorkerEvent::GenerationFailed(format!(
                    "字幕 {current}/{total}：{error}"
                )));
                return;
            }
        };

        if clip.len() > available_samples {
            // Never alter pitch or drop spoken words to force a cue into a short
            // subtitle slot. The next cue remains anchored when possible; if
            // this clip runs long, TimelineMp3Encoder naturally starts it after
            // the preceding speech instead of overlapping or truncating it.
            overflow_count += 1;
        }
        if let Err(error) = encoder.write_clip(&clip) {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    }

    if let Err(error) =
        encoder.write_silence_until(timeline_audio::milliseconds_to_samples(timeline_end_ms))
    {
        let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
        return;
    }
    let mp3 = match encoder.finish() {
        Ok(mp3) => mp3,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    };
    let byte_count = mp3.len();
    match tokio::fs::write(&output_path, mp3).await {
        Ok(()) => {
            let _ = event_tx.send(WorkerEvent::SubtitleGenerationFinished {
                output_path,
                byte_count,
                overflow_count,
            });
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(format!(
                "Qwen3-TTS 字幕音频已完成，但无法保存 MP3：{error}"
            )));
        }
    }
}

async fn generate_qwen_clone_subtitle_mp3(
    model_cache: &mut Option<LocalQwenModel>,
    prompt_cache: &mut Option<CachedQwenClonePrompt>,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    cues: Vec<SubtitleCue>,
    selection: QwenCloneSelection,
    settings: QwenOutputSettings,
) {
    if cues.is_empty() {
        let _ = event_tx.send(WorkerEvent::GenerationFailed(
            "导入的字幕没有可合成的时间轴文本。".to_owned(),
        ));
        return;
    }
    let (model, prompt) =
        match ensure_qwen_clone_context(model_cache, prompt_cache, event_tx, &selection) {
            Ok(context) => context,
            Err(error) => {
                let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
                return;
            }
        };

    let total = cues.len();
    // Deliberately lock both the language strategy and cloned prompt for the
    // complete subtitle task. Chinese/English cue boundaries never select a
    // different speaker or rebuild the prompt.
    let language =
        qwen_local::synthesis_language_for_clone(cues.iter().map(|cue| cue.text.as_str()));
    let timeline_end_ms = cues.iter().map(|cue| cue.end_ms).max().unwrap_or(0);
    let mut encoder = timeline_audio::TimelineMp3Encoder::new();
    let mut overflow_count = 0;

    for (index, cue) in cues.iter().enumerate() {
        let current = index + 1;
        let _ = event_tx.send(WorkerEvent::SubtitleProgress { current, total });
        let cue_start_sample = timeline_audio::milliseconds_to_samples(cue.start_ms);
        if let Err(error) = encoder.write_silence_until(cue_start_sample) {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
        let cue_slot_end_ms = cues
            .get(index + 1)
            .map(|next| cue.end_ms.min(next.start_ms))
            .unwrap_or(cue.end_ms)
            .max(cue.start_ms + 1);
        let available_samples =
            timeline_audio::milliseconds_to_samples(cue_slot_end_ms.saturating_sub(cue.start_ms))
                as usize;
        let clip = match synthesize_qwen_clone_pcm(
            model,
            prompt,
            &cue.text,
            language,
            settings.rate_percent,
            settings.volume_percent,
            settings.qwen_settings,
        ) {
            Ok(clip) => clip,
            Err(error) => {
                let _ = event_tx.send(WorkerEvent::GenerationFailed(format!(
                    "字幕 {current}/{total}：{error}"
                )));
                return;
            }
        };
        if clip.len() > available_samples {
            overflow_count += 1;
        }
        if let Err(error) = encoder.write_clip(&clip) {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    }

    if let Err(error) =
        encoder.write_silence_until(timeline_audio::milliseconds_to_samples(timeline_end_ms))
    {
        let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
        return;
    }
    let mp3 = match encoder.finish() {
        Ok(mp3) => mp3,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    };
    let byte_count = mp3.len();
    match tokio::fs::write(&settings.output_path, mp3).await {
        Ok(()) => {
            let _ = event_tx.send(WorkerEvent::SubtitleGenerationFinished {
                output_path: settings.output_path,
                byte_count,
                overflow_count,
            });
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(format!(
                "克隆音色字幕已完成合成，但无法保存 MP3：{error}"
            )));
        }
    }
}

async fn fetch_voices(
    client: &EdgeTtsClient,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    cache_path: &Path,
) {
    match client.list_voices().await {
        Ok(voices) => {
            let mut choices: Vec<_> = voices
                .into_iter()
                .map(|voice| VoiceChoice {
                    short_name: voice.short_name,
                    locale: voice.locale,
                    gender: voice.gender,
                    friendly_name: voice.friendly_name,
                })
                .collect();
            choices.sort_by(|left, right| {
                left.locale
                    .cmp(&right.locale)
                    .then_with(|| left.short_name.cmp(&right.short_name))
            });

            if let Err(error) = save_voice_cache(cache_path, &choices).await {
                let _ = event_tx.send(WorkerEvent::VoicesLoaded {
                    voices: choices,
                    from_cache: false,
                    warning: Some(format!(
                        "Voices loaded, but the local cache could not be saved: {error}"
                    )),
                });
            } else {
                let _ = event_tx.send(WorkerEvent::VoicesLoaded {
                    voices: choices,
                    from_cache: false,
                    warning: None,
                });
            }
        }
        Err(network_error) => match load_voice_cache(cache_path).await {
            Ok(voices) if !voices.is_empty() => {
                let _ = event_tx.send(WorkerEvent::VoicesLoaded {
                    voices,
                    from_cache: true,
                    warning: Some(format!(
                        "Could not refresh voices ({network_error}). Using the last cached list."
                    )),
                });
            }
            _ => {
                let _ = event_tx.send(WorkerEvent::VoicesFailed(format!(
                    "Could not load Edge TTS voices: {network_error}. Check your internet connection and try again."
                )));
            }
        },
    }
}

async fn generate_mp3(
    client: &EdgeTtsClient,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    text: String,
    voice: String,
    rate_percent: i32,
    volume_percent: i32,
    output_path: PathBuf,
) {
    let options = SpeakOptions {
        voice,
        rate: signed_percent(rate_percent),
        volume: signed_percent(volume_percent),
        ..SpeakOptions::default()
    };

    match client.synthesize(text, options).await {
        Ok(result) => {
            let byte_count = result.audio.len();
            match tokio::fs::write(&output_path, &result.audio).await {
                Ok(()) => {
                    let _ = event_tx.send(WorkerEvent::GenerationFinished {
                        output_path,
                        byte_count,
                    });
                }
                Err(error) => {
                    let _ = event_tx.send(WorkerEvent::GenerationFailed(format!(
                        "Speech was generated, but the MP3 file could not be saved: {error}"
                    )));
                }
            }
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(format!(
                "Could not generate speech: {error}"
            )));
        }
    }
}

async fn generate_subtitle_mp3(
    client: &EdgeTtsClient,
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    cues: Vec<SubtitleCue>,
    voice: String,
    rate_percent: i32,
    volume_percent: i32,
    output_path: PathBuf,
) {
    let progress_tx = event_tx.clone();
    let report = match subtitle_pipeline::generate_subtitle_audio(
        client,
        &cues,
        &voice,
        rate_percent,
        volume_percent,
        move |current, total| {
            let _ = progress_tx.send(WorkerEvent::SubtitleProgress { current, total });
        },
    )
    .await
    {
        Ok(report) => report,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(error));
            return;
        }
    };
    let byte_count = report.mp3.len();
    match tokio::fs::write(&output_path, report.mp3).await {
        Ok(()) => {
            let _ = event_tx.send(WorkerEvent::SubtitleGenerationFinished {
                output_path,
                byte_count,
                overflow_count: report.overflow_count,
            });
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(format!(
                "The subtitle audio was generated, but the MP3 file could not be saved: {error}"
            )));
        }
    }
}

async fn transcribe_media_locally(
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    model_cache: &mut Option<(RecognitionLanguage, Whisper)>,
    input_path: PathBuf,
    language: RecognitionLanguage,
) {
    let needs_model = model_cache
        .as_ref()
        .is_none_or(|(loaded_language, _)| *loaded_language != language);
    if needs_model {
        // Loading and inference stay on this Tokio worker. egui only receives
        // progress events and therefore remains responsive even on first use.
        let progress_tx = event_tx.clone();
        let mut last_progress_event = Instant::now() - Duration::from_secs(1);
        let mut previous_source = String::new();
        let model = match asr::load_model(language, move |progress| match progress {
            ModelLoadingProgress::Downloading { source, progress } => {
                let source_changed = source != previous_source;
                let finished = progress.progress >= progress.size;
                if source_changed
                    || finished
                    || last_progress_event.elapsed() >= Duration::from_millis(150)
                {
                    previous_source.clone_from(&source);
                    last_progress_event = Instant::now();
                    let _ = progress_tx.send(WorkerEvent::AsrModelProgress {
                        source,
                        downloaded_bytes: progress.progress,
                        total_bytes: progress.size,
                    });
                }
            }
            ModelLoadingProgress::Loading { progress } => {
                if last_progress_event.elapsed() >= Duration::from_millis(150) || progress >= 1.0 {
                    last_progress_event = Instant::now();
                    let _ = progress_tx.send(WorkerEvent::AsrModelLoading {
                        progress: progress.clamp(0.0, 1.0),
                    });
                }
            }
        })
        .await
        {
            Ok(model) => model,
            Err(error) => {
                let _ = event_tx.send(WorkerEvent::TranscriptionFailed(error));
                return;
            }
        };
        *model_cache = Some((language, model));
    }

    let _ = event_tx.send(WorkerEvent::AsrModelReady);
    let progress_tx = event_tx.clone();
    let Some((_, model)) = model_cache.as_ref() else {
        let _ = event_tx.send(WorkerEvent::TranscriptionFailed(
            "The local model was not available after loading.".to_owned(),
        ));
        return;
    };
    let report = match asr::transcribe_media(model, &input_path, move |progress, remaining| {
        let _ = progress_tx.send(WorkerEvent::TranscriptionProgress {
            progress,
            remaining_seconds: remaining,
        });
    })
    .await
    {
        Ok(report) => report,
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::TranscriptionFailed(error));
            return;
        }
    };

    let _ = event_tx.send(WorkerEvent::TranscriptionFinished {
        cues: report.cues,
        audio_duration_ms: report.audio_duration_ms,
    });
}

async fn save_transcription_locally(
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    output_path: PathBuf,
    cues: Vec<SubtitleCue>,
    format: SubtitleExportFormat,
) {
    let subtitle_text = asr::render_subtitles(&cues, format);
    if let Some(parent) = output_path.parent()
        && let Err(error) = tokio::fs::create_dir_all(parent).await
    {
        let _ = event_tx.send(WorkerEvent::TranscriptionSaveFailed(format!(
            "The output folder could not be created: {error}"
        )));
        return;
    }
    match tokio::fs::write(&output_path, subtitle_text.as_bytes()).await {
        Ok(()) => {
            let _ = event_tx.send(WorkerEvent::TranscriptionSaved(output_path));
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::TranscriptionSaveFailed(format!(
                "The subtitle file could not be written: {error}"
            )));
        }
    }
}

async fn save_voice_cache(path: &Path, voices: &[VoiceChoice]) -> Result<(), String> {
    let json = serde_json::to_vec_pretty(voices).map_err(|error| error.to_string())?;
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| error.to_string())?;
    }
    tokio::fs::write(path, json)
        .await
        .map_err(|error| error.to_string())
}

async fn load_voice_cache(path: &Path) -> Result<Vec<VoiceChoice>, String> {
    let json = tokio::fs::read(path)
        .await
        .map_err(|error| error.to_string())?;
    serde_json::from_slice(&json).map_err(|error| error.to_string())
}

fn voice_cache_path() -> PathBuf {
    ProjectDirs::from("com", "Aura Labs", "Edge TTS Studio")
        .map(|dirs| dirs.cache_dir().join("voices.json"))
        .unwrap_or_else(|| std::env::temp_dir().join("edge-tts-studio-voices.json"))
}

fn ensure_mp3_extension(mut path: PathBuf) -> PathBuf {
    let is_mp3 = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mp3"));
    if !is_mp3 {
        path.set_extension("mp3");
    }
    path
}

fn ensure_subtitle_extension(mut path: PathBuf, format: SubtitleExportFormat) -> PathBuf {
    let expected = format.extension();
    let matches = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case(expected));
    if !matches {
        path.set_extension(expected);
    }
    path
}

async fn load_generated_audio(event_tx: &mpsc::UnboundedSender<WorkerEvent>, input_path: PathBuf) {
    let decode_path = input_path.clone();
    match tokio::task::spawn_blocking(move || asr::decode_media_mono(&decode_path)).await {
        Ok(Ok(samples)) if !samples.is_empty() => {
            let _ = event_tx.send(WorkerEvent::GeneratedAudioLoaded {
                input_path,
                samples,
                sample_rate: timeline_audio::TIMELINE_SAMPLE_RATE,
            });
        }
        Ok(Ok(_)) => {
            let _ = event_tx.send(WorkerEvent::GeneratedAudioLoadFailed {
                input_path,
                error: "文件中没有可播放的音频采样。".to_owned(),
            });
        }
        Ok(Err(error)) => {
            let _ = event_tx.send(WorkerEvent::GeneratedAudioLoadFailed { input_path, error });
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GeneratedAudioLoadFailed {
                input_path,
                error: format!("音频解码任务异常：{error}"),
            });
        }
    }
}

fn localized_transcription_error(error: &str) -> String {
    if error.contains("No speech was recognized") {
        "没有识别到清晰语音。中英文交替的音频请选择“中英混合”，并确认音量足够且人声清楚。"
            .to_owned()
    } else if error.contains("does not contain decodable audio")
        || error.contains("Could not decode the MP3 file")
        || error.contains("audio codec is not supported")
        || error.contains("No supported audio track")
    {
        format!(
            "无法解码所选媒体的音轨，请确认文件完整、未受保护且音频编码受支持。详细信息：{error}"
        )
    } else if error.contains("Could not load the local Whisper model") {
        format!("无法加载本地 Whisper 模型，请检查首次下载是否完成。详细信息：{error}")
    } else if error.contains("Could not read the MP3 file")
        || error.contains("Could not open the selected media file")
    {
        format!("无法读取所选媒体，请检查文件权限。详细信息：{error}")
    } else {
        error.to_owned()
    }
}

fn configure_egui_for_platform(ctx: &egui::Context) {
    // The interface uses a deliberate light palette so the custom card and
    // status colors stay legible and consistent across macOS appearances.
    ctx.set_theme(egui::Theme::Light);

    for theme in [egui::Theme::Light, egui::Theme::Dark] {
        let mut style = (*ctx.style_of(theme)).clone();
        style.spacing.item_spacing = egui::vec2(8.0, 8.0);
        style.spacing.button_padding = egui::vec2(14.0, 7.0);
        style.spacing.interact_size.y = 34.0;
        style.visuals.selection.bg_fill = PRIMARY;
        style.visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
        style
            .text_styles
            .insert(egui::TextStyle::Body, egui::FontId::proportional(14.0));
        style
            .text_styles
            .insert(egui::TextStyle::Button, egui::FontId::proportional(14.0));
        ctx.set_style_of(theme, style);
    }

    // egui's compact default font set does not include CJK glyphs. Load a
    // native fallback on each desktop OS so the Chinese-first UI is readable.
    let mut font_candidates = Vec::<PathBuf>::new();
    #[cfg(target_os = "macos")]
    font_candidates.extend(
        [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ]
        .into_iter()
        .map(PathBuf::from),
    );
    #[cfg(target_os = "windows")]
    {
        let windows_dir = std::env::var_os("WINDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
        font_candidates.extend(
            ["msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc"]
                .into_iter()
                .map(|name| windows_dir.join("Fonts").join(name)),
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    font_candidates.extend(
        [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]
        .into_iter()
        .map(PathBuf::from),
    );

    if let Some(font_bytes) = font_candidates
        .iter()
        .find_map(|path| std::fs::read(path).ok())
    {
        let fallback_name = "system-cjk".to_owned();
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            fallback_name.clone(),
            egui::FontData::from_owned(font_bytes).into(),
        );
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .push(fallback_name.clone());
        fonts
            .families
            .entry(egui::FontFamily::Monospace)
            .or_default()
            .push(fallback_name);
        ctx.set_fonts(fonts);
    }
}

fn main() -> eframe::Result {
    // Finder-launched macOS apps do not inherit the shell proxy variables used
    // by reqwest/tokio-tungstenite. Bridge the native system proxy first,
    // before eframe or Tokio has had an opportunity to start any threads.
    system_proxy::configure_from_macos_settings();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 820.0])
            .with_min_inner_size([960.0, 720.0]),
        // Do not let a previously persisted compact window override the
        // intended full-height startup layout.
        persist_window: false,
        ..Default::default()
    };

    eframe::run_native(
        APP_NAME,
        native_options,
        Box::new(|creation_context| Ok(Box::new(TtsApp::new(creation_context)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mp3_extension_is_added_or_preserved() {
        assert_eq!(
            ensure_mp3_extension(PathBuf::from("speech")),
            PathBuf::from("speech.mp3")
        );
        assert_eq!(
            ensure_mp3_extension(PathBuf::from("speech.MP3")),
            PathBuf::from("speech.MP3")
        );
        assert_eq!(
            ensure_mp3_extension(PathBuf::from("speech.wav")),
            PathBuf::from("speech.mp3")
        );
    }

    #[test]
    fn voice_filter_is_case_insensitive() {
        let voice = VoiceChoice {
            short_name: "zh-CN-XiaoxiaoNeural".to_owned(),
            locale: "zh-CN".to_owned(),
            gender: "Female".to_owned(),
            friendly_name: Some("Microsoft Xiaoxiao".to_owned()),
        };

        assert!(voice.matches("xiaoxiao"));
        assert!(voice.matches("晓晓"));
        assert!(voice.matches("中文"));
        assert!(voice.matches("ZH-cn"));
        assert!(voice.matches("female"));
        assert!(!voice.matches("en-US"));

        assert!(voice.label(UiLanguage::Chinese).contains("晓晓 Xiaoxiao"));
        assert!(voice.label(UiLanguage::English).contains("Xiaoxiao"));
    }

    #[test]
    fn preview_text_uses_a_short_unicode_safe_excerpt() {
        let long_text = "你好，欢迎试听这个音色。".repeat(12);
        let preview = preview_text(&long_text, UiLanguage::Chinese);
        assert_eq!(preview.chars().count(), 81);
        assert!(preview.ends_with('…'));
        assert_eq!(
            preview_text("   ", UiLanguage::English),
            "Hello, this is a preview of the selected voice."
        );
        assert_eq!(signed_percent(-25), "-25%");
        assert_eq!(signed_percent(0), "+0%");
        assert_eq!(signed_percent(80), "+80%");
    }

    #[test]
    fn voice_card_compacts_at_the_default_macos_workspace_height() {
        let compact = VoiceCardLayout::for_height(565.0);
        let roomy = VoiceCardLayout::for_height(640.0);

        assert!(compact.field_gap < roomy.field_gap);
        assert!(compact.metadata_height < roomy.metadata_height);
        assert!(compact.settings_margin < roomy.settings_margin);
        assert!(compact.preview_margin < roomy.preview_margin);
        assert!(compact.waveform_height < roomy.waveform_height);
        assert!(compact.preview_button_height < roomy.preview_button_height);
    }
}
