#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use std::{
    path::{Path, PathBuf},
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
mod subtitle_pipeline;
mod subtitles;
mod system_proxy;
mod timeline_audio;

use asr::{RecognitionLanguage, SubtitleExportFormat};
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
        voice: String,
        rate_percent: i32,
        volume_percent: i32,
    },
    Generate {
        text: String,
        voice: String,
        rate_percent: i32,
        volume_percent: i32,
        output_path: PathBuf,
    },
    GenerateSubtitles {
        cues: Vec<SubtitleCue>,
        voice: String,
        rate_percent: i32,
        volume_percent: i32,
        output_path: PathBuf,
    },
    TranscribeMp3 {
        input_path: PathBuf,
        output_path: PathBuf,
        language: RecognitionLanguage,
        format: SubtitleExportFormat,
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
        adjusted_count: usize,
        truncated_count: usize,
    },
    AsrModelProgress {
        source: String,
        downloaded_bytes: u64,
        total_bytes: u64,
    },
    AsrModelReady,
    TranscriptionProgress {
        progress: f32,
        remaining_seconds: u64,
    },
    TranscriptionFinished {
        output_path: PathBuf,
        cues: Vec<SubtitleCue>,
        audio_duration_ms: u64,
    },
    TranscriptionFailed(String),
    GenerationFailed(String),
    WorkerFailed(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputMode {
    Text,
    Subtitles,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkspaceMode {
    TextToSpeech,
    AudioToSubtitles,
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
    voice_filter: String,
    text: String,
    workspace_mode: WorkspaceMode,
    input_mode: InputMode,
    subtitle_track: Option<SubtitleTrack>,
    subtitle_path: Option<PathBuf>,
    subtitle_progress: Option<(usize, usize)>,
    rate_percent: i32,
    volume_percent: i32,
    ui_language: UiLanguage,
    fetching_voices: bool,
    previewing: bool,
    generating: bool,
    last_generated_audio: Option<PathBuf>,
    asr_input_path: Option<PathBuf>,
    recognition_language: RecognitionLanguage,
    subtitle_export_format: SubtitleExportFormat,
    loading_asr_model: bool,
    transcribing: bool,
    transcription_progress: f32,
    transcription_remaining_seconds: u64,
    transcription_cues: Vec<SubtitleCue>,
    transcription_output_path: Option<PathBuf>,
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
            ui_language: UiLanguage::Chinese,
            fetching_voices,
            previewing: false,
            generating: false,
            last_generated_audio: None,
            asr_input_path: None,
            recognition_language: RecognitionLanguage::MixedChineseEnglish,
            subtitle_export_format: SubtitleExportFormat::Srt,
            loading_asr_model: false,
            transcribing: false,
            transcription_progress: 0.0,
            transcription_remaining_seconds: 0,
            transcription_cues: Vec::new(),
            transcription_output_path: None,
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
                WorkerEvent::PreviewFinished => {
                    self.previewing = false;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Success,
                        "音色试听播放完成。",
                        "Voice preview finished.",
                    ));
                }
                WorkerEvent::PreviewFailed(error) => {
                    self.previewing = false;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Error,
                        format!("试听失败：{error}"),
                        format!("Preview failed: {error}"),
                    ));
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
                    adjusted_count,
                    truncated_count,
                } => {
                    self.generating = false;
                    self.subtitle_progress = None;
                    self.last_generated_audio = Some(output_path.clone());
                    let kind = if truncated_count > 0 {
                        StatusKind::Warning
                    } else {
                        StatusKind::Success
                    };
                    self.status = Some(StatusMessage::new(
                        kind,
                        format!(
                            "字幕音频已保存（{:.1} KB）；自动调速 {adjusted_count} 条，截断 {truncated_count} 条：{}",
                            byte_count as f64 / 1024.0,
                            output_path.display()
                        ),
                        format!(
                            "Saved subtitle audio ({:.1} KB); {adjusted_count} cues auto-fitted and {truncated_count} truncated: {}",
                            byte_count as f64 / 1024.0,
                            output_path.display()
                        ),
                    ));
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
                    self.status = Some(StatusMessage::new(
                        StatusKind::Info,
                        format!(
                            "正在准备本地模型：{}（{:.1}/{:.1} MB）",
                            compact_model_source(&source),
                            downloaded_bytes as f64 / 1_048_576.0,
                            total_bytes as f64 / 1_048_576.0
                        ),
                        format!(
                            "Preparing local model: {} ({:.1}/{:.1} MB)",
                            compact_model_source(&source),
                            downloaded_bytes as f64 / 1_048_576.0,
                            total_bytes as f64 / 1_048_576.0
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
                    output_path,
                    cues,
                    audio_duration_ms,
                } => {
                    self.loading_asr_model = false;
                    self.transcribing = false;
                    self.transcription_progress = 1.0;
                    self.transcription_remaining_seconds = 0;
                    self.transcription_cues = cues;
                    self.transcription_output_path = Some(output_path.clone());
                    self.status = Some(StatusMessage::new(
                        StatusKind::Success,
                        format!(
                            "已在本机生成 {} 条字幕（音频 {}）：{}",
                            self.transcription_cues.len(),
                            format_timestamp(audio_duration_ms),
                            output_path.display()
                        ),
                        format!(
                            "Generated {} subtitle cues locally from {} of audio: {}",
                            self.transcription_cues.len(),
                            format_timestamp(audio_duration_ms),
                            output_path.display()
                        ),
                    ));
                }
                WorkerEvent::TranscriptionFailed(error) => {
                    self.loading_asr_model = false;
                    self.transcribing = false;
                    self.transcription_progress = 0.0;
                    self.transcription_remaining_seconds = 0;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Error,
                        format!("生成字幕失败：{}", localized_transcription_error(&error)),
                        format!("Subtitle generation failed: {error}"),
                    ));
                }
                WorkerEvent::GenerationFailed(error) => {
                    self.generating = false;
                    self.subtitle_progress = None;
                    self.status = Some(StatusMessage::new(
                        StatusKind::Error,
                        format!("生成语音失败：{error}"),
                        error,
                    ));
                }
                WorkerEvent::WorkerFailed(error) => {
                    self.fetching_voices = false;
                    self.previewing = false;
                    self.generating = false;
                    self.loading_asr_model = false;
                    self.transcribing = false;
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
        if self.fetching_voices || self.previewing || self.generating || self.transcribing {
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

    fn start_generation(&mut self) {
        if self.generating || self.previewing || self.fetching_voices || self.transcribing {
            return;
        }

        let Some(voice) = self
            .selected_voice
            .and_then(|index| self.voices.get(index))
            .map(|voice| voice.short_name.clone())
        else {
            self.status = Some(StatusMessage::new(
                StatusKind::Error,
                "请先选择一个音色。",
                "Choose a voice before generating audio.",
            ));
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
                    self.ui_language.text("语音合成.mp3", "edge-tts-output.mp3"),
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
                output_path,
            },
            GenerationContent::Subtitles(cues) => WorkerCommand::GenerateSubtitles {
                cues,
                voice,
                rate_percent: self.rate_percent,
                volume_percent: self.volume_percent,
                output_path,
            },
        };

        match self.command_tx.send(command) {
            Ok(()) => {
                self.generating = true;
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
        if self.fetching_voices || self.previewing || self.generating || self.transcribing {
            return;
        }

        let Some(voice) = self
            .selected_voice
            .and_then(|index| self.voices.get(index))
            .map(|voice| voice.short_name.clone())
        else {
            self.status = Some(StatusMessage::new(
                StatusKind::Error,
                "请先选择一个音色。",
                "Choose a voice before previewing.",
            ));
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

    fn choose_audio_for_transcription(&mut self) {
        if self.transcribing {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .set_title(
                self.ui_language
                    .text("选择需要生成字幕的 MP3", "Choose an MP3 to transcribe"),
            )
            .add_filter(self.ui_language.text("MP3 音频", "MP3 audio"), &["mp3"])
            .pick_file()
        else {
            return;
        };
        self.asr_input_path = Some(path);
        self.transcription_cues.clear();
        self.transcription_output_path = None;
        self.status = Some(StatusMessage::new(
            StatusKind::Info,
            "音频已选择。模型首次使用会下载到本机，之后可以离线识别。",
            "Audio selected. The model is downloaded once, then transcription works offline.",
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
    }

    fn start_transcription(&mut self) {
        if self.transcribing || self.generating || self.previewing {
            return;
        }
        let Some(input_path) = self.asr_input_path.clone() else {
            self.status = Some(StatusMessage::new(
                StatusKind::Error,
                "请先选择需要识别的 MP3 音频。",
                "Choose an MP3 audio file before transcribing.",
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

        let stem = input_path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("transcript");
        let default_name = format!("{stem}.{}", self.subtitle_export_format.extension());
        let Some(output_path) = rfd::FileDialog::new()
            .set_title(
                self.ui_language
                    .text("保存本地识别字幕", "Save local transcription"),
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

        let command = WorkerCommand::TranscribeMp3 {
            input_path,
            output_path,
            language: self.recognition_language,
            format: self.subtitle_export_format,
        };
        match self.command_tx.send(command) {
            Ok(()) => {
                self.transcribing = true;
                self.loading_asr_model = true;
                self.transcription_progress = 0.0;
                self.transcription_remaining_seconds = 0;
                self.transcription_cues.clear();
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

    fn selected_voice_label(&self) -> String {
        self.selected_voice
            .and_then(|index| self.voices.get(index))
            .map(|voice| voice.label(self.ui_language))
            .unwrap_or_else(|| {
                self.ui_language
                    .text("尚未选择音色", "No voice selected")
                    .to_owned()
            })
    }

    fn matching_voice_count(&self) -> usize {
        self.voices
            .iter()
            .filter(|voice| voice.matches(&self.voice_filter))
            .count()
    }
}

impl eframe::App for TtsApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.process_worker_events();

        // Poll only while work is active. This keeps the idle app at near-zero
        // repaint CPU while still noticing worker results promptly.
        if self.fetching_voices || self.previewing || self.generating || self.transcribing {
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
                            WorkspaceMode::TextToSpeech => 98.0,
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
    }
}

impl TtsApp {
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
                            "使用本地 Whisper 模型把 MP3 转换为精准字幕",
                            "Create accurate subtitles with a local Whisper model",
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
        card_frame().show(ui, |ui| {
            ui.set_min_height((card_height - 36.0).max(0.0));
            let busy =
                self.fetching_voices || self.previewing || self.generating || self.transcribing;

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new(language.text("选择音色", "Choose a voice"))
                            .size(17.0)
                            .strong()
                            .color(TEXT_PRIMARY),
                    );
                    ui.label(
                        egui::RichText::new(language.text(
                            "支持按中文名称、语言、地区或代码搜索",
                            "Search by name, language, region, or locale code",
                        ))
                        .size(13.0)
                        .color(TEXT_SECONDARY),
                    );
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let refresh = ui.add_enabled(
                        !busy,
                        egui::Button::new(
                            egui::RichText::new(language.text("刷新音色", "Refresh"))
                                .color(PRIMARY),
                        )
                        .fill(PRIMARY_SOFT)
                        .stroke(egui::Stroke::NONE)
                        .corner_radius(8)
                        .min_size(egui::vec2(88.0, 34.0)),
                    );
                    if refresh.clicked() {
                        self.reload_voices();
                    }
                    if self.fetching_voices {
                        ui.label(
                            egui::RichText::new(language.text("加载中…", "Loading…"))
                                .size(11.0)
                                .color(PRIMARY),
                        );
                    }
                });
            });

            ui.add_space(10.0);
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
                                .hint_text(language.text(
                                    "搜索音色，如：晓晓、英语、en-US",
                                    "Search voices, e.g. Xiaoxiao, English, en-US",
                                ))
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

                ui.add_space(10.0);
                let selected_text = self.selected_voice_label();
                let mut selection = self.selected_voice;
                ui.scope(|ui| {
                    configure_voice_combo_style(ui);
                    egui::ComboBox::from_id_salt("voice-combo")
                        .width(ui.available_width())
                        .height(280.0)
                        .truncate()
                        .selected_text(egui::RichText::new(selected_text).color(TEXT_PRIMARY))
                        .popup_style(voice_popup_style())
                        .show_ui(ui, |ui| {
                            let mut match_count = 0;
                            for (index, voice) in self.voices.iter().enumerate() {
                                if voice.matches(&self.voice_filter) {
                                    match_count += 1;
                                    ui.selectable_value(
                                        &mut selection,
                                        Some(index),
                                        voice.label(language),
                                    );
                                }
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
                self.selected_voice = selection;
            });

            ui.add_space(9.0);
            let matching = self.matching_voice_count();
            let summary = if language == UiLanguage::Chinese {
                format!("共 {} 个音色，当前显示 {} 个", self.voices.len(), matching)
            } else {
                format!(
                    "{} voices available · {} shown",
                    self.voices.len(),
                    matching
                )
            };
            ui.label(
                egui::RichText::new(summary)
                    .size(12.0)
                    .color(TEXT_SECONDARY),
            );

            ui.add_space(8.0);
            egui::Frame::new()
                .fill(EDITOR_BACKGROUND)
                .stroke(egui::Stroke::new(1.0, BORDER))
                .corner_radius(10)
                .inner_margin(egui::Margin::same(10))
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
                                self.rate_percent = 0;
                                self.volume_percent = 0;
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
                });

            ui.add_space(8.0);
            egui::Frame::new()
                .fill(PRIMARY_SOFT)
                .stroke(egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgb(213, 220, 255),
                ))
                .corner_radius(11)
                .inner_margin(egui::Margin::same(8))
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
                    ui.add_space(2.0);

                    let (waveform_rect, _) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), 22.0),
                        egui::Sense::hover(),
                    );
                    paint_preview_waveform(ui, waveform_rect, self.previewing);
                    ui.add_space(2.0);

                    let can_preview = !busy && self.selected_voice.is_some();
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
                    .min_size(egui::vec2(ui.available_width(), 34.0));

                    if ui.add(preview_button).clicked() && can_preview {
                        self.start_preview();
                    }
                });
        });
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
                                    language.text("尚未选择 MP3", "No MP3 selected")
                                }))
                                .size(13.0)
                                .strong()
                                .color(TEXT_PRIMARY),
                            );
                            ui.label(
                                egui::RichText::new(language.text(
                                    "支持应用生成的 MP3 音频",
                                    "Supports MP3 audio generated by the app",
                                ))
                                .size(11.0)
                                .color(TEXT_SECONDARY),
                            );
                        });
                    });
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        let choose = secondary_action_button(
                            language.text("选择 MP3", "Choose MP3"),
                            108.0,
                        );
                        if ui.add_enabled(!self.transcribing, choose).clicked() {
                            self.choose_audio_for_transcription();
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
                    ui.label(
                        egui::RichText::new("Whisper Large-v3 Turbo · Q8")
                            .size(12.0)
                            .strong()
                            .color(TEXT_PRIMARY),
                    );
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
                            "按语音时间戳自动分句，可直接用于剪辑软件",
                            "Timestamped and split into editor-friendly cues",
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
                                        .show_percentage(),
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
                                        "选择 MP3 后开始本地识别",
                                        "Choose an MP3 to start local transcription",
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
                                for (index, cue) in self.transcription_cues.iter().enumerate() {
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
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(&cue.text)
                                                    .size(13.0)
                                                    .color(TEXT_PRIMARY),
                                            )
                                            .wrap(),
                                        );
                                    });
                                    if index + 1 < self.transcription_cues.len() {
                                        ui.separator();
                                    }
                                }
                            });
                    }
                });
        });
    }

    fn show_transcription_area(&mut self, ui: &mut egui::Ui, language: UiLanguage) {
        let can_transcribe = !self.transcribing
            && !self.generating
            && !self.previewing
            && self
                .asr_input_path
                .as_ref()
                .is_some_and(|path| path.is_file());
        ui.horizontal(|ui| {
            let label = if self.transcribing {
                language.text("正在生成字幕…", "Generating subtitles…")
            } else {
                language.text("生成字幕文件", "Generate subtitle file")
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

            if self.transcribing {
                ui.spinner();
            }
            self.show_inline_status(ui, language);
        });
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(language.text(
                "隐私说明：识别时不调用云端 API，音频和字幕均保留在本机。",
                "Privacy: no cloud API is used; audio and subtitles stay on this Mac.",
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
            && !self.fetching_voices
            && self.selected_voice.is_some()
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
                        if language == UiLanguage::Chinese {
                            format!("正在合成第 {current}/{total} 条字幕")
                        } else {
                            format!("Synthesizing subtitle {current}/{total}")
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

        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(language.text(
                "需要联网；文本会发送至 Microsoft Edge 朗读服务进行语音合成。",
                "Internet required. Text is sent to Microsoft Edge Read Aloud for synthesis.",
            ))
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

fn spawn_tts_worker() -> (
    mpsc::UnboundedSender<WorkerCommand>,
    mpsc::UnboundedReceiver<WorkerEvent>,
    Option<String>,
) {
    let (command_tx, mut command_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let thread_event_tx = event_tx.clone();

    let spawn_result = thread::Builder::new()
        .name("edge-tts-tokio".to_owned())
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
                        } => {
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
                        WorkerCommand::Generate {
                            text,
                            voice,
                            rate_percent,
                            volume_percent,
                            output_path,
                        } => {
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
                        WorkerCommand::GenerateSubtitles {
                            cues,
                            voice,
                            rate_percent,
                            volume_percent,
                            output_path,
                        } => {
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
                        WorkerCommand::TranscribeMp3 {
                            input_path,
                            output_path,
                            language,
                            format,
                        } => {
                            transcribe_mp3_locally(
                                &thread_event_tx,
                                &mut whisper_model,
                                input_path,
                                output_path,
                                language,
                                format,
                            )
                            .await;
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

    #[cfg(target_os = "macos")]
    {
        let preview_path = std::env::temp_dir().join(format!(
            "edge-tts-studio-preview-{}.mp3",
            std::process::id()
        ));
        if let Err(error) = tokio::fs::write(&preview_path, &result.audio).await {
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
            let samples = timeline_audio::decode_mp3_mono_preserving_silence(&result.audio)
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
        let _ = result;
        let _ = event_tx.send(WorkerEvent::PreviewFailed(
            "Voice preview playback is currently available on macOS and Windows.".to_owned(),
        ));
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
                adjusted_count: report.adjusted_count,
                truncated_count: report.truncated_count,
            });
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::GenerationFailed(format!(
                "The subtitle audio was generated, but the MP3 file could not be saved: {error}"
            )));
        }
    }
}

async fn transcribe_mp3_locally(
    event_tx: &mpsc::UnboundedSender<WorkerEvent>,
    model_cache: &mut Option<(RecognitionLanguage, Whisper)>,
    input_path: PathBuf,
    output_path: PathBuf,
    language: RecognitionLanguage,
    format: SubtitleExportFormat,
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
                    let scaled = (progress.clamp(0.0, 1.0) * 100.0) as u64;
                    let _ = progress_tx.send(WorkerEvent::AsrModelProgress {
                        source: "Whisper model".to_owned(),
                        downloaded_bytes: scaled,
                        total_bytes: 100,
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
    let report = match asr::transcribe_mp3(model, &input_path, move |progress, remaining| {
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

    let subtitle_text = asr::render_subtitles(&report.cues, format);
    if let Some(parent) = output_path.parent()
        && let Err(error) = tokio::fs::create_dir_all(parent).await
    {
        let _ = event_tx.send(WorkerEvent::TranscriptionFailed(format!(
            "The subtitles were recognized, but the output folder could not be created: {error}"
        )));
        return;
    }
    match tokio::fs::write(&output_path, subtitle_text.as_bytes()).await {
        Ok(()) => {
            let _ = event_tx.send(WorkerEvent::TranscriptionFinished {
                output_path,
                cues: report.cues,
                audio_duration_ms: report.audio_duration_ms,
            });
        }
        Err(error) => {
            let _ = event_tx.send(WorkerEvent::TranscriptionFailed(format!(
                "The subtitles were recognized, but the file could not be saved: {error}"
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

fn localized_transcription_error(error: &str) -> String {
    if error.contains("No speech was recognized") {
        "没有识别到清晰语音。中英文交替的音频请选择“中英混合”，并确认音量足够且人声清楚。"
            .to_owned()
    } else if error.contains("does not contain decodable audio")
        || error.contains("Could not decode the MP3 file")
    {
        format!("无法解码这个 MP3，请确认文件完整且不是受保护音频。详细信息：{error}")
    } else if error.contains("Could not load the local Whisper model") {
        format!("无法加载本地 Whisper 模型，请检查首次下载是否完成。详细信息：{error}")
    } else if error.contains("Could not read the MP3 file") {
        format!("无法读取所选 MP3，请检查文件权限。详细信息：{error}")
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
}
