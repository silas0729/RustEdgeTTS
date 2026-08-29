use std::{
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use directories::ProjectDirs;
use edge_tts_rust::{EdgeTtsClient, SpeakOptions};
use eframe::egui;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

mod system_proxy;

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
    GenerationFailed(String),
    WorkerFailed(String),
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
    rate_percent: i32,
    volume_percent: i32,
    ui_language: UiLanguage,
    fetching_voices: bool,
    previewing: bool,
    generating: bool,
    status: Option<StatusMessage>,
}

impl TtsApp {
    fn new(creation_context: &eframe::CreationContext<'_>) -> Self {
        configure_egui_for_macos(&creation_context.egui_ctx);

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
            voice_filter: String::new(),
            text: DEFAULT_TEXT.to_owned(),
            rate_percent: 0,
            volume_percent: 0,
            ui_language: UiLanguage::Chinese,
            fetching_voices,
            previewing: false,
            generating: false,
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
                WorkerEvent::GenerationFailed(error) => {
                    self.generating = false;
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
        if self.fetching_voices || self.previewing || self.generating {
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
        if self.generating || self.previewing || self.fetching_voices {
            return;
        }

        let text = self.text.trim().to_owned();
        if text.is_empty() {
            self.status = Some(StatusMessage::new(
                StatusKind::Error,
                "请先输入需要转换的文字。",
                "Enter some text before generating audio.",
            ));
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

        // rfd uses macOS's native NSSavePanel. It is intentionally opened on
        // the UI thread; network work and file writing remain on Tokio.
        let Some(output_path) = rfd::FileDialog::new()
            .set_title(
                self.ui_language
                    .text("保存生成的语音", "Save generated speech"),
            )
            .set_file_name(self.ui_language.text("语音合成.mp3", "edge-tts-output.mp3"))
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

        match self.command_tx.send(WorkerCommand::Generate {
            text,
            voice,
            rate_percent: self.rate_percent,
            volume_percent: self.volume_percent,
            output_path,
        }) {
            Ok(()) => {
                self.generating = true;
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

    fn start_preview(&mut self) {
        if self.fetching_voices || self.previewing || self.generating {
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

        let text = preview_text(&self.text, self.ui_language);
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
        if self.fetching_voices || self.previewing || self.generating {
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
                        ui.add_space(18.0);

                        let footer_height = 88.0;
                        let workspace_height = (ui.available_height() - footer_height).max(360.0);
                        let gap = 14.0;
                        let total_width = ui.available_width();
                        let voice_width = (total_width * 0.39).clamp(360.0, 420.0);
                        let text_width = (total_width - voice_width - gap).max(420.0);

                        ui.horizontal_top(|ui| {
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

                        ui.add_space(14.0);
                        self.show_generate_area(ui, language);
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
                    egui::RichText::new(self.ui_language.text(
                        "把文字快速转换为自然流畅的 MP3 语音",
                        "Turn text into natural-sounding MP3 speech",
                    ))
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

    fn show_voice_card(&mut self, ui: &mut egui::Ui, language: UiLanguage, card_height: f32) {
        card_frame().show(ui, |ui| {
            ui.set_min_height((card_height - 36.0).max(0.0));
            let busy = self.fetching_voices || self.previewing || self.generating;

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

            ui.add_space(14.0);
            ui.add_enabled_ui(!busy, |ui| {
                input_frame().show(ui, |ui| {
                    ui.horizontal(|ui| {
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
                                .desired_width(f32::INFINITY)
                                .frame(egui::Frame::NONE)
                                .text_color(TEXT_PRIMARY),
                        );
                        if !self.voice_filter.is_empty()
                            && ui
                                .add(
                                    egui::Button::new(
                                        egui::RichText::new(language.text("清除", "Clear"))
                                            .size(12.0)
                                            .color(TEXT_SECONDARY),
                                    )
                                    .fill(egui::Color32::TRANSPARENT)
                                    .stroke(egui::Stroke::NONE)
                                    .corner_radius(7),
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

            ui.add_space(12.0);
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

            ui.add_space(12.0);
            let preview_panel_height = ui.available_height().max(112.0);
            egui::Frame::new()
                .fill(PRIMARY_SOFT)
                .stroke(egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgb(213, 220, 255),
                ))
                .corner_radius(11)
                .inner_margin(egui::Margin::same(10))
                .show(ui, |ui| {
                    ui.set_min_height((preview_panel_height - 20.0).max(0.0));
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
                    ui.add_space(4.0);

                    let waveform_height = (ui.available_height() - 44.0).clamp(24.0, 40.0);
                    let (waveform_rect, _) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), waveform_height),
                        egui::Sense::hover(),
                    );
                    paint_preview_waveform(ui, waveform_rect, self.previewing);
                    ui.add_space(4.0);

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
                        egui::RichText::new(language.text("输入文本", "Enter text"))
                            .size(17.0)
                            .strong()
                            .color(TEXT_PRIMARY),
                    );
                    ui.label(
                        egui::RichText::new(language.text(
                            "可输入或粘贴中文、英文及混合内容",
                            "Type or paste Chinese, English, or mixed text",
                        ))
                        .size(13.0)
                        .color(TEXT_SECONDARY),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
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

            ui.add_space(12.0);
            let editor_height = (ui.available_height() - 16.0).max(220.0);
            egui::Frame::new()
                .fill(EDITOR_BACKGROUND)
                .stroke(egui::Stroke::new(1.0, BORDER))
                .corner_radius(9)
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    // The editor is inside its own fixed-height ScrollArea. Long
                    // documents scroll here instead of stretching the whole app.
                    egui::ScrollArea::vertical()
                        .id_salt("text-editor-scroll")
                        .max_height(editor_height)
                        .min_scrolled_height(editor_height)
                        .auto_shrink([false, false])
                        .scroll_bar_visibility(
                            egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded,
                        )
                        .show(ui, |ui| {
                            ui.add_enabled(
                                !self.generating,
                                egui::TextEdit::multiline(&mut self.text)
                                    .desired_width(f32::INFINITY)
                                    .desired_rows(10)
                                    .cursor_at_end(false)
                                    .frame(egui::Frame::NONE)
                                    .margin(egui::Margin::symmetric(4, 3))
                                    .text_color(TEXT_PRIMARY)
                                    .hint_text(language.text(
                                        "在这里输入或粘贴需要转换的文字…",
                                        "Type or paste the text to synthesize…",
                                    )),
                            );
                        });
                });
        });
    }

    fn show_generate_area(&mut self, ui: &mut egui::Ui, language: UiLanguage) {
        let can_generate = !self.generating
            && !self.previewing
            && !self.fetching_voices
            && self.selected_voice.is_some()
            && !self.text.trim().is_empty();

        ui.horizontal(|ui| {
            let button = egui::Button::new(
                egui::RichText::new(language.text("生成 MP3", "Generate MP3"))
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
                    egui::RichText::new(
                        language.text("正在合成并保存…", "Synthesizing and saving…"),
                    )
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

    #[cfg(not(target_os = "macos"))]
    {
        let _ = result;
        let _ = event_tx.send(WorkerEvent::PreviewFailed(
            "Voice preview playback is currently available on macOS.".to_owned(),
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

fn configure_egui_for_macos(ctx: &egui::Context) {
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

    // egui's compact default font set does not include CJK glyphs. On macOS,
    // add a system Chinese font as a fallback so pasted Chinese text renders.
    #[cfg(target_os = "macos")]
    {
        const FONT_CANDIDATES: &[&str] = &[
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ];

        if let Some(font_bytes) = FONT_CANDIDATES
            .iter()
            .find_map(|path| std::fs::read(path).ok())
        {
            let mut fonts = egui::FontDefinitions::default();
            fonts.font_data.insert(
                "macos-cjk".to_owned(),
                egui::FontData::from_owned(font_bytes).into(),
            );
            fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .push("macos-cjk".to_owned());
            fonts
                .families
                .entry(egui::FontFamily::Monospace)
                .or_default()
                .push("macos-cjk".to_owned());
            ctx.set_fonts(fonts);
        }
    }
}

fn main() -> eframe::Result {
    // Finder-launched macOS apps do not inherit the shell proxy variables used
    // by reqwest/tokio-tungstenite. Bridge the native system proxy first,
    // before eframe or Tokio has had an opportunity to start any threads.
    system_proxy::configure_from_macos_settings();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 720.0])
            .with_min_inner_size([900.0, 620.0]),
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
