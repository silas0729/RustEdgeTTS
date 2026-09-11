use std::{
    collections::VecDeque,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc,
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use directories::ProjectDirs;

use crate::download_control::{
    DOWNLOAD_CANCELLED_ERROR, DownloadControl, DownloadState, model_download_control,
};

pub const OFFICIAL_TAG: &str = "v2.5.0";
const OFFICIAL_COMMIT: &str = "39207d91c30899cad1e7c1b9eb678c241f678e55";
const OFFICIAL_REPOSITORY: &str = "https://github.com/index-tts/index-tts.git";
const RUNTIME_READY_MARKER: &str = ".aura_runtime_ready";
const PREPARE_SCRIPT: &str = include_str!("../assets/indextts_prepare.py");
const BRIDGE_SCRIPT: &str = include_str!("../assets/indextts_bridge.py");

#[derive(Debug)]
pub enum IndexTtsProgress {
    Phase(String),
    ModelDownload {
        phase: String,
        downloaded_bytes: u64,
        total_bytes: u64,
    },
    Inference {
        current: usize,
        total: usize,
    },
}

/// Controls accepted by the pinned official IndexTTS-2.5 `infer_v2_5.py`.
/// Keeping these values in Rust makes the desktop UI and the offline bridge
/// share one validated contract.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexTtsSettings {
    pub duration_factor: f64,
    pub text_normalization: bool,
    pub max_text_tokens_per_segment: usize,
    pub interval_silence_ms: usize,
    pub use_random: bool,
    pub emo_alpha: f64,
    pub use_emo_text: bool,
    pub emo_text: String,
    pub do_sample: bool,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f64,
    pub length_penalty: f64,
    pub num_beams: usize,
    pub max_mel_tokens: usize,
}

impl Default for IndexTtsSettings {
    fn default() -> Self {
        Self {
            duration_factor: 1.0,
            text_normalization: true,
            max_text_tokens_per_segment: 120,
            interval_silence_ms: 200,
            use_random: false,
            emo_alpha: 1.0,
            use_emo_text: false,
            emo_text: String::new(),
            do_sample: true,
            temperature: 0.8,
            top_k: 30,
            top_p: 0.8,
            repetition_penalty: 10.0,
            length_penalty: 0.0,
            num_beams: 3,
            max_mel_tokens: 1_500,
        }
    }
}

impl IndexTtsSettings {
    pub fn validated(&self) -> Self {
        Self {
            duration_factor: finite_clamp(self.duration_factor, 0.5, 2.0, 1.0),
            text_normalization: self.text_normalization,
            max_text_tokens_per_segment: self.max_text_tokens_per_segment.clamp(16, 512),
            interval_silence_ms: self.interval_silence_ms.min(5_000),
            use_random: self.use_random,
            emo_alpha: finite_clamp(self.emo_alpha, 0.0, 1.0, 1.0),
            use_emo_text: self.use_emo_text,
            emo_text: self.emo_text.chars().take(500).collect(),
            do_sample: self.do_sample,
            temperature: finite_clamp(self.temperature, 0.0, 2.0, 0.8),
            top_k: self.top_k.clamp(1, 200),
            top_p: finite_clamp(self.top_p, 0.01, 1.0, 0.8),
            repetition_penalty: finite_clamp(self.repetition_penalty, 0.1, 20.0, 10.0),
            length_penalty: finite_clamp(self.length_penalty, -2.0, 2.0, 0.0),
            num_beams: self.num_beams.clamp(1, 8),
            max_mel_tokens: self.max_mel_tokens.clamp(128, 4_000),
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

pub struct IndexTtsRuntime {
    source_dir: PathBuf,
    model_dir: PathBuf,
}

impl IndexTtsRuntime {
    pub fn new() -> Result<Self, String> {
        let project_dirs = ProjectDirs::from("com", "Aura Labs", "Edge TTS Studio")
            .ok_or_else(|| "无法确定 IndexTTS-2.5 本地缓存目录。".to_owned())?;
        let root = project_dirs.cache_dir().join("indextts-2.5");
        Ok(Self {
            source_dir: root.join("official-v2.5.0"),
            model_dir: root.join("models"),
        })
    }

    pub fn is_ready(&self) -> bool {
        runtime_python(&self.source_dir).is_file()
            && self.source_dir.join(RUNTIME_READY_MARKER).is_file()
            && model_is_ready(&self.model_dir)
    }

    /// Import a complete IndexTTS-2.5 model directory without contacting
    /// Hugging Face. This is useful when the machine running the app has no
    /// access to the model host but another machine has already downloaded the
    /// official checkpoint tree.
    pub fn import_offline_model(
        &self,
        selected_dir: &Path,
        mut on_progress: impl FnMut(IndexTtsProgress),
    ) -> Result<PathBuf, String> {
        let control = model_download_control();
        let _download_session = control.begin();
        control.checkpoint()?;
        let source_dir = locate_model_source(selected_dir)?;
        validate_model_dir(&source_dir)?;
        std::fs::create_dir_all(&self.model_dir)
            .map_err(|error| format!("无法创建 IndexTTS 模型目录：{error}"))?;

        let files = required_model_files();
        let total_bytes = files
            .iter()
            .map(|(relative, _)| file_size(&source_dir.join(relative)))
            .sum::<u64>();
        let mut copied_bytes = 0_u64;
        on_progress(IndexTtsProgress::ModelDownload {
            phase: "正在导入 IndexTTS-2.5 离线模型".to_owned(),
            downloaded_bytes: 0,
            total_bytes,
        });
        for (relative, minimum_size) in files {
            control.checkpoint()?;
            let source = source_dir.join(relative);
            let destination = self.model_dir.join(relative);
            let size = file_size(&source);
            import_file(
                &source,
                &destination,
                *minimum_size,
                control,
                &mut |file_bytes| {
                    let progress = copied_bytes.saturating_add(file_bytes).min(total_bytes);
                    on_progress(IndexTtsProgress::ModelDownload {
                        phase: format!("正在导入 {relative}"),
                        downloaded_bytes: progress,
                        total_bytes,
                    });
                },
            )?;
            copied_bytes = copied_bytes.saturating_add(size);
            on_progress(IndexTtsProgress::ModelDownload {
                phase: format!("已导入 {relative}"),
                downloaded_bytes: copied_bytes.min(total_bytes),
                total_bytes,
            });
        }
        validate_model_dir(&self.model_dir)?;
        on_progress(IndexTtsProgress::Phase(
            "IndexTTS-2.5 离线模型导入完成".to_owned(),
        ));
        Ok(self.model_dir.clone())
    }

    pub fn ensure_ready(
        &self,
        mut on_progress: impl FnMut(IndexTtsProgress),
    ) -> Result<(), String> {
        let control = model_download_control();
        let _download_session = control.begin();
        control.checkpoint()?;
        std::fs::create_dir_all(
            self.source_dir
                .parent()
                .ok_or_else(|| "IndexTTS 缓存路径无效。".to_owned())?,
        )
        .map_err(|error| format!("无法创建 IndexTTS 缓存目录：{error}"))?;

        if !self.source_dir.join(".git").is_dir() {
            require_tool("git", &["--version"], "Git")?;
            on_progress(IndexTtsProgress::Phase(
                "正在获取官方 IndexTTS-2.5 v2.5.0 源码".to_owned(),
            ));
            let staging = self.source_dir.with_extension("downloading");
            if staging.exists() {
                std::fs::remove_dir_all(&staging)
                    .map_err(|error| format!("无法清理未完成的 IndexTTS 下载：{error}"))?;
            }
            let mut clone = Command::new("git");
            clone.args([
                "clone",
                "--depth",
                "1",
                "--branch",
                OFFICIAL_TAG,
                OFFICIAL_REPOSITORY,
            ]);
            clone.arg(&staging);
            run_download_command(&mut clone, control, |_| {})?;
            std::fs::rename(&staging, &self.source_dir)
                .map_err(|error| format!("无法安装 IndexTTS 官方源码：{error}"))?;
        }
        verify_official_revision(&self.source_dir)?;

        if !runtime_python(&self.source_dir).is_file()
            || !self.source_dir.join(RUNTIME_READY_MARKER).is_file()
        {
            require_tool("uv", &["--version"], "uv")?;
            on_progress(IndexTtsProgress::Phase(
                "正在创建 IndexTTS-2.5 官方 Python/PyTorch 运行环境".to_owned(),
            ));
            let mut sync = Command::new("uv");
            sync.args(["sync", "--project"]).arg(&self.source_dir);
            run_download_command(&mut sync, control, |_| {})?;
            std::fs::write(
                self.source_dir.join(RUNTIME_READY_MARKER),
                format!("{OFFICIAL_COMMIT}\n"),
            )
            .map_err(|error| format!("无法记录 IndexTTS 运行环境状态：{error}"))?;
        }

        if !model_is_ready(&self.model_dir) {
            std::fs::create_dir_all(&self.model_dir)
                .map_err(|error| format!("无法创建 IndexTTS 模型目录：{error}"))?;
            let prepare_path = self.source_dir.join(".aura_indextts_prepare.py");
            std::fs::write(&prepare_path, PREPARE_SCRIPT)
                .map_err(|error| format!("无法准备 IndexTTS 下载桥接器：{error}"))?;
            // Invoke the venv interpreter directly. Besides avoiding another
            // launcher process, this gives the download controller one stable
            // process to suspend and resume on every desktop platform.
            let mut prepare = Command::new(runtime_python(&self.source_dir));
            prepare
                .arg(&prepare_path)
                .arg(&self.model_dir)
                .current_dir(&self.source_dir);
            run_download_command(&mut prepare, control, |line| {
                if let Some(progress) = parse_model_progress(line) {
                    on_progress(progress);
                }
            })?;
            if !model_is_ready(&self.model_dir) {
                return Err(
                    "IndexTTS-2.5 下载命令已结束，但模型文件仍不完整；可重试以断点续传。"
                        .to_owned(),
                );
            }
        }
        on_progress(IndexTtsProgress::Phase(
            "IndexTTS-2.5 官方模型已就绪".to_owned(),
        ));
        Ok(())
    }

    #[allow(dead_code)]
    pub fn synthesize_batch(
        &self,
        texts: &[String],
        reference_audio: &Path,
        rate_percent: i32,
        on_progress: impl FnMut(IndexTtsProgress),
    ) -> Result<Vec<Vec<f32>>, String> {
        let settings = IndexTtsSettings {
            duration_factor: (1.0 / (1.0 + rate_percent as f64 / 100.0)).clamp(0.5, 2.0),
            ..IndexTtsSettings::default()
        };
        self.synthesize_batch_with_settings(texts, reference_audio, &settings, on_progress)
    }

    pub fn synthesize_batch_with_settings(
        &self,
        texts: &[String],
        reference_audio: &Path,
        settings: &IndexTtsSettings,
        mut on_progress: impl FnMut(IndexTtsProgress),
    ) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Err("IndexTTS-2.5 没有收到可合成的文字。".to_owned());
        }
        if !self.is_ready() {
            return Err("IndexTTS-2.5 模型尚未准备完成。".to_owned());
        }
        if !reference_audio.is_file() {
            return Err("IndexTTS-2.5 的参考音频已不存在。".to_owned());
        }

        let task_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let task_dir = self
            .source_dir
            .parent()
            .expect("validated IndexTTS cache parent")
            .join("tasks")
            .join(format!("{}-{task_id}", std::process::id()));
        std::fs::create_dir_all(&task_dir)
            .map_err(|error| format!("无法创建 IndexTTS 合成任务目录：{error}"))?;
        let bridge_path = self.source_dir.join(".aura_indextts_bridge.py");
        std::fs::write(&bridge_path, BRIDGE_SCRIPT)
            .map_err(|error| format!("无法准备 IndexTTS 推理桥接器：{error}"))?;

        let items: Vec<_> = texts
            .iter()
            .enumerate()
            .map(|(index, text)| {
                serde_json::json!({
                    "text": text,
                    "language": language_for_text(text),
                    "output": task_dir.join(format!("{index:05}.wav")),
                })
            })
            .collect();
        let settings = settings.validated();
        let manifest = serde_json::json!({
            "reference_audio": reference_audio,
            "duration_factor": settings.duration_factor,
            "text_normalization": settings.text_normalization,
            "max_text_tokens_per_segment": settings.max_text_tokens_per_segment,
            "interval_silence_ms": settings.interval_silence_ms,
            "use_random": settings.use_random,
            "emo_alpha": settings.emo_alpha,
            "use_emo_text": settings.use_emo_text,
            "emo_text": settings.emo_text,
            "generation": {
                "do_sample": settings.do_sample,
                "temperature": settings.temperature,
                "top_k": settings.top_k,
                "top_p": settings.top_p,
                "repetition_penalty": settings.repetition_penalty,
                "length_penalty": settings.length_penalty,
                "num_beams": settings.num_beams,
                "max_mel_tokens": settings.max_mel_tokens,
            },
            "items": items,
        });
        let manifest_path = task_dir.join("manifest.json");
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest)
                .map_err(|error| format!("无法创建 IndexTTS 任务清单：{error}"))?,
        )
        .map_err(|error| format!("无法写入 IndexTTS 任务清单：{error}"))?;

        let mut bridge = Command::new("uv");
        bridge
            .args(["run", "--project"])
            .arg(&self.source_dir)
            .arg("python")
            .arg(&bridge_path)
            .arg(&self.model_dir)
            .arg(&manifest_path)
            .current_dir(&self.source_dir);
        run_command(&mut bridge, |line| {
            if let Some(rest) = line.strip_prefix("AURA_INFERENCE\t") {
                on_progress(IndexTtsProgress::Phase(rest.to_owned()));
            } else if let Some(rest) = line.strip_prefix("AURA_INFERENCE_PROGRESS\t") {
                let mut fields = rest.split('\t');
                if let (Some(current), Some(total)) = (fields.next(), fields.next())
                    && let (Ok(current), Ok(total)) = (current.parse(), total.parse())
                {
                    on_progress(IndexTtsProgress::Inference { current, total });
                }
            }
        })?;

        let mut audio = Vec::with_capacity(texts.len());
        for index in 0..texts.len() {
            let path = task_dir.join(format!("{index:05}.wav"));
            let samples = crate::asr::decode_media_mono(&path)
                .map_err(|error| format!("无法读取 IndexTTS 第 {} 段输出：{error}", index + 1))?;
            audio.push(samples);
        }
        if let Err(error) = std::fs::remove_dir_all(&task_dir) {
            eprintln!(
                "Could not clean the IndexTTS task cache {}: {error}",
                task_dir.display()
            );
        }
        Ok(audio)
    }
}

pub fn models_directory() -> Result<PathBuf, String> {
    ProjectDirs::from("com", "Aura Labs", "Edge TTS Studio")
        .map(|project_dirs| project_dirs.cache_dir().join("indextts-2.5/models"))
        .ok_or_else(|| "无法确定 IndexTTS-2.5 本地缓存目录。".to_owned())
}

fn runtime_python(source_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        source_dir.join(".venv/Scripts/python.exe")
    } else {
        source_dir.join(".venv/bin/python")
    }
}

fn model_is_ready(model_dir: &Path) -> bool {
    validate_model_dir(model_dir).is_ok()
}

fn required_model_files() -> &'static [(&'static str, u64)] {
    &[
        ("config.yaml", 100),
        ("codec.pth", 100_000_000),
        ("gpt.pth", 100_000_000),
        ("multilingual_zh_ja_yue_char_del.tiktoken", 100_000),
        ("s2mel.pth", 100_000_000),
        ("wav2vec2bert_stats.pt", 100),
        ("feat1.pt", 100),
        ("feat2.pt", 100),
        ("hf_cache/campplus_cn_common.bin", 1_000_000),
        ("hf_cache/semantic_codec_model.safetensors", 1_000_000),
        ("hf_cache/bigvgan/config.json", 100),
        ("hf_cache/bigvgan/bigvgan_generator.pt", 100_000_000),
        ("hf_cache/w2v-bert-2.0/config.json", 100),
        ("hf_cache/w2v-bert-2.0/model.safetensors", 100_000_000),
        ("qwen0.6bemo4-merge/model.safetensors", 100_000_000),
    ]
}

fn locate_model_source(selected_dir: &Path) -> Result<PathBuf, String> {
    let candidates = [
        selected_dir.to_path_buf(),
        selected_dir.join("IndexTTS-2.5"),
        selected_dir.join("models"),
        selected_dir.join("IndexTeam/IndexTTS-2.5"),
    ];
    candidates
        .into_iter()
        .find(|candidate| candidate.join("config.yaml").is_file())
        .ok_or_else(|| {
            "所选文件夹中没有找到 IndexTTS-2.5 模型。请选择包含 config.yaml 的完整模型目录。"
                .to_owned()
        })
}

fn file_size(path: &Path) -> u64 {
    std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0)
}

fn validate_model_dir(model_dir: &Path) -> Result<(), String> {
    let missing: Vec<_> = required_model_files()
        .iter()
        .filter(|(relative, minimum_size)| {
            let path = model_dir.join(relative);
            !path.is_file() || file_size(&path) < *minimum_size
        })
        .map(|(relative, _)| *relative)
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "IndexTTS-2.5 模型不完整，缺少或尺寸异常：{}",
            missing.join("、")
        ))
    }
}

fn import_file(
    source: &Path,
    destination: &Path,
    minimum_size: u64,
    control: &DownloadControl,
    on_progress: &mut impl FnMut(u64),
) -> Result<(), String> {
    let source_size = file_size(source);
    if !source.is_file() || source_size < minimum_size {
        return Err(format!("离线模型文件缺失或不完整：{}", source.display()));
    }
    if source == destination {
        on_progress(source_size);
        return Ok(());
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("无法创建 IndexTTS 模型目录：{error}"))?;
    }
    let part_path = destination.with_extension("import");
    let _ = std::fs::remove_file(&part_path);
    let resolved_source = std::fs::canonicalize(source).unwrap_or_else(|_| source.to_path_buf());
    if std::fs::hard_link(&resolved_source, &part_path).is_ok() {
        on_progress(source_size);
    } else {
        let mut input =
            std::fs::File::open(source).map_err(|error| format!("无法读取离线模型：{error}"))?;
        let mut output = std::fs::File::create(&part_path)
            .map_err(|error| format!("无法写入离线模型：{error}"))?;
        let mut copied = 0_u64;
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            control.checkpoint()?;
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
            on_progress(copied);
        }
        output
            .flush()
            .map_err(|error| format!("写入离线模型失败：{error}"))?;
    }
    if file_size(&part_path) < minimum_size {
        let _ = std::fs::remove_file(&part_path);
        return Err("导入后的 IndexTTS 模型文件不完整。".to_owned());
    }
    let _ = std::fs::remove_file(destination);
    std::fs::rename(&part_path, destination)
        .map_err(|error| format!("无法安装 IndexTTS 模型文件：{error}"))?;
    Ok(())
}

fn verify_official_revision(source_dir: &Path) -> Result<(), String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(source_dir)
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|error| format!("无法验证 IndexTTS 官方版本：{error}"))?;
    let revision = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if output.status.success() && revision == OFFICIAL_COMMIT {
        Ok(())
    } else {
        Err(format!(
            "IndexTTS 源码版本校验失败：需要 {OFFICIAL_TAG} ({OFFICIAL_COMMIT})，当前为 {revision}。"
        ))
    }
}

fn require_tool(program: &str, arguments: &[&str], display_name: &str) -> Result<(), String> {
    match Command::new(program).args(arguments).output() {
        Ok(output) if output.status.success() => Ok(()),
        _ => Err(format!(
            "IndexTTS-2.5 官方运行环境需要 {display_name}。请先安装 {display_name}，然后重试；已下载的模型进度会保留。"
        )),
    }
}

fn parse_model_progress(line: &str) -> Option<IndexTtsProgress> {
    let rest = line.strip_prefix("AURA_MODEL_PROGRESS\t")?;
    let mut fields = rest.split('\t');
    let phase = fields.next()?.to_owned();
    let downloaded_bytes = fields
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let total_bytes = fields
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    if total_bytes > 0 {
        Some(IndexTtsProgress::ModelDownload {
            phase,
            downloaded_bytes,
            total_bytes,
        })
    } else {
        Some(IndexTtsProgress::Phase(phase))
    }
}

fn run_command(command: &mut Command, mut on_line: impl FnMut(&str)) -> Result<(), String> {
    run_command_impl(command, None, &mut on_line)
}

fn run_download_command(
    command: &mut Command,
    control: &DownloadControl,
    mut on_line: impl FnMut(&str),
) -> Result<(), String> {
    control.checkpoint()?;
    run_command_impl(command, Some(control), &mut on_line)
}

fn run_command_impl(
    command: &mut Command,
    control: Option<&DownloadControl>,
    on_line: &mut impl FnMut(&str),
) -> Result<(), String> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(unix)]
    if control.is_some() {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let display = format!("{command:?}");
    let mut child = command
        .spawn()
        .map_err(|error| format!("无法启动 {display}：{error}"))?;
    let process_id = child.id();
    if let Some(control) = control {
        control.register_process(process_id);
    }
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let (line_tx, line_rx) = mpsc::channel();
    let stdout_tx = line_tx.clone();
    let stdout_thread = thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = stdout_tx.send(line);
        }
    });
    let stderr_thread = thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            let _ = line_tx.send(line);
        }
    });
    let mut tail = VecDeque::with_capacity(12);
    for line in line_rx {
        on_line(&line);
        if tail.len() == 12 {
            tail.pop_front();
        }
        tail.push_back(line);
    }
    let _ = stdout_thread.join();
    let _ = stderr_thread.join();
    let status = child
        .wait()
        .map_err(|error| format!("等待 {display} 完成时失败：{error}"))?;
    if let Some(control) = control {
        control.clear_process(process_id);
        if control.state() == DownloadState::Cancelled {
            return Err(DOWNLOAD_CANCELLED_ERROR.to_owned());
        }
    }
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "IndexTTS 命令执行失败（{status}）：{}",
            tail.into_iter().collect::<Vec<_>>().join("\n")
        ))
    }
}

fn language_for_text(text: &str) -> &'static str {
    if text
        .chars()
        .any(|character| matches!(character as u32, 0x3040..=0x30ff | 0x31f0..=0x31ff))
    {
        "JA"
    } else if text
        .chars()
        .any(|character| matches!(character as u32, 0x0600..=0x06ff | 0x0750..=0x077f))
    {
        "AR"
    } else if text
        .chars()
        .any(|character| matches!(character as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff))
    {
        "ZH"
    } else if text
        .chars()
        .any(|character| matches!(character, 'ñ' | 'Ñ' | '¿' | '¡'))
    {
        "ES"
    } else {
        "EN"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_languages_supported_by_index_tts_2_5() {
        assert_eq!(language_for_text("你好 IndexTTS"), "ZH");
        assert_eq!(language_for_text("Hello IndexTTS"), "EN");
        assert_eq!(language_for_text("こんにちは"), "JA");
        assert_eq!(language_for_text("مرحبا"), "AR");
        assert_eq!(language_for_text("¿Qué tal?"), "ES");
    }

    #[test]
    fn parses_download_progress_protocol() {
        let progress = parse_model_progress("AURA_MODEL_PROGRESS\tdownloading\t25\t100");
        assert!(matches!(
            progress,
            Some(IndexTtsProgress::ModelDownload {
                downloaded_bytes: 25,
                total_bytes: 100,
                ..
            })
        ));
    }

    #[test]
    fn generation_settings_are_finite_and_bounded() {
        let settings = IndexTtsSettings {
            duration_factor: f64::NAN,
            max_text_tokens_per_segment: 0,
            interval_silence_ms: usize::MAX,
            emo_alpha: f64::INFINITY,
            temperature: -1.0,
            top_k: usize::MAX,
            top_p: f64::NAN,
            repetition_penalty: f64::INFINITY,
            length_penalty: -10.0,
            num_beams: 0,
            max_mel_tokens: usize::MAX,
            ..IndexTtsSettings::default()
        }
        .validated();
        assert_eq!(settings.duration_factor, 1.0);
        assert_eq!(settings.max_text_tokens_per_segment, 16);
        assert_eq!(settings.interval_silence_ms, 5_000);
        assert_eq!(settings.emo_alpha, 1.0);
        assert_eq!(settings.temperature, 0.0);
        assert_eq!(settings.top_k, 200);
        assert_eq!(settings.top_p, 0.8);
        assert_eq!(settings.repetition_penalty, 10.0);
        assert_eq!(settings.length_penalty, -2.0);
        assert_eq!(settings.num_beams, 1);
        assert_eq!(settings.max_mel_tokens, 4_000);
    }
}
