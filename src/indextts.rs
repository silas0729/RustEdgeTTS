use std::{
    collections::VecDeque,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc,
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use directories::ProjectDirs;

pub const OFFICIAL_TAG: &str = "v2.5.0";
const OFFICIAL_COMMIT: &str = "39207d91c30899cad1e7c1b9eb678c241f678e55";
const OFFICIAL_REPOSITORY: &str = "https://github.com/index-tts/index-tts.git";
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
        runtime_python(&self.source_dir).is_file() && model_is_ready(&self.model_dir)
    }

    pub fn ensure_ready(
        &self,
        mut on_progress: impl FnMut(IndexTtsProgress),
    ) -> Result<(), String> {
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
            run_command(&mut clone, |_| {})?;
            std::fs::rename(&staging, &self.source_dir)
                .map_err(|error| format!("无法安装 IndexTTS 官方源码：{error}"))?;
        }
        verify_official_revision(&self.source_dir)?;

        if !runtime_python(&self.source_dir).is_file() {
            require_tool("uv", &["--version"], "uv")?;
            on_progress(IndexTtsProgress::Phase(
                "正在创建 IndexTTS-2.5 官方 Python/PyTorch 运行环境".to_owned(),
            ));
            let mut sync = Command::new("uv");
            sync.args(["sync", "--project"]).arg(&self.source_dir);
            run_command(&mut sync, |_| {})?;
        }

        if !model_is_ready(&self.model_dir) {
            std::fs::create_dir_all(&self.model_dir)
                .map_err(|error| format!("无法创建 IndexTTS 模型目录：{error}"))?;
            let prepare_path = self.source_dir.join(".aura_indextts_prepare.py");
            std::fs::write(&prepare_path, PREPARE_SCRIPT)
                .map_err(|error| format!("无法准备 IndexTTS 下载桥接器：{error}"))?;
            let mut prepare = Command::new("uv");
            prepare
                .args(["run", "--project"])
                .arg(&self.source_dir)
                .arg("python")
                .arg(&prepare_path)
                .arg(&self.model_dir)
                .current_dir(&self.source_dir);
            run_command(&mut prepare, |line| {
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

    pub fn synthesize_batch(
        &self,
        texts: &[String],
        reference_audio: &Path,
        rate_percent: i32,
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
        let duration_factor = (1.0 / (1.0 + rate_percent as f64 / 100.0)).clamp(0.5, 2.0);
        let manifest = serde_json::json!({
            "reference_audio": reference_audio,
            "duration_factor": duration_factor,
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

fn runtime_python(source_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        source_dir.join(".venv/Scripts/python.exe")
    } else {
        source_dir.join(".venv/bin/python")
    }
}

fn model_is_ready(model_dir: &Path) -> bool {
    [
        "config.yaml",
        "codec.pth",
        "gpt.pth",
        "multilingual_zh_ja_yue_char_del.tiktoken",
        "s2mel.pth",
        "wav2vec2bert_stats.pt",
        "feat1.pt",
        "feat2.pt",
        "hf_cache/campplus_cn_common.bin",
        "hf_cache/semantic_codec_model.safetensors",
        "hf_cache/bigvgan/config.json",
        "hf_cache/bigvgan/bigvgan_generator.pt",
        "hf_cache/w2v-bert-2.0/config.json",
        "hf_cache/w2v-bert-2.0/model.safetensors",
        "qwen0.6bemo4-merge/model.safetensors",
    ]
    .iter()
    .all(|relative| model_dir.join(relative).is_file())
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
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let display = format!("{command:?}");
    let mut child = command
        .spawn()
        .map_err(|error| format!("无法启动 {display}：{error}"))?;
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
}
