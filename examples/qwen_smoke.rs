#![allow(dead_code)]

#[path = "../src/qwen_local.rs"]
mod qwen_local;
#[path = "../src/timeline_audio.rs"]
mod timeline_audio;

use qwen_local::{LocalQwenModel, QwenModelKind, QwenModelVersion, QwenVoice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    configure_macos_proxy();

    let version = match std::env::args().nth(1).as_deref() {
        Some("1.7") | Some("1.7B") | Some("1.7b") => QwenModelVersion::Large1_7B,
        Some("0.6") | Some("0.6B") | Some("0.6b") | None => QwenModelVersion::Small0_6B,
        Some(other) => {
            return Err(format!("Unsupported model version '{other}'; use 0.6 or 1.7").into());
        }
    };
    if let Some(source_dir) = std::env::args().nth(2) {
        println!("Importing offline model folder: {source_dir}");
        let installed = qwen_local::import_offline_model(
            version,
            QwenModelKind::CustomVoice,
            std::path::Path::new(&source_dir),
            |_| {},
        )?;
        println!("Offline model installed at {}.", installed.display());
    }

    println!(
        "Preparing {} (the first run downloads {})…",
        version.model_id(QwenModelKind::CustomVoice),
        version.download_size_label()
    );
    let mut previous_file = "";
    let mut previous_percent = u64::MAX;
    let model = LocalQwenModel::load(version, QwenModelKind::CustomVoice, |progress| {
        let percent = progress
            .downloaded_bytes
            .saturating_mul(100)
            .checked_div(progress.total_bytes)
            .unwrap_or(0);
        if progress.file != previous_file || percent >= previous_percent.saturating_add(5) {
            println!(
                "{}: {}% ({:.0}/{:.0} MB)",
                progress.file,
                percent,
                progress.downloaded_bytes as f64 / 1_048_576.0,
                progress.total_bytes as f64 / 1_048_576.0
            );
            previous_file = progress.file;
            previous_percent = percent;
        }
    })?;
    println!("Loading complete on {}.", model.device_label());
    let sample_segments = [
        "大家好，我是一名独立 iOS 开发者。",
        "项目的完整代码已经上传到 GitHub。",
        "这款工具基于 iOS Packet Tunnel。",
        "Thanks for watching and welcome to star the project.",
    ];
    let voice = QwenVoice::Vivian;
    let language = qwen_local::synthesis_language(sample_segments, voice);
    let mut encoder = timeline_audio::TimelineMp3Encoder::new();
    for (index, sample_text) in sample_segments.iter().enumerate() {
        println!("Synthesizing consistency segment {}/4…", index + 1);
        let audio = model.synthesize(sample_text, voice, language)?;
        if audio.samples.is_empty() || audio.sample_rate != 24_000 {
            return Err("Qwen3-TTS returned invalid audio".into());
        }
        encoder.write_clip(&audio.samples)?;
    }
    let mp3 = encoder.finish()?;
    let decoded = timeline_audio::decode_mp3_mono_preserving_silence(&mp3)?;
    if decoded.is_empty() {
        return Err("the generated Qwen3 MP3 could not be decoded".into());
    }
    let output = std::env::temp_dir().join(format!(
        "edge-tts-studio-qwen3-{}-smoke.mp3",
        version.short_label().to_lowercase()
    ));
    std::fs::write(&output, mp3)?;
    println!(
        "Generated and decoded {:.2}s of local Qwen3 MP3 audio: {}",
        decoded.len() as f32 / timeline_audio::TIMELINE_SAMPLE_RATE as f32,
        output.display()
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn configure_macos_proxy() {
    if ["https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]
        .iter()
        .any(|key| std::env::var_os(key).is_some_and(|value| !value.is_empty()))
    {
        return;
    }
    let Ok(Some(configuration)) = proxy_cfg::get_proxy_config() else {
        return;
    };
    let Ok(url) = url::Url::parse("https://huggingface.co/") else {
        return;
    };
    let Some(proxy_url) = configuration.get_proxy_for_url(&url) else {
        return;
    };
    if !proxy_url.is_empty() {
        // SAFETY: this is executed before the model library starts any threads.
        unsafe { std::env::set_var("https_proxy", proxy_url) };
    }
}

#[cfg(not(target_os = "macos"))]
fn configure_macos_proxy() {}
