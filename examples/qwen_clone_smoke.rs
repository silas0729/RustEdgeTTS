#![allow(dead_code)]

#[path = "../src/download_control.rs"]
mod download_control;
#[path = "../src/qwen_local.rs"]
mod qwen_local;
#[path = "../src/timeline_audio.rs"]
mod timeline_audio;

use qwen_local::{LocalQwenModel, QwenModelKind, QwenModelVersion};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    configure_macos_proxy();
    let mut args = std::env::args().skip(1);
    let version = match args.next().as_deref() {
        Some("1.7") | Some("1.7B") | Some("1.7b") => QwenModelVersion::Large1_7B,
        Some("0.6") | Some("0.6B") | Some("0.6b") => QwenModelVersion::Small0_6B,
        _ => {
            return Err(
                "usage: qwen_clone_smoke <0.6|1.7> <reference.wav|mp3> <exact transcript>".into(),
            );
        }
    };
    let reference_path = args.next().ok_or("missing WAV/MP3 reference path")?;
    let reference_text = args.next().ok_or("missing exact reference transcript")?;

    let model = LocalQwenModel::load(version, QwenModelKind::VoiceClone, |progress| {
        let percent = progress
            .downloaded_bytes
            .saturating_mul(100)
            .checked_div(progress.total_bytes)
            .unwrap_or(0);
        println!("{}: {percent}%", progress.file);
    })?;
    let reference_audio = qwen_local::load_reference_audio(std::path::Path::new(&reference_path))?;
    let prompt = model.create_voice_clone_prompt(&reference_audio, Some(&reference_text))?;
    let segments = [
        "这两段内容会复用同一个本地克隆提示。",
        "The English segment keeps the same cloned speaker.",
    ];
    let language = qwen_local::synthesis_language_for_clone(segments);
    let mut encoder = timeline_audio::TimelineMp3Encoder::new();
    for segment in segments {
        let audio = model.synthesize_voice_clone(segment, &prompt, language)?;
        encoder.write_clip(&audio.samples)?;
    }
    let output = std::env::temp_dir().join(format!(
        "edge-tts-studio-qwen3-{}-clone-smoke.mp3",
        version.short_label().to_lowercase()
    ));
    std::fs::write(&output, encoder.finish()?)?;
    println!("Voice clone smoke output: {}", output.display());
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
        // SAFETY: this runs before the model library starts any threads.
        unsafe { std::env::set_var("https_proxy", proxy_url) };
    }
}

#[cfg(not(target_os = "macos"))]
fn configure_macos_proxy() {}
