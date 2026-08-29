use edge_tts_rust::{EdgeTtsClient, SpeakOptions};

#[path = "../src/system_proxy.rs"]
mod system_proxy;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    system_proxy::configure_from_macos_settings();

    let client = EdgeTtsClient::new()?;
    let voices = client.list_voices().await?;
    let voice = voices
        .iter()
        .find(|voice| voice.short_name == "zh-CN-XiaoxiaoNeural")
        .or_else(|| voices.first())
        .ok_or("Edge returned an empty voice list")?;

    println!(
        "Fetched {} voices; testing {}",
        voices.len(),
        voice.short_name
    );

    let result = client
        .synthesize(
            "你好，这是 Edge TTS Studio 的连通性测试。",
            SpeakOptions {
                voice: voice.short_name.clone(),
                ..SpeakOptions::default()
            },
        )
        .await?;

    if result.audio.len() < 1_024 {
        return Err(format!(
            "Audio response was unexpectedly small: {} bytes",
            result.audio.len()
        )
        .into());
    }

    let output_path = std::env::temp_dir().join("edge-tts-studio-smoke-test.mp3");
    tokio::fs::write(&output_path, &result.audio).await?;
    println!(
        "Live synthesis passed: {} bytes written to {}",
        result.audio.len(),
        output_path.display()
    );
    Ok(())
}
