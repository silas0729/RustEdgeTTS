/// Makes the current macOS HTTPS proxy visible to Rust HTTP/WebSocket clients.
/// Existing user-provided proxy environment variables always win.
#[cfg(target_os = "macos")]
pub fn configure_from_macos_settings() {
    const PROXY_ENVIRONMENT_KEYS: &[&str] =
        &["https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"];

    if PROXY_ENVIRONMENT_KEYS
        .iter()
        .any(|key| std::env::var_os(key).is_some_and(|value| !value.is_empty()))
    {
        return;
    }

    let Ok(Some(configuration)) = proxy_cfg::get_proxy_config() else {
        return;
    };
    let Ok(edge_url) = url::Url::parse("https://speech.platform.bing.com/") else {
        return;
    };
    let Some(proxy_url) = configuration.get_proxy_for_url(&edge_url) else {
        return;
    };
    if proxy_url.is_empty() {
        return;
    }

    // SAFETY: this function is the first call made by main, before eframe,
    // Tokio, or application worker threads exist. No concurrent environment
    // read/write can therefore occur inside this process.
    unsafe {
        std::env::set_var("https_proxy", proxy_url);
    }
}

#[cfg(not(target_os = "macos"))]
pub fn configure_from_macos_settings() {}
