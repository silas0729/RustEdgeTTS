use std::path::Path;

use encoding_rs::GBK;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubtitleFormat {
    Srt,
    WebVtt,
    Ass,
    Lrc,
}

impl SubtitleFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::Srt => "SRT",
            Self::WebVtt => "WebVTT",
            Self::Ass => "ASS / SSA",
            Self::Lrc => "LRC",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubtitleCue {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct SubtitleTrack {
    pub format: SubtitleFormat,
    pub cues: Vec<SubtitleCue>,
}

impl SubtitleTrack {
    pub fn duration_ms(&self) -> u64 {
        self.cues.iter().map(|cue| cue.end_ms).max().unwrap_or(0)
    }
}

/// Decode the encodings most often encountered in Chinese subtitle files.
/// UTF-8 and UTF-16 are detected first; legacy files fall back to GBK/GB18030.
pub fn decode_subtitle_bytes(bytes: &[u8]) -> Result<String, String> {
    if let Some(body) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8(body.to_vec())
            .map_err(|error| format!("Invalid UTF-8 subtitle: {error}"));
    }
    if let Some(body) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return decode_utf16(body, true);
    }
    if let Some(body) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return decode_utf16(body, false);
    }
    if let Ok(text) = std::str::from_utf8(bytes) {
        return Ok(text.to_owned());
    }

    let (text, _, had_errors) = GBK.decode(bytes);
    if had_errors {
        Err("The subtitle encoding is not recognized. Save it as UTF-8, UTF-16, or GBK and try again.".to_owned())
    } else {
        Ok(text.into_owned())
    }
}

fn decode_utf16(bytes: &[u8], little_endian: bool) -> Result<String, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err("The UTF-16 subtitle has an incomplete final byte.".to_owned());
    }
    let (pairs, remainder) = bytes.as_chunks::<2>();
    debug_assert!(remainder.is_empty());
    let units = pairs.iter().map(|pair| {
        if little_endian {
            u16::from_le_bytes([pair[0], pair[1]])
        } else {
            u16::from_be_bytes([pair[0], pair[1]])
        }
    });
    String::from_utf16(&units.collect::<Vec<_>>())
        .map_err(|error| format!("Invalid UTF-16 subtitle: {error}"))
}

pub fn parse_subtitle(path: &Path, bytes: &[u8]) -> Result<SubtitleTrack, String> {
    let text = decode_subtitle_bytes(bytes)?;
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    let (format, cues) = match extension.as_str() {
        "srt" | "str" => (SubtitleFormat::Srt, parse_srt_or_vtt(&text)),
        "vtt" => (SubtitleFormat::WebVtt, parse_srt_or_vtt(&text)),
        "ass" | "ssa" => (SubtitleFormat::Ass, parse_ass(&text)),
        "lrc" => (SubtitleFormat::Lrc, parse_lrc(&text)),
        _ if text.trim_start().starts_with("WEBVTT") => {
            (SubtitleFormat::WebVtt, parse_srt_or_vtt(&text))
        }
        _ if text.contains("[Events]") && text.contains("Dialogue:") => {
            (SubtitleFormat::Ass, parse_ass(&text))
        }
        _ if text
            .lines()
            .any(|line| parse_lrc_timestamps(line).is_some()) =>
        {
            (SubtitleFormat::Lrc, parse_lrc(&text))
        }
        _ if text.contains("-->") => (SubtitleFormat::Srt, parse_srt_or_vtt(&text)),
        _ => {
            return Err(
                "Unsupported subtitle format. Choose an SRT/STR, WebVTT, ASS/SSA, or LRC file."
                    .to_owned(),
            );
        }
    };

    let cues = normalize_cues(cues);
    if cues.is_empty() {
        return Err("No timed subtitle lines were found in this file.".to_owned());
    }
    Ok(SubtitleTrack { format, cues })
}

fn parse_srt_or_vtt(text: &str) -> Vec<SubtitleCue> {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut cues = Vec::new();
    let lines: Vec<_> = normalized.lines().collect();
    let mut index = 0;

    while index < lines.len() {
        let line = lines[index].trim();
        if line.starts_with("NOTE") || line == "STYLE" || line == "REGION" {
            index += 1;
            while index < lines.len() && !lines[index].trim().is_empty() {
                index += 1;
            }
            continue;
        }

        let Some((left, right)) = line.split_once("-->") else {
            index += 1;
            continue;
        };
        let Some(start_ms) = parse_clock(left.trim()) else {
            index += 1;
            continue;
        };
        let Some(end_token) = right.split_whitespace().next() else {
            index += 1;
            continue;
        };
        let Some(end_ms) = parse_clock(end_token) else {
            index += 1;
            continue;
        };

        index += 1;
        let mut text_lines = Vec::new();
        while index < lines.len() && !lines[index].trim().is_empty() {
            text_lines.push(lines[index].trim());
            index += 1;
        }
        cues.push(SubtitleCue {
            start_ms,
            end_ms,
            text: clean_caption_text(&text_lines.join(" ")),
        });
    }
    cues
}

fn parse_ass(text: &str) -> Vec<SubtitleCue> {
    let mut in_events = false;
    let mut fields: Vec<String> = vec![
        "layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let mut cues = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_events = line.eq_ignore_ascii_case("[Events]");
            continue;
        }
        if !in_events {
            continue;
        }
        if let Some(format) = line.strip_prefix("Format:") {
            fields = format
                .split(',')
                .map(|field| field.trim().to_ascii_lowercase())
                .collect();
            continue;
        }
        let Some(dialogue) = line.strip_prefix("Dialogue:") else {
            continue;
        };
        let values: Vec<_> = dialogue.trim().splitn(fields.len(), ',').collect();
        if values.len() != fields.len() {
            continue;
        }
        let Some(start_index) = fields.iter().position(|field| field == "start") else {
            continue;
        };
        let Some(end_index) = fields.iter().position(|field| field == "end") else {
            continue;
        };
        let Some(text_index) = fields.iter().position(|field| field == "text") else {
            continue;
        };
        let (Some(start_ms), Some(end_ms)) = (
            parse_clock(values[start_index].trim()),
            parse_clock(values[end_index].trim()),
        ) else {
            continue;
        };
        cues.push(SubtitleCue {
            start_ms,
            end_ms,
            text: clean_caption_text(values[text_index]),
        });
    }
    cues
}

fn parse_lrc(text: &str) -> Vec<SubtitleCue> {
    let mut entries = Vec::new();
    for line in text.lines() {
        let Some((timestamps, caption)) = parse_lrc_timestamps(line) else {
            continue;
        };
        let caption = clean_caption_text(caption);
        if caption.is_empty() {
            continue;
        }
        for start_ms in timestamps {
            entries.push((start_ms, caption.clone()));
        }
    }
    entries.sort_by_key(|entry| entry.0);

    // Multiple LRC tags may point to the same lyric. Merge them to avoid two
    // voices being mixed at the same timestamp.
    let mut merged: Vec<(u64, String)> = Vec::new();
    for (start_ms, caption) in entries {
        if let Some((last_start, last_caption)) = merged.last_mut()
            && *last_start == start_ms
        {
            if !last_caption.contains(&caption) {
                last_caption.push(' ');
                last_caption.push_str(&caption);
            }
            continue;
        }
        merged.push((start_ms, caption));
    }

    merged
        .iter()
        .enumerate()
        .map(|(index, (start_ms, caption))| {
            let end_ms = merged
                .get(index + 1)
                .map(|next| next.0)
                .filter(|next| *next > *start_ms)
                .unwrap_or_else(|| start_ms + estimate_lrc_duration_ms(caption));
            SubtitleCue {
                start_ms: *start_ms,
                end_ms,
                text: caption.clone(),
            }
        })
        .collect()
}

fn parse_lrc_timestamps(line: &str) -> Option<(Vec<u64>, &str)> {
    let mut rest = line.trim();
    let mut timestamps = Vec::new();
    while let Some(after_open) = rest.strip_prefix('[') {
        let close = after_open.find(']')?;
        let tag = &after_open[..close];
        if let Some(timestamp) = parse_clock(tag) {
            timestamps.push(timestamp);
        } else if timestamps.is_empty() {
            return None;
        }
        rest = &after_open[close + 1..];
    }
    (!timestamps.is_empty()).then_some((timestamps, rest))
}

fn parse_clock(value: &str) -> Option<u64> {
    let value = value.trim().replace(',', ".");
    let parts: Vec<_> = value.split(':').collect();
    let (hours, minutes, seconds) = match parts.as_slice() {
        [minutes, seconds] => (0_u64, minutes.parse::<u64>().ok()?, parse_seconds(seconds)?),
        [hours, minutes, seconds] => (
            hours.parse::<u64>().ok()?,
            minutes.parse::<u64>().ok()?,
            parse_seconds(seconds)?,
        ),
        _ => return None,
    };
    Some(hours * 3_600_000 + minutes * 60_000 + seconds)
}

fn parse_seconds(value: &str) -> Option<u64> {
    let (seconds, fraction) = value.split_once('.').unwrap_or((value, ""));
    let seconds: u64 = seconds.parse().ok()?;
    let fraction_digits: String = fraction
        .chars()
        .take(3)
        .chain(std::iter::repeat('0'))
        .take(3)
        .collect();
    Some(seconds * 1_000 + fraction_digits.parse::<u64>().unwrap_or(0))
}

fn clean_caption_text(value: &str) -> String {
    let value = value
        .replace("\\N", " ")
        .replace("\\n", " ")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">");
    let mut result = String::with_capacity(value.len());
    let mut in_html_tag = false;
    let mut in_ass_tag = false;
    let mut in_lrc_word_tag = false;
    for character in value.chars() {
        match character {
            '<' => {
                in_html_tag = true;
                in_lrc_word_tag = true;
            }
            '>' if in_html_tag || in_lrc_word_tag => {
                in_html_tag = false;
                in_lrc_word_tag = false;
            }
            '{' => in_ass_tag = true,
            '}' if in_ass_tag => in_ass_tag = false,
            _ if !in_html_tag && !in_ass_tag && !in_lrc_word_tag => result.push(character),
            _ => {}
        }
    }
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn estimate_lrc_duration_ms(text: &str) -> u64 {
    let characters = text.chars().count() as u64;
    (characters.saturating_mul(260) + 900).clamp(2_000, 8_000)
}

fn normalize_cues(mut cues: Vec<SubtitleCue>) -> Vec<SubtitleCue> {
    cues.retain(|cue| !cue.text.trim().is_empty() && cue.end_ms > cue.start_ms);
    cues.sort_by_key(|cue| (cue.start_ms, cue.end_ms));
    let mut merged: Vec<SubtitleCue> = Vec::with_capacity(cues.len());
    for cue in cues {
        if let Some(previous) = merged.last_mut()
            && previous.start_ms == cue.start_ms
        {
            if !previous.text.contains(&cue.text) {
                previous.text.push(' ');
                previous.text.push_str(&cue.text);
            }
            previous.end_ms = previous.end_ms.max(cue.end_ms);
            continue;
        }
        merged.push(cue);
    }
    merged
}

pub fn format_timestamp(milliseconds: u64) -> String {
    let hours = milliseconds / 3_600_000;
    let minutes = (milliseconds / 60_000) % 60;
    let seconds = (milliseconds / 1_000) % 60;
    let millis = milliseconds % 1_000;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
    } else {
        format!("{minutes:02}:{seconds:02}.{millis:03}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_named(name: &str, text: &str) -> SubtitleTrack {
        parse_subtitle(Path::new(name), text.as_bytes()).unwrap()
    }

    #[test]
    fn parses_srt_with_multiline_text_and_comma_milliseconds() {
        let track = parse_named(
            "demo.srt",
            "1\r\n00:00:01,250 --> 00:00:03,500\r\n你好，<b>世界</b>！\r\nSecond line\r\n\r\n",
        );
        assert_eq!(track.format, SubtitleFormat::Srt);
        assert_eq!(track.cues.len(), 1);
        assert_eq!(track.cues[0].start_ms, 1_250);
        assert_eq!(track.cues[0].end_ms, 3_500);
        assert_eq!(track.cues[0].text, "你好，世界！ Second line");
    }

    #[test]
    fn parses_webvtt_with_settings() {
        let track = parse_named(
            "demo.vtt",
            "WEBVTT\n\nintro\n00:01.000 --> 00:03.200 position:20%\nHello world\n",
        );
        assert_eq!(track.format, SubtitleFormat::WebVtt);
        assert_eq!(track.cues[0].start_ms, 1_000);
        assert_eq!(track.cues[0].end_ms, 3_200);
    }

    #[test]
    fn parses_ass_dialogue_and_removes_override_tags() {
        let track = parse_named(
            "demo.ass",
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:02.10,0:00:04.25,Default,,0,0,0,,{\\an8}你好，世界\\NHello\n",
        );
        assert_eq!(track.format, SubtitleFormat::Ass);
        assert_eq!(track.cues[0].start_ms, 2_100);
        assert_eq!(track.cues[0].end_ms, 4_250);
        assert_eq!(track.cues[0].text, "你好，世界 Hello");
    }

    #[test]
    fn parses_lrc_and_derives_end_times() {
        let track = parse_named("demo.lrc", "[00:01.00]第一句\n[00:04.50]Second line\n");
        assert_eq!(track.format, SubtitleFormat::Lrc);
        assert_eq!(track.cues.len(), 2);
        assert_eq!(track.cues[0].end_ms, 4_500);
        assert!(track.cues[1].end_ms > track.cues[1].start_ms);
    }

    #[test]
    fn decodes_utf16_le_bom() {
        let text = "1\n00:00:00,000 --> 00:00:01,000\n你好\n";
        let mut bytes = vec![0xFF, 0xFE];
        for unit in text.encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        assert_eq!(decode_subtitle_bytes(&bytes).unwrap(), text);
    }
}
