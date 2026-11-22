//! Voice messages: an `m.audio` event is saved in the room's folder and
//! turned into text by an outside transcriber, then handled like a text
//! message (`🎤 <text>`).
//!
//! The transcriber is `parley-transcribe <file>` (faster-whisper, built
//! next to `parley` by the flake): it prints the text on stdout. Set
//! `PARLEY_TRANSCRIBE` to use another program.

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// What a room does with voice messages (`voice` in room.toml).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Voice {
    /// Transcribe, and show the transcript as a reply to the voice message,
    /// so its sender sees what was heard.
    #[default]
    Reply,
    /// Transcribe without showing it; the agent still gets the text.
    Quiet,
    /// Ignore voice messages.
    Off,
}

/// How long one transcription may take.
pub const TIMEOUT: Duration = Duration::from_secs(600);

/// The transcriber to run: `PARLEY_TRANSCRIBE`, or `parley-transcribe`
/// next to this executable (as started: the flake's `bin/` holds symlinks
/// into two packages, so the resolved path would miss it), or on PATH.
pub fn transcriber() -> PathBuf {
    if let Some(p) = std::env::var_os("PARLEY_TRANSCRIBE") {
        return PathBuf::from(p);
    }
    sibling("parley-transcribe")
}

/// The program `name` next to this executable, or else on PATH.
pub fn sibling(name: &str) -> PathBuf {
    let started = std::env::args_os().next().map(PathBuf::from);
    [started, std::env::current_exe().ok()]
        .into_iter()
        .flatten()
        .filter_map(|exe| exe.parent().map(|d| d.join(name)))
        .find(|p| p.is_absolute() && p.exists())
        .unwrap_or_else(|| PathBuf::from(name))
}

/// The text `program` hears in the audio file `file`.
pub async fn transcribe(program: &Path, file: &Path, timeout: Duration) -> Result<String> {
    let child = tokio::process::Command::new(program)
        .arg(file)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("could not start {}", program.display()))?;
    let out = tokio::time::timeout(timeout, child.wait_with_output())
        .await
        .with_context(|| format!("transcription took over {}s", timeout.as_secs()))??;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = err.lines().rev().take(3).collect();
        let tail: Vec<&str> = tail.into_iter().rev().collect();
        bail!("transcriber failed ({}): {}", out.status, tail.join(" / "));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// The file extension for audio of `mimetype` (or named `filename`).
pub fn extension(mimetype: Option<&str>, filename: &str) -> String {
    let by_mime = match mimetype.map(|m| m.split(';').next().unwrap_or(m).trim()) {
        Some("audio/ogg" | "audio/opus") => Some("ogg"),
        Some("audio/mp4" | "audio/m4a" | "audio/x-m4a" | "audio/aac") => Some("m4a"),
        Some("audio/mpeg" | "audio/mp3") => Some("mp3"),
        Some("audio/webm") => Some("webm"),
        Some("audio/wav" | "audio/x-wav" | "audio/wave") => Some("wav"),
        _ => None,
    };
    if let Some(e) = by_mime {
        return e.to_owned();
    }
    match Path::new(filename).extension().and_then(|e| e.to_str()) {
        Some(e)
            if !e.is_empty() && e.len() <= 5 && e.chars().all(|c| c.is_ascii_alphanumeric()) =>
        {
            e.to_ascii_lowercase()
        }
        _ => "audio".to_owned(),
    }
}

/// The message body for a transcript.
pub fn heard(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        "🎤 _(niets verstaan)_".to_owned()
    } else {
        format!("🎤 {text}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn script(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("transcribe");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[tokio::test]
    async fn transcribes_what_the_program_prints() {
        let dir = tempfile::tempdir().unwrap();
        let prog = script(dir.path(), r#"echo "  hallo $1  ""#);
        let text = transcribe(&prog, Path::new("x.ogg"), TIMEOUT)
            .await
            .unwrap();
        assert_eq!(text, "hallo x.ogg");
    }

    #[tokio::test]
    async fn a_failing_program_is_an_error_with_its_stderr() {
        let dir = tempfile::tempdir().unwrap();
        let prog = script(dir.path(), "echo 'no model' >&2; exit 3");
        let err = transcribe(&prog, Path::new("x.ogg"), TIMEOUT)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no model"), "{err}");
    }

    #[tokio::test]
    async fn a_slow_program_times_out() {
        let dir = tempfile::tempdir().unwrap();
        let prog = script(dir.path(), "sleep 5");
        let err = transcribe(&prog, Path::new("x.ogg"), Duration::from_millis(100))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("took over"), "{err}");
    }

    #[test]
    fn extensions_come_from_the_mimetype_then_the_name() {
        assert_eq!(extension(Some("audio/ogg; codecs=opus"), "x"), "ogg");
        assert_eq!(extension(Some("audio/mp4"), "Voice.m4a"), "m4a");
        assert_eq!(extension(None, "Voice message.OGG"), "ogg");
        assert_eq!(extension(Some("audio/strange"), "../../x"), "audio");
    }

    #[test]
    fn an_empty_transcript_says_so() {
        assert_eq!(heard(" hoi "), "🎤 hoi");
        assert_eq!(heard(""), "🎤 _(niets verstaan)_");
    }
}
