//! Speaking: the `speak` tool turns text into a Matrix voice message.
//!
//! The voice is an outside program, so it can be swapped without touching
//! parley: `<program> <out.ogg>` gets the text on stdin and writes Ogg Opus
//! to the file. The default is `parley-speak` (Piper, on the CPU, built next
//! to `parley` by the flake). Set `PARLEY_SPEAK` to use another, for example
//! a script that calls ElevenLabs.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    sync::OnceLock,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use tokio::{io::AsyncWriteExt, sync::Semaphore};

/// Longer texts are refused: a voice message is for a few sentences.
pub const MAX_CHARS: usize = 3000;
/// Larger results are refused.
const MAX_BYTES: u64 = 10 * 1024 * 1024;
/// How long making one message may take.
pub const TIMEOUT: Duration = Duration::from_secs(180);

/// A voice message, ready to send.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spoken {
    pub data: Vec<u8>,
    pub duration: Duration,
}

/// The program to run: `PARLEY_SPEAK`, or `parley-speak` next to this
/// executable, or on PATH.
pub fn speaker() -> PathBuf {
    if let Some(p) = std::env::var_os("PARLEY_SPEAK") {
        return PathBuf::from(p);
    }
    crate::voice::sibling("parley-speak")
}

/// One voice at a time: it runs in parley's own cgroup, outside the limits
/// of an agent run.
fn gate() -> &'static Semaphore {
    static GATE: OnceLock<Semaphore> = OnceLock::new();
    GATE.get_or_init(|| Semaphore::new(1))
}

/// `text` as a voice message, made by `program`.
pub async fn speak(program: &Path, text: &str, timeout: Duration) -> Result<Spoken> {
    let text = text.trim();
    if text.is_empty() {
        bail!("the text is empty");
    }
    let chars = text.chars().count();
    if chars > MAX_CHARS {
        bail!("the text is {chars} characters; the limit is {MAX_CHARS}");
    }
    let _turn = gate().acquire().await.context("speech is shut down")?;
    let dir = tempfile::tempdir().context("no temporary directory")?;
    let out = dir.path().join("voice.ogg");
    tokio::time::timeout(timeout, run(program, text, &out))
        .await
        .with_context(|| format!("speech took over {}s", timeout.as_secs()))??;
    let size = tokio::fs::metadata(&out)
        .await
        .context("the speech program wrote no file")?
        .len();
    if size > MAX_BYTES {
        bail!("the voice message is {size} bytes; the limit is {MAX_BYTES}");
    }
    let data = tokio::fs::read(&out).await?;
    let duration = opus_duration(&data).context("the speech program did not write Ogg Opus")?;
    Ok(Spoken { data, duration })
}

async fn run(program: &Path, text: &str, out: &Path) -> Result<()> {
    let mut child = tokio::process::Command::new(program)
        .arg(out)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("could not start {}", program.display()))?;
    let mut stdin = child.stdin.take().context("no stdin")?;
    let input = text.as_bytes().to_vec();
    // A program that exits without reading is reported by its status.
    let feed = tokio::spawn(async move {
        let _ = stdin.write_all(&input).await;
    });
    let done = child.wait_with_output().await?;
    let _ = feed.await;
    if !done.status.success() {
        let err = String::from_utf8_lossy(&done.stderr);
        let tail: Vec<&str> = err.lines().rev().take(3).collect();
        let tail: Vec<&str> = tail.into_iter().rev().collect();
        bail!("speech failed ({}): {}", done.status, tail.join(" / "));
    }
    Ok(())
}

/// The length of an Ogg Opus file: the last page's granule position (always
/// counted at 48 kHz) minus the pre-skip in the `OpusHead` packet.
fn opus_duration(data: &[u8]) -> Option<Duration> {
    let (mut at, mut pre_skip, mut last) = (0, None, None);
    while at + 27 <= data.len() {
        if &data[at..at + 4] != b"OggS" {
            return None;
        }
        let granule = u64::from_le_bytes(data[at + 6..at + 14].try_into().ok()?);
        let segments = data[at + 26] as usize;
        let table = data.get(at + 27..at + 27 + segments)?;
        let body = at + 27 + segments;
        let len: usize = table.iter().map(|&s| s as usize).sum();
        let page = data.get(body..body + len)?;
        if pre_skip.is_none() {
            if page.len() < 12 || &page[..8] != b"OpusHead" {
                return None;
            }
            pre_skip = Some(u16::from_le_bytes([page[10], page[11]]) as u64);
        }
        // All ones: no packet ends on this page.
        if granule != u64::MAX {
            last = Some(granule);
        }
        at = body + len;
    }
    let samples = last?.saturating_sub(pre_skip?);
    (samples > 0).then(|| Duration::from_millis(samples / 48))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn page(granule: u64, body: &[u8]) -> Vec<u8> {
        assert!(body.len() < 255);
        let mut p = b"OggS\0\0".to_vec();
        p.extend(granule.to_le_bytes());
        p.extend([0; 12]);
        p.extend([1, body.len() as u8]);
        p.extend(body);
        p
    }

    /// Ogg Opus of 1.5 s, with a pre-skip of 312 samples.
    fn ogg() -> Vec<u8> {
        let mut head = b"OpusHead\x01\x01".to_vec();
        head.extend(312u16.to_le_bytes());
        head.extend([0; 7]);
        let mut d = page(0, &head);
        d.extend(page(u64::MAX, b"partial"));
        d.extend(page(72_000 + 312, b"audio"));
        d
    }

    fn script(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("speak");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[test]
    fn reads_the_length_of_ogg_opus() {
        assert_eq!(opus_duration(&ogg()), Some(Duration::from_millis(1500)));
        assert_eq!(opus_duration(b"RIFF....WAVE"), None);
        let vorbis = page(48_000, b"\x01vorbis......");
        assert_eq!(opus_duration(&vorbis), None);
    }

    #[tokio::test]
    async fn the_program_gets_the_text_and_writes_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src.ogg");
        std::fs::write(&src, ogg()).unwrap();
        let heard = dir.path().join("heard");
        let prog = script(
            dir.path(),
            &format!("cat > {}; cp {} \"$1\"", heard.display(), src.display()),
        );
        let s = speak(&prog, "  hallo  ", TIMEOUT).await.unwrap();
        assert_eq!(s.duration, Duration::from_millis(1500));
        assert_eq!(std::fs::read_to_string(heard).unwrap(), "hallo");
    }

    #[tokio::test]
    async fn failures_are_errors() {
        let dir = tempfile::tempdir().unwrap();
        let prog = script(dir.path(), "echo 'no voice' >&2; exit 3");
        let err = speak(&prog, "hoi", TIMEOUT).await.unwrap_err();
        assert!(err.to_string().contains("no voice"), "{err}");

        let junk = script(dir.path(), "cat >/dev/null; echo junk > \"$1\"");
        assert!(speak(&junk, "hoi", TIMEOUT).await.is_err());

        let slow = script(dir.path(), "sleep 5");
        let err = speak(&slow, "hoi", Duration::from_millis(100))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("took over"), "{err}");
    }

    #[tokio::test]
    async fn refuses_empty_and_too_long_texts() {
        let p = Path::new("/nonexistent");
        assert!(speak(p, "  ", TIMEOUT).await.is_err());
        let long = "a".repeat(MAX_CHARS + 1);
        let err = speak(p, &long, TIMEOUT).await.unwrap_err();
        assert!(err.to_string().contains("limit"), "{err}");
    }
}
