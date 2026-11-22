//! Files an agent posts with the `file` tool: anything on the lab, such as a
//! PDF, a song or an image. The type decides how clients show it, so an image
//! is recognised by its content and the rest by its extension; what is not
//! known goes out as a plain download.

use std::path::Path;

use anyhow::{Context, Result, bail};

/// The homeserver's `max_request_size` (continuwuity, in matrix.nix): a larger
/// file would only fail at the upload.
pub const MAX_BYTES: u64 = 20_000_000;

/// A file, ready to send.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loaded {
    pub filename: String,
    pub mimetype: &'static str,
    pub data: Vec<u8>,
    /// Width and height in pixels, where the format makes that cheap (PNG).
    pub size: Option<(u32, u32)>,
}

/// The image type of `data`, from its first bytes.
fn sniff_image(data: &[u8]) -> Option<&'static str> {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if data.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

/// The type of a file that is not an image, from its extension.
fn by_extension(path: &Path) -> &'static str {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase());
    match ext.as_deref() {
        Some("mp3") => "audio/mpeg",
        Some("ogg" | "oga" | "opus") => "audio/ogg",
        Some("wav") => "audio/wav",
        Some("flac") => "audio/flac",
        Some("m4a") => "audio/mp4",
        Some("mp4" | "m4v") => "video/mp4",
        Some("webm") => "video/webm",
        Some("mov") => "video/quicktime",
        Some("pdf") => "application/pdf",
        Some("zip") => "application/zip",
        Some("json") => "application/json",
        Some("txt" | "log") => "text/plain",
        Some("md") => "text/markdown",
        Some("csv") => "text/csv",
        _ => "application/octet-stream",
    }
}

/// Width and height from a PNG's header.
fn png_size(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 24 || &data[12..16] != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes(data[16..20].try_into().ok()?);
    let h = u32::from_be_bytes(data[20..24].try_into().ok()?);
    Some((w, h))
}

/// Read the file at `path`, which must be absolute.
pub async fn load(path: &str) -> Result<Loaded> {
    let p = Path::new(path);
    if !p.is_absolute() {
        bail!("`path` must be absolute, not {path:?}");
    }
    let meta = tokio::fs::metadata(p)
        .await
        .with_context(|| format!("cannot read {path}"))?;
    if !meta.is_file() {
        bail!("{path} is not a file");
    }
    if meta.len() > MAX_BYTES {
        bail!("{path} is {} bytes; the limit is {MAX_BYTES}", meta.len());
    }
    let data = tokio::fs::read(p)
        .await
        .with_context(|| format!("cannot read {path}"))?;
    let mimetype = sniff_image(&data).unwrap_or_else(|| match by_extension(p) {
        // Named like an image type we do not recognise: stays a download.
        t if t.starts_with("image/") => "application/octet-stream",
        t => t,
    });
    let size = (mimetype == "image/png").then(|| png_size(&data)).flatten();
    let filename = p
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_owned());
    Ok(Loaded {
        filename,
        mimetype,
        data,
        size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(w: u32, h: u32) -> Vec<u8> {
        let mut d = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
        d.extend(w.to_be_bytes());
        d.extend(h.to_be_bytes());
        d
    }

    #[test]
    fn sniffs_images_by_content() {
        assert_eq!(sniff_image(&png(1, 1)), Some("image/png"));
        assert_eq!(sniff_image(&[0xff, 0xd8, 0xff, 0xe0]), Some("image/jpeg"));
        assert_eq!(sniff_image(b"GIF89a..."), Some("image/gif"));
        assert_eq!(sniff_image(b"RIFF\0\0\0\0WEBPVP8 "), Some("image/webp"));
        assert_eq!(sniff_image(b"[Interface]\nPrivateKey = x"), None);
    }

    #[test]
    fn knows_types_by_extension() {
        assert_eq!(by_extension(Path::new("/a/Lied.MP3")), "audio/mpeg");
        assert_eq!(by_extension(Path::new("/a/r.pdf")), "application/pdf");
        assert_eq!(by_extension(Path::new("/a/v.mp4")), "video/mp4");
        assert_eq!(
            by_extension(Path::new("/a/noext")),
            "application/octet-stream"
        );
    }

    #[tokio::test]
    async fn loads_any_file_and_refuses_bad_paths() {
        let dir = tempfile::tempdir().unwrap();
        let img = dir.path().join("qr.txt");
        std::fs::write(&img, png(290, 310)).unwrap();
        let l = load(img.to_str().unwrap()).await.unwrap();
        assert_eq!(l.mimetype, "image/png");
        assert_eq!(l.size, Some((290, 310)));
        assert_eq!(l.filename, "qr.txt");

        let song = dir.path().join("lied.ogg");
        std::fs::write(&song, "OggS").unwrap();
        let l = load(song.to_str().unwrap()).await.unwrap();
        assert_eq!(l.mimetype, "audio/ogg");
        assert_eq!(l.size, None);

        assert!(load("relative.pdf").await.is_err());
        assert!(load(dir.path().to_str().unwrap()).await.is_err());
    }

    #[tokio::test]
    async fn refuses_a_file_over_the_limit() {
        let dir = tempfile::tempdir().unwrap();
        let big = dir.path().join("big.bin");
        let f = std::fs::File::create(&big).unwrap();
        f.set_len(MAX_BYTES + 1).unwrap();
        assert!(load(big.to_str().unwrap()).await.is_err());
    }
}
