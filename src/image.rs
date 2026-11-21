//! Images an agent posts with the `image` tool: a file on the lab, checked
//! by its content (not its name) so that the tool cannot be used to post
//! arbitrary files.

use std::path::Path;

use anyhow::{Context, Result, bail};

/// Larger files are refused; a chat image does not need more.
pub const MAX_BYTES: u64 = 10 * 1024 * 1024;

/// An image file, ready to send.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loaded {
    pub filename: String,
    pub mimetype: &'static str,
    pub data: Vec<u8>,
    /// Width and height in pixels, where the format makes that cheap (PNG).
    pub size: Option<(u32, u32)>,
}

/// The image type of `data`, from its first bytes.
pub fn sniff(data: &[u8]) -> Option<&'static str> {
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

/// Width and height from a PNG's header.
fn png_size(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 24 || &data[12..16] != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes(data[16..20].try_into().ok()?);
    let h = u32::from_be_bytes(data[20..24].try_into().ok()?);
    Some((w, h))
}

/// Read the image at `path`, which must be absolute.
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
    let Some(mimetype) = sniff(&data) else {
        bail!("{path} is not a PNG, JPEG, GIF or WebP image");
    };
    let size = (mimetype == "image/png").then(|| png_size(&data)).flatten();
    let filename = p
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".to_owned());
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
    fn sniffs_by_content() {
        assert_eq!(sniff(&png(1, 1)), Some("image/png"));
        assert_eq!(sniff(&[0xff, 0xd8, 0xff, 0xe0]), Some("image/jpeg"));
        assert_eq!(sniff(b"GIF89a..."), Some("image/gif"));
        assert_eq!(sniff(b"RIFF\0\0\0\0WEBPVP8 "), Some("image/webp"));
        assert_eq!(sniff(b"[Interface]\nPrivateKey = x"), None);
    }

    #[tokio::test]
    async fn loads_an_image_and_refuses_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("qr.txt");
        std::fs::write(&good, png(290, 310)).unwrap();
        let l = load(good.to_str().unwrap()).await.unwrap();
        assert_eq!(l.mimetype, "image/png");
        assert_eq!(l.size, Some((290, 310)));
        assert_eq!(l.filename, "qr.txt");

        let bad = dir.path().join("x.png");
        std::fs::write(&bad, "[Interface]").unwrap();
        assert!(load(bad.to_str().unwrap()).await.is_err());
        assert!(load("relative.png").await.is_err());
        assert!(load(dir.path().to_str().unwrap()).await.is_err());
    }
}
