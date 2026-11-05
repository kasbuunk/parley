//! Standing clients: agents outside parley's own runs (a Claude Code
//! session on the lab, or on the Mac over ssh) that use the same tools as
//! a run: `post`, `read`, `rooms` and the schedule tools.
//!
//! One file per client, `PARLEY_DATA_DIR/clients/<name>.toml` (keep it
//! 0600: the token is the client's only credential):
//!
//! ```toml
//! home = "nixos"   # the room `post` and `read` use when none is named
//! token = "…"      # at least 32 hex characters
//!
//! [[access]]       # same tables as in room.toml; home is always open
//! room = "life"
//! read = true
//! ```
//!
//! parley loads them when it starts, so a change needs a restart. The
//! client starts `parley mcp --client <name>`, which reads the token from
//! the same file and talks to the running parley over `post.sock`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::access::Grant;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Client {
    /// A room's `name` (from its room.toml) or ID.
    pub home: String,
    pub token: String,
    #[serde(default)]
    pub access: Vec<Grant>,
}

pub fn parse(text: &str) -> Result<Client> {
    let c: Client = toml::from_str(text).context("invalid client file")?;
    if c.token.len() < 32 || !c.token.chars().all(|ch| ch.is_ascii_hexdigit()) {
        bail!("token must be at least 32 hex characters");
    }
    Ok(c)
}

/// Every `*.toml` in `dir`, by name, parsed; sorted by name. A missing
/// directory means no clients.
pub fn load(dir: &Path) -> Vec<(String, Result<Client>)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<_> = entries
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            if path.extension()? != "toml" {
                return None;
            }
            let name = path.file_stem()?.to_string_lossy().into_owned();
            let client = std::fs::read_to_string(&path)
                .context("unreadable")
                .and_then(|t| parse(&t));
            Some((name, client))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The token of client `name` in `dir`, for `parley mcp --client <name>`.
pub fn token(dir: &Path, name: &str) -> Result<String> {
    if name.is_empty() || name.contains('/') || name.contains("..") {
        bail!("bad client name {name:?}");
    }
    let path = dir.join(format!("{name}.toml"));
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    Ok(parse(&text)?.token)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKEN: &str = "0123456789abcdef0123456789abcdef";

    #[test]
    fn parses_a_client() {
        let c = parse(&format!(
            "home = \"nixos\"\ntoken = \"{TOKEN}\"\n\n[[access]]\nroom = \"life\"\nread = true\n"
        ))
        .unwrap();
        assert_eq!(c.home, "nixos");
        assert_eq!(c.access.len(), 1);
        assert!(c.access[0].read && !c.access[0].write);
    }

    #[test]
    fn rejects_weak_tokens_and_unknown_fields() {
        assert!(parse("home = \"x\"\ntoken = \"abc\"\n").is_err());
        assert!(parse(&format!("home = \"x\"\ntoken = \"{}\"\n", "z".repeat(40))).is_err());
        assert!(
            parse(&format!(
                "home = \"x\"\ntoken = \"{TOKEN}\"\nadmin = true\n"
            ))
            .is_err()
        );
    }

    #[test]
    fn loads_toml_files_by_name_and_finds_tokens() {
        let dir = std::env::temp_dir().join(format!("parley-clients-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("b.toml"),
            format!("home = \"x\"\ntoken = \"{TOKEN}\"\n"),
        )
        .unwrap();
        std::fs::write(dir.join("a.toml"), "home = \"x\"\n").unwrap();
        std::fs::write(dir.join("notes.txt"), "ignored").unwrap();
        let loaded = load(&dir);
        let names: Vec<_> = loaded.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
        assert!(loaded[0].1.is_err() && loaded[1].1.is_ok());
        assert_eq!(token(&dir, "b").unwrap(), TOKEN);
        assert!(token(&dir, "../b").is_err());
        assert!(token(&dir, "a").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(load(&dir).is_empty());
    }
}
