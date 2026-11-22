//! Runs as another Linux user: `run_as = "paul"` in room.toml.
//!
//! Every other room runs as parley's own user, with its environment, so a
//! `workdir` is no boundary. A room with `run_as` starts its harness as
//! `sudo -n -u <user> /run/current-system/sw/bin/<user>-claude <args>` with
//! a cleared environment; the wrapper (from the NixOS config) loads the
//! user's own credentials and picks the working directory.
//!
//! What crosses the boundary is kept out of the command line, which sudo
//! logs: the message goes over stdin, and the MCP config (with the turn's
//! token) is a file in `/run/parley-<user>` (owned parley:<user>, 2750).
//! The turn's socket sits next to it and answers only that turn's token, so
//! a token of another room's run is worth nothing here. The main `post.sock`
//! stays closed to the user.
//!
//! It fails closed: if anything here fails, the turn fails. It never runs
//! as parley's user instead.

use std::{
    fs,
    io::Write,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, bail};

use crate::post::{self, Posts};

/// On NixOS the setuid sudo is here; a service's PATH need not have it.
pub const SUDO: &str = "/run/wrappers/bin/sudo";

/// Told to a confined run: what it does not have compared to other rooms.
pub const PROMPT_NOTE: &str = "\n\n## Your limits here\n\n\
You run as your own Linux user and cannot read parley's files. The `file`, \
`task` and `tasks` tools do not work here. For the room's past, use the \
`read` tool.\n";

/// A user name that is safe in a sudo argument and in a path.
pub fn check_user(user: &str) -> Result<()> {
    let mut chars = user.chars();
    let first_ok = chars.next().is_some_and(|c| c.is_ascii_lowercase());
    let rest_ok = chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    if !first_ok || !rest_ok || user.len() > 31 {
        bail!("invalid run_as user {user:?}");
    }
    Ok(())
}

/// The wrapper that starts the harness as `user`.
pub fn wrapper(user: &str) -> String {
    format!("/run/current-system/sw/bin/{user}-claude")
}

/// Where `user`'s runs get their socket and MCP config.
pub fn run_dir(user: &str) -> Result<PathBuf> {
    check_user(user)?;
    Ok(PathBuf::from(format!("/run/parley-{user}")))
}

/// One turn's socket and MCP config, in the user's run directory. Dropping
/// it stops the socket and removes both.
pub struct RunTools {
    socket: PathBuf,
    config: PathBuf,
    server: tokio::task::JoinHandle<()>,
}

impl RunTools {
    /// Must be called on the runtime: it starts the socket's server.
    pub fn create(dir: &Path, exe: &Path, posts: Arc<Posts>, token: &str) -> Result<Self> {
        let id = post::random_token()?;
        let socket = dir.join(format!("run-{id}.sock"));
        let config = dir.join(format!("run-{id}.json"));
        let listener = tokio::net::UnixListener::bind(&socket)
            .with_context(|| format!("could not bind {}", socket.display()))?;
        let server = tokio::spawn(post::serve_token(listener, posts, token.to_owned()));
        // From here on, dropping it cleans up, also when a step below fails.
        let tools = Self {
            socket,
            config,
            server,
        };
        fs::set_permissions(&tools.socket, fs::Permissions::from_mode(0o660))
            .context("could not set the socket's mode")?;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o640)
            .open(&tools.config)
            .with_context(|| format!("could not write {}", tools.config.display()))?;
        file.write_all(post::mcp_config(exe, &tools.socket, token).as_bytes())?;
        Ok(tools)
    }

    /// The value for `--mcp-config`: a path, so the token stays off the
    /// command line.
    pub fn config_arg(&self) -> String {
        self.config.display().to_string()
    }
}

impl Drop for RunTools {
    fn drop(&mut self) {
        self.server.abort();
        let _ = fs::remove_file(&self.socket);
        let _ = fs::remove_file(&self.config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    #[test]
    fn only_plain_user_names() {
        for good in ["paul", "a", "paul-2"] {
            check_user(good).unwrap();
        }
        for bad in ["", "Paul", "../x", "a/b", "-n", "a b", "root;id", "2a"] {
            assert!(check_user(bad).is_err(), "{bad:?}");
        }
        assert!(run_dir("../etc").is_err());
        assert_eq!(run_dir("paul").unwrap(), Path::new("/run/parley-paul"));
        assert_eq!(wrapper("paul"), "/run/current-system/sw/bin/paul-claude");
    }

    #[tokio::test]
    async fn run_tools_have_the_agreed_modes_and_clean_up() {
        let tmp = tempfile::tempdir().unwrap();
        let posts = Arc::new(Posts::default());
        let tools = RunTools::create(tmp.path(), Path::new("/bin/parley"), posts, "tok").unwrap();
        let mode = |p: &Path| fs::metadata(p).unwrap().mode() & 0o777;
        assert_eq!(mode(&tools.socket), 0o660);
        assert_eq!(mode(&tools.config), 0o640);

        let config: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&tools.config).unwrap()).unwrap();
        let env = &config["mcpServers"]["parley"]["env"];
        assert_eq!(env["PARLEY_TOKEN"], "tok");
        assert_eq!(env["PARLEY_SOCKET"], tools.socket.to_str().unwrap());
        assert_eq!(tools.config_arg(), tools.config.to_str().unwrap());

        drop(tools);
        assert_eq!(fs::read_dir(tmp.path()).unwrap().count(), 0);
    }
}
