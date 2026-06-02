//! User-config file mutation: appending `[[match]]` blocks.

#[cfg(feature = "config")]
use std::path::PathBuf;

#[cfg(feature = "config")]
use super::discovery::user_config_path;

/// Append a `[[match]]` block to the user config file.
///
/// At least one of `git_remote` or `path_glob` must be `Some`.
/// Parent directories are created if they do not exist.
/// Returns the path of the config file that was written.
#[cfg(feature = "config")]
pub fn append_match_to_user_config(
    git_remote: Option<&str>,
    path_glob: Option<&str>,
    project: &str,
) -> crate::Result<PathBuf> {
    if git_remote.is_none() && path_glob.is_none() {
        return Err(crate::Error::config(
            "at least one of git_remote or path_glob must be provided",
        ));
    }

    let config_path = user_config_path();

    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let existing = match std::fs::read_to_string(&config_path) {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e.into()),
    };

    let mut block = String::from("[[match]]\n");
    if let Some(remote) = git_remote {
        block.push_str(&format!(
            "git-remote = \"{}\"\n",
            toml_escape_string(remote)
        ));
    }
    if let Some(glob) = path_glob {
        block.push_str(&format!("path = \"{}\"\n", toml_escape_string(glob)));
    }
    block.push_str(&format!("project = \"{}\"\n", toml_escape_string(project)));

    // Build the final content: preserve existing, add a blank-line separator, append block.
    if !existing.is_empty() {
        let trimmed = existing.trim_end_matches('\n');
        let final_content = format!("{trimmed}\n\n{block}");
        std::fs::write(&config_path, final_content)?;
    } else {
        std::fs::write(&config_path, &block)?;
    }

    Ok(config_path)
}

/// Escape a string value for safe embedding inside a TOML basic string (double-quoted).
///
/// Handles the named escapes `\\`, `\"`, `\n`, `\r`, `\t` and `\uXXXX`-encodes any
/// remaining control character (`U+0000`–`U+001F` and `U+007F`), which TOML forbids
/// bare inside a basic string. Any value written without these escapes produces
/// invalid TOML that won't round-trip.
#[cfg(feature = "config")]
pub(super) fn toml_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c == '\u{7f}' => {
                out.push_str(&format!("\\u{:04X}", c as u32));
            }
            other => out.push(other),
        }
    }
    out
}
