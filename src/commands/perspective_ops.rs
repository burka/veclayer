//! Perspective management commands.

use super::*;

/// List all perspectives.
pub fn perspective_list(data_dir: &Path) -> Result<()> {
    let perspectives = crate::perspective::load(data_dir)?;
    if perspectives.is_empty() {
        println!("No perspectives defined.");
        return Ok(());
    }
    for p in &perspectives {
        let tag = if p.builtin { " [builtin]" } else { "" };
        println!("  {} -- {}{}", p.id, p.hint, tag);
    }
    println!("\n{} perspective(s)", perspectives.len());
    Ok(())
}

/// Add a custom perspective.
pub fn perspective_add(data_dir: &Path, id: &str, name: &str, hint: &str) -> Result<()> {
    crate::perspective::add(
        data_dir,
        crate::perspective::Perspective::new(id, name, hint),
    )?;
    println!("Added perspective '{}'", id);
    Ok(())
}

/// Remove a custom perspective.
pub fn perspective_remove(data_dir: &Path, id: &str) -> Result<()> {
    crate::perspective::remove(data_dir, id)?;
    println!("Removed perspective '{}'", id);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn init_dir() -> TempDir {
        let dir = TempDir::new().unwrap();
        crate::perspective::init(dir.path()).unwrap();
        dir
    }

    // ── perspective_list ──────────────────────────────────────────────────────

    #[test]
    fn test_perspective_list_shows_defaults() {
        let dir = init_dir();
        perspective_list(dir.path()).unwrap();
    }

    #[test]
    fn test_perspective_list_empty_dir_shows_defaults() {
        // No init — falls back to defaults
        let dir = TempDir::new().unwrap();
        perspective_list(dir.path()).unwrap();
    }

    // ── perspective_add ───────────────────────────────────────────────────────

    #[test]
    fn test_perspective_add_creates_new_perspective() {
        let dir = init_dir();
        perspective_add(dir.path(), "custom", "Custom", "A custom lens").unwrap();

        let perspectives = crate::perspective::load(dir.path()).unwrap();
        assert!(
            perspectives.iter().any(|p| p.id == "custom"),
            "custom perspective should exist"
        );
    }

    #[test]
    fn test_perspective_add_duplicate_id_returns_error() {
        let dir = init_dir();
        perspective_add(dir.path(), "myp", "My P", "hint").unwrap();

        let err = perspective_add(dir.path(), "myp", "My P Again", "hint2").unwrap_err();
        assert!(
            err.to_string().contains("already exists"),
            "expected 'already exists', got: {err}"
        );
    }

    #[test]
    fn test_perspective_add_builtin_id_returns_error() {
        let dir = init_dir();
        // "decisions" is a builtin — adding it again should fail with 'already exists'
        let err = perspective_add(dir.path(), "decisions", "Decisions", "hint").unwrap_err();
        assert!(
            err.to_string().contains("already exists"),
            "expected 'already exists' for duplicate builtin id, got: {err}"
        );
    }

    // ── perspective_remove ────────────────────────────────────────────────────

    #[test]
    fn test_perspective_remove_custom_perspective() {
        let dir = init_dir();
        perspective_add(dir.path(), "toremove", "ToRemove", "hint").unwrap();

        perspective_remove(dir.path(), "toremove").unwrap();

        let perspectives = crate::perspective::load(dir.path()).unwrap();
        assert!(
            !perspectives.iter().any(|p| p.id == "toremove"),
            "removed perspective should not exist"
        );
    }

    #[test]
    fn test_perspective_remove_builtin_returns_error() {
        let dir = init_dir();
        let err = perspective_remove(dir.path(), "decisions").unwrap_err();
        assert!(
            err.to_string().contains("builtin"),
            "expected 'builtin' error, got: {err}"
        );
    }

    #[test]
    fn test_perspective_remove_nonexistent_returns_error() {
        let dir = init_dir();
        let err = perspective_remove(dir.path(), "does-not-exist").unwrap_err();
        assert!(
            err.to_string().contains("not found"),
            "expected 'not found', got: {err}"
        );
    }

    // ── round-trip: add then list then remove ─────────────────────────────────

    #[test]
    fn test_perspective_add_list_remove_round_trip() {
        let dir = init_dir();

        let default_count = crate::perspective::load(dir.path()).unwrap().len();

        perspective_add(dir.path(), "rt-test", "RT Test", "round trip test").unwrap();
        let after_add = crate::perspective::load(dir.path()).unwrap();
        assert_eq!(after_add.len(), default_count + 1);

        perspective_remove(dir.path(), "rt-test").unwrap();
        let after_remove = crate::perspective::load(dir.path()).unwrap();
        assert_eq!(after_remove.len(), default_count);
    }

    // ── perspective_list with custom entry ────────────────────────────────────

    #[test]
    fn test_perspective_list_shows_custom_entry() {
        let dir = init_dir();
        perspective_add(dir.path(), "custom-vis", "Custom Vis", "visible in list").unwrap();
        // Must not panic
        perspective_list(dir.path()).unwrap();
    }
}
