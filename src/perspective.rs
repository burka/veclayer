//! Perspectives: named lenses over the same knowledge.
//!
//! A perspective is a facet like "decisions", "people", or "learnings".
//! Entries can belong to multiple perspectives. Search can be filtered
//! by perspective to see knowledge from a specific angle.
//!
//! VecLayer ships 7 defaults. Custom perspectives can be added at runtime.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// A named perspective (lens) over knowledge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Perspective {
    /// Unique slug identifier (e.g. "decisions", "people")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Hint for LLMs: what kind of content belongs in this perspective
    pub hint: String,
    /// Whether this is a built-in default perspective
    #[serde(default)]
    pub builtin: bool,
}

impl Perspective {
    pub fn new(id: impl Into<String>, name: impl Into<String>, hint: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            hint: hint.into(),
            builtin: false,
        }
    }

    fn builtin(id: &str, name: &str, hint: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            hint: hint.to_string(),
            builtin: true,
        }
    }
}

/// The 7 default perspective IDs.
pub const DEFAULT_PERSPECTIVE_IDS: &[&str] = &[
    "intentions",
    "people",
    "temporal",
    "knowledge",
    "decisions",
    "learnings",
    "session",
];

/// Create the 7 default perspectives.
pub fn defaults() -> Vec<Perspective> {
    vec![
        Perspective::builtin(
            "intentions",
            "Intentions",
            "Goals, motivations, plans, aspirations",
        ),
        Perspective::builtin(
            "people",
            "People",
            "Persons, relationships, roles, organizations",
        ),
        Perspective::builtin(
            "temporal",
            "Temporal",
            "Timelines, milestones, chronology, evolution",
        ),
        Perspective::builtin(
            "knowledge",
            "Knowledge",
            "Durable expertise, definitions, concepts, references",
        ),
        Perspective::builtin(
            "decisions",
            "Decisions",
            "Decisions, trade-offs, alternatives, rationale",
        ),
        Perspective::builtin(
            "learnings",
            "Learnings",
            "Insights, mistakes, lessons learned, aha-moments",
        ),
        Perspective::builtin(
            "session",
            "Session",
            "Work sessions, context summaries, handoffs",
        ),
    ]
}

/// Persistent store for perspectives (JSON file in data dir).
#[derive(Debug, Serialize, Deserialize)]
struct PerspectiveFile {
    perspectives: Vec<Perspective>,
}

const FILENAME: &str = "perspectives.json";

/// Load perspectives from the data directory.
/// Returns defaults if file doesn't exist yet.
pub fn load(data_dir: &Path) -> crate::Result<Vec<Perspective>> {
    let path = data_dir.join(FILENAME);
    if !path.exists() {
        return Ok(defaults());
    }
    let contents = std::fs::read_to_string(&path)
        .map_err(|e| crate::Error::config(format!("Failed to read {}: {}", path.display(), e)))?;
    let file: PerspectiveFile = serde_json::from_str(&contents)
        .map_err(|e| crate::Error::config(format!("Invalid {}: {}", FILENAME, e)))?;
    Ok(file.perspectives)
}

/// Save perspectives to the data directory.
pub fn save(data_dir: &Path, perspectives: &[Perspective]) -> crate::Result<()> {
    let path = data_dir.join(FILENAME);
    let file = PerspectiveFile {
        perspectives: perspectives.to_vec(),
    };
    let json = serde_json::to_string_pretty(&file)
        .map_err(|e| crate::Error::config(format!("Failed to serialize perspectives: {}", e)))?;
    std::fs::write(&path, json)
        .map_err(|e| crate::Error::config(format!("Failed to write {}: {}", path.display(), e)))?;
    Ok(())
}

/// Initialize perspectives: write defaults if file doesn't exist.
pub fn init(data_dir: &Path) -> crate::Result<()> {
    let path = data_dir.join(FILENAME);
    if !path.exists() {
        save(data_dir, &defaults())?;
    }
    Ok(())
}

/// Add a custom perspective. Returns error if ID already exists.
pub fn add(data_dir: &Path, perspective: Perspective) -> crate::Result<()> {
    let mut all = load(data_dir)?;
    if all.iter().any(|p| p.id == perspective.id) {
        return Err(crate::Error::config(format!(
            "Perspective '{}' already exists",
            perspective.id
        )));
    }
    all.push(perspective);
    save(data_dir, &all)
}

/// Remove a perspective by ID. Cannot remove builtins.
pub fn remove(data_dir: &Path, id: &str) -> crate::Result<()> {
    let all = load(data_dir)?;
    let target = all.iter().find(|p| p.id == id);
    match target {
        None => Err(crate::Error::not_found(format!(
            "Perspective '{}' not found",
            id
        ))),
        Some(p) if p.builtin => Err(crate::Error::config(format!(
            "Cannot remove builtin perspective '{}'",
            id
        ))),
        _ => {
            let filtered: Vec<_> = all.into_iter().filter(|p| p.id != id).collect();
            save(data_dir, &filtered)
        }
    }
}

/// Get a perspective by ID.
pub fn get(data_dir: &Path, id: &str) -> crate::Result<Option<Perspective>> {
    let all = load(data_dir)?;
    Ok(all.into_iter().find(|p| p.id == id))
}

/// Validate that all given perspective IDs exist.
pub fn validate_ids(data_dir: &Path, ids: &[String]) -> crate::Result<()> {
    let all = load(data_dir)?;
    let known: std::collections::HashSet<&str> = all.iter().map(|p| p.id.as_str()).collect();
    for id in ids {
        if !known.contains(id.as_str()) {
            return Err(crate::Error::config(format!(
                "Unknown perspective '{}'. Available: {}",
                id,
                all.iter()
                    .map(|p| p.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_defaults_count() {
        assert_eq!(defaults().len(), 7);
    }

    #[test]
    fn test_defaults_ids() {
        let all = defaults();
        let ids: Vec<&str> = all.iter().map(|p| p.id.as_str()).collect();
        assert!(ids.contains(&"intentions"));
        assert!(ids.contains(&"people"));
        assert!(ids.contains(&"temporal"));
        assert!(ids.contains(&"knowledge"));
        assert!(ids.contains(&"decisions"));
        assert!(ids.contains(&"learnings"));
        assert!(ids.contains(&"session"));
    }

    #[test]
    fn test_defaults_are_builtin() {
        for p in defaults() {
            assert!(p.builtin, "Default '{}' should be builtin", p.id);
        }
    }

    #[test]
    fn test_load_returns_defaults_when_no_file() {
        let dir = TempDir::new().unwrap();
        let perspectives = load(dir.path()).unwrap();
        assert_eq!(perspectives.len(), 7);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = TempDir::new().unwrap();
        let original = defaults();
        save(dir.path(), &original).unwrap();
        let loaded = load(dir.path()).unwrap();
        assert_eq!(original, loaded);
    }

    #[test]
    fn test_init_creates_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(FILENAME);
        assert!(!path.exists());
        init(dir.path()).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_init_idempotent() {
        let dir = TempDir::new().unwrap();
        init(dir.path()).unwrap();
        // Add a custom one
        add(dir.path(), Perspective::new("custom", "Custom", "hint")).unwrap();
        // Init again should NOT overwrite
        init(dir.path()).unwrap();
        let all = load(dir.path()).unwrap();
        assert_eq!(all.len(), 8); // 7 defaults + 1 custom
    }

    #[test]
    fn test_add_custom_perspective() {
        let dir = TempDir::new().unwrap();
        init(dir.path()).unwrap();
        add(
            dir.path(),
            Perspective::new("emotions", "Emotions", "Feelings, moods, reactions"),
        )
        .unwrap();
        let all = load(dir.path()).unwrap();
        assert_eq!(all.len(), 8);
        let custom = all.iter().find(|p| p.id == "emotions").unwrap();
        assert!(!custom.builtin);
    }

    #[test]
    fn test_add_duplicate_fails() {
        let dir = TempDir::new().unwrap();
        init(dir.path()).unwrap();
        let result = add(
            dir.path(),
            Perspective::new("decisions", "Dup", "duplicate"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_custom() {
        let dir = TempDir::new().unwrap();
        init(dir.path()).unwrap();
        add(dir.path(), Perspective::new("temp", "Temp", "temporary")).unwrap();
        assert_eq!(load(dir.path()).unwrap().len(), 8);
        remove(dir.path(), "temp").unwrap();
        assert_eq!(load(dir.path()).unwrap().len(), 7);
    }

    #[test]
    fn test_remove_builtin_fails() {
        let dir = TempDir::new().unwrap();
        init(dir.path()).unwrap();
        let result = remove(dir.path(), "decisions");
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_nonexistent_fails() {
        let dir = TempDir::new().unwrap();
        init(dir.path()).unwrap();
        let result = remove(dir.path(), "nope");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_existing() {
        let dir = TempDir::new().unwrap();
        init(dir.path()).unwrap();
        let p = get(dir.path(), "decisions").unwrap();
        assert!(p.is_some());
        assert_eq!(p.unwrap().name, "Decisions");
    }

    #[test]
    fn test_get_nonexistent() {
        let dir = TempDir::new().unwrap();
        init(dir.path()).unwrap();
        let p = get(dir.path(), "nope").unwrap();
        assert!(p.is_none());
    }

    #[test]
    fn test_validate_ids_ok() {
        let dir = TempDir::new().unwrap();
        init(dir.path()).unwrap();
        let ids = vec!["decisions".to_string(), "learnings".to_string()];
        validate_ids(dir.path(), &ids).unwrap();
    }

    #[test]
    fn test_validate_ids_unknown() {
        let dir = TempDir::new().unwrap();
        init(dir.path()).unwrap();
        let ids = vec!["decisions".to_string(), "nope".to_string()];
        assert!(validate_ids(dir.path(), &ids).is_err());
    }

    #[test]
    fn test_perspective_serde_roundtrip() {
        let p = Perspective::new("test", "Test", "test hint");
        let json = serde_json::to_string(&p).unwrap();
        let parsed: Perspective = serde_json::from_str(&json).unwrap();
        assert_eq!(p, parsed);
    }
}
