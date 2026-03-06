//! Parse and render `config.toml` from the memory branch.
//!
//! The config lives at `config.toml` in the root of the memory branch and is
//! read via git plumbing (`git show branch:config.toml`) or from the worktree.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{GitError, DEFAULT_BRANCH};

// ---------------------------------------------------------------------------
// Push mode
// ---------------------------------------------------------------------------

/// Push mode for sync operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PushMode {
    /// Push automatically after every commit.
    Always,
    /// Accumulate commits, open pull request for review (future).
    PullRequest,
    /// Auto-stage to local branch, wait for explicit `sync --push`.
    #[default]
    Review,
    /// No auto-staging; user explicitly runs `sync --stage <id>`.
    Manual,
    /// Git is never used.
    Off,
}

impl Serialize for PushMode {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            PushMode::Always => serializer.serialize_str("always"),
            PushMode::PullRequest => serializer.serialize_str("pull-request"),
            PushMode::Review => serializer.serialize_str("review"),
            PushMode::Manual => serializer.serialize_str("manual"),
            PushMode::Off => serializer.serialize_str("off"),
        }
    }
}

impl<'de> Deserialize<'de> for PushMode {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "always" => Ok(PushMode::Always),
            "auto" => {
                tracing::warn!("push_mode 'auto' is deprecated, use 'always' instead");
                Ok(PushMode::Always)
            }
            "pull-request" => Ok(PushMode::PullRequest),
            "review" => Ok(PushMode::Review),
            "manual" => Ok(PushMode::Manual),
            "off" => Ok(PushMode::Off),
            other => Err(serde::de::Error::custom(format!(
                "unknown push mode '{other}'. Valid: always, pull-request, review, manual, off"
            ))),
        }
    }
}

impl PushMode {
    /// Whether this mode auto-stages entries to the local git branch.
    pub fn auto_stages(&self) -> bool {
        matches!(
            self,
            PushMode::Always | PushMode::Review | PushMode::PullRequest
        )
    }

    /// Whether this mode pushes to remote immediately after staging.
    pub fn auto_pushes(&self) -> bool {
        matches!(self, PushMode::Always)
    }

    /// Whether git is used at all.
    pub fn uses_git(&self) -> bool {
        !matches!(self, PushMode::Off)
    }
}

impl std::fmt::Display for PushMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PushMode::Always => write!(f, "always"),
            PushMode::PullRequest => write!(f, "pull-request"),
            PushMode::Review => write!(f, "review"),
            PushMode::Manual => write!(f, "manual"),
            PushMode::Off => write!(f, "off"),
        }
    }
}

// ---------------------------------------------------------------------------
// Config structs
// ---------------------------------------------------------------------------

/// Configuration parsed from `config.toml` on the memory branch.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BranchConfig {
    #[serde(default)]
    pub memory: MemorySection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySection {
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
    #[serde(default)]
    pub sync: SyncConfig,
    #[serde(default)]
    pub company: Option<CompanyConfig>,
    #[serde(default)]
    pub prompts: PromptsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    #[serde(default)]
    pub push_mode: PushMode,
    #[serde(default = "default_branch")]
    pub branch: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompanyConfig {
    pub remote: String,
    pub branch: String,
    #[serde(default)]
    pub push_mode: PushMode,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptsConfig {
    pub recall: Option<String>,
    pub store: Option<String>,
    pub focus: Option<String>,
    pub think: Option<String>,
    pub priming: Option<String>,
    #[serde(default)]
    pub perspectives: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Default functions (required by serde `default = "..."`)
// ---------------------------------------------------------------------------

fn default_version() -> String {
    "1".to_string()
}

fn default_embedding_model() -> String {
    "bge-small-en-v1.5".to_string()
}

fn default_branch() -> String {
    DEFAULT_BRANCH.to_string()
}

// ---------------------------------------------------------------------------
// Default trait impls
// ---------------------------------------------------------------------------

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            push_mode: PushMode::default(),
            branch: default_branch(),
        }
    }
}

impl Default for MemorySection {
    fn default() -> Self {
        Self {
            version: default_version(),
            embedding_model: default_embedding_model(),
            sync: SyncConfig::default(),
            company: None,
            prompts: PromptsConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse TOML bytes into a [`BranchConfig`].
///
/// Missing fields are filled with their defaults.
pub fn parse(toml_bytes: &[u8]) -> Result<BranchConfig, GitError> {
    let text = std::str::from_utf8(toml_bytes).map_err(|e| GitError::CommandFailed {
        command: "parse config.toml".to_string(),
        stderr: format!("config.toml is not valid UTF-8: {e}"),
        exit_code: -1,
    })?;

    toml::from_str(text).map_err(|e| GitError::CommandFailed {
        command: "parse config.toml".to_string(),
        stderr: format!("config.toml parse error: {e}"),
        exit_code: -1,
    })
}

/// Return a [`BranchConfig`] populated entirely from defaults.
pub fn default_config() -> BranchConfig {
    BranchConfig::default()
}

/// Serialize a [`BranchConfig`] to a TOML string.
pub fn render(config: &BranchConfig) -> Result<String, GitError> {
    toml::to_string_pretty(config).map_err(|e| GitError::CommandFailed {
        command: "render config.toml".to_string(),
        stderr: format!("config.toml serialization error: {e}"),
        exit_code: -1,
    })
}

// ---------------------------------------------------------------------------
// BranchConfig convenience methods
// ---------------------------------------------------------------------------

impl BranchConfig {
    /// Effective push mode: company overrides sync when both are present.
    pub fn push_mode(&self) -> PushMode {
        self.memory
            .company
            .as_ref()
            .map(|c| c.push_mode)
            .unwrap_or(self.memory.sync.push_mode)
    }

    /// Name of the embedding model to use.
    pub fn embedding_model(&self) -> &str {
        &self.memory.embedding_model
    }

    /// Memory branch name from the sync section.
    pub fn branch(&self) -> &str {
        &self.memory.sync.branch
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const FULL_TOML: &str = r#"
[memory]
version = "1"
embedding_model = "bge-small-en-v1.5"

[memory.sync]
push_mode = "always"
branch = "veclayer-memory"

[memory.company]
remote = "git@github.com:acme/shared-memory.git"
branch = "veclayer-memory"
push_mode = "review"

[memory.prompts]
recall = "Search the team's shared knowledge base..."
store = "Always tag decisions with project name..."
priming = "You have access to the team's shared memory..."

[memory.prompts.perspectives]
decisions = "Architectural and design choices with rationale"
playbook = "Team-specific patterns and conventions"
"#;

    #[test]
    fn test_parse_full_config() {
        let config = parse(FULL_TOML.as_bytes()).unwrap();
        let b = &config.memory;

        assert_eq!(b.version, "1");
        assert_eq!(b.embedding_model, "bge-small-en-v1.5");
        assert_eq!(b.sync.push_mode, PushMode::Always);
        assert_eq!(b.sync.branch, "veclayer-memory");

        let company = b.company.as_ref().expect("company section missing");
        assert_eq!(company.remote, "git@github.com:acme/shared-memory.git");
        assert_eq!(company.branch, "veclayer-memory");
        assert_eq!(company.push_mode, PushMode::Review);

        let prompts = &b.prompts;
        assert_eq!(
            prompts.recall.as_deref(),
            Some("Search the team's shared knowledge base...")
        );
        assert_eq!(
            prompts.store.as_deref(),
            Some("Always tag decisions with project name...")
        );
        assert_eq!(
            prompts.priming.as_deref(),
            Some("You have access to the team's shared memory...")
        );
        assert_eq!(
            prompts.perspectives.get("decisions").map(String::as_str),
            Some("Architectural and design choices with rationale")
        );
        assert_eq!(
            prompts.perspectives.get("playbook").map(String::as_str),
            Some("Team-specific patterns and conventions")
        );
    }

    #[test]
    fn test_parse_minimal_config() {
        let toml = "[memory]\n";
        let config = parse(toml.as_bytes()).unwrap();
        let b = &config.memory;

        assert_eq!(b.version, "1");
        assert_eq!(b.embedding_model, "bge-small-en-v1.5");
        assert_eq!(b.sync.push_mode, PushMode::Review);
        assert_eq!(b.sync.branch, DEFAULT_BRANCH);
        assert!(b.company.is_none());
        assert!(b.prompts.recall.is_none());
        assert!(b.prompts.perspectives.is_empty());
    }

    #[test]
    fn test_default_config() {
        let config = default_config();
        let b = &config.memory;

        assert_eq!(b.version, "1");
        assert_eq!(b.embedding_model, "bge-small-en-v1.5");
        assert_eq!(b.sync.branch, DEFAULT_BRANCH);
        assert_eq!(b.sync.push_mode, PushMode::Review);
        assert!(b.company.is_none());
    }

    #[test]
    fn test_roundtrip() {
        let first = parse(FULL_TOML.as_bytes()).unwrap();
        let rendered = render(&first).unwrap();
        let second = parse(rendered.as_bytes()).unwrap();

        let b1 = &first.memory;
        let b2 = &second.memory;

        assert_eq!(b1.version, b2.version);
        assert_eq!(b1.embedding_model, b2.embedding_model);
        assert_eq!(b1.sync.push_mode, b2.sync.push_mode);
        assert_eq!(b1.sync.branch, b2.sync.branch);
        assert_eq!(
            b1.company.as_ref().map(|c| &c.remote),
            b2.company.as_ref().map(|c| &c.remote)
        );
        assert_eq!(
            b1.company.as_ref().map(|c| c.push_mode),
            b2.company.as_ref().map(|c| c.push_mode)
        );
        assert_eq!(b1.prompts.recall, b2.prompts.recall);
        assert_eq!(b1.prompts.perspectives, b2.prompts.perspectives);
    }

    #[test]
    fn test_push_mode_variants() {
        let cases = [
            (r#"push_mode = "always""#, PushMode::Always),
            (r#"push_mode = "auto""#, PushMode::Always), // backward compat
            (r#"push_mode = "pull-request""#, PushMode::PullRequest),
            (r#"push_mode = "review""#, PushMode::Review),
            (r#"push_mode = "manual""#, PushMode::Manual),
            (r#"push_mode = "off""#, PushMode::Off),
        ];

        for (snippet, expected) in cases {
            let toml =
                format!("[memory]\n[memory.sync]\n{snippet}\nbranch = \"veclayer-memory\"\n");
            let config = parse(toml.as_bytes())
                .unwrap_or_else(|e| panic!("failed to parse '{snippet}': {e}"));
            assert_eq!(
                config.memory.sync.push_mode, expected,
                "wrong push_mode for '{snippet}'"
            );
        }
    }

    #[test]
    fn test_push_mode_helpers() {
        assert!(PushMode::Always.auto_stages());
        assert!(PushMode::Always.auto_pushes());
        assert!(PushMode::Always.uses_git());

        assert!(PushMode::Review.auto_stages());
        assert!(!PushMode::Review.auto_pushes());
        assert!(PushMode::Review.uses_git());

        assert!(PushMode::PullRequest.auto_stages());
        assert!(!PushMode::PullRequest.auto_pushes());
        assert!(PushMode::PullRequest.uses_git());

        assert!(!PushMode::Manual.auto_stages());
        assert!(!PushMode::Manual.auto_pushes());
        assert!(PushMode::Manual.uses_git());

        assert!(!PushMode::Off.auto_stages());
        assert!(!PushMode::Off.auto_pushes());
        assert!(!PushMode::Off.uses_git());
    }

    #[test]
    fn test_push_mode_display() {
        assert_eq!(PushMode::Always.to_string(), "always");
        assert_eq!(PushMode::PullRequest.to_string(), "pull-request");
        assert_eq!(PushMode::Review.to_string(), "review");
        assert_eq!(PushMode::Manual.to_string(), "manual");
        assert_eq!(PushMode::Off.to_string(), "off");
    }
}
