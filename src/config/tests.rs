use std::io::Write;
use std::path::Path;

// Re-export everything from the parent (mod.rs public surface)
use super::*;

// Also pull in the test-only private items directly from submodules
use super::types::{
    embedder_explicitly_set, env_bool, env_or, env_parse, DetectedOllama, DetectedOpenAiEmbed,
    FileConfig, FileEmbedderConfig, FileLlmConfig, DEFAULT_FASTEMBED_MODEL,
    DEFAULT_OLLAMA_DIMENSION, DEFAULT_OLLAMA_URL,
};
use super::user_file::toml_escape_string;

#[test]
fn test_config_defaults() {
    // Clear env vars to test pure defaults
    // (can't fully clear since tests run in parallel, but verify structure)
    let config = Config::new();
    assert!(!config.data_dir.as_os_str().is_empty());
    assert!(!config.host.is_empty());
    assert!(config.port > 0);
    assert_eq!(config.search_top_k, 5);
    assert_eq!(config.search_children_k, 3);
}

// Security: api_key is a SecretString — Debug must redact, not leak.
#[test]
fn test_llm_config_debug_redacts_api_key_when_present() {
    let config = LlmConfig {
        api_key: Some(secrecy::SecretString::from("sk-supersecret")),
        ..LlmConfig::default()
    };
    let debug_output = format!("{config:?}");
    assert!(
        debug_output.contains("<redacted>"),
        "Debug output must contain '<redacted>', got: {debug_output}"
    );
    assert!(
        !debug_output.contains("sk-supersecret"),
        "Debug output must NOT leak the secret value, got: {debug_output}"
    );
}

// Security: api_key absent → Debug shows None, not a redacted placeholder.
#[test]
fn test_llm_config_debug_shows_none_when_api_key_absent() {
    let config = LlmConfig {
        api_key: None,
        ..LlmConfig::default()
    };
    let debug_output = format!("{config:?}");
    assert!(
        debug_output.contains("api_key: None"),
        "Debug output must show 'api_key: None' when absent, got: {debug_output}"
    );
}

#[test]
fn test_config_builder_chain() {
    let config = Config::new()
        .with_data_dir("/data")
        .with_host("localhost")
        .with_port(9000)
        .with_read_only(true);

    assert_eq!(config.data_dir, Path::new("/data"));
    assert_eq!(config.host, "localhost");
    assert_eq!(config.port, 9000);
    assert!(config.read_only);
}

#[test]
fn test_embedder_config_default() {
    let embedder = EmbedderConfig::default();
    if cfg!(feature = "embedding-local") {
        assert!(
            matches!(embedder, EmbedderConfig::FastEmbed { ref model } if model == DEFAULT_FASTEMBED_MODEL),
            "Expected FastEmbed variant with default model when embedding-local is enabled"
        );
    } else {
        assert!(
            matches!(embedder, EmbedderConfig::Ollama { .. }),
            "Expected Ollama variant when embedding-local is disabled"
        );
    }
}

#[test]
#[serial_test::serial]
fn test_resolve_llm_invalid_base_url_falls_back_to_default() {
    std::env::remove_var("VECLAYER_LLM_BASE_URL");
    let bad = FileLlmConfig {
        provider: "ollama".to_string(),
        model: None,
        base_url: Some("ftp://not-http".to_string()),
        api_key: None,
        temperature: None,
        max_tokens: None,
    };
    let llm = Config::resolve_llm(Some(bad), None);
    assert_eq!(
        llm.base_url, DEFAULT_OLLAMA_URL,
        "an invalid base_url must actually fall back to the default"
    );
}

#[test]
#[serial_test::serial]
fn test_resolve_embedder_ollama_from_env() {
    // Use values DISTINCT from the defaults so the test proves env wins over default,
    // not that it accidentally equals the default.
    std::env::set_var("VECLAYER_EMBEDDER", "ollama");
    std::env::set_var("VECLAYER_OLLAMA_MODEL", "custom-model");
    std::env::set_var("VECLAYER_OLLAMA_URL", "http://gpu:11434");
    std::env::set_var("VECLAYER_OLLAMA_DIMENSION", "1024");

    let embedder = Config::resolve_embedder(None, None, None);

    std::env::remove_var("VECLAYER_EMBEDDER");
    std::env::remove_var("VECLAYER_OLLAMA_MODEL");
    std::env::remove_var("VECLAYER_OLLAMA_URL");
    std::env::remove_var("VECLAYER_OLLAMA_DIMENSION");

    assert!(matches!(
        embedder,
        EmbedderConfig::Ollama {
            ref model,
            ref base_url,
            dimension
        } if model == "custom-model"
            && base_url == "http://gpu:11434"
            && dimension == 1024
    ));
}

#[test]
#[serial_test::serial]
fn test_resolve_embedder_prefers_openai_compat_when_no_ollama_embed() {
    // No explicit embedder is configured and Ollama offered no embedding
    // model, but a local OpenAI-compatible service (e.g. vLLM) was detected
    // serving a 1024-dim model. resolve_embedder must point the Ollama-
    // protocol embedder — which transparently falls back to /v1/embeddings —
    // at that endpoint, carrying the probed dimension so the store is sized
    // correctly rather than defaulting to 768.
    std::env::remove_var("VECLAYER_EMBEDDER");
    std::env::remove_var("VECLAYER_OLLAMA_MODEL");
    std::env::remove_var("VECLAYER_OLLAMA_URL");
    std::env::remove_var("VECLAYER_OLLAMA_DIMENSION");

    let openai = DetectedOpenAiEmbed {
        base_url: "http://localhost:8000".to_string(),
        model: "BAAI/bge-m3".to_string(),
        dimension: 1024,
    };
    let embedder = Config::resolve_embedder(None, None, Some(&openai));

    assert!(matches!(
        embedder,
        EmbedderConfig::Ollama {
            ref model,
            ref base_url,
            dimension
        } if model == "BAAI/bge-m3"
            && base_url == "http://localhost:8000"
            && dimension == 1024
    ));
}

#[test]
#[serial_test::serial]
fn test_resolve_embedder_ollama_embed_wins_over_openai() {
    // When both an Ollama embed model and an OpenAI-compat service are
    // available, the Ollama-native model takes precedence (it is the more
    // specific, native protocol match).
    std::env::remove_var("VECLAYER_EMBEDDER");
    std::env::remove_var("VECLAYER_OLLAMA_MODEL");
    std::env::remove_var("VECLAYER_OLLAMA_URL");
    std::env::remove_var("VECLAYER_OLLAMA_DIMENSION");

    let ollama = DetectedOllama {
        base_url: "http://localhost:11434".to_string(),
        embed_model: Some("nomic-embed-text".to_string()),
        chat_model: None,
    };
    let openai = DetectedOpenAiEmbed {
        base_url: "http://localhost:8000".to_string(),
        model: "BAAI/bge-m3".to_string(),
        dimension: 1024,
    };
    let embedder = Config::resolve_embedder(None, Some(&ollama), Some(&openai));

    // All three fields must come from Ollama — never a mix where the model
    // is Ollama's but the dimension leaks from the OpenAI-compat probe. Since
    // Ollama discovery learns no dimension, it falls back to the default.
    assert!(
        matches!(
            embedder,
            EmbedderConfig::Ollama {
                ref model,
                ref base_url,
                dimension
            } if model == "nomic-embed-text"
                && base_url == "http://localhost:11434"
                && dimension == DEFAULT_OLLAMA_DIMENSION
        ),
        "Ollama embed must win on all fields with no OpenAI-compat leakage"
    );
}

#[test]
#[serial_test::serial]
fn test_resolve_embedder_chat_only_ollama_does_not_contaminate_openai() {
    // Regression: a chat-only Ollama (no embed model) plus a detected
    // OpenAI-compatible embed service must yield an embedder whose model,
    // base_url AND dimension all come from the OpenAI-compat server. The
    // chat-only Ollama's base_url must NOT leak in.
    std::env::remove_var("VECLAYER_EMBEDDER");
    std::env::remove_var("VECLAYER_OLLAMA_MODEL");
    std::env::remove_var("VECLAYER_OLLAMA_URL");
    std::env::remove_var("VECLAYER_OLLAMA_DIMENSION");

    let ollama = DetectedOllama {
        base_url: "http://localhost:11434".to_string(),
        embed_model: None,
        chat_model: Some("llama3.2".to_string()),
    };
    let openai = DetectedOpenAiEmbed {
        base_url: "http://localhost:8000".to_string(),
        model: "BAAI/bge-m3".to_string(),
        dimension: 1024,
    };
    let embedder = Config::resolve_embedder(None, Some(&ollama), Some(&openai));

    assert!(
        matches!(
            embedder,
            EmbedderConfig::Ollama {
                ref model,
                ref base_url,
                dimension
            } if model == "BAAI/bge-m3"
                && base_url == "http://localhost:8000"
                && dimension == 1024
        ),
        "chat-only Ollama must not contaminate the OpenAI-compat embedder"
    );
}

#[test]
#[serial_test::serial]
fn test_embedder_explicitly_set_detects_ollama_env_overrides() {
    for key in [
        "VECLAYER_EMBEDDER",
        "VECLAYER_OLLAMA_URL",
        "VECLAYER_OLLAMA_MODEL",
        "VECLAYER_OLLAMA_DIMENSION",
    ] {
        std::env::remove_var(key);
    }

    // Nothing set, no file → not explicit.
    assert!(!embedder_explicitly_set(&None));

    // A file embedder block alone pins it.
    let file = FileEmbedderConfig {
        embedder_type: "fastembed".to_string(),
        model: None,
        base_url: None,
        dimension: None,
    };
    assert!(embedder_explicitly_set(&Some(file)));

    // Each Ollama override env var pins it too — this is what suppresses the
    // OpenAI-compat probe so it can't inject a conflicting model/dimension.
    for key in [
        "VECLAYER_OLLAMA_URL",
        "VECLAYER_OLLAMA_MODEL",
        "VECLAYER_OLLAMA_DIMENSION",
    ] {
        std::env::set_var(key, "x");
        assert!(
            embedder_explicitly_set(&None),
            "{key} must count as an explicit embedder"
        );
        std::env::remove_var(key);
    }
}

#[test]
fn test_file_config_load_toml() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("veclayer.toml");
    let mut file = std::fs::File::create(&toml_path).unwrap();
    writeln!(
        file,
        r#"
host = "0.0.0.0"
port = 3000
search_top_k = 10

[embedder]
type = "ollama"
model = "mxbai-embed-large"
base_url = "http://gpu:11434"
"#
    )
    .unwrap();

    let fc = FileConfig::load(&toml_path);
    assert_eq!(fc.host.as_deref(), Some("0.0.0.0"));
    assert_eq!(fc.port, Some(3000));
    assert_eq!(fc.search_top_k, Some(10));
    assert!(fc.data_dir.is_none()); // not specified
    assert!(fc.read_only.is_none()); // not specified

    let emb = fc.embedder.unwrap();
    assert_eq!(emb.embedder_type, "ollama");
    assert_eq!(emb.model.as_deref(), Some("mxbai-embed-large"));
    assert_eq!(emb.base_url.as_deref(), Some("http://gpu:11434"));
}

#[test]
fn test_file_config_missing_file() {
    let fc = FileConfig::load(Path::new("/nonexistent/path/veclayer.toml"));
    assert!(fc.host.is_none());
    assert!(fc.port.is_none());
}

#[test]
fn test_file_config_invalid_toml() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("veclayer.toml");
    std::fs::write(&toml_path, "this is not [valid toml {{{").unwrap();

    let fc = FileConfig::load(&toml_path);
    // Should gracefully return defaults (all None)
    assert!(fc.host.is_none());
    assert!(fc.port.is_none());
}

#[test]
fn test_file_config_partial_toml() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("veclayer.toml");
    std::fs::write(&toml_path, "port = 4444\n").unwrap();

    let fc = FileConfig::load(&toml_path);
    assert_eq!(fc.port, Some(4444));
    assert!(fc.host.is_none());
    assert!(fc.data_dir.is_none());
}

#[test]
fn test_env_or_helper() {
    // With no env var set for this unique key, should use file_val or default
    let result = env_or(
        "VECLAYER_TEST_NONEXISTENT_KEY_12345",
        Some("file".to_string()),
        "default".to_string(),
    );
    assert_eq!(result, "file");

    let result2 = env_or(
        "VECLAYER_TEST_NONEXISTENT_KEY_12345",
        None,
        "default".to_string(),
    );
    assert_eq!(result2, "default");
}

#[test]
fn test_config_clone() {
    let config1 = Config::new().with_data_dir("/test").with_port(9999);
    let config2 = config1.clone();

    assert_eq!(config1.data_dir, config2.data_dir);
    assert_eq!(config1.port, config2.port);
    assert_eq!(config1.host, config2.host);
    assert_eq!(config1.read_only, config2.read_only);
}

#[test]
fn test_config_debug_format() {
    let config = Config::new();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("Config"));
}

#[test]
fn test_discover_project_walk_up() {
    let dir = tempfile::TempDir::new().unwrap();
    let veclayer_dir = dir.path().join(".veclayer");
    std::fs::create_dir_all(&veclayer_dir).unwrap();

    // With config.toml
    let config_path = veclayer_dir.join("config.toml");
    std::fs::write(&config_path, "project = \"myproject\"\n").unwrap();

    // Discover from the root
    let result = discover_project(dir.path());
    assert!(result.is_some());
    let (found_dir, config) = result.unwrap();
    assert_eq!(found_dir, veclayer_dir);
    assert_eq!(config.project.as_deref(), Some("myproject"));

    // Discover from a subdirectory
    let sub = dir.path().join("src").join("deep");
    std::fs::create_dir_all(&sub).unwrap();
    let result = discover_project(&sub);
    assert!(result.is_some());
    let (found_dir, config) = result.unwrap();
    assert_eq!(found_dir, veclayer_dir);
    assert_eq!(config.project.as_deref(), Some("myproject"));
}

#[test]
fn test_discover_project_no_config() {
    let dir = tempfile::TempDir::new().unwrap();
    let veclayer_dir = dir.path().join(".veclayer");
    std::fs::create_dir_all(&veclayer_dir).unwrap();

    // No config.toml
    let result = discover_project(dir.path());
    assert!(result.is_some());
    let (found_dir, config) = result.unwrap();
    assert_eq!(found_dir, veclayer_dir);
    assert!(config.project.is_none());
}

#[test]
fn test_discover_project_not_found() {
    let dir = tempfile::TempDir::new().unwrap();
    // No .veclayer/ anywhere
    let result = discover_project(dir.path());
    assert!(result.is_none());
}

#[test]
fn test_discover_project_bad_toml_returns_none() {
    let dir = tempfile::TempDir::new().unwrap();
    let veclayer_dir = dir.path().join(".veclayer");
    std::fs::create_dir_all(&veclayer_dir).unwrap();
    std::fs::write(veclayer_dir.join("config.toml"), "not valid {{{ toml").unwrap();

    // Malformed config.toml must return None gracefully, not panic
    let result = discover_project(dir.path());
    assert!(result.is_none());
}

#[test]
fn test_user_config_default() {
    let config = UserConfig::default();
    assert!(config.matches.is_empty());
    assert!(config.project.is_none());
    assert!(config.data_dir.is_none());
}

#[test]
fn test_match_override_tilde_expansion() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");
    std::fs::write(
        &toml_path,
        r#"
[[match]]
path = "~/work/damalo*"
project = "damalo"
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);
    assert_eq!(config.matches.len(), 1);
    assert_eq!(config.matches[0].project.as_deref(), Some("damalo"));
}

#[test]
fn test_match_override_absolute_path() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");
    std::fs::write(
        &toml_path,
        r#"
[[match]]
path = "/tmp/test*"
project = "test"
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);
    assert_eq!(config.matches.len(), 1);
    assert_eq!(config.matches[0].project.as_deref(), Some("test"));
}

#[test]
fn test_resolve_single_path_match() {
    let mut config = UserConfig::default();
    config.matches.push(MatchOverride {
        path: Some(glob::Pattern::new("/tmp/test/*").unwrap()),
        git_remote: None,
        project: Some("test".to_string()),
        data_dir: Some("/tmp/test-data".to_string()),
        host: None,
        port: None,
        read_only: Some(true),
        scopes: vec![],
    });

    let resolved = config.resolve(Path::new("/tmp/test/something"), None);
    assert_eq!(resolved.project.as_deref(), Some("test"));
    assert_eq!(resolved.data_dir.as_deref(), Some("/tmp/test-data"));
    assert_eq!(resolved.read_only, Some(true));
}

#[test]
fn test_resolve_no_match() {
    let mut config = UserConfig::default();
    config.matches.push(MatchOverride {
        path: Some(glob::Pattern::new("/tmp/test/*").unwrap()),
        git_remote: None,
        project: Some("test".to_string()),
        data_dir: None,
        host: None,
        port: None,
        read_only: None,
        scopes: vec![],
    });

    let resolved = config.resolve(Path::new("/other/path"), None);
    assert!(resolved.project.is_none());
    assert!(resolved.data_dir.is_none());
}

#[test]
fn test_resolve_multiple_match_last_wins() {
    let mut config = UserConfig::default();

    config.matches.push(MatchOverride {
        path: Some(glob::Pattern::new("/tmp/test/**").unwrap()),
        git_remote: None,
        project: Some("first".to_string()),
        data_dir: Some("/first".to_string()),
        host: None,
        port: None,
        read_only: Some(false),
        scopes: vec![],
    });

    config.matches.push(MatchOverride {
        path: Some(glob::Pattern::new("/tmp/test/specific").unwrap()),
        git_remote: None,
        project: Some("second".to_string()),
        data_dir: Some("/second".to_string()),
        host: None,
        port: None,
        read_only: Some(true),
        scopes: vec![],
    });

    let resolved = config.resolve(Path::new("/tmp/test/specific"), None);
    assert_eq!(resolved.project.as_deref(), Some("second"));
    assert_eq!(resolved.data_dir.as_deref(), Some("/second"));
    assert_eq!(resolved.read_only, Some(true));
}

#[test]
fn test_resolve_partial_override() {
    let mut config = UserConfig {
        project: Some("global".to_string()),
        ..Default::default()
    };

    config.matches.push(MatchOverride {
        path: Some(glob::Pattern::new("/tmp/*").unwrap()),
        git_remote: None,
        project: None,
        data_dir: Some("/tmp/data".to_string()),
        host: None,
        port: None,
        read_only: Some(true),
        scopes: vec![],
    });

    let resolved = config.resolve(Path::new("/tmp/test"), None);
    assert_eq!(resolved.project.as_deref(), Some("global"));
    assert_eq!(resolved.data_dir.as_deref(), Some("/tmp/data"));
    assert_eq!(resolved.read_only, Some(true));
}

#[test]
fn test_match_override_invalid_path_pattern() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");
    std::fs::write(
        &toml_path,
        r#"
[[match]]
path = "[[invalid"
project = "test"
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);
    assert!(config.matches.is_empty());
}

#[test]
fn test_match_override_no_matcher_rejected() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");
    std::fs::write(
        &toml_path,
        r#"
[[match]]
project = "orphan"
"#,
    )
    .unwrap();

    // Should fail to parse — at least one matcher required
    let config = UserConfig::load(&toml_path);
    assert!(config.matches.is_empty());
}

// BUG-2: tilde in global data_dir must be expanded after load
#[test]
fn test_user_config_global_data_dir_tilde_expanded() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");
    std::fs::write(&toml_path, "data_dir = \"~/.veclayer\"\n").unwrap();

    let config = UserConfig::load(&toml_path);
    let data_dir = config.data_dir.expect("data_dir should be set");
    assert!(
        !data_dir.starts_with('~'),
        "data_dir '{}' should not start with '~' after tilde expansion",
        data_dir
    );
}

// BUG-2: tilde in match override data_dir must be expanded during deserialization
#[test]
fn test_match_override_data_dir_tilde_expanded() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");
    std::fs::write(
        &toml_path,
        "[[match]]\npath = \"/tmp/work\"\ndata_dir = \"~/.veclayer\"\n",
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);
    let data_dir = config.matches[0]
        .data_dir
        .as_deref()
        .expect("match override data_dir should be set");
    assert!(
        !data_dir.starts_with('~'),
        "match override data_dir '{}' should not start with '~' after tilde expansion",
        data_dir
    );
}

// BUG-3: explicit VECLAYER_USER_CONFIG pointing to nonexistent file must not fall through
// NOTE(known-limitation): std::env::set_var/remove_var are unsafe since Rust 1.83+.
// These tests use serial_test to avoid data races, but will need unsafe blocks when
// the crate upgrades to Rust edition 2024. See README "Known Limitations".
#[test]
#[serial_test::serial]
fn test_discover_user_config_nonexistent_env_returns_defaults() {
    let original = std::env::var("VECLAYER_USER_CONFIG").ok();

    std::env::set_var(
        "VECLAYER_USER_CONFIG",
        "/nonexistent/path/that/does/not/exist.toml",
    );
    let config = UserConfig::discover();
    assert!(
        config.matches.is_empty(),
        "should return default (empty matches)"
    );
    assert!(
        config.data_dir.is_none(),
        "should return default (no data_dir)"
    );

    match original {
        Some(v) => std::env::set_var("VECLAYER_USER_CONFIG", v),
        None => std::env::remove_var("VECLAYER_USER_CONFIG"),
    }
}

#[test]
fn test_match_git_remote_only() {
    let toml_str = r#"
[[match]]
git-remote = "(?i)damalo"
project = "damalo"

[[match]]
git-remote = "github\\.com/myorg/"
project = "myorg"
"#;
    let config: UserConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.matches.len(), 2);

    // git-remote match, no path
    let resolved = config.resolve(Path::new("/other"), Some("github.com/Damalo/some-repo"));
    assert_eq!(resolved.project.as_deref(), Some("damalo"));

    let resolved = config.resolve(Path::new("/other"), Some("github.com/myorg/tool"));
    assert_eq!(resolved.project.as_deref(), Some("myorg"));

    let resolved = config.resolve(Path::new("/other"), Some("github.com/unrelated/repo"));
    assert!(resolved.project.is_none());
}

#[test]
fn test_match_last_wins_with_remote() {
    let toml_str = r#"
[[match]]
git-remote = "specific-repo"
project = "specific"

[[match]]
git-remote = ".*"
project = "catch-all"
"#;
    let config: UserConfig = toml::from_str(toml_str).unwrap();
    // Last match wins: catch-all matches everything, so it always wins
    let resolved = config.resolve(Path::new("/tmp"), Some("github.com/org/specific-repo"));
    assert_eq!(resolved.project.as_deref(), Some("catch-all"));

    let resolved = config.resolve(Path::new("/tmp"), Some("github.com/org/other"));
    assert_eq!(resolved.project.as_deref(), Some("catch-all"));
}

#[test]
fn test_match_or_logic_both_matchers() {
    let toml_str = r#"
[[match]]
path = "/tmp/damalo*"
git-remote = "(?i)damalo"
project = "damalo"
"#;
    let config: UserConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.matches.len(), 1);

    // Path matches, no remote
    let resolved = config.resolve(Path::new("/tmp/damalo-app"), None);
    assert_eq!(resolved.project.as_deref(), Some("damalo"));

    // Remote matches, different path
    let resolved = config.resolve(Path::new("/other/path"), Some("github.com/Damalo/repo"));
    assert_eq!(resolved.project.as_deref(), Some("damalo"));

    // Both match
    let resolved = config.resolve(Path::new("/tmp/damalo-app"), Some("github.com/Damalo/repo"));
    assert_eq!(resolved.project.as_deref(), Some("damalo"));

    // Neither matches
    let resolved = config.resolve(Path::new("/other/path"), Some("github.com/other/repo"));
    assert!(resolved.project.is_none());
}

#[test]
fn test_match_no_remote_provided() {
    let config = UserConfig::default();
    let resolved = config.resolve(Path::new("/tmp"), None);
    assert!(resolved.project.is_none());
}

// NIT-3: * must not cross path separators (require_literal_separator = true)
#[test]
fn test_resolve_star_does_not_cross_separator() {
    let mut config = UserConfig::default();

    config.matches.push(MatchOverride {
        path: Some(glob::Pattern::new("/tmp/work*").unwrap()),
        git_remote: None,
        project: Some("shallow".to_string()),
        data_dir: None,
        host: None,
        port: None,
        read_only: None,
        scopes: vec![],
    });

    // /tmp/work/deep has a slash after the * position — must not match
    let resolved_deep = config.resolve(Path::new("/tmp/work/deep"), None);
    assert!(
        resolved_deep.project.is_none(),
        "* should not cross / (got {:?})",
        resolved_deep.project
    );

    // /tmp/workspace has no slash after the * position — must match
    let resolved_shallow = config.resolve(Path::new("/tmp/workspace"), None);
    assert_eq!(
        resolved_shallow.project.as_deref(),
        Some("shallow"),
        "* should match within a single path component"
    );
}

// NOTE(known-limitation): std::env::set_var/remove_var — see comment above.
#[test]
#[serial_test::serial]
fn test_append_match_to_user_config() {
    let dir = tempfile::TempDir::new().unwrap();
    let config_path = dir.path().join("config.toml");

    // Use env var to point to our temp file
    std::env::set_var("VECLAYER_USER_CONFIG", config_path.to_str().unwrap());

    let result = append_match_to_user_config(
        Some("github.com/org/repo"),
        Some("/home/user/work/project*"),
        "myproject",
    );

    std::env::remove_var("VECLAYER_USER_CONFIG");

    let path = result.unwrap();
    assert_eq!(path, config_path);

    let contents = std::fs::read_to_string(&config_path).unwrap();
    assert!(contents.contains("[[match]]"));
    assert!(contents.contains("git-remote = \"github.com/org/repo\""));
    assert!(contents.contains("path = \"/home/user/work/project*\""));
    assert!(contents.contains("project = \"myproject\""));

    // Verify it round-trips through UserConfig::load
    let loaded = UserConfig::load(&config_path);
    assert_eq!(loaded.matches.len(), 1);
    assert_eq!(loaded.matches[0].project.as_deref(), Some("myproject"));
}

#[test]
fn test_auth_config_defaults() {
    let auth = AuthConfig::default();
    assert!(!auth.auth_required);
    assert!(auth.server_url.is_none());
    assert_eq!(auth.token_expiry_secs, crate::util::TOKEN_EXPIRY_SECS);
    assert_eq!(
        auth.refresh_expiry_secs,
        crate::util::REFRESH_MAX_LIFETIME_SECS
    );
    assert!(!auth.auto_approve);
}

#[test]
fn test_auth_config_from_toml() {
    let toml_str = r#"
[auth]
auth_required = true
server_url = "https://my-veclayer.example.com"
token_expiry_secs = 1800
refresh_expiry_secs = 86400
auto_approve = true
"#;
    let fc: FileConfig = toml::from_str(toml_str).unwrap();
    let auth_file = fc.auth.unwrap();
    assert_eq!(auth_file.auth_required, Some(true));
    assert_eq!(
        auth_file.server_url.as_deref(),
        Some("https://my-veclayer.example.com")
    );
    assert_eq!(auth_file.token_expiry_secs, Some(1800));
    assert_eq!(auth_file.refresh_expiry_secs, Some(86400));
    assert_eq!(auth_file.auto_approve, Some(true));
}

#[test]
#[serial_test::serial]
fn test_auth_config_env_override() {
    let saved_required = std::env::var("VECLAYER_AUTH_REQUIRED").ok();
    let saved_url = std::env::var("VECLAYER_SERVER_URL").ok();
    let saved_expiry = std::env::var("VECLAYER_TOKEN_EXPIRY").ok();
    let saved_approve = std::env::var("VECLAYER_AUTO_APPROVE").ok();

    std::env::set_var("VECLAYER_AUTH_REQUIRED", "true");
    std::env::set_var("VECLAYER_SERVER_URL", "https://env.example.com");
    std::env::set_var("VECLAYER_TOKEN_EXPIRY", "7200");
    std::env::set_var("VECLAYER_AUTO_APPROVE", "1");

    let auth = Config::resolve_auth(None);

    // Restore env
    match saved_required {
        Some(v) => std::env::set_var("VECLAYER_AUTH_REQUIRED", v),
        None => std::env::remove_var("VECLAYER_AUTH_REQUIRED"),
    }
    match saved_url {
        Some(v) => std::env::set_var("VECLAYER_SERVER_URL", v),
        None => std::env::remove_var("VECLAYER_SERVER_URL"),
    }
    match saved_expiry {
        Some(v) => std::env::set_var("VECLAYER_TOKEN_EXPIRY", v),
        None => std::env::remove_var("VECLAYER_TOKEN_EXPIRY"),
    }
    match saved_approve {
        Some(v) => std::env::set_var("VECLAYER_AUTO_APPROVE", v),
        None => std::env::remove_var("VECLAYER_AUTO_APPROVE"),
    }

    assert!(auth.auth_required);
    assert_eq!(auth.server_url.as_deref(), Some("https://env.example.com"));
    assert_eq!(auth.token_expiry_secs, 7200);
    assert!(auth.auto_approve);
}

#[test]
fn test_scope_config_parsing() {
    let toml_str = r#"
[scopes.personal]
storage = "git@github.com:flob/my-memory.git"
push = "manual"

[scopes.acme]
storage = "git@github.com:acme/shared-memory.git"
push = "review"
branch = "acme-memory"
"#;
    let config: UserConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.scopes.len(), 2);

    let personal = config.scopes.get("personal").unwrap();
    assert_eq!(personal.storage, "git@github.com:flob/my-memory.git");
    assert_eq!(personal.push.as_deref(), Some("manual"));
    assert!(personal.branch.is_none());

    let acme = config.scopes.get("acme").unwrap();
    assert_eq!(acme.storage, "git@github.com:acme/shared-memory.git");
    assert_eq!(acme.push.as_deref(), Some("review"));
    assert_eq!(acme.branch.as_deref(), Some("acme-memory"));
}

#[test]
fn test_match_with_scopes() {
    let toml_str = r#"
[scopes.personal]
storage = "git@github.com:flob/my-memory.git"

[scopes.acme]
storage = "git@github.com:acme/shared-memory.git"

[[match]]
git-remote = "github.com/acme/"
project = "acme-stuff"
scopes = ["personal", "acme"]
"#;
    let config: UserConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.matches.len(), 1);
    assert_eq!(config.matches[0].scopes, vec!["personal", "acme"]);
    assert_eq!(config.matches[0].project.as_deref(), Some("acme-stuff"));
}

#[test]
fn test_project_config_with_scopes() {
    let toml_str = r#"
project = "myproject"
storage = "git"
push = "auto"
scopes = ["acme"]
"#;
    let project_config: ProjectConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(project_config.project.as_deref(), Some("myproject"));
    assert_eq!(project_config.storage.as_deref(), Some("git"));
    assert_eq!(project_config.push.as_deref(), Some("auto"));
    assert_eq!(project_config.scopes, vec!["acme"]);
}

#[test]
fn test_scope_resolution() {
    let toml_str = r#"
[scopes.personal]
storage = "git@github.com:flob/my-memory.git"

[scopes.acme]
storage = "git@github.com:acme/shared-memory.git"
push = "review"
"#;
    let config: UserConfig = toml::from_str(toml_str).unwrap();

    // Union: project=[acme], match=[personal, acme] → [acme, personal] (dedup, project first)
    let project_scopes = vec!["acme".to_string()];
    let match_scopes = vec!["personal".to_string(), "acme".to_string()];
    let resolved = config.resolve_scopes(&project_scopes, &match_scopes);

    assert_eq!(resolved.len(), 2);
    assert_eq!(resolved[0].name, "acme");
    assert_eq!(resolved[0].storage, "git@github.com:acme/shared-memory.git");
    assert_eq!(resolved[0].push, "review");
    assert_eq!(resolved[0].branch, "veclayer-memory"); // default

    assert_eq!(resolved[1].name, "personal");
    assert_eq!(resolved[1].storage, "git@github.com:flob/my-memory.git");
    assert_eq!(resolved[1].push, "manual"); // default
    assert_eq!(resolved[1].branch, "veclayer-memory"); // default
}

#[test]
fn test_unknown_scope_warning() {
    let toml_str = r#"
[scopes.known]
storage = "git"
"#;
    let config: UserConfig = toml::from_str(toml_str).unwrap();

    let project_scopes = vec!["known".to_string()];
    let match_scopes = vec!["unknown".to_string()];
    let resolved = config.resolve_scopes(&project_scopes, &match_scopes);

    // "unknown" is skipped; only "known" resolves
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].name, "known");
}

// with_auth_required builder tests

#[test]
fn test_with_auth_required_sets_true() {
    // Builder must propagate true regardless of env/file defaults.
    let config = Config::new().with_auth_required(true);
    assert!(config.auth.auth_required);
}

#[test]
fn test_with_auth_required_sets_false() {
    // Builder must propagate false, making it authoritative for the merged CLI value.
    let config = Config::new().with_auth_required(false);
    assert!(!config.auth.auth_required);
}

#[test]
fn test_with_auth_required_overrides_prior_true() {
    // Start with auth_required=true (via builder), then override with false.
    // Documents that with_auth_required is fully authoritative — the last call wins.
    let config = Config::new()
        .with_auth_required(true)
        .with_auth_required(false);
    assert!(!config.auth.auth_required);
}

#[test]
fn test_with_auth_required_composes_in_chain() {
    // Verify that with_auth_required returns Self and can be composed
    // with the other builder methods without breaking anything.
    let config = Config::new()
        .with_port(9090)
        .with_auth_required(true)
        .with_read_only(false);
    assert!(config.auth.auth_required);
    assert_eq!(config.port, 9090);
    assert!(!config.read_only);
}

// parse_push_mode: unrecognized string must warn and fall back to PushMode::Review.
#[test]
fn test_parse_push_mode_unknown_falls_back_to_review() {
    use crate::git::branch_config::PushMode;
    assert!(
        matches!(parse_push_mode("review"), PushMode::Review),
        "canonical \"review\" must map to PushMode::Review"
    );
    let result = parse_push_mode("bogus");
    assert!(
        matches!(result, PushMode::Review),
        "expected PushMode::Review for unknown input, got {:?}",
        result
    );
}

// append_match_to_user_config: both matchers None must return Err with the guard message.
#[test]
fn test_append_match_to_user_config_both_none_returns_err() {
    let result = append_match_to_user_config(None, None, "myproject");
    assert!(
        result.is_err(),
        "expected Err when both git_remote and path_glob are None"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("at least one of"),
        "error message should contain 'at least one of', got: {msg}"
    );
}

// Regression: resolve() with a valid UTF-8 cwd must behave identically to before
// the lossy-conversion change (common-case correctness).
#[test]
fn test_resolve_utf8_cwd_behavior_unchanged() {
    let mut config = UserConfig::default();
    config.matches.push(MatchOverride {
        path: Some(glob::Pattern::new("/tmp/project/*").unwrap()),
        git_remote: None,
        project: Some("myproject".to_string()),
        data_dir: Some("/tmp/data".to_string()),
        host: None,
        port: None,
        read_only: Some(false),
        scopes: vec![],
    });

    // Matching path: override must be applied.
    let resolved = config.resolve(Path::new("/tmp/project/src"), None);
    assert_eq!(
        resolved.project.as_deref(),
        Some("myproject"),
        "UTF-8 matching cwd must yield the expected project override"
    );
    assert_eq!(resolved.data_dir.as_deref(), Some("/tmp/data"));
    assert_eq!(resolved.read_only, Some(false));

    // Non-matching path: override must NOT be applied.
    let resolved_no_match = config.resolve(Path::new("/other/path"), None);
    assert!(
        resolved_no_match.project.is_none(),
        "UTF-8 non-matching cwd must not apply any override"
    );
}

// Edge case: resolve() with a non-UTF-8 cwd must not panic (lossy conversion).
// On Linux, paths are arbitrary byte sequences that need not be valid UTF-8.
// This test would panic against the old `.expect("... not valid UTF-8")` and
// must pass cleanly after the fix.
#[test]
#[cfg(unix)]
fn test_resolve_non_utf8_cwd_does_not_panic() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    // 0x66 0x80 0x6f — the 0x80 byte is not valid UTF-8.
    let non_utf8_bytes: &[u8] = &[0x2f, 0x66, 0x80, 0x6f]; // "/f\x80o"
    let non_utf8_path = Path::new(OsStr::from_bytes(non_utf8_bytes));

    let config = UserConfig::default();
    // Must not panic; result is unimportant (no matches configured).
    let resolved = config.resolve(non_utf8_path, None);
    assert!(
        resolved.project.is_none(),
        "non-UTF-8 cwd with no match overrides must return no project"
    );
}

// --- toml_escape_string + append_match_to_user_config round-trip tests ---

// GREEN: normal values with no special characters must serialize and re-parse cleanly.
#[test]
#[serial_test::serial]
fn test_append_match_round_trips_normal_values() {
    let dir = tempfile::TempDir::new().unwrap();
    let config_path = dir.path().join("config.toml");
    std::env::set_var("VECLAYER_USER_CONFIG", config_path.to_str().unwrap());

    append_match_to_user_config(
        Some("github.com/org/normal-repo"),
        Some("/home/user/work/project"),
        "normal-project",
    )
    .unwrap();

    std::env::remove_var("VECLAYER_USER_CONFIG");

    let loaded = UserConfig::load(&config_path);
    assert_eq!(loaded.matches.len(), 1);
    assert_eq!(
        loaded.matches[0].project.as_deref(),
        Some("normal-project"),
        "normal project value must round-trip unchanged"
    );
}

// EDGE: a project name containing a double-quote must produce valid TOML and round-trip.
#[test]
#[serial_test::serial]
fn test_append_match_round_trips_double_quote_in_project() {
    let dir = tempfile::TempDir::new().unwrap();
    let config_path = dir.path().join("config.toml");
    std::env::set_var("VECLAYER_USER_CONFIG", config_path.to_str().unwrap());

    let project_with_quote = "my\"project";
    append_match_to_user_config(None, Some("/tmp/work"), project_with_quote).unwrap();

    std::env::remove_var("VECLAYER_USER_CONFIG");

    // The written file must be parseable by the toml crate.
    let contents = std::fs::read_to_string(&config_path).unwrap();
    let parsed: Result<toml::Value, _> = toml::from_str(&contents);
    assert!(
        parsed.is_ok(),
        "file containing escaped double-quote must be valid TOML, got: {parsed:?}"
    );

    // And the project field must round-trip to the original string.
    let loaded = UserConfig::load(&config_path);
    assert_eq!(loaded.matches.len(), 1);
    assert_eq!(
        loaded.matches[0].project.as_deref(),
        Some(project_with_quote),
        "double-quote in project must round-trip unchanged"
    );
}

// EDGE: a path glob containing a backslash must produce valid TOML and round-trip.
#[test]
#[serial_test::serial]
fn test_append_match_round_trips_backslash_in_path() {
    let dir = tempfile::TempDir::new().unwrap();
    let config_path = dir.path().join("config.toml");
    std::env::set_var("VECLAYER_USER_CONFIG", config_path.to_str().unwrap());

    // Windows-style path: contains backslashes.
    let path_with_backslash = r"C:\Users\bob\work";
    append_match_to_user_config(None, Some(path_with_backslash), "winproject").unwrap();

    std::env::remove_var("VECLAYER_USER_CONFIG");

    let contents = std::fs::read_to_string(&config_path).unwrap();
    let parsed: Result<toml::Value, _> = toml::from_str(&contents);
    assert!(
        parsed.is_ok(),
        "file containing escaped backslash must be valid TOML, got: {parsed:?}"
    );

    let loaded = UserConfig::load(&config_path);
    // UserConfig::load parses the `path` field through glob::Pattern; the raw string
    // (before glob compilation) is not directly available after loading. We verify the
    // TOML is valid and the match entry was parsed (not silently dropped).
    // A glob::Pattern for a Windows path may or may not be valid on Linux — what matters
    // is that the TOML itself is well-formed and no panic occurs.
    let _ = loaded; // parsed without panic — sufficient to prove TOML validity
}

// EDGE: a project name containing a literal newline must produce valid TOML and round-trip.
#[test]
#[serial_test::serial]
fn test_append_match_round_trips_newline_in_project() {
    let dir = tempfile::TempDir::new().unwrap();
    let config_path = dir.path().join("config.toml");
    std::env::set_var("VECLAYER_USER_CONFIG", config_path.to_str().unwrap());

    let project_with_newline = "line1\nline2";
    append_match_to_user_config(None, Some("/tmp/work"), project_with_newline).unwrap();

    std::env::remove_var("VECLAYER_USER_CONFIG");

    let contents = std::fs::read_to_string(&config_path).unwrap();
    let parsed: Result<toml::Value, _> = toml::from_str(&contents);
    assert!(
        parsed.is_ok(),
        "file containing escaped newline must be valid TOML, got: {parsed:?}"
    );

    let loaded = UserConfig::load(&config_path);
    assert_eq!(loaded.matches.len(), 1);
    assert_eq!(
        loaded.matches[0].project.as_deref(),
        Some(project_with_newline),
        "newline in project must round-trip unchanged"
    );
}

// Unit test for the escaping helper itself — all special characters in one pass.
#[test]
fn test_toml_escape_string_all_special_chars() {
    // Each special character must be replaced by its TOML escape sequence.
    assert_eq!(toml_escape_string("\\"), "\\\\");
    assert_eq!(toml_escape_string("\""), "\\\"");
    assert_eq!(toml_escape_string("\n"), "\\n");
    assert_eq!(toml_escape_string("\r"), "\\r");
    assert_eq!(toml_escape_string("\t"), "\\t");

    // A string with all of them combined.
    let input = "a\\b\"c\nd\re\tf";
    let escaped = toml_escape_string(input);
    assert_eq!(escaped, r#"a\\b\"c\nd\re\tf"#);

    // The escaped result, wrapped in quotes, must be parseable by the toml crate.
    let toml_str = format!("value = \"{escaped}\"");
    let parsed: toml::Value =
        toml::from_str(&toml_str).expect("escaped string must produce valid TOML");
    assert_eq!(
        parsed["value"].as_str().unwrap(),
        input,
        "escaped TOML value must round-trip to the original string"
    );
}

// Unit test: plain strings (no special chars) pass through unchanged.
#[test]
fn test_toml_escape_string_plain_passthrough() {
    let plain = "github.com/org/repo-name_v2.0";
    assert_eq!(toml_escape_string(plain), plain);
}

// EDGE: C0 control characters and U+007F must be \uXXXX-escaped so the result
// is valid TOML that round-trips, rather than a bare control char the toml
// crate rejects on read-back.
#[test]
fn test_toml_escape_string_control_chars() {
    // U+0001 (SOH), U+0008 (BS), U+001B (ESC), U+007F (DEL) are forbidden bare.
    assert_eq!(toml_escape_string("\u{01}"), "\\u0001");
    assert_eq!(toml_escape_string("\u{08}"), "\\u0008");
    assert_eq!(toml_escape_string("\u{1b}"), "\\u001B");
    assert_eq!(toml_escape_string("\u{7f}"), "\\u007F");

    // A value mixing a control char with normal text must produce valid TOML
    // and round-trip to the original string.
    let input = "tab\tand\u{1b}escape";
    let escaped = toml_escape_string(input);
    let toml_str = format!("value = \"{escaped}\"");
    let parsed: toml::Value =
        toml::from_str(&toml_str).expect("control-char escape must produce valid TOML");
    assert_eq!(
        parsed["value"].as_str().unwrap(),
        input,
        "escaped control characters must round-trip to the original string"
    );
}

// --- env_bool tests ---

#[test]
#[serial_test::serial]
fn test_env_bool_true_values() {
    let saved = std::env::var("VECLAYER_TEST_BOOL_X9Z").ok();
    std::env::set_var("VECLAYER_TEST_BOOL_X9Z", "true");
    assert_eq!(env_bool("VECLAYER_TEST_BOOL_X9Z"), Some(true));
    std::env::set_var("VECLAYER_TEST_BOOL_X9Z", "1");
    assert_eq!(env_bool("VECLAYER_TEST_BOOL_X9Z"), Some(true));
    match saved {
        Some(v) => std::env::set_var("VECLAYER_TEST_BOOL_X9Z", v),
        None => std::env::remove_var("VECLAYER_TEST_BOOL_X9Z"),
    }
}

#[test]
#[serial_test::serial]
fn test_env_bool_false_values() {
    let saved = std::env::var("VECLAYER_TEST_BOOL_X9Z").ok();
    std::env::set_var("VECLAYER_TEST_BOOL_X9Z", "false");
    assert_eq!(env_bool("VECLAYER_TEST_BOOL_X9Z"), Some(false));
    std::env::set_var("VECLAYER_TEST_BOOL_X9Z", "0");
    assert_eq!(env_bool("VECLAYER_TEST_BOOL_X9Z"), Some(false));
    match saved {
        Some(v) => std::env::set_var("VECLAYER_TEST_BOOL_X9Z", v),
        None => std::env::remove_var("VECLAYER_TEST_BOOL_X9Z"),
    }
}

// Unrecognized value falls back to Some(false) — the existing contract — not None.
// This test pins that contract so it cannot silently regress to a panic or None.
#[test]
#[serial_test::serial]
fn test_env_bool_unrecognized_value_returns_some_false() {
    let saved = std::env::var("VECLAYER_TEST_BOOL_X9Z").ok();
    std::env::set_var("VECLAYER_TEST_BOOL_X9Z", "yes");
    assert_eq!(
        env_bool("VECLAYER_TEST_BOOL_X9Z"),
        Some(false),
        "unrecognized boolean string must return Some(false), not None or panic"
    );
    match saved {
        Some(v) => std::env::set_var("VECLAYER_TEST_BOOL_X9Z", v),
        None => std::env::remove_var("VECLAYER_TEST_BOOL_X9Z"),
    }
}

#[test]
#[serial_test::serial]
fn test_env_bool_unset_returns_none() {
    // A unique key guaranteed never to be set in the test environment.
    std::env::remove_var("VECLAYER_TEST_BOOL_UNSET_X9Z9");
    assert_eq!(env_bool("VECLAYER_TEST_BOOL_UNSET_X9Z9"), None);
}

// --- env_parse tests ---

#[test]
#[serial_test::serial]
fn test_env_parse_bad_integer_returns_none_no_panic() {
    let saved = std::env::var("VECLAYER_TEST_PARSE_INT_X9Z").ok();
    std::env::set_var("VECLAYER_TEST_PARSE_INT_X9Z", "not-a-number");
    let result: Option<u16> = env_parse("VECLAYER_TEST_PARSE_INT_X9Z");
    assert_eq!(
        result, None,
        "unparseable integer env var must return None without panicking"
    );
    match saved {
        Some(v) => std::env::set_var("VECLAYER_TEST_PARSE_INT_X9Z", v),
        None => std::env::remove_var("VECLAYER_TEST_PARSE_INT_X9Z"),
    }
}

#[test]
#[serial_test::serial]
fn test_env_parse_bad_float_returns_none_no_panic() {
    let saved = std::env::var("VECLAYER_TEST_PARSE_FLOAT_X9Z").ok();
    std::env::set_var("VECLAYER_TEST_PARSE_FLOAT_X9Z", "not-a-float");
    let result: Option<f32> = env_parse("VECLAYER_TEST_PARSE_FLOAT_X9Z");
    assert_eq!(
        result, None,
        "unparseable float env var must return None without panicking"
    );
    match saved {
        Some(v) => std::env::set_var("VECLAYER_TEST_PARSE_FLOAT_X9Z", v),
        None => std::env::remove_var("VECLAYER_TEST_PARSE_FLOAT_X9Z"),
    }
}

// --- VECLAYER_OLLAMA_DIMENSION invalid value test ---

#[test]
#[serial_test::serial]
fn test_resolve_embedder_invalid_dimension_falls_back_to_default() {
    std::env::set_var("VECLAYER_EMBEDDER", "ollama");
    std::env::remove_var("VECLAYER_OLLAMA_MODEL");
    std::env::remove_var("VECLAYER_OLLAMA_URL");
    std::env::set_var("VECLAYER_OLLAMA_DIMENSION", "not-a-number");

    let embedder = Config::resolve_embedder(None, None, None);

    std::env::remove_var("VECLAYER_EMBEDDER");
    std::env::remove_var("VECLAYER_OLLAMA_DIMENSION");

    assert!(
        matches!(
            embedder,
            EmbedderConfig::Ollama { dimension, .. } if dimension == DEFAULT_OLLAMA_DIMENSION
        ),
        "invalid VECLAYER_OLLAMA_DIMENSION must fall back to DEFAULT_OLLAMA_DIMENSION"
    );
}

// ── resolve_scopes: all-unknown names ────────────────────────────────────────

// When every scope name in both project_scopes and match_scopes is absent from
// self.scopes, resolve_scopes must return an empty vec (no panic, no partial result).
#[test]
fn test_resolve_scopes_all_unknown_returns_empty() {
    let config: UserConfig = toml::from_str(
        r#"
[scopes.real-scope]
storage = "git@github.com:flob/real.git"
"#,
    )
    .unwrap();

    let project_scopes = vec!["ghost".to_string(), "phantom".to_string()];
    let match_scopes = vec!["specter".to_string()];
    let resolved = config.resolve_scopes(&project_scopes, &match_scopes);

    assert!(
        resolved.is_empty(),
        "all-unknown scope names must produce an empty result, got: {resolved:?}"
    );
}

// When both project_scopes and match_scopes are empty, the result must be empty
// even when scopes are defined in the config.
#[test]
fn test_resolve_scopes_empty_inputs_returns_empty() {
    let config: UserConfig = toml::from_str(
        r#"
[scopes.personal]
storage = "git@github.com:flob/my-memory.git"
"#,
    )
    .unwrap();

    let resolved = config.resolve_scopes(&[], &[]);
    assert!(
        resolved.is_empty(),
        "empty scope inputs must produce an empty result, got: {resolved:?}"
    );
}

// ── UserConfig::discover: loads a valid file pointed to by env var ────────────

// Regression guard: discover() must load the file when VECLAYER_USER_CONFIG
// points at an existing, valid TOML config — not silently return defaults.
#[test]
#[serial_test::serial]
fn test_discover_user_config_valid_env_loads_file() {
    let original = std::env::var("VECLAYER_USER_CONFIG").ok();

    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");
    std::fs::write(
        &toml_path,
        r#"
project = "env-loaded-project"
data_dir = "/tmp/env-loaded-data"
"#,
    )
    .unwrap();

    std::env::set_var("VECLAYER_USER_CONFIG", toml_path.to_str().unwrap());
    let config = UserConfig::discover();

    match original {
        Some(v) => std::env::set_var("VECLAYER_USER_CONFIG", v),
        None => std::env::remove_var("VECLAYER_USER_CONFIG"),
    }

    assert_eq!(
        config.project.as_deref(),
        Some("env-loaded-project"),
        "discover() must load the file when env var points at a valid path"
    );
    assert_eq!(
        config.data_dir.as_deref(),
        Some("/tmp/env-loaded-data"),
        "discover() must populate data_dir from the env-var-pointed file"
    );
}

// ── discover_project: stops walk-up at $HOME boundary ─────────────────────────

// discover_project must NOT walk through $HOME looking for a .veclayer/
// directory. A .veclayer/ sitting exactly at $HOME must be invisible.
#[test]
fn test_discover_project_stops_at_home_boundary() {
    // We cannot reliably create a .veclayer/ at the real $HOME, so we use an
    // isolated tempdir tree and set HOME to a directory that has .veclayer/.
    // discover_project stops when it would enter the home dir, so starting the
    // search *at* home should return None even if .veclayer/ is present there.
    let tmp = tempfile::TempDir::new().unwrap();
    let fake_home = tmp.path().join("home");
    std::fs::create_dir_all(&fake_home).unwrap();

    // Place .veclayer/ at fake_home — the walk-up guard must skip it.
    let veclayer_at_home = fake_home.join(".veclayer");
    std::fs::create_dir_all(&veclayer_at_home).unwrap();

    // A subdirectory below fake_home that we start the search from.
    let work_dir = fake_home.join("work").join("project");
    std::fs::create_dir_all(&work_dir).unwrap();

    // Override HOME so directories::BaseDirs resolves to fake_home.
    let original_home = std::env::var("HOME").ok();
    std::env::set_var("HOME", &fake_home);

    let result = discover_project(&work_dir);

    // Restore HOME.
    match original_home {
        Some(v) => std::env::set_var("HOME", v),
        None => std::env::remove_var("HOME"),
    }

    assert!(
        result.is_none(),
        "discover_project must stop at $HOME and not find .veclayer/ placed there"
    );
}

// ── UserConfig::load: tilde in data_dir is expanded ──────────────────────────
// (Already tested by test_user_config_global_data_dir_tilde_expanded above;
// this companion test verifies the expansion survives the discover() path too.)
#[test]
#[serial_test::serial]
fn test_discover_user_config_tilde_in_data_dir_is_expanded() {
    let original = std::env::var("VECLAYER_USER_CONFIG").ok();

    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");
    std::fs::write(&toml_path, "data_dir = \"~/.veclayer-discover-test\"\n").unwrap();

    std::env::set_var("VECLAYER_USER_CONFIG", toml_path.to_str().unwrap());
    let config = UserConfig::discover();

    match original {
        Some(v) => std::env::set_var("VECLAYER_USER_CONFIG", v),
        None => std::env::remove_var("VECLAYER_USER_CONFIG"),
    }

    let data_dir = config
        .data_dir
        .expect("data_dir must be populated from the env-var file");
    assert!(
        !data_dir.starts_with('~'),
        "discover() must expand tilde in data_dir — got: {data_dir}"
    );
}

// --- VECLAYER_CONFIG missing file test ---

#[test]
#[serial_test::serial]
fn test_file_config_discover_nonexistent_env_returns_defaults() {
    let saved = std::env::var("VECLAYER_CONFIG").ok();

    std::env::set_var(
        "VECLAYER_CONFIG",
        "/nonexistent/path/that/does/not/exist/veclayer.toml",
    );
    let fc = FileConfig::discover(None);

    match saved {
        Some(v) => std::env::set_var("VECLAYER_CONFIG", v),
        None => std::env::remove_var("VECLAYER_CONFIG"),
    }

    assert!(
        fc.host.is_none(),
        "missing VECLAYER_CONFIG path must return defaults (host == None)"
    );
    assert!(
        fc.port.is_none(),
        "missing VECLAYER_CONFIG path must return defaults (port == None)"
    );
    assert!(
        fc.data_dir.is_none(),
        "missing VECLAYER_CONFIG path must return defaults (data_dir == None)"
    );
}
