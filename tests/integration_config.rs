//! Integration tests for cascading match-based configuration.

use std::path::Path;

use serial_test::serial;
use veclayer::config::{discover_project, UserConfig};

#[test]
fn test_backwards_compatibility_no_user_config() {
    let dir = tempfile::TempDir::new().unwrap();
    let cwd = dir.path();

    let user_config = UserConfig::discover();
    let resolved = user_config.resolve(cwd, None);

    assert!(resolved.project.is_none());
    assert!(resolved.data_dir.is_none());
    assert!(resolved.host.is_none());
    assert!(resolved.port.is_none());
    assert!(resolved.read_only.is_none());
}

#[test]
fn test_cascade_global_defaults() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");

    std::fs::write(
        &toml_path,
        r#"
project = "global-project"
data_dir = "/global/data"
host = "0.0.0.0"
port = 9000
read_only = true
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);
    let resolved = config.resolve(Path::new("/any/path"), None);

    assert_eq!(resolved.project.as_deref(), Some("global-project"));
    assert_eq!(resolved.data_dir.as_deref(), Some("/global/data"));
    assert_eq!(resolved.host.as_deref(), Some("0.0.0.0"));
    assert_eq!(resolved.port, Some(9000));
    assert_eq!(resolved.read_only, Some(true));
}

#[test]
fn test_cascade_path_override() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");

    std::fs::write(
        &toml_path,
        r#"
project = "global"
data_dir = "/global/data"

[[match]]
path = "/tmp/special*"
project = "special"
data_dir = "/special/data"
host = "192.168.1.1"
port = 8081
read_only = true
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);
    let resolved = config.resolve(Path::new("/tmp/special-project"), None);

    assert_eq!(resolved.project.as_deref(), Some("special"));
    assert_eq!(resolved.data_dir.as_deref(), Some("/special/data"));
    assert_eq!(resolved.host.as_deref(), Some("192.168.1.1"));
    assert_eq!(resolved.port, Some(8081));
    assert_eq!(resolved.read_only, Some(true));
}

#[test]
fn test_cascade_no_match_globals_used() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");

    std::fs::write(
        &toml_path,
        r#"
project = "global"
data_dir = "/global/data"
port = 9000

[[match]]
path = "/tmp/special*"
project = "special"
read_only = true
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);
    let resolved = config.resolve(Path::new("/other/path"), None);

    assert_eq!(resolved.project.as_deref(), Some("global"));
    assert_eq!(resolved.data_dir.as_deref(), Some("/global/data"));
    assert_eq!(resolved.port, Some(9000));
    assert!(resolved.read_only.is_none());
}

#[test]
fn test_cascade_partial_override() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");

    std::fs::write(
        &toml_path,
        r#"
project = "global"
data_dir = "/global/data"
host = "127.0.0.1"
port = 8080
read_only = false

[[match]]
path = "/tmp/readonly*"
read_only = true

[[match]]
path = "/tmp/special*"
project = "special"
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);

    let resolved = config.resolve(Path::new("/tmp/readonly-thing"), None);
    assert_eq!(resolved.project.as_deref(), Some("global"));
    assert_eq!(resolved.data_dir.as_deref(), Some("/global/data"));
    assert_eq!(resolved.host.as_deref(), Some("127.0.0.1"));
    assert_eq!(resolved.port, Some(8080));
    assert_eq!(resolved.read_only, Some(true));

    let resolved = config.resolve(Path::new("/tmp/special-thing"), None);
    assert_eq!(resolved.project.as_deref(), Some("special"));
    assert_eq!(resolved.data_dir.as_deref(), Some("/global/data"));
    assert_eq!(resolved.host.as_deref(), Some("127.0.0.1"));
    assert_eq!(resolved.port, Some(8080));
    assert_eq!(resolved.read_only, Some(false));
}

#[test]
fn test_cascade_last_match_wins() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");

    std::fs::write(
        &toml_path,
        r#"
[[match]]
path = "/tmp/test/**"
project = "first"
data_dir = "/first"

[[match]]
path = "/tmp/test/specific"
project = "second"
data_dir = "/second"

[[match]]
path = "/tmp/test/**"
project = "third"
read_only = true
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);
    let resolved = config.resolve(Path::new("/tmp/test/specific"), None);

    assert_eq!(resolved.project.as_deref(), Some("third"));
    assert_eq!(resolved.data_dir.as_deref(), Some("/second"));
    assert_eq!(resolved.read_only, Some(true));
}

#[test]
fn test_tilde_expansion_in_user_config() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");

    std::fs::write(
        &toml_path,
        r#"
[[match]]
path = "~/work/veclayer*"
project = "veclayer"
data_dir = "~/work/veclayer-one/.veclayer"
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);

    let home = std::env::var("HOME").unwrap();
    let test_path = format!("{}/work/veclayer-project", home);

    let resolved = config.resolve(Path::new(&test_path), None);
    assert_eq!(resolved.project.as_deref(), Some("veclayer"));
    let data_dir = resolved.data_dir.as_deref().unwrap();
    assert!(
        !data_dir.starts_with('~'),
        "Expected tilde-expanded data_dir, got {:?}",
        data_dir
    );
    assert!(
        data_dir.contains("work/veclayer-one"),
        "Expected data_dir to contain 'work/veclayer-one', got {:?}",
        data_dir
    );
}

#[test]
fn test_malformed_user_config_no_matcher() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");

    // [[match]] without path or git-remote is rejected
    std::fs::write(
        &toml_path,
        r#"
[[match]]
project = "test"
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);
    assert!(config.matches.is_empty());
}

#[test]
fn test_malformed_user_config_invalid_toml() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");

    std::fs::write(&toml_path, "not [ valid toml {{{").unwrap();

    let config = UserConfig::load(&toml_path);
    assert!(config.matches.is_empty());
}

#[test]
#[serial]
fn test_user_config_discover_env_var() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");

    std::fs::write(
        &toml_path,
        r#"
project = "from_env"
"#,
    )
    .unwrap();

    std::env::set_var(
        "VECLAYER_USER_CONFIG",
        toml_path.to_string_lossy().to_string(),
    );
    let config = UserConfig::discover();
    std::env::remove_var("VECLAYER_USER_CONFIG");

    assert_eq!(config.project.as_deref(), Some("from_env"));
}

#[test]
#[serial]
fn test_user_config_discover_xdg_config_home() {
    let dir = tempfile::TempDir::new().unwrap();
    let config_dir = dir.path().join("config/veclayer");
    std::fs::create_dir_all(&config_dir).unwrap();
    let toml_path = config_dir.join("config.toml");

    std::fs::write(
        &toml_path,
        r#"
project = "from_xdg"
"#,
    )
    .unwrap();

    std::env::set_var(
        "XDG_CONFIG_HOME",
        dir.path().join("config").to_string_lossy().to_string(),
    );
    let config = UserConfig::discover();
    std::env::remove_var("XDG_CONFIG_HOME");

    assert_eq!(config.project.as_deref(), Some("from_xdg"));
}

#[test]
fn test_project_config_discovery_still_works() {
    let dir = tempfile::TempDir::new().unwrap();
    let veclayer_dir = dir.path().join(".veclayer");
    std::fs::create_dir_all(&veclayer_dir).unwrap();

    let config_path = veclayer_dir.join("config.toml");
    std::fs::write(&config_path, "project = \"local-project\"\n").unwrap();

    let result = discover_project(dir.path());
    assert!(result.is_some());
    let (_, config) = result.unwrap();
    assert_eq!(config.project.as_deref(), Some("local-project"));
}

#[test]
fn test_match_override_with_star_wildcard() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");

    std::fs::write(
        &toml_path,
        r#"
[[match]]
path = "/tmp/**/*.rs"
project = "rust"
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);

    let resolved = config.resolve(Path::new("/tmp/deep/nested/main.rs"), None);
    assert_eq!(resolved.project.as_deref(), Some("rust"));

    let resolved = config.resolve(Path::new("/tmp/deep/nested/main.c"), None);
    assert!(resolved.project.is_none());
}

#[test]
fn test_match_override_with_question_mark() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");

    std::fs::write(
        &toml_path,
        r#"
[[match]]
path = "/tmp/test?"
project = "single-char"
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);

    let resolved = config.resolve(Path::new("/tmp/test1"), None);
    assert_eq!(resolved.project.as_deref(), Some("single-char"));

    let resolved = config.resolve(Path::new("/tmp/test12"), None);
    assert!(resolved.project.is_none());
}

#[test]
fn test_match_override_with_character_class() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");

    std::fs::write(
        &toml_path,
        r#"
[[match]]
path = "/tmp/[abc]*"
project = "abc-start"
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);

    let resolved = config.resolve(Path::new("/tmp/apple"), None);
    assert_eq!(resolved.project.as_deref(), Some("abc-start"));

    let resolved = config.resolve(Path::new("/tmp/banana"), None);
    assert_eq!(resolved.project.as_deref(), Some("abc-start"));

    let resolved = config.resolve(Path::new("/tmp/dragon"), None);
    assert!(resolved.project.is_none());
}

#[test]
fn test_match_git_remote_override() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");

    std::fs::write(
        &toml_path,
        r#"
[[match]]
git-remote = "(?i)damalo"
project = "damalo"
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);

    let resolved = config.resolve(Path::new("/tmp"), Some("github.com/Damalo/repo"));
    assert_eq!(resolved.project.as_deref(), Some("damalo"));

    let resolved = config.resolve(Path::new("/tmp"), Some("github.com/other/repo"));
    assert!(resolved.project.is_none());

    let resolved = config.resolve(Path::new("/tmp"), None);
    assert!(resolved.project.is_none());
}

#[test]
fn test_match_combined_path_and_git_remote() {
    let dir = tempfile::TempDir::new().unwrap();
    let toml_path = dir.path().join("user.toml");

    std::fs::write(
        &toml_path,
        r#"
[[match]]
path = "/home/flob/work/damalo*"
git-remote = "(?i)damalo"
project = "damalo"
data_dir = "/damalo/data"
"#,
    )
    .unwrap();

    let config = UserConfig::load(&toml_path);

    // Path match only
    let resolved = config.resolve(Path::new("/home/flob/work/damalo-app"), None);
    assert_eq!(resolved.project.as_deref(), Some("damalo"));
    assert_eq!(resolved.data_dir.as_deref(), Some("/damalo/data"));

    // Remote match only
    let resolved = config.resolve(Path::new("/other/path"), Some("github.com/Damalo/repo"));
    assert_eq!(resolved.project.as_deref(), Some("damalo"));

    // Neither
    let resolved = config.resolve(Path::new("/other/path"), Some("github.com/other/repo"));
    assert!(resolved.project.is_none());
}
