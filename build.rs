//! Build script: git metadata + optional Markdown landing page compilation.
//!
//! Always extracts git hash/date for `VECLAYER_GIT_HASH` / `VECLAYER_GIT_DATE`.
//! When the `landing-page` feature is enabled, also compiles website/index.md
//! into HTML via pulldown-cmark and writes $OUT_DIR/index.html.

use std::process::Command;

fn main() {
    // ── Git metadata (always) ─────────────────────────────────────────────
    let hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let date = Command::new("git")
        .args(["log", "-1", "--format=%Y-%m-%d"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    println!("cargo:rustc-env=VECLAYER_GIT_HASH={hash}");
    println!("cargo:rustc-env=VECLAYER_GIT_DATE={date}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");
    println!("cargo:rerun-if-changed=.git/packed-refs");

    // ── Landing page (feature-gated) ──────────────────────────────────────
    #[cfg(feature = "landing-page")]
    {
        use std::fs;
        use std::path::Path;

        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let out_dir = std::env::var("OUT_DIR").unwrap();

        let website_dir = Path::new(&manifest_dir).join("website");
        let md_path = website_dir.join("index.md");
        let template_path = website_dir.join("template.html");

        println!("cargo:rerun-if-changed=website/index.md");
        println!("cargo:rerun-if-changed=website/template.html");

        let md_source = fs::read_to_string(&md_path)
            .expect(&format!("Failed to read {}", md_path.display()));

        let template = fs::read_to_string(&template_path)
            .expect(&format!("Failed to read {}", template_path.display()));

        // Render Markdown → HTML
        let parser = pulldown_cmark::Parser::new_ext(&md_source, pulldown_cmark::Options::all());
        let mut html_content = String::new();
        pulldown_cmark::html::push_html(&mut html_content, parser);

        // Escape the raw Markdown source as a JSON string for the copy button
        let md_json = serde_json::to_string(&md_source).unwrap();

        // Insert into template
        let page = template
            .replace("{{CONTENT}}", &html_content)
            .replace("{{MD_SOURCE_JSON}}", &md_json);

        let out_path = Path::new(&out_dir).join("index.html");
        fs::write(&out_path, &page)
            .expect(&format!("Failed to write {}", out_path.display()));
    }

    // When landing-page is disabled, write a minimal fallback so include_str! still compiles
    // if any code path references it (though it shouldn't with proper cfg gating).
    #[cfg(not(feature = "landing-page"))]
    {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let out_path = std::path::Path::new(&out_dir).join("index.html");
        std::fs::write(&out_path, "<!DOCTYPE html><html><body><p>Landing page disabled.</p></body></html>").ok();
    }
}
