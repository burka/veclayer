//! MCP/HTTP server startup.

use super::*;

/// Start the MCP server (stdio transport) or the experimental HTTP server.
///
/// **Stable:** `--mcp-stdio` (default for Claude Code / MCP clients — use this).
/// **Experimental / WIP:** HTTP mode (no `--mcp-stdio`) — requires the `http`
/// feature and is not production-ready.
pub async fn serve(data_dir: &Path, options: &ServeOptions) -> Result<()> {
    if !data_dir.exists() {
        std::fs::create_dir_all(data_dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(data_dir, std::fs::Permissions::from_mode(0o700))?;
        }
    }

    let config = Config::new()
        .with_data_dir(data_dir)
        .with_host(&options.host)
        .with_port(options.port)
        .with_read_only(options.read_only)
        .with_project(options.project.clone())
        .with_branch(options.branch.clone())
        .with_storage(options.storage.clone())
        .with_push_mode(options.push.as_deref())
        .with_auth_required(options.auth_required);

    if options.mcp_stdio {
        crate::mcp::run_stdio(config).await
    } else {
        #[cfg(feature = "http")]
        {
            crate::mcp::run_http(config).await
        }
        #[cfg(not(feature = "http"))]
        {
            let _ = config;
            Err(crate::Error::InvalidOperation(
                "HTTP server mode is experimental and requires the 'http' feature \
(not included in the default build). Use --mcp-stdio for the stable MCP transport, \
or build with `--features http` to enable the HTTP server.".to_string(),
            ))
        }
    }
}

/// Compact LanceDB on startup, draining version backlog before accepting requests.
/// Non-blocking: completes before the server starts.
pub async fn startup_compact(store: &crate::store::StoreBackend) -> Result<()> {
    eprintln!("[veclayer] compaction started");
    match store.force_compact().await {
        Ok(stats) => {
            eprintln!("[veclayer] compaction complete: {} versions pruned, {} bytes reclaimed",
                stats.versions_removed,
                crate::util::format_bytes(stats.bytes_reclaimed)
            );
        }
        Err(e) => {
            eprintln!("[veclayer] compaction failed (non-fatal): {}", e);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serve_options_default() {
        let opts = ServeOptions::default();
        assert_eq!(opts.host, "127.0.0.1");
        assert_eq!(opts.port, 8080);
        assert!(!opts.read_only);
        assert!(!opts.mcp_stdio);
    }
}
