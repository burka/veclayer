//! MCP/HTTP server startup.

use super::*;

/// Start the MCP/HTTP server.
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
        .with_push_mode(options.push.as_deref());

    if options.mcp_stdio {
        crate::mcp::run_stdio(config).await
    } else {
        crate::mcp::run_http(config).await
    }
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
