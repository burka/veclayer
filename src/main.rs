use std::io::IsTerminal;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[cfg(feature = "llm")]
use veclayer::commands::think;
#[cfg(feature = "auth")]
use veclayer::commands::{auth_login, auth_status, auth_token, identity_init, identity_show};
use veclayer::commands::{
    add, archive, browse, compact, export_entries, focus, history, import_entries, init, merge,
    orientation, perspective_add, perspective_list, perspective_remove, print_sources,
    rebuild_index, reflect, search, serve, show_config, status, think_aging_apply,
    think_aging_configure, think_demote, think_discover, think_promote, think_relate, AddOptions,
    CompactAction, CompactOptions, ExportOptions, FocusOptions, ImportOptions, MergeOptions,
    SearchOptions, ServeOptions,
};
use veclayer::Result;

#[derive(Clone, Debug, clap::ValueEnum)]
enum EntryKind {
    Raw,
    Summary,
    Meta,
    Impression,
}

impl std::fmt::Display for EntryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntryKind::Raw => write!(f, "raw"),
            EntryKind::Summary => write!(f, "summary"),
            EntryKind::Meta => write!(f, "meta"),
            EntryKind::Impression => write!(f, "impression"),
        }
    }
}

#[derive(Parser)]
#[command(name = "veclayer")]
#[command(about = "Persistent memory for AI agents — recall, store, focus, think, share")]
#[command(version)]
struct Cli {
    /// Data directory (default: .veclayer/ if present, else platform data dir)
    #[arg(short, long, env = "VECLAYER_DATA_DIR")]
    data_dir: Option<PathBuf>,

    /// Enable verbose output (DEBUG for veclayer, INFO for dependencies; default is WARN)
    #[arg(short, long)]
    verbose: bool,

    /// Suppress all output except errors
    #[arg(short, long, conflicts_with = "verbose")]
    quiet: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Show resolved configuration for current working directory
    Config,

    /// Initialize a new VecLayer store
    Init {
        #[arg(long, help = "Enable git-based memory sharing for this project")]
        share: bool,
    },

    /// Store knowledge (text, file, or directory)
    #[command(alias = "add")]
    Store {
        /// Text content, file path, or directory to add
        input: String,

        /// Disable recursive directory processing
        #[arg(long)]
        no_recursive: bool,

        /// Disable cluster summarization
        #[arg(long)]
        no_summarize: bool,

        /// Ollama model for summarization
        #[arg(long, env = "VECLAYER_OLLAMA_MODEL", default_value = "llama3.2")]
        model: String,

        /// Visibility: "normal" (default), "always", "deep_only", "seasonal", or custom
        #[arg(long)]
        visibility: Option<String>,

        /// Entry type: raw (default), summary, meta, impression
        #[arg(long)]
        entry_type: Option<EntryKind>,

        /// Tag with perspectives (comma-separated or repeated)
        #[arg(short = 'P', long = "perspective")]
        perspectives: Vec<String>,

        /// Parent entry ID for hierarchy placement
        #[arg(long)]
        parent_id: Option<String>,

        /// Heading/title for the entry
        #[arg(long)]
        heading: Option<String>,

        /// Impression hint (e.g. "uncertain", "confident", "exploratory")
        #[arg(long)]
        impression_hint: Option<String>,

        /// Impression strength: 0.0–1.0 (default 1.0, only for entry_type=impression)
        #[arg(long, default_value = "1.0")]
        impression_strength: f32,

        /// This entry supersedes the target (auto-demotes target)
        #[arg(long, value_name = "ID", help_heading = "Relations")]
        rel_supersedes: Vec<String>,

        /// This entry summarizes the target
        #[arg(long, value_name = "ID", help_heading = "Relations")]
        rel_summarizes: Vec<String>,

        /// Bidirectional related_to link
        #[arg(long, value_name = "ID", help_heading = "Relations")]
        rel_to: Vec<String>,

        /// This entry is derived from the target (forward only)
        #[arg(long, value_name = "ID", help_heading = "Relations")]
        rel_derived_from: Vec<String>,

        /// This is a new version of the target (auto-demotes target)
        #[arg(long, value_name = "ID", help_heading = "Relations")]
        rel_version_of: Vec<String>,

        /// Custom relation: KIND:ID (forward on self only).
        /// Known kinds: supersedes, summarizes, related_to, derived_from, version_of.
        /// Custom kinds are also accepted.
        #[arg(
            short = 'R',
            long = "rel",
            value_name = "KIND:ID",
            help_heading = "Relations"
        )]
        rel_custom: Vec<String>,

        /// Follow symbolic links when recursing into directories
        #[arg(long)]
        follow_links: bool,
    },

    /// Recall knowledge — semantic search with hierarchical results
    #[command(alias = "search", alias = "s")]
    Recall {
        /// The search query (omit to browse all entries)
        query: Option<String>,

        /// Number of top results to return
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,

        /// Show full hierarchy path
        #[arg(short = 'p', long)]
        show_path: bool,

        /// Search within a specific subtree (entry ID or short ID)
        #[arg(long)]
        subtree: Option<String>,

        /// Deep search: include all visibilities
        #[arg(long)]
        deep: bool,

        /// Recency window: 24h, 7d, 30d
        #[arg(long)]
        recent: Option<String>,

        /// Filter by perspective (e.g. "decisions", "learnings")
        #[arg(short = 'P', long)]
        perspective: Option<String>,

        /// Search for entries similar to this entry ID (mutually exclusive with query)
        #[arg(long)]
        similar_to: Option<String>,

        /// Minimum salience (soft filter: excludes from salience boosting but not from results)
        #[arg(long = "min-salience")]
        min_salience: Option<f32>,

        /// Minimum search score (hard filter applied to pre-blend vector score)
        #[arg(long = "min-score")]
        min_score: Option<f32>,

        /// Only entries created after this ISO 8601 date or epoch seconds
        #[arg(long)]
        since: Option<String>,

        /// Only entries created before this ISO 8601 date or epoch seconds
        #[arg(long)]
        until: Option<String>,

        /// Filter to open threads (unresolved items)
        #[arg(long)]
        ongoing: bool,
    },

    /// Focus on an entry — show details and children, optionally reranked by a question
    #[command(alias = "f")]
    Focus {
        /// Entry ID (full hash or short 7-char prefix)
        id: String,

        /// Optional question to rerank children by relevance
        #[arg(short, long)]
        question: Option<String>,

        /// Max children to show
        #[arg(short = 'k', long, default_value = "10")]
        limit: usize,
    },

    /// Start the MCP/HTTP server
    Serve {
        /// Port to listen on
        #[arg(short, long, env = "VECLAYER_PORT")]
        port: Option<u16>,

        /// Host to bind to
        #[arg(long, env = "VECLAYER_HOST")]
        host: Option<String>,

        /// Run in read-only mode
        #[arg(long)]
        read_only: bool,

        /// Enable MCP stdio transport
        #[arg(long)]
        mcp_stdio: bool,

        /// Project scope for memory isolation
        #[arg(long)]
        project: Option<String>,

        /// Require authentication for all API access
        #[arg(long, env = "VECLAYER_AUTH_REQUIRED")]
        auth_required: bool,

        /// Public server URL for OAuth metadata
        #[arg(long, env = "VECLAYER_SERVER_URL")]
        server_url: Option<String>,

        /// Auto-approve OAuth requests (testing only)
        #[arg(long, env = "VECLAYER_AUTO_APPROVE", hide = true)]
        auto_approve: bool,
    },

    /// Show store statistics
    Status,

    /// List all indexed source files
    Sources,

    /// Manage perspectives (list, add, remove)
    #[command(alias = "p")]
    Perspective {
        #[command(subcommand)]
        action: PerspectiveAction,
    },

    /// Show version history of an entry (relations chain)
    History {
        /// Entry ID (full hash or short 7-char prefix)
        id: String,
    },

    /// Archive entries (demote to deep_only)
    Archive {
        /// Entry IDs to archive
        ids: Vec<String>,
    },

    /// Export all entries to JSONL (stdout), sorted by id
    Export {
        /// Filter by perspective (e.g. "decisions", "learnings")
        #[arg(short = 'P', long)]
        perspective: Option<String>,
    },

    /// Import entries from a JSONL file (or stdin with "-")
    Import {
        /// Path to JSONL file, or "-" to read from stdin
        path: String,
    },

    /// Rebuild the Lance index from the blob store
    #[command(alias = "reindex")]
    RebuildIndex,

    /// Merge blobs from another VecLayer store
    Merge {
        /// Path to the source .veclayer directory
        source: PathBuf,

        /// Project scope to tag merged entries with
        #[arg(long)]
        project: Option<String>,

        /// Preview only — show what would be merged without changing anything
        #[arg(long)]
        dry_run: bool,

        /// Merge without project scope (suppress warning)
        #[arg(long)]
        force: bool,
    },

    /// Manage cryptographic identity (DID, keypairs)
    Identity {
        #[command(subcommand)]
        action: IdentityAction,
    },

    /// Manage authentication tokens
    Auth {
        #[command(subcommand)]
        action: AuthAction,
    },

    /// Reflect — identity snapshot, salience ranking, archive candidates
    #[command(alias = "id")]
    Reflect {
        #[command(subcommand)]
        action: Option<ReflectAction>,
    },

    /// Think — curate memory: promote, demote, relate, aging, LLM consolidation
    Think {
        #[command(subcommand)]
        action: Option<ThinkAction>,
    },

    /// Sync memory from git scopes into the local index
    Sync {
        /// Only sync a specific scope by name
        #[arg(long)]
        scope: Option<String>,

        /// Export local entries to the git memory branch
        #[arg(long, conflicts_with_all = ["pending", "push", "stage", "reject"])]
        migrate: bool,

        /// List entries on the local branch not yet pushed to remote
        #[arg(long, conflicts_with_all = ["migrate", "push", "stage", "reject"])]
        pending: bool,

        /// Push the local git memory branch to remote
        #[arg(long, conflicts_with_all = ["migrate", "pending", "stage", "reject"])]
        push: bool,

        /// Stage a LanceDB entry to the git branch by ID (for manual push mode)
        #[arg(long, value_name = "ID", conflicts_with_all = ["migrate", "pending", "push", "reject"])]
        stage: Option<String>,

        /// Remove an entry from the git branch by ID (unstage)
        #[arg(long, value_name = "ID", conflicts_with_all = ["migrate", "pending", "push", "stage"])]
        reject: Option<String>,

        /// Filter migrate to entries with this perspective
        #[arg(long, value_name = "NAME")]
        perspective: Option<String>,

        /// Filter migrate to exclude entries with this perspective
        #[arg(long, value_name = "NAME")]
        exclude_perspective: Option<String>,

        /// Filter migrate to entries created after this date (ISO 8601, epoch seconds, or relative: 7d, 1m)
        #[arg(long, value_name = "DATE")]
        since: Option<String>,

        /// Show what would be synced without making changes
        #[arg(long)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum PerspectiveAction {
    /// List all perspectives
    #[command(alias = "ls")]
    List,

    /// Add a custom perspective
    Add {
        /// Unique slug ID (e.g. "emotions")
        id: String,
        /// Human-readable name (e.g. "Emotions")
        name: String,
        /// Hint for LLMs
        hint: String,
    },

    /// Remove a custom perspective
    #[command(alias = "rm")]
    Remove {
        /// Perspective ID to remove
        id: String,
    },
}

#[derive(Subcommand)]
enum IdentityAction {
    /// Generate a new Ed25519 keypair and store it encrypted
    Init {
        /// Overwrite existing identity
        #[arg(long)]
        force: bool,
    },
    /// Show current identity (DID, public key, keystore path)
    Show,
}

#[derive(Subcommand)]
enum AuthAction {
    /// Mint a JWT access token (for server operators)
    Token {
        /// Capability: read, write, admin
        #[arg(long, default_value = "read")]
        can: String,
        /// Expiry duration: 1h, 30d, 3600 (seconds)
        #[arg(long, default_value = "1h")]
        expires: String,
        /// Target server DID (defaults to own DID)
        #[arg(long)]
        audience: Option<String>,
    },
    /// Authenticate against a remote VecLayer server
    Login {
        /// Server URL (e.g. https://my-veclayer.fly.dev)
        #[arg(long)]
        server: String,
    },
    /// Show current authentication status
    Status,
}

#[derive(Subcommand)]
enum ReflectAction {
    /// Show salience scores for most important entries
    Salience {
        /// Max entries to show
        #[arg(short = 'k', long, default_value = "20")]
        limit: usize,
    },

    /// Show entries that are candidates for archival (low salience)
    #[command(alias = "candidates")]
    ArchiveCandidates {
        /// Max entries to show
        #[arg(short = 'k', long, default_value = "20")]
        limit: usize,

        /// Salience threshold (entries below this are candidates)
        #[arg(short, long, default_value = "0.1")]
        threshold: f32,
    },
}

#[derive(Subcommand)]
enum ThinkAction {
    /// Promote an entry (set visibility to 'always')
    Promote {
        /// Entry ID (full or short prefix)
        id: String,

        /// Target visibility (default: always)
        #[arg(long, default_value = "always")]
        visibility: String,
    },

    /// Demote an entry (set visibility to 'deep_only')
    Demote {
        /// Entry ID (full or short prefix)
        id: String,

        /// Target visibility (default: deep_only)
        #[arg(long, default_value = "deep_only")]
        visibility: String,
    },

    /// Add a relation between two entries
    Relate {
        /// Source entry ID (full or short prefix)
        source: String,

        /// Target entry ID (full or short prefix)
        target: String,

        /// Relation kind: supersedes, summarizes, related_to, derived_from, version_of
        #[arg(long, default_value = "related_to")]
        kind: String,
    },

    /// Discover hidden connections: find similar-but-unlinked entries
    Discover {
        /// Max pairs to show
        #[arg(short = 'k', long, default_value = "10")]
        limit: usize,
    },

    /// Manage aging rules
    Aging {
        #[command(subcommand)]
        action: AgingAction,
    },
}

#[derive(Subcommand)]
enum AgingAction {
    /// Apply aging rules: degrade stale entries
    Apply,

    /// Configure aging parameters
    Configure {
        /// Days without access before degradation
        #[arg(long)]
        days: Option<u32>,

        /// Target visibility after degradation (default: deep_only)
        #[arg(long)]
        to: Option<String>,
    },

    /// Rotate: roll access-profile buckets and apply aging
    Rotate,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Disable ANSI escape codes when stdout is not a TTY (e.g. piped output)
    if !std::io::stdout().is_terminal() {
        owo_colors::set_override(false);
    }

    let cli = Cli::parse();

    // Configuration precedence (highest to lowest):
    // 1. CLI flags (--port, --host, --data-dir, etc.)
    // 2. Environment variables (VECLAYER_PORT, VECLAYER_HOST, etc.)
    // 3. User config path overrides (~/.config/veclayer/config.toml [[path]])
    // 4. User config globals (~/.config/veclayer/config.toml top-level)
    // 5. Project-local config (.veclayer/config.toml)
    // 6. Git auto-detection (remote → project, branch → scope)
    // 7. Platform defaults

    let cwd = std::env::current_dir().expect("Failed to determine current directory");
    let git_info = veclayer::git::detect::detect(&cwd);

    let user_config = veclayer::config::UserConfig::discover();
    let user_resolved = user_config.resolve(&cwd, git_info.remote.as_deref());

    let user_project = user_resolved.project.clone();
    let user_data_dir = user_resolved.data_dir.clone();

    let (
        data_dir,
        discovered_project,
        discovered_branch,
        project_scopes,
        project_storage,
        project_push,
        using_global_store_fallback,
    ) = match cli.data_dir {
        Some(dir) => (
            dir,
            user_project.or(git_info.remote.clone()),
            git_info.branch.clone(),
            Vec::<String>::new(),
            None::<String>,
            None::<String>,
            false,
        ),
        None => {
            let project_local = veclayer::config::discover_project(&cwd);

            let project = project_local
                .as_ref()
                .and_then(|(_, pc)| pc.project.clone())
                .or(user_project)
                .or(git_info.remote.clone());

            let scopes = project_local
                .as_ref()
                .map(|(_, pc)| pc.scopes.clone())
                .unwrap_or_default();
            let storage = project_local
                .as_ref()
                .and_then(|(_, pc)| pc.storage.clone());
            let push = project_local.as_ref().and_then(|(_, pc)| pc.push.clone());

            let fallback = project_local.is_none() && user_data_dir.is_none();
            let data_dir = project_local
                .map(|(d, _)| d)
                .or_else(|| user_data_dir.as_ref().map(PathBuf::from))
                .unwrap_or_else(veclayer::default_data_dir);

            (
                data_dir,
                project,
                git_info.branch.clone(),
                scopes,
                storage,
                push,
                fallback,
            )
        }
    };

    let resolved_scopes = user_config.resolve_scopes(
        &project_scopes,
        &user_resolved
            .scopes
            .iter()
            .map(|s| s.name.clone())
            .collect::<Vec<_>>(),
    );

    init_logging(cli.verbose, cli.quiet);

    let command = match cli.command {
        Some(cmd) => cmd,
        None => {
            if using_global_store_fallback {
                eprintln!(
                    "Warning: no local store found, using global store at {}. Run 'veclayer init' to create a project store.",
                    data_dir.display()
                );
            }
            orientation(&data_dir).await?;
            return Ok(());
        }
    };

    if using_global_store_fallback && !matches!(command, Commands::Init { .. }) {
        eprintln!(
            "Warning: no local store found, using global store at {}. Run 'veclayer init' to create a project store.",
            data_dir.display()
        );
    }

    match command {
        Commands::Config => {
            show_config(
                &cwd,
                &user_config,
                &user_resolved,
                git_info.remote.as_deref(),
                git_info.branch.as_deref(),
            )?;
        }
        Commands::Init { share } => {
            init(&cwd, &data_dir, share)?;
        }
        Commands::Store {
            input,
            no_recursive,
            no_summarize,
            model,
            visibility,
            entry_type,
            perspectives,
            parent_id,
            heading,
            impression_hint,
            impression_strength,
            rel_supersedes,
            rel_summarizes,
            rel_to,
            rel_derived_from,
            rel_version_of,
            rel_custom,
            follow_links,
        } => {
            let options = AddOptions {
                recursive: !no_recursive,
                follow_links,
                summarize: !no_summarize,
                model,
                visibility,
                entry_type: entry_type
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "raw".to_string()),
                perspectives,
                parent_id,
                heading,
                impression_hint,
                impression_strength,
                rel_supersedes,
                rel_summarizes,
                rel_to,
                rel_derived_from,
                rel_version_of,
                rel_custom,
            };
            let git_store = if project_storage.as_deref() == Some("git") {
                veclayer::git::detect::find_git_dir(&cwd).and_then(|git_dir| {
                    match veclayer::git::memory_store::MemoryStore::open(&git_dir, None) {
                        Ok(s) => Some(s),
                        Err(e) => {
                            tracing::warn!("Failed to open git memory store: {e}");
                            None
                        }
                    }
                })
            } else {
                None
            };
            add(&data_dir, &input, options, git_store.as_ref()).await?;
        }
        Commands::Recall {
            query,
            top_k,
            show_path,
            subtree,
            deep,
            recent,
            perspective,
            similar_to,
            min_salience,
            min_score,
            since,
            until,
            ongoing,
        } => {
            let options = SearchOptions {
                top_k,
                show_path,
                subtree,
                deep,
                recent,
                perspective,
                similar_to,
                min_salience,
                min_score,
                since,
                until,
                ongoing,
            };
            if options.similar_to.is_some() {
                search(&data_dir, query.as_deref().unwrap_or(""), &options).await?
            } else {
                match query {
                    Some(q) => search(&data_dir, &q, &options).await?,
                    None => browse(&data_dir, &options).await?,
                }
            }
        }
        Commands::Focus {
            id,
            question,
            limit,
        } => {
            let options = FocusOptions { question, limit };
            focus(&data_dir, &id, &options).await?;
        }
        Commands::Serve {
            port,
            host,
            read_only,
            mcp_stdio,
            project,
            auth_required,
            server_url,
            auto_approve,
        } => {
            let auth_config = veclayer::config::Config::new().auth;
            let options = ServeOptions {
                host: host
                    .or(user_resolved.host.clone())
                    .unwrap_or_else(|| "127.0.0.1".to_string()),
                port: port.or(user_resolved.port).unwrap_or(8080),
                read_only: read_only || user_resolved.read_only.unwrap_or(false),
                mcp_stdio,
                project: project.or(discovered_project),
                branch: discovered_branch,
                auth_required: auth_required || auth_config.auth_required,
                server_url: server_url.or(auth_config.server_url),
                auto_approve: auto_approve || auth_config.auto_approve,
                token_expiry_secs: auth_config.token_expiry_secs,
                refresh_expiry_secs: auth_config.refresh_expiry_secs,
                storage: project_storage.clone(),
                push: project_push.clone(),
            };
            serve(&data_dir, &options).await?;
        }
        Commands::Status => {
            status(&data_dir).await?;
        }
        Commands::Sources => {
            print_sources(&data_dir).await?;
        }
        Commands::Perspective { action } => match action {
            PerspectiveAction::List => {
                perspective_list(&data_dir)?;
            }
            PerspectiveAction::Add { id, name, hint } => {
                perspective_add(&data_dir, &id, &name, &hint)?;
            }
            PerspectiveAction::Remove { id } => {
                perspective_remove(&data_dir, &id)?;
            }
        },
        Commands::History { id } => {
            history(&data_dir, &id).await?;
        }
        Commands::Archive { ids } => {
            archive(&data_dir, &ids).await?;
        }
        Commands::Export { perspective } => {
            let options = ExportOptions { perspective };
            export_entries(&data_dir, &options).await?;
        }
        Commands::Import { path } => {
            let options = ImportOptions { path };
            import_entries(&data_dir, &options).await?;
        }
        Commands::RebuildIndex => {
            rebuild_index(&data_dir).await?;
        }
        Commands::Merge {
            source,
            project,
            dry_run,
            force,
        } => {
            let options = MergeOptions {
                project,
                dry_run,
                force,
            };
            merge(&data_dir, &source, &options).await?;
        }
        Commands::Identity { action } => {
            #[cfg(feature = "auth")]
            match action {
                IdentityAction::Init { force } => {
                    identity_init(&data_dir, force).await?;
                }
                IdentityAction::Show => {
                    identity_show(&data_dir).await?;
                }
            }
            #[cfg(not(feature = "auth"))]
            {
                let _ = action;
                eprintln!("Error: `identity` commands require the 'auth' feature.");
                eprintln!("Build with `cargo build` (default features) or `cargo build --features auth`.");
                std::process::exit(1);
            }
        }
        Commands::Auth { action } => {
            #[cfg(feature = "auth")]
            match action {
                AuthAction::Token {
                    can,
                    expires,
                    audience,
                } => {
                    auth_token(&data_dir, &can, &expires, audience.as_deref()).await?;
                }
                AuthAction::Login { server } => {
                    auth_login(&data_dir, &server).await?;
                }
                AuthAction::Status => {
                    auth_status(&data_dir).await?;
                }
            }
            #[cfg(not(feature = "auth"))]
            {
                let _ = action;
                eprintln!("Error: `auth` commands require the 'auth' feature.");
                eprintln!("Build with `cargo build` (default features) or `cargo build --features auth`.");
                std::process::exit(1);
            }
        }
        Commands::Reflect { action } => match action {
            None => {
                reflect(&data_dir).await?;
            }
            Some(ReflectAction::Salience { limit }) => {
                let options = CompactOptions {
                    limit,
                    ..Default::default()
                };
                compact(&data_dir, CompactAction::Salience, &options).await?;
            }
            Some(ReflectAction::ArchiveCandidates { limit, threshold }) => {
                let options = CompactOptions {
                    limit,
                    archive_threshold: threshold,
                };
                compact(&data_dir, CompactAction::ArchiveCandidates, &options).await?;
            }
        },
        Commands::Think { action } => match action {
            None => {
                #[cfg(feature = "llm")]
                {
                    think(&data_dir).await?;
                }
                #[cfg(not(feature = "llm"))]
                {
                    eprintln!("Error: `think` (without subcommand) requires the 'llm' feature.");
                    eprintln!("Build with `cargo build` (default features) or `cargo build --features llm`.");
                    std::process::exit(1);
                }
            }
            Some(ThinkAction::Promote { id, visibility }) => {
                think_promote(&data_dir, &id, &visibility).await?;
            }
            Some(ThinkAction::Demote { id, visibility }) => {
                think_demote(&data_dir, &id, &visibility).await?;
            }
            Some(ThinkAction::Relate {
                source,
                target,
                kind,
            }) => {
                think_relate(&data_dir, &source, &target, &kind).await?;
            }
            Some(ThinkAction::Discover { limit }) => {
                think_discover(&data_dir, limit).await?;
            }
            Some(ThinkAction::Aging { action }) => match action {
                AgingAction::Apply => {
                    think_aging_apply(&data_dir).await?;
                }
                AgingAction::Configure { days, to } => {
                    think_aging_configure(&data_dir, days, to.as_deref()).await?;
                }
                AgingAction::Rotate => {
                    compact(&data_dir, CompactAction::Rotate, &CompactOptions::default()).await?;
                }
            },
        },
        Commands::Sync {
            scope,
            migrate,
            pending,
            push,
            stage,
            reject,
            perspective,
            exclude_perspective,
            since,
            dry_run,
        } => {
            if pending {
                veclayer::commands::sync::show_pending().await?;
                return Ok(());
            }

            if push {
                veclayer::commands::sync::push_to_remote().await?;
                return Ok(());
            }

            if let Some(ref id) = stage {
                veclayer::commands::sync::stage_entry(&data_dir, id).await?;
                return Ok(());
            }

            if let Some(ref id) = reject {
                veclayer::commands::sync::reject_entry(id).await?;
                return Ok(());
            }

            if migrate {
                let since_ts = since.as_deref().and_then(veclayer::resolve::parse_temporal);
                let filters = veclayer::commands::sync::MigrateFilters {
                    perspective,
                    exclude_perspective,
                    since: since_ts,
                };
                veclayer::commands::sync::migrate(&data_dir, &filters).await?;
                return Ok(());
            }

            let mut scopes = resolved_scopes;

            // Inject implicit "project" scope when the project config uses git storage
            // and it isn't already present from the named-scope resolution.
            if let Some(ref storage) = project_storage {
                if storage == "git" && !scopes.iter().any(|s| s.name == "project") {
                    scopes.insert(
                        0,
                        veclayer::config::ResolvedScope {
                            name: "project".to_string(),
                            storage: "git".to_string(),
                            branch: "veclayer-memory".to_string(),
                            push: project_push.unwrap_or_else(|| {
                                veclayer::git::branch_config::PushMode::default().to_string()
                            }),
                        },
                    );
                }
            }

            veclayer::commands::sync::sync(&data_dir, &scopes, scope.as_deref(), dry_run).await?;
        }
    }

    Ok(())
}

fn init_logging(verbose: bool, quiet: bool) {
    // Respect RUST_LOG env if set; otherwise use flag-based defaults.
    // Default: only show warnings. --verbose enables veclayer DEBUG.
    // --quiet suppresses everything except errors.
    // Dependency crates (lancedb, lance, ort, etc.) stay at WARN unless --verbose.
    // Always write to stderr so logs never pollute stdout (e.g. `veclayer export > file.jsonl`).
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else if quiet {
        EnvFilter::new("error")
    } else if verbose {
        EnvFilter::new("veclayer=debug,info")
    } else {
        EnvFilter::new("warn")
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
}
