use std::io::IsTerminal;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use veclayer::commands::{
    AddOptions, CompactAction, CompactOptions, ExportOptions, FocusOptions, ImportOptions,
    SearchOptions, ServeOptions,
};
use veclayer::Result;

#[derive(Parser)]
#[command(name = "veclayer")]
#[command(about = "Hierarchical memory for AI agents")]
#[command(version)]
struct Cli {
    /// Data directory for VecLayer storage
    #[arg(
        short,
        long,
        env = "VECLAYER_DATA_DIR",
        default_value = "./veclayer-data"
    )]
    data_dir: PathBuf,

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
    /// Initialize a new VecLayer store
    Init,

    /// Add knowledge (text, file, or directory)
    #[command(alias = "store")]
    Add {
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

        /// Visibility (e.g. "always", "deep_only", or custom)
        #[arg(long)]
        visibility: Option<String>,

        /// Entry type: raw (default), meta, impression
        #[arg(long, default_value = "raw")]
        entry_type: String,

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
        #[arg(long, value_name = "ID")]
        rel_supersedes: Vec<String>,

        /// This entry summarizes the target
        #[arg(long, value_name = "ID")]
        rel_summarizes: Vec<String>,

        /// Bidirectional related_to link
        #[arg(long, value_name = "ID")]
        rel_to: Vec<String>,

        /// This entry is derived from the target (forward only)
        #[arg(long, value_name = "ID")]
        rel_derived_from: Vec<String>,

        /// This is a new version of the target (auto-demotes target)
        #[arg(long, value_name = "ID")]
        rel_version_of: Vec<String>,

        /// Custom relation: KIND:ID (forward on self only)
        #[arg(short = 'R', long = "rel", value_name = "KIND:ID")]
        rel_custom: Vec<String>,
    },

    /// Semantic search with hierarchical results
    #[command(alias = "s", alias = "recall")]
    Search {
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

        /// Search for entries similar to this entry ID (uses its embedding as query)
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
    },

    /// Focus on an entry: show details and children
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
        #[arg(short, long, env = "VECLAYER_PORT", default_value = "8080")]
        port: u16,

        /// Host to bind to
        #[arg(long, env = "VECLAYER_HOST", default_value = "127.0.0.1")]
        host: String,

        /// Run in read-only mode
        #[arg(long)]
        read_only: bool,

        /// Enable MCP stdio transport
        #[arg(long)]
        mcp_stdio: bool,
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

    /// Read-only identity operations: reflection, salience, candidates
    #[command(alias = "id")]
    Reflect {
        #[command(subcommand)]
        action: Option<ReflectAction>,
    },

    /// Curate memory: promote, demote, relate, aging, LLM consolidation
    Think {
        #[command(subcommand)]
        action: Option<ThinkAction>,
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

    let is_mcp_stdio = matches!(
        &cli.command,
        Some(Commands::Serve {
            mcp_stdio: true,
            ..
        })
    );

    init_logging(cli.verbose, cli.quiet, is_mcp_stdio);

    let command = match cli.command {
        Some(cmd) => cmd,
        None => {
            // No subcommand: show orientation
            veclayer::commands::orientation(&cli.data_dir).await?;
            return Ok(());
        }
    };

    match command {
        Commands::Init => {
            veclayer::commands::init(&cli.data_dir)?;
        }
        Commands::Add {
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
        } => {
            let options = AddOptions {
                recursive: !no_recursive,
                summarize: !no_summarize,
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
            };
            veclayer::commands::add(&cli.data_dir, &input, &options).await?;
        }
        Commands::Search {
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
            };
            if options.similar_to.is_some() {
                veclayer::commands::search(&cli.data_dir, query.as_deref().unwrap_or(""), &options)
                    .await?
            } else {
                match query {
                    Some(q) => veclayer::commands::search(&cli.data_dir, &q, &options).await?,
                    None => veclayer::commands::browse(&cli.data_dir, &options).await?,
                }
            }
        }
        Commands::Focus {
            id,
            question,
            limit,
        } => {
            let options = FocusOptions { question, limit };
            veclayer::commands::focus(&cli.data_dir, &id, &options).await?;
        }
        Commands::Serve {
            port,
            host,
            read_only,
            mcp_stdio,
        } => {
            let options = ServeOptions {
                host,
                port,
                read_only,
                mcp_stdio,
            };
            veclayer::commands::serve(&cli.data_dir, &options).await?;
        }
        Commands::Status => {
            veclayer::commands::status(&cli.data_dir).await?;
        }
        Commands::Sources => {
            veclayer::commands::print_sources(&cli.data_dir).await?;
        }
        Commands::Perspective { action } => match action {
            PerspectiveAction::List => {
                veclayer::commands::perspective_list(&cli.data_dir)?;
            }
            PerspectiveAction::Add { id, name, hint } => {
                veclayer::commands::perspective_add(&cli.data_dir, &id, &name, &hint)?;
            }
            PerspectiveAction::Remove { id } => {
                veclayer::commands::perspective_remove(&cli.data_dir, &id)?;
            }
        },
        Commands::History { id } => {
            veclayer::commands::history(&cli.data_dir, &id).await?;
        }
        Commands::Archive { ids } => {
            veclayer::commands::archive(&cli.data_dir, &ids).await?;
        }
        Commands::Export { perspective } => {
            let options = ExportOptions { perspective };
            veclayer::commands::export_entries(&cli.data_dir, &options).await?;
        }
        Commands::Import { path } => {
            let options = ImportOptions { path };
            veclayer::commands::import_entries(&cli.data_dir, &options).await?;
        }
        Commands::Reflect { action } => match action {
            None => {
                // Full identity + reflection report (merged reflect + id)
                veclayer::commands::reflect(&cli.data_dir).await?;
            }
            Some(ReflectAction::Salience { limit }) => {
                let options = CompactOptions {
                    limit,
                    ..Default::default()
                };
                veclayer::commands::compact(&cli.data_dir, CompactAction::Salience, &options)
                    .await?;
            }
            Some(ReflectAction::ArchiveCandidates { limit, threshold }) => {
                let options = CompactOptions {
                    limit,
                    archive_threshold: threshold,
                };
                veclayer::commands::compact(
                    &cli.data_dir,
                    CompactAction::ArchiveCandidates,
                    &options,
                )
                .await?;
            }
        },
        Commands::Think { action } => match action {
            None => {
                // Full LLM sleep cycle
                #[cfg(feature = "llm")]
                {
                    veclayer::commands::think(&cli.data_dir).await?;
                }
                #[cfg(not(feature = "llm"))]
                {
                    eprintln!("Error: `think` (without subcommand) requires the 'llm' feature.");
                    eprintln!("Build with `cargo build` (default features) or `cargo build --features llm`.");
                    std::process::exit(1);
                }
            }
            Some(ThinkAction::Promote { id, visibility }) => {
                veclayer::commands::think_promote(&cli.data_dir, &id, &visibility).await?;
            }
            Some(ThinkAction::Demote { id, visibility }) => {
                veclayer::commands::think_demote(&cli.data_dir, &id, &visibility).await?;
            }
            Some(ThinkAction::Relate {
                source,
                target,
                kind,
            }) => {
                veclayer::commands::think_relate(&cli.data_dir, &source, &target, &kind).await?;
            }
            Some(ThinkAction::Aging { action }) => match action {
                AgingAction::Apply => {
                    veclayer::commands::think_aging_apply(&cli.data_dir).await?;
                }
                AgingAction::Configure { days, to } => {
                    veclayer::commands::think_aging_configure(&cli.data_dir, days, to.as_deref())
                        .await?;
                }
                AgingAction::Rotate => {
                    veclayer::commands::compact(
                        &cli.data_dir,
                        CompactAction::Rotate,
                        &CompactOptions::default(),
                    )
                    .await?;
                }
            },
        },
    }

    Ok(())
}

fn init_logging(verbose: bool, quiet: bool, use_stderr: bool) {
    // Respect RUST_LOG env if set; otherwise use flag-based defaults.
    // Default: only show warnings. --verbose enables veclayer DEBUG.
    // --quiet suppresses everything except errors.
    // Dependency crates (lancedb, lance, ort, etc.) stay at WARN unless --verbose.
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else if quiet {
        EnvFilter::new("error")
    } else if verbose {
        EnvFilter::new("veclayer=debug,info")
    } else {
        EnvFilter::new("warn")
    };

    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false);

    if use_stderr {
        builder.with_writer(std::io::stderr).init();
    } else {
        builder.init();
    }
}
