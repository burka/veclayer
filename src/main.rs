use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing::Level;
use tracing_subscriber::EnvFilter;

use veclayer::commands::{
    AddOptions, CompactAction, CompactOptions, FocusOptions, SearchOptions, ServeOptions,
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

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

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

        /// This entry summarizes the given entry ID
        #[arg(long)]
        summarizes: Option<String>,

        /// This entry supersedes the given entry ID
        #[arg(long)]
        supersedes: Option<String>,

        /// This is a new version of the given entry ID
        #[arg(long)]
        version_of: Option<String>,
    },

    /// Semantic search with hierarchical results
    #[command(alias = "s")]
    Search {
        /// The search query
        query: String,

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

    /// Compact: rotate access profiles, compute salience, suggest archival
    #[command(alias = "c")]
    Compact {
        #[command(subcommand)]
        action: CompactActionCmd,
    },

    /// Show identity summary (perspectives, core knowledge, open threads)
    Id,

    /// Generate a comprehensive reflection report (priming-ready)
    Reflect,

    /// Run one think cycle: reflect → LLM → add → compact (requires LLM)
    Think,
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
enum CompactActionCmd {
    /// Roll access-profile time buckets and apply aging rules
    Rotate,

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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let is_mcp_stdio = matches!(&cli.command, Some(Commands::Serve { mcp_stdio: true, .. }));

    init_logging(cli.verbose, is_mcp_stdio);

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
            summarizes,
            supersedes,
            version_of,
        } => {
            let options = AddOptions {
                recursive: !no_recursive,
                summarize: !no_summarize,
                model,
                visibility,
                entry_type,
                perspectives,
                summarizes,
                supersedes,
                version_of,
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
        } => {
            let options = SearchOptions {
                top_k,
                show_path,
                subtree,
                deep,
                recent,
                perspective,
            };
            veclayer::commands::search(&cli.data_dir, &query, &options).await?;
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
        Commands::Compact { action } => {
            let (compact_action, options) = match action {
                CompactActionCmd::Rotate => (CompactAction::Rotate, CompactOptions::default()),
                CompactActionCmd::Salience { limit } => (
                    CompactAction::Salience,
                    CompactOptions {
                        limit,
                        ..Default::default()
                    },
                ),
                CompactActionCmd::ArchiveCandidates { limit, threshold } => (
                    CompactAction::ArchiveCandidates,
                    CompactOptions {
                        limit,
                        archive_threshold: threshold,
                        ..Default::default()
                    },
                ),
            };
            veclayer::commands::compact(&cli.data_dir, compact_action, &options).await?;
        }
        Commands::Id => {
            veclayer::commands::identity(&cli.data_dir).await?;
        }
        Commands::Reflect => {
            veclayer::commands::reflect(&cli.data_dir).await?;
        }
        #[cfg(feature = "llm")]
        Commands::Think => {
            veclayer::commands::think(&cli.data_dir).await?;
        }
        #[cfg(not(feature = "llm"))]
        Commands::Think => {
            eprintln!("Error: `think` requires the 'llm' feature. Build with `cargo build` (default features) or `cargo build --features llm`.");
            std::process::exit(1);
        }
    }

    Ok(())
}

fn init_logging(verbose: bool, use_stderr: bool) {
    let filter = if verbose {
        EnvFilter::new(Level::DEBUG.to_string())
    } else {
        EnvFilter::new(Level::INFO.to_string())
    };

    let builder = tracing_subscriber::fmt().with_env_filter(filter).with_target(false);

    if use_stderr {
        builder.with_writer(std::io::stderr).init();
    } else {
        builder.init();
    }
}