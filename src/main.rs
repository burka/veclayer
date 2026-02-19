use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing::Level;
use tracing_subscriber::EnvFilter;

use veclayer::commands::{AddOptions, FocusOptions, SearchOptions, ServeOptions};
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
    command: Commands,
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    init_logging(cli.verbose);

    match cli.command {
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
        } => {
            let options = AddOptions {
                recursive: !no_recursive,
                summarize: !no_summarize,
                model,
                visibility,
                entry_type,
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
        } => {
            let options = SearchOptions {
                top_k,
                show_path,
                subtree,
                deep,
                recent,
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
    }

    Ok(())
}

fn init_logging(verbose: bool) {
    let filter = if verbose {
        EnvFilter::new(Level::DEBUG.to_string())
    } else {
        EnvFilter::new(Level::INFO.to_string())
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}
