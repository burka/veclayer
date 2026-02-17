use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing::Level;
use tracing_subscriber::EnvFilter;

use veclayer::commands::{IngestOptions, QueryOptions, ServeOptions};
use veclayer::Result;

#[derive(Parser)]
#[command(name = "veclayer")]
#[command(about = "Hierarchical vector indexing for documents")]
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
    /// Ingest documents into the vector store (recursive + summarize by default)
    Ingest {
        /// Path to a file or directory to ingest
        path: PathBuf,

        /// Disable recursive directory processing
        #[arg(long)]
        no_recursive: bool,

        /// Disable cluster summarization (faster, but no cross-doc links)
        #[arg(long)]
        no_summarize: bool,

        /// Ollama model for summarization
        #[arg(long, env = "VECLAYER_OLLAMA_MODEL", default_value = "llama3.2")]
        model: String,

        /// Visibility to assign to all ingested chunks (e.g. "always", "deep_only", or any custom value)
        #[arg(long)]
        visibility: Option<String>,
    },

    /// Query the vector store with hierarchical search
    Query {
        /// The search query
        query: String,

        /// Number of top results to return
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,

        /// Show full hierarchy path
        #[arg(short = 'p', long)]
        show_path: bool,

        /// Search within a specific subtree (parent chunk ID)
        #[arg(long)]
        subtree: Option<String>,

        /// Deep search: include all visibilities (deep_only, expired, custom)
        #[arg(long)]
        deep: bool,
    },

    /// Start the MCP server
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

        /// Enable MCP stdio transport (for Claude integration)
        #[arg(long)]
        mcp_stdio: bool,
    },

    /// Show statistics about the vector store
    Stats,

    /// List all indexed source files
    Sources,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    init_logging(cli.verbose);

    match cli.command {
        Commands::Ingest {
            path,
            no_recursive,
            no_summarize,
            model,
            visibility,
        } => {
            let options = IngestOptions {
                recursive: !no_recursive,
                summarize: !no_summarize,
                model,
                visibility,
            };
            veclayer::commands::ingest(&cli.data_dir, &path, &options).await?;
        }
        Commands::Query {
            query,
            top_k,
            show_path,
            subtree,
            deep,
        } => {
            let options = QueryOptions {
                top_k,
                show_path,
                subtree,
                deep,
            };
            let results = veclayer::commands::query(&cli.data_dir, &query, &options).await?;

            if results.is_empty() {
                println!("No results found.");
                return Ok(());
            }

            println!("\nSearch results for: \"{}\"\n", query);
            println!("{}", "=".repeat(60));

            for (i, result) in results.iter().enumerate() {
                println!(
                    "\n{}. [Score: {:.3}] {}",
                    i + 1,
                    result.score,
                    result.chunk.level
                );

                if show_path && !result.hierarchy_path.is_empty() {
                    let path: Vec<&str> = result
                        .hierarchy_path
                        .iter()
                        .filter_map(|c| c.heading.as_deref())
                        .collect();
                    println!("   Path: {}", path.join(" > "));
                }

                println!("   Source: {}", result.chunk.source_file);

                if let Some(ref heading) = result.chunk.heading {
                    println!("   Heading: {}", heading);
                }

                let content = if result.chunk.content.len() > 200 {
                    format!("{}...", &result.chunk.content[..200])
                } else {
                    result.chunk.content.clone()
                };
                println!("   Content: {}", content.replace('\n', " "));

                if !result.relevant_children.is_empty() {
                    println!("   Relevant children:");
                    for child in &result.relevant_children {
                        let child_preview = if child.chunk.content.len() > 80 {
                            format!("{}...", &child.chunk.content[..80])
                        } else {
                            child.chunk.content.clone()
                        };
                        println!(
                            "     - [Score: {:.3}] {}",
                            child.score,
                            child_preview.replace('\n', " ")
                        );
                    }
                }

                println!("   ID: {}", result.chunk.id);
            }
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
        Commands::Stats => {
            let stats = veclayer::commands::stats(&cli.data_dir).await?;

            println!("VecLayer Statistics");
            println!("{}", "=".repeat(40));
            println!("Total chunks: {}", stats.total_chunks);
            println!("\nChunks by level:");
            for level in 1..=7 {
                if let Some(count) = stats.chunks_by_level.get(&level) {
                    let level_name = if level <= 6 {
                        format!("H{}", level)
                    } else {
                        "Content".to_string()
                    };
                    println!("  {}: {}", level_name, count);
                }
            }
            println!("\nSource files: {}", stats.source_files.len());
        }
        Commands::Sources => {
            let sources = veclayer::commands::sources(&cli.data_dir).await?;

            if sources.is_empty() {
                println!("No files indexed.");
            } else {
                println!("Indexed source files:");
                for file in &sources {
                    println!("  {}", file);
                }
            }
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
