use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber::EnvFilter;

use veclayer::embedder::FastEmbedder;
use veclayer::parser::MarkdownParser;
use veclayer::search::{HierarchicalSearch, SearchConfig};
use veclayer::store::LanceStore;
use veclayer::{Config, DocumentParser, Embedder, Result, VectorStore};

#[derive(Parser)]
#[command(name = "veclayer")]
#[command(about = "Hierarchical vector indexing for documents")]
#[command(version)]
struct Cli {
    /// Data directory for VecLayer storage
    #[arg(short, long, env = "VECLAYER_DATA_DIR", default_value = "./veclayer-data")]
    data_dir: PathBuf,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest documents into the vector store
    Ingest {
        /// Path to a file or directory to ingest
        path: PathBuf,

        /// Recursively process directories
        #[arg(short, long)]
        recursive: bool,
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

    // Initialize logging
    let filter = if cli.verbose {
        EnvFilter::new(Level::DEBUG.to_string())
    } else {
        EnvFilter::new(Level::INFO.to_string())
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    match cli.command {
        Commands::Ingest { path, recursive } => {
            cmd_ingest(&cli.data_dir, &path, recursive).await?;
        }
        Commands::Query {
            query,
            top_k,
            show_path,
            subtree,
        } => {
            cmd_query(&cli.data_dir, &query, top_k, show_path, subtree).await?;
        }
        Commands::Serve {
            port,
            host,
            read_only,
            mcp_stdio,
        } => {
            cmd_serve(&cli.data_dir, &host, port, read_only, mcp_stdio).await?;
        }
        Commands::Stats => {
            cmd_stats(&cli.data_dir).await?;
        }
        Commands::Sources => {
            cmd_sources(&cli.data_dir).await?;
        }
    }

    Ok(())
}

async fn cmd_ingest(data_dir: &PathBuf, path: &PathBuf, recursive: bool) -> Result<()> {
    info!("Initializing embedder...");
    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();

    info!("Opening vector store at {:?}...", data_dir);
    let store = LanceStore::open(data_dir, dimension).await?;

    let parser = MarkdownParser::new();

    let files = collect_files(path, recursive, &parser)?;
    info!("Found {} files to process", files.len());

    for file in files {
        info!("Processing {:?}...", file);

        // Delete existing chunks from this file
        let deleted = store.delete_by_source(&file.to_string_lossy()).await?;
        if deleted > 0 {
            info!("  Removed {} existing chunks", deleted);
        }

        // Parse the file
        let mut chunks = parser.parse_file(&file)?;
        info!("  Parsed {} chunks", chunks.len());

        if chunks.is_empty() {
            continue;
        }

        // Generate embeddings
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings = embedder.embed(&texts)?;

        // Attach embeddings to chunks
        for (chunk, embedding) in chunks.iter_mut().zip(embeddings.into_iter()) {
            chunk.embedding = Some(embedding);
        }

        // Insert into store
        store.insert_chunks(chunks).await?;
        info!("  Indexed successfully");
    }

    info!("Ingestion complete!");
    Ok(())
}

async fn cmd_query(
    data_dir: &PathBuf,
    query: &str,
    top_k: usize,
    show_path: bool,
    subtree: Option<String>,
) -> Result<()> {
    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = LanceStore::open(data_dir, dimension).await?;

    let config = SearchConfig {
        top_k,
        children_k: 3,
        max_depth: 3,
        min_score: 0.0,
    };

    let search = HierarchicalSearch::new(store, embedder).with_config(config);

    let results = if let Some(ref parent_id) = subtree {
        search.search_subtree(query, parent_id).await?
    } else {
        search.search(query).await?
    };

    if results.is_empty() {
        println!("No results found.");
        return Ok(());
    }

    println!("\nSearch results for: \"{}\"\n", query);
    println!("{}", "=".repeat(60));

    for (i, result) in results.iter().enumerate() {
        println!("\n{}. [Score: {:.3}] {}", i + 1, result.score, result.chunk.level);

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

        // Truncate content for display
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

    Ok(())
}

async fn cmd_serve(
    data_dir: &PathBuf,
    host: &str,
    port: u16,
    read_only: bool,
    mcp_stdio: bool,
) -> Result<()> {
    let config = Config::default()
        .with_data_dir(data_dir)
        .with_host(host)
        .with_port(port)
        .with_read_only(read_only);

    if mcp_stdio {
        info!("Starting MCP server on stdio...");
        veclayer::mcp::run_stdio(config).await
    } else {
        info!("Starting HTTP server on {}:{}...", host, port);
        veclayer::mcp::run_http(config).await
    }
}

async fn cmd_stats(data_dir: &PathBuf) -> Result<()> {
    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = LanceStore::open(data_dir, dimension).await?;

    let stats = store.stats().await?;

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

    Ok(())
}

async fn cmd_sources(data_dir: &PathBuf) -> Result<()> {
    let embedder = FastEmbedder::new()?;
    let dimension = embedder.dimension();
    let store = LanceStore::open(data_dir, dimension).await?;

    let stats = store.stats().await?;

    if stats.source_files.is_empty() {
        println!("No files indexed.");
    } else {
        println!("Indexed source files:");
        for file in &stats.source_files {
            println!("  {}", file);
        }
    }

    Ok(())
}

fn collect_files(
    path: &PathBuf,
    recursive: bool,
    parser: &impl DocumentParser,
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if path.is_file() {
        if parser.can_parse(path) {
            files.push(path.clone());
        }
    } else if path.is_dir() {
        if recursive {
            for entry in walkdir::WalkDir::new(path)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let entry_path = entry.path();
                if entry_path.is_file() && parser.can_parse(entry_path) {
                    files.push(entry_path.to_path_buf());
                }
            }
        } else {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();
                if entry_path.is_file() && parser.can_parse(&entry_path) {
                    files.push(entry_path);
                }
            }
        }
    }

    Ok(files)
}
