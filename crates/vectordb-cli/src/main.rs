use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "vdb", about = "vectordb CLI — interact with a running vectordb-server")]
struct Cli {
    /// Base URL of the vectordb server
    #[arg(long, env = "VDB_HOST", default_value = "http://localhost:8080")]
    host: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List all collections
    List,

    /// Create a new collection
    Create {
        name: String,
        #[arg(long)]
        dimensions: usize,
        /// Metric: l2 | cosine | dot_product
        #[arg(long, default_value = "cosine")]
        metric: String,
        /// Index type: flat | hnsw
        #[arg(long, default_value = "hnsw")]
        index: String,
    },

    /// Delete a collection
    Drop { name: String },

    /// Insert or update a vector (comma-separated floats)
    Insert {
        collection: String,
        #[arg(long)]
        id: u64,
        /// e.g. "0.1,0.2,0.3"
        #[arg(long)]
        vector: String,
    },

    /// Search for nearest neighbours
    Search {
        collection: String,
        /// Query vector as comma-separated floats
        #[arg(long)]
        vector: String,
        #[arg(long, default_value = "5")]
        k: usize,
    },

    /// Delete a vector by ID
    Delete {
        collection: String,
        #[arg(long)]
        id: u64,
    },
}

fn parse_vector(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|x| x.trim().parse::<f32>().context("invalid float in vector"))
        .collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = reqwest::Client::new();
    let base = cli.host.trim_end_matches('/');

    match cli.command {
        Commands::List => {
            let resp: Vec<String> = client
                .get(format!("{base}/collections"))
                .send()
                .await?
                .json()
                .await?;
            for name in resp {
                println!("{name}");
            }
        }

        Commands::Create { name, dimensions, metric, index } => {
            let body = serde_json::json!({
                "dimensions": dimensions,
                "metric": metric,
                "index_type": index,
            });
            let status = client
                .post(format!("{base}/collections/{name}"))
                .json(&body)
                .send()
                .await?
                .status();
            println!("HTTP {status}");
        }

        Commands::Drop { name } => {
            let status = client
                .delete(format!("{base}/collections/{name}"))
                .send()
                .await?
                .status();
            println!("HTTP {status}");
        }

        Commands::Insert { collection, id, vector } => {
            let vec = parse_vector(&vector)?;
            let body = serde_json::json!({ "id": id, "vector": vec });
            let status = client
                .post(format!("{base}/collections/{collection}/vectors"))
                .json(&body)
                .send()
                .await?
                .status();
            println!("HTTP {status}");
        }

        Commands::Search { collection, vector, k } => {
            let vec = parse_vector(&vector)?;
            let body = serde_json::json!({ "vector": vec, "k": k });
            let resp: serde_json::Value = client
                .post(format!("{base}/collections/{collection}/search"))
                .json(&body)
                .send()
                .await?
                .json()
                .await?;
            println!("{}", serde_json::to_string_pretty(&resp)?);
        }

        Commands::Delete { collection, id } => {
            let status = client
                .delete(format!("{base}/collections/{collection}/vectors/{id}"))
                .send()
                .await?
                .status();
            println!("HTTP {status}");
        }
    }

    Ok(())
}
