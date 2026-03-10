//! Ollama local embedding provider.
//!
//! Calls the Ollama REST API (`POST /api/embeddings`) which runs entirely
//! locally.  Start Ollama and pull a model first:
//!
//! ```bash
//! ollama pull nomic-embed-text
//! ```
//!
//! Default base URL: `http://localhost:11434`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{EmbeddingError, EmbeddingProvider};

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// Ollama local-inference embedding provider.
#[derive(Clone)]
pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaProvider {
    /// Create a provider pointing at the default local Ollama instance.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            model: model.into(),
        }
    }

    /// Use the default `http://localhost:11434` base URL.
    pub fn local(model: impl Into<String>) -> Self {
        Self::new(DEFAULT_BASE_URL, model)
    }
}

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct ApiError {
    error: String,
}

// ── Trait impl ────────────────────────────────────────────────────────────────

#[async_trait]
impl EmbeddingProvider for OllamaProvider {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> Option<usize> {
        // Ollama doesn't advertise dimensions; caller must check after the first embed.
        None
    }

    async fn embed(&self, text: String) -> Result<Vec<f32>, EmbeddingError> {
        let url = format!("{}/api/embeddings", self.base_url);
        let body = EmbedRequest { model: &self.model, prompt: &text };

        let resp = self.client.post(&url).json(&body).send().await?;

        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            let message = serde_json::from_str::<ApiError>(&text)
                .map(|e| e.error)
                .unwrap_or(text);
            return Err(EmbeddingError::Api { status, message });
        }

        let data: EmbedResponse = resp.json().await?;
        if data.embedding.is_empty() {
            return Err(EmbeddingError::Format("Ollama returned empty embedding".into()));
        }
        Ok(data.embedding)
    }

    /// Ollama's `/api/embeddings` is single-input only; fall back to sequential calls.
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t).await?);
        }
        Ok(out)
    }
}
