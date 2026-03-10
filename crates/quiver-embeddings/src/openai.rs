//! OpenAI Embeddings API provider.
//!
//! Supported models: `text-embedding-3-small`, `text-embedding-3-large`,
//! `text-embedding-ada-002`.
//!
//! Requires the `OPENAI_API_KEY` environment variable (or pass the key
//! explicitly via [`OpenAiProvider::with_api_key`]).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{EmbeddingError, EmbeddingProvider};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// OpenAI-compatible embeddings provider.
///
/// Works with the official OpenAI API and any compatible endpoint
/// (Azure OpenAI, local proxies, etc.) by setting a custom base URL.
#[derive(Clone)]
pub struct OpenAiProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    dimensions: Option<usize>,
}

impl OpenAiProvider {
    /// Create a provider using the `OPENAI_API_KEY` environment variable.
    ///
    /// # Errors
    /// Returns [`EmbeddingError::Config`] if the env var is not set.
    pub fn from_env(model: impl Into<String>) -> Result<Self, EmbeddingError> {
        let key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| EmbeddingError::Config("OPENAI_API_KEY env var not set".into()))?;
        Ok(Self::with_api_key(key, model))
    }

    /// Create a provider with an explicit API key.
    pub fn with_api_key(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let model = model.into();
        // text-embedding-3-small → 1536, text-embedding-3-large → 3072, ada-002 → 1536
        let dimensions = match model.as_str() {
            "text-embedding-3-small" | "text-embedding-ada-002" => Some(1536),
            "text-embedding-3-large" => Some(3072),
            _ => None,
        };
        Self {
            client: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            model,
            dimensions,
        }
    }

    /// Override the base URL (for Azure OpenAI, local proxies, etc.).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Override the known dimensions (useful for custom-dimension models).
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = Some(dims);
        self
    }
}

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [String],
    model: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct ApiError {
    error: ApiErrorBody,
}

#[derive(Deserialize)]
struct ApiErrorBody {
    message: String,
}

// ── Trait impl ────────────────────────────────────────────────────────────────

#[async_trait]
impl EmbeddingProvider for OpenAiProvider {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> Option<usize> {
        self.dimensions
    }

    async fn embed(&self, text: String) -> Result<Vec<f32>, EmbeddingError> {
        let mut vecs = self.embed_batch(vec![text]).await?;
        vecs.pop().ok_or_else(|| EmbeddingError::Format("empty response".into()))
    }

    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let url = format!("{}/embeddings", self.base_url);
        let body = EmbedRequest { input: &texts, model: &self.model };

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            let message = serde_json::from_str::<ApiError>(&text)
                .map(|e| e.error.message)
                .unwrap_or(text);
            return Err(EmbeddingError::Api { status, message });
        }

        let data: EmbedResponse = resp.json().await?;
        Ok(data.data.into_iter().map(|d| d.embedding).collect())
    }
}
