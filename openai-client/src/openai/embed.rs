use super::ask::{DEFAULT_BASE_URL, authorize, deserialize_value, join_url, read_json};
use super::error::OpenAIResponseError;
use super::types::embedding::*;
use derive_getters::{Dissolve, Getters};
use reqwest::Client;

/// Client for generating embeddings using any OpenAI compatible `/embeddings` endpoint.
///
/// # Example
/// ```no_run
/// use open_client::openai::embed::OpenAIEmbedding;
///
/// # async fn run() {
/// let embedder = OpenAIEmbedding::new("YOUR_API_KEY", "text-embedding-3-small")
///     .set_dimensions(256);
///
/// let response = embedder.embed_text("Hello, world!").await.unwrap();
/// println!("Embedding dimension: {}", response.embedding().values().len());
/// # }
/// ```
/// `Debug` masks `api_key`, so it can't leak through logs or a panic message.
#[derive(Clone, Getters, Dissolve)]
pub struct OpenAIEmbedding {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    dimensions: Option<u32>,
}
impl std::fmt::Debug for OpenAIEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIEmbedding")
            .field("client", &self.client)
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("dimensions", &self.dimensions)
            .finish()
    }
}

impl OpenAIEmbedding {
    /// Creates a new `OpenAIEmbedding` client for OpenAI API. Use `set_base_url()` for any other API.
    ///
    /// # Arguments
    /// * `api_key` - Your API key. Empty key sends no `Authorization` header.
    /// * `model` - The embedding model to use (e.g., `"text-embedding-3-small"`).
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new_with_client(api_key, model, Client::default())
    }
    /// Creates a new `OpenAIEmbedding` client with a custom `reqwest::Client`.
    ///
    /// # Arguments
    /// * `api_key` - Your API key.
    /// * `model` - The embedding model to use.
    /// * `client` - A custom `reqwest::Client` for making requests.
    pub fn new_with_client(
        api_key: impl Into<String>,
        model: impl Into<String>,
        client: Client,
    ) -> Self {
        Self {
            client,
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: model.into(),
            dimensions: None,
        }
    }
    /// Sets the base URL of any OpenAI compatible API. Requests are sent to `{base_url}/embeddings`.
    pub fn set_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }
    /// Sets the number of dimensions of the output embeddings. Only supported by some models,
    /// e.g. `text-embedding-3` and later.
    pub fn set_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
    pub fn set_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }
    pub fn set_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Generates one embedding per text, in the same order.
    ///
    /// # Errors
    /// Returns `OpenAIResponseError` on network failure or API error.
    pub async fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> Result<EmbeddingResponse, OpenAIResponseError> {
        let request = self
            .client
            .post(join_url(&self.base_url, "embeddings"))
            .json(&EmbeddingRequest {
                model: &self.model,
                input: &texts,
                dimensions: self.dimensions,
            });
        let response = authorize(request, &self.api_key).send().await?;
        let value = read_json(response).await?;
        let mut response: EmbeddingResponse = deserialize_value(&value)?;
        if response.embeddings().len() != texts.len() {
            return Err(OpenAIResponseError::InvalidResponseFormat(format!(
                "Expected {} embeddings, received {}",
                texts.len(),
                response.embeddings().len()
            )));
        }
        response.sort_by_index();
        Ok(response)
    }

    /// Convenience method to generate an embedding for a single text string.
    ///
    /// # Example
    /// ```no_run
    /// # use open_client::openai::embed::OpenAIEmbedding;
    /// # async fn run() {
    /// let embedder = OpenAIEmbedding::new("YOUR_API_KEY", "text-embedding-3-small");
    /// let response = embedder.embed_text("What is the meaning of life?").await.unwrap();
    /// println!("Got {} dimensions", response.embedding().values().len());
    /// # }
    /// ```
    pub async fn embed_text(
        &self,
        text: impl Into<String>,
    ) -> Result<EmbeddingResponse, OpenAIResponseError> {
        self.embed_texts(vec![text.into()]).await
    }
}
