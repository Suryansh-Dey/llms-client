use super::ask::BASE_URL;
use super::error::GeminiResponseError;
use super::types::embedding::*;
use super::types::request::Part;
use reqwest::Client;

/// Client for generating embeddings using Gemini embedding models.
///
/// # Example
/// ```no_run
/// use gemini_client_api::gemini::embed::GeminiEmbedding;
/// use gemini_client_api::gemini::types::embedding::TaskType;
///
/// # async fn run() {
/// let embedder = GeminiEmbedding::new("YOUR_API_KEY", "gemini-embedding-001")
///     .set_task_type(TaskType::RetrievalDocument);
///
/// let response = embedder.embed_text("Hello, world!").await.unwrap();
/// println!("Embedding dimension: {}", response.embedding().values().len());
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct GeminiEmbedding {
    client: Client,
    api_key: String,
    model: String,
    config: Option<EmbedContentConfig>,
}

impl GeminiEmbedding {
    /// Creates a new `GeminiEmbedding` client.
    ///
    /// # Arguments
    /// * `api_key` - Your Gemini API key. Get one from [Google AI studio](https://aistudio.google.com/app/apikey).
    /// * `model` - The embedding model to use (e.g., `"gemini-embedding-001"`).
    ///   See [embedding models](https://ai.google.dev/gemini-api/docs/models#gemini-embedding).
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::default(),
            api_key: api_key.into(),
            model: model.into(),
            config: None,
        }
    }
    /// Creates a new `GeminiEmbedding` client with a custom `reqwest::Client`.
    ///
    /// # Arguments
    /// * `api_key` - Your Gemini API key.
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
            model: model.into(),
            config: None,
        }
    }
    /// Sets the task type for the embedding.
    ///
    /// The task type helps the model produce better embeddings tailored for the specific use case.
    pub fn set_task_type(mut self, task_type: TaskType) -> Self {
        let output_dimensionality = self
            .config
            .as_ref()
            .and_then(|c| c.output_dimensionality().clone());
        self.config = Some(EmbedContentConfig::new(
            Some(task_type),
            output_dimensionality,
        ));
        self
    }
    /// Sets the output dimensionality for the embedding.
    ///
    /// Allows reducing the embedding dimension for storage/performance optimization
    /// via [Matryoshka Representation Learning](https://ai.google.dev/gemini-api/docs/embeddings#matryoshka).
    pub fn set_output_dimensionality(mut self, output_dimensionality: u32) -> Self {
        let task_type = self.config.as_ref().and_then(|c| c.task_type().clone());
        self.config = Some(EmbedContentConfig::new(
            task_type,
            Some(output_dimensionality),
        ));
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
    /// Sets the full embedding configuration, replacing any previously set config.
    pub fn set_config(mut self, config: EmbedContentConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Generates an embedding for the given content parts.
    ///
    /// # Arguments
    /// * `content` - The content parts to embed (e.g., text, inline data).
    ///
    /// # Errors
    /// Returns `GeminiResponseError` on network failure or API error.
    pub async fn embed(
        &self,
        content: Vec<Part>,
    ) -> Result<EmbedContentResponse, GeminiResponseError> {
        let req_url = format!(
            "{BASE_URL}/{}:embedContent?key={}",
            self.model, self.api_key
        );

        let request_body = EmbedContentRequest {
            model: format!("models/{}", self.model),
            content: Content::new(content),
            task_type: self.config.as_ref().and_then(|c| c.task_type().clone()),
            output_dimensionality: self
                .config
                .as_ref()
                .and_then(|c| c.output_dimensionality().clone()),
        };

        let response = self
            .client
            .post(req_url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| GeminiResponseError::ReqwestError(e))?;

        if !response.status().is_success() {
            let error = response
                .json()
                .await
                .map_err(|e| GeminiResponseError::ReqwestError(e))?;
            return Err(GeminiResponseError::StatusNotOk(error));
        }

        let embed_response: EmbedContentResponse = response
            .json()
            .await
            .map_err(|e| GeminiResponseError::ReqwestError(e))?;
        Ok(embed_response)
    }

    /// Convenience method to generate an embedding for a single text string.
    ///
    /// # Arguments
    /// * `text` - The text to embed.
    ///
    /// # Example
    /// ```no_run
    /// # use gemini_client_api::gemini::embed::GeminiEmbedding;
    /// # async fn run() {
    /// let embedder = GeminiEmbedding::new("YOUR_API_KEY", "gemini-embedding-001");
    /// let response = embedder.embed_text("What is the meaning of life?").await.unwrap();
    /// println!("Got {} dimensions", response.embedding().values().len());
    /// # }
    /// ```
    pub async fn embed_text(
        &self,
        text: impl Into<String>,
    ) -> Result<EmbedContentResponse, GeminiResponseError> {
        let part: Part = text.into().into();
        self.embed(vec![part]).await
    }
}
