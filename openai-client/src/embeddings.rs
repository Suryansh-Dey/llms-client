//! Client and types of the `/embeddings` endpoint.
#[cfg(feature = "reqwest")]
pub use client::Embedder;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Text, or a list of texts, to embed.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Text(String),
    Texts(Vec<String>),
}
impl EmbeddingInput {
    /// Count of texts, i.e. of embeddings in the response.
    pub fn len(&self) -> usize {
        match self {
            Self::Text(_) => 1,
            Self::Texts(texts) => texts.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
impl From<&str> for EmbeddingInput {
    fn from(text: &str) -> Self {
        Self::Text(text.into())
    }
}
impl From<String> for EmbeddingInput {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}
impl From<Vec<String>> for EmbeddingInput {
    fn from(texts: Vec<String>) -> Self {
        Self::Texts(texts)
    }
}
impl From<Vec<&str>> for EmbeddingInput {
    fn from(texts: Vec<&str>) -> Self {
        Self::Texts(texts.into_iter().map(Into::into).collect())
    }
}

/// Embedding of one input.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Embedding {
    pub embedding: Vec<f32>,
    /// Position of the input it belongs to.
    #[serde(default)]
    pub index: usize,
}

/// Response of `/embeddings`. Has one embedding per input, in the order of inputs.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EmbeddingResponse {
    pub data: Vec<Embedding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
}
impl EmbeddingResponse {
    /// Embedding of the first input. Empty if there is none.
    pub fn embedding(&self) -> &[f32] {
        self.data.first().map_or(&[], |data| &data.embedding)
    }
    /// Embeddings in the order of inputs.
    pub fn into_embeddings(self) -> Vec<Vec<f32>> {
        self.data.into_iter().map(|data| data.embedding).collect()
    }
}

#[cfg(feature = "reqwest")]
mod client {
    use super::{EmbeddingInput, EmbeddingResponse};
    use crate::http::{authorize, deserialize_value, join_url, read_env, read_json};
    use crate::{DEFAULT_BASE_URL, EnvVarError, Error};
    use reqwest::Client;
    use serde::Serialize;
    use serde::ser::{SerializeMap, Serializer};
    use serde_json::{Map, Value};

    /// Keys of the request body set by `Embedder` itself, so they are ignored in its parameters.
    const RESERVED_KEYS: [&str; 2] = ["model", "input"];

    struct EmbeddingRequest<'a> {
        model: &'a str,
        input: &'a EmbeddingInput,
        /// Entries are added to the top level of the body.
        params: &'a Map<String, Value>,
    }
    impl Serialize for EmbeddingRequest<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(None)?;
            map.serialize_entry("model", self.model)?;
            map.serialize_entry("input", self.input)?;
            for (key, value) in self.params {
                if !RESERVED_KEYS.contains(&key.as_str()) {
                    map.serialize_entry(key, value)?;
                }
            }
            map.end()
        }
    }

    /// Client of any OpenAI compatible `/embeddings` endpoint.
    ///
    /// # Example
    /// ```no_run
    /// use open_client::embeddings::Embedder;
    /// # async fn run() -> Result<(), open_client::Error> {
    /// let mut embedder = Embedder::new("API_KEY", "text-embedding-3-small");
    /// embedder.dimensions(256);
    /// let response = embedder.embed("Hello, world!").await?;
    /// println!("{} dimensions", response.embedding().len());
    /// # Ok(())
    /// # }
    /// ```
    /// `Debug` masks the API key, so it can't leak through logs or a panic message.
    #[derive(Clone)]
    pub struct Embedder {
        client: Client,
        api_key: String,
        base_url: String,
        model: String,
        params: Map<String, Value>,
    }
    impl std::fmt::Debug for Embedder {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Embedder")
                .field("client", &self.client)
                .field("api_key", &"***")
                .field("base_url", &self.base_url)
                .field("model", &self.model)
                .field("params", &self.params)
                .finish()
        }
    }

    impl Embedder {
        /// A client of the OpenAI API. Set `base_url()` for any other API.
        ///
        /// # Arguments
        /// * `api_key` - Your API key. Use `with_no_api_key()` for servers without one.
        /// * `model` - The embedding model to use (e.g., `"text-embedding-3-small"`).
        pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
            Self::with_client(api_key, model, Client::default())
        }
        /// A client configured by environment variables, like the official OpenAI SDKs:
        /// * `OPENAI_API_KEY` - Your API key. Required.
        /// * `OPENAI_BASE_URL` - Base URL of any OpenAI compatible API. Default is
        ///   `DEFAULT_BASE_URL`.
        ///
        /// # Errors
        /// If `OPENAI_API_KEY` is not set, or any of them isn't valid unicode.
        pub fn from_env(model: impl Into<String>) -> Result<Self, EnvVarError> {
            let (api_key, base_url) = read_env()?;
            let mut embedder = Self::new(api_key, model);
            embedder.base_url(base_url);
            Ok(embedder)
        }
        /// A client sending no `Authorization` header, for servers without API keys, e.g. local
        /// servers. Set their URL with `base_url()`.
        pub fn with_no_api_key(model: impl Into<String>) -> Self {
            Self::new("", model)
        }
        /// Like `new()`, with a `reqwest::Client` configured e.g. with a timeout or default
        /// headers.
        pub fn with_client(
            api_key: impl Into<String>,
            model: impl Into<String>,
            client: Client,
        ) -> Self {
            Self {
                client,
                api_key: api_key.into(),
                base_url: DEFAULT_BASE_URL.to_string(),
                model: model.into(),
                params: Map::new(),
            }
        }

        /// Base URL of any OpenAI compatible API. Requests are sent to `{base_url}/embeddings`.
        pub fn base_url(&mut self, base_url: impl Into<String>) -> &mut Self {
            self.base_url = base_url.into().trim_end_matches('/').to_string();
            self
        }
        pub fn api_key(&mut self, api_key: impl Into<String>) -> &mut Self {
            self.api_key = api_key.into();
            self
        }
        pub fn model(&mut self, model: impl Into<String>) -> &mut Self {
            self.model = model.into();
            self
        }
        /// Sets any parameter of the request body, like `AI::param()`. `model` and `input` are
        /// ignored as `Embedder` sets them.
        ///
        /// # Panics
        /// If `value` can't be serialized to JSON, e.g. a map with non-string keys.
        pub fn param(&mut self, key: impl Into<String>, value: impl Serialize) -> &mut Self {
            let key = key.into();
            let value = serde_json::to_value(value)
                .unwrap_or_else(|e| panic!("parameter `{key}` can't be serialized to JSON: {e}"));
            if value.is_null() {
                self.params.remove(&key);
            } else {
                self.params.insert(key, value);
            }
            self
        }
        /// Parameters added to the top level of the request body, for editing nested values in
        /// place.
        pub fn params_mut(&mut self) -> &mut Map<String, Value> {
            &mut self.params
        }
        /// Number of dimensions of the embeddings. Only supported by some models, e.g.
        /// `text-embedding-3` and later.
        pub fn dimensions(&mut self, dimensions: u32) -> &mut Self {
            self.param("dimensions", dimensions)
        }

        pub fn get_base_url(&self) -> &str {
            &self.base_url
        }
        pub fn get_model(&self) -> &str {
            &self.model
        }
        pub fn get_params(&self) -> &Map<String, Value> {
            &self.params
        }

        /// Embeds a text, or a list of texts, e.g. `embed("Hi")` or `embed(vec!["Hi", "Bye"])`.
        /// The response has one embedding per text, in the same order.
        pub async fn embed(
            &self,
            input: impl Into<EmbeddingInput>,
        ) -> Result<EmbeddingResponse, Error> {
            let input = input.into();
            let body = EmbeddingRequest {
                model: &self.model,
                input: &input,
                params: &self.params,
            };
            let request = self
                .client
                .post(join_url(&self.base_url, "embeddings"))
                .json(&body);
            let response = authorize(request, &self.api_key).send().await?;
            let mut response: EmbeddingResponse = deserialize_value(&read_json(response).await?)?;
            if response.data.len() != input.len() {
                return Err(Error::InvalidResponse(format!(
                    "Expected {} embeddings, received {}",
                    input.len(),
                    response.data.len()
                )));
            }
            response.data.sort_by_key(|data| data.index);
            Ok(response)
        }
    }
}
