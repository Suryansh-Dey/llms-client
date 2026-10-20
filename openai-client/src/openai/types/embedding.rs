use derive_getters::{Dissolve, Getters};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Request body of the `/embeddings` endpoint.
#[cfg(feature = "reqwest")]
#[derive(Serialize, Debug, Clone)]
pub(crate) struct EmbeddingRequest<'a> {
    pub model: &'a str,
    pub input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
}

/// A list of floats representing the embedding of one input.
#[derive(Dissolve, Serialize, Deserialize, Debug, Clone, Getters)]
pub struct Embedding {
    #[serde(rename = "embedding")]
    values: Vec<f32>,
    /// Position of the input it belongs to.
    #[serde(default)]
    index: usize,
}
impl Embedding {
    pub fn new(values: Vec<f32>, index: usize) -> Self {
        Self { values, index }
    }
    pub fn values_owned(self) -> Vec<f32> {
        self.values
    }
    pub fn dimension(&self) -> usize {
        self.values().len()
    }
}

/// Response from the `/embeddings` endpoint. Has one embedding per input, in the order of inputs.
#[derive(Dissolve, Serialize, Deserialize, Debug, Clone, Getters)]
pub struct EmbeddingResponse {
    #[serde(rename = "data")]
    embeddings: Vec<Embedding>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<Value>,
}
impl EmbeddingResponse {
    /// Embedding of the first input. Use `embeddings()` for all of them.
    pub fn embedding(&self) -> &Embedding {
        &self.embeddings[0]
    }
    pub fn embedding_owned(self) -> Embedding {
        self.embeddings.into_iter().next().unwrap()
    }
    pub fn embeddings_owned(self) -> Vec<Embedding> {
        self.embeddings
    }
    #[cfg(feature = "reqwest")]
    pub(crate) fn sort_by_index(&mut self) {
        self.embeddings.sort_by_key(|embedding| embedding.index);
    }
}
