use super::request::Part;
use derive_getters::{Dissolve, Getters};
use derive_new::new;
use serde::{Deserialize, Serialize};

/// The type of task for which the embedding will be used.
///
/// See [Gemini docs](https://ai.google.dev/gemini-api/docs/embeddings#task-types) for details.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TaskType {
    /// Specifies the given text is a query in a search/retrieval setting.
    RetrievalQuery,
    /// Specifies the given text is a document from the corpus being searched.
    RetrievalDocument,
    /// Specifies the given text will be used for Semantic Textual Similarity (STS).
    SemanticSimilarity,
    /// Specifies that the given text will be classified.
    Classification,
    /// Specifies that the embeddings will be used for clustering.
    Clustering,
    /// Specifies the given text is a query for code retrieval.
    CodeRetrievalQuery,
    /// Specifies the given text will be used for fact verification.
    FactVerification,
    /// Specifies the given text will be used for question answering.
    QuestionAnswering,
}

/// Configuration for the embedding request.
///
/// See [Gemini docs](https://ai.google.dev/api/embeddings#EmbedContentConfig) for details.
#[derive(Dissolve, Serialize, Deserialize, Debug, Clone, Getters)]
#[serde(rename_all = "camelCase")]
pub struct EmbedContentConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<TaskType>,
    /// Optional reduced dimension for the output embedding.
    /// Supported by models that use [Matryoshka Representation Learning](https://ai.google.dev/gemini-api/docs/embeddings#matryoshka).
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimensionality: Option<u32>,
}
impl EmbedContentConfig {
    pub fn new(task_type: Option<TaskType>, output_dimensionality: Option<u32>) -> Self {
        Self {
            task_type,
            output_dimensionality,
        }
    }
    pub fn from_task_type(task_type: TaskType) -> Self {
        Self {
            task_type: Some(task_type),
            output_dimensionality: None,
        }
    }
    pub fn from_output_dimensionality(output_dimensionality: u32) -> Self {
        Self {
            task_type: None,
            output_dimensionality: Some(output_dimensionality),
        }
    }
}

/// The content to embed, matching the Gemini API `Content` structure.
#[derive(Dissolve, Serialize, Deserialize, Debug, Clone, Getters)]
pub struct Content {
    parts: Vec<Part>,
}
impl Content {
    pub fn new(parts: Vec<Part>) -> Self {
        Self { parts }
    }
}
impl From<String> for Content {
    fn from(text: String) -> Self {
        Self {
            parts: vec![text.into()],
        }
    }
}
impl From<&str> for Content {
    fn from(text: &str) -> Self {
        Self {
            parts: vec![text.into()],
        }
    }
}

/// Request body for the `embedContent` endpoint.
#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub(crate) struct EmbedContentRequest {
    pub model: String,
    pub content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<TaskType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimensionality: Option<u32>,
}

/// A list of floats representing the embedding.
#[derive(Dissolve, Serialize, Deserialize, Debug, Clone, Getters, new)]
pub struct ContentEmbedding {
    values: Vec<f32>,
}
impl ContentEmbedding {
    pub fn values_owned(self) -> Vec<f32> {
        self.values
    }
    pub fn dimension(&self) -> usize {
        self.values().len()
    }
}

/// Response from the `embedContent` endpoint.
#[derive(Dissolve, Serialize, Deserialize, Debug, Clone, Getters)]
#[serde(rename_all = "camelCase")]
pub struct EmbedContentResponse {
    embedding: ContentEmbedding,
}
impl EmbedContentResponse {
    pub fn embedding_owned(self) -> ContentEmbedding {
        self.embedding
    }
}
