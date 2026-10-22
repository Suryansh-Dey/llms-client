//! Client for any OpenAI compatible API: OpenAI, DeepSeek, Ollama, vLLM, OpenRouter, Gemini..
//!
//! - [`chat`]: chat completions with [`chat::AI`], and conversation history with
//!   [`chat::session::Session`].
//! - [`embeddings`]: embeddings with [`embeddings::Embedder`].
//!
//! Types follow the [Chat Completions](https://platform.openai.com/docs/api-reference/chat)
//! specification, so they are sent as they are.

// Lets the proc-macros' `::open_client::..` paths resolve inside this crate too.
extern crate self as open_client;

pub mod chat;
pub mod embeddings;
mod error;
#[cfg(feature = "reqwest")]
mod http;
mod schema;
#[cfg(test)]
mod tests;

#[cfg(feature = "reqwest")]
pub use error::Error;
pub use error::{ApiError, EnvVarError};
pub use open_client_proc_macros::{
    execute_function_calls, execute_function_calls_with_callback, openai_function, openai_schema,
};
pub use schema::OpenAISchema;

pub use futures;
pub use mime;
#[cfg(feature = "reqwest")]
pub use reqwest;
pub use serde;
pub use serde_json;

/// Base URL of the OpenAI API, used unless another one is set.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
