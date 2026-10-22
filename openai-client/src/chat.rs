//! Chat completions (`/chat/completions`): the [`AI`] client, its options, replies and streams.
//! Messages are in [`message`], and the conversation history in [`session`].
#[cfg(feature = "reqwest")]
mod client;
mod completion;
pub mod message;
mod params;
pub mod session;
mod stream;
#[cfg(feature = "reqwest")]
pub(crate) mod wire;

#[cfg(feature = "reqwest")]
pub use client::AI;
pub use completion::{ChatCompletion, Choice, FinishReason, Usage};
pub use params::{ReasoningEffort, ResponseFormat, Tool, ToolChoice};
#[cfg(feature = "reqwest")]
pub use stream::ChatStream;
pub use stream::{ChatCompletionChunk, ChunkChoice, Delta, FunctionCallDelta, ToolCallDelta};
