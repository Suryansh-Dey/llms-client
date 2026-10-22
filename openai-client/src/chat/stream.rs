use super::completion::{FinishReason, Usage};
use super::message::RawMessage;
#[cfg(feature = "reqwest")]
use super::message::{AssistantMessage, FunctionCall, ToolCall, generate_call_id};
#[cfg(feature = "reqwest")]
use super::session::{OverflowHandler, Session, SlidingWindow};
#[cfg(feature = "reqwest")]
use crate::{ApiError, Error};
#[cfg(feature = "reqwest")]
use bytes::Bytes;
#[cfg(feature = "reqwest")]
use futures::{Stream, StreamExt};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
#[cfg(feature = "reqwest")]
use std::{
    pin::Pin,
    task::{Context, Poll},
};

/// Newly received part of a streamed reply.
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    /// See `AssistantMessage::reasoning`.
    #[serde(rename = "reasoning_content", skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallDelta>,
}
/// Parsed leniently, like `AssistantMessage`.
impl<'de> Deserialize<'de> for Delta {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = Option::<RawMessage>::deserialize(deserializer)?.unwrap_or_default();
        let (content, reasoning, refusal) = raw.texts();
        let tool_calls = raw
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|call| ToolCallDelta {
                index: call.index,
                id: call.id,
                function: FunctionCallDelta {
                    arguments: call.function.arguments(),
                    name: call.function.name,
                },
            })
            .collect();
        Ok(Self {
            content,
            refusal,
            reasoning,
            tool_calls,
        })
    }
}
/// Piece of a tool call. The first piece has `id` and `name`, and `arguments` arrive in pieces.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct ToolCallDelta {
    /// Position of the call in the reply, shared by the pieces of a call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default)]
    pub function: FunctionCallDelta,
}
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct FunctionCallDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChunkChoice {
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub delta: Delta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
}

/// One event of a streamed response of `/chat/completions`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChatCompletionChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Empty in a chunk with only `usage`.
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    /// Sent with `AI::include_usage(true)`. OpenAI sends it in the last chunk, others in every
    /// chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// All other fields, e.g. `created` or provider specific ones.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
const EMPTY_DELTA: &Delta = &Delta {
    content: None,
    refusal: None,
    reasoning: None,
    tool_calls: Vec::new(),
};
impl ChatCompletionChunk {
    /// Delta of the first choice. Empty if there is no choice.
    pub fn delta(&self) -> &Delta {
        self.choices
            .first()
            .map_or(EMPTY_DELTA, |choice| &choice.delta)
    }
    /// Newly received content, or `""` if there is none.
    pub fn text(&self) -> &str {
        self.delta().content.as_deref().unwrap_or_default()
    }
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.choices.first()?.finish_reason.as_ref()
    }
}

/// Reply assembled from deltas. Pieces of a tool call are matched by `index`, as the official
/// OpenAI SDKs do: the first `id` and `name` are kept, as some servers repeat them, and `arguments`
/// are appended. A call without `index` is complete, e.g. Gemini sends calls whole.
#[cfg(feature = "reqwest")]
#[derive(Default)]
struct Reply {
    message: AssistantMessage,
    /// `index` of each call in `message.tool_calls`.
    indexes: Vec<Option<u32>>,
}
#[cfg(feature = "reqwest")]
impl Reply {
    fn apply(&mut self, delta: &Delta) {
        fn append(text: &mut Option<String>, delta: &Option<String>) {
            if let Some(delta) = delta {
                text.get_or_insert_default().push_str(delta);
            }
        }
        append(&mut self.message.content, &delta.content);
        append(&mut self.message.reasoning, &delta.reasoning);
        append(&mut self.message.refusal, &delta.refusal);
        for piece in &delta.tool_calls {
            let position = piece
                .index
                .and_then(|index| self.indexes.iter().position(|i| *i == Some(index)));
            match position {
                Some(position) => {
                    let call = &mut self.message.tool_calls[position].function;
                    if call.name.is_empty()
                        && let Some(name) = &piece.function.name
                    {
                        call.name = name.clone();
                    }
                    if let Some(arguments) = &piece.function.arguments {
                        call.arguments.push_str(arguments);
                    }
                }
                None => {
                    self.message.tool_calls.push(ToolCall {
                        id: piece
                            .id
                            .clone()
                            .filter(|id| !id.is_empty())
                            .unwrap_or_else(generate_call_id),
                        function: FunctionCall {
                            name: piece.function.name.clone().unwrap_or_default(),
                            arguments: piece.function.arguments.clone().unwrap_or_default(),
                        },
                    });
                    self.indexes.push(piece.index);
                }
            }
        }
    }
}

#[cfg(feature = "reqwest")]
pub(crate) type ByteStream = Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send>;

/// Returns the position and length of the blank line ending the first event.
#[cfg(feature = "reqwest")]
fn find_event_end(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer.windows(2).position(|w| w == b"\n\n").map(|p| (p, 2));
    let crlf = buffer
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|p| (p, 4));
    match (lf, crlf) {
        (Some(lf), Some(crlf)) => Some(if lf.0 < crlf.0 { lf } else { crlf }),
        (lf, crlf) => lf.or(crlf),
    }
}

/// Streamed reply, returned by `AI::stream()`. Each item is a chunk with the newly received part
/// of the reply.
///
/// The whole reply is added to the session when the stream ends, or when it is dropped (with what
/// was received so far). `message()` has the reply received so far.
#[cfg(feature = "reqwest")]
pub struct ChatStream<'a, H: OverflowHandler = SlidingWindow> {
    bytes: ByteStream,
    session: &'a mut Session<H>,
    buffer: Vec<u8>,
    /// Set after `[DONE]` or end of the response.
    ended: bool,
    reply: Reply,
    /// Set once a chunk with a choice arrives, as only then there is a reply to add.
    received: bool,
    added: bool,
}
#[cfg(feature = "reqwest")]
impl<'a, H: OverflowHandler> ChatStream<'a, H> {
    pub(crate) fn new(bytes: ByteStream, session: &'a mut Session<H>) -> Self {
        Self {
            bytes,
            session,
            buffer: Vec::new(),
            ended: false,
            reply: Reply::default(),
            received: false,
            added: false,
        }
    }
    /// Reply received so far, with tool calls assembled from their pieces.
    pub fn message(&self) -> &AssistantMessage {
        &self.reply.message
    }
    /// The session, which has the reply once the stream ended.
    pub fn session(&self) -> &Session<H> {
        self.session
    }
    fn add_reply(&mut self) {
        if self.received && !self.added {
            self.added = true;
            self.session.add_reply(self.reply.message.clone());
        }
    }
    /// Parses the data of one event. `None` for an event without data, e.g. a comment.
    fn parse_event(&mut self, event: &[u8]) -> Option<Result<ChatCompletionChunk, Error>> {
        let event = String::from_utf8_lossy(event);
        // Lines other than `data:` are comments (keep alive), `event:`, `id:` etc.
        let data = event
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(str::trim)
            .collect::<Vec<&str>>()
            .join("\n");
        if data.is_empty() {
            return None;
        }
        let value: Value = match serde_json::from_str(&data) {
            Ok(value) => value,
            Err(e) => {
                let error = format!("JSON parsing error [{e}]: {data}");
                return Some(Err(Error::InvalidResponse(error)));
            }
        };
        if value.get("error").is_some() && value.get("choices").is_none() {
            let error = ApiError::from_value(&value);
            return Some(Err(Error::Api {
                status: None,
                error,
            }));
        }
        let chunk = match ChatCompletionChunk::deserialize(&value) {
            Ok(chunk) => chunk,
            Err(e) => {
                let error = format!("Unexpected response format [{e}]: {data}");
                return Some(Err(Error::InvalidResponse(error)));
            }
        };
        if let Some(choice) = chunk.choices.iter().find(|choice| choice.index == 0) {
            self.reply.apply(&choice.delta);
            self.received = true;
        }
        Some(Ok(chunk))
    }
}
#[cfg(feature = "reqwest")]
impl<H: OverflowHandler> Stream for ChatStream<'_, H> {
    type Item = Result<ChatCompletionChunk, Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some((end_of_event, delimiter_length)) = find_event_end(&this.buffer) {
                let event: Vec<u8> = this
                    .buffer
                    .drain(..end_of_event + delimiter_length)
                    .collect();
                let event = &event[..end_of_event];
                if event.trim_ascii() == b"data: [DONE]" {
                    this.ended = true;
                    this.buffer.clear();
                    continue;
                }
                match this.parse_event(event) {
                    Some(item) => return Poll::Ready(Some(item)),
                    None => continue,
                }
            }
            if this.ended {
                if this.buffer.iter().all(u8::is_ascii_whitespace) {
                    this.buffer.clear();
                    this.add_reply();
                    return Poll::Ready(None);
                }
                // The last event may not be followed by a blank line.
                this.buffer.extend_from_slice(b"\n\n");
                continue;
            }
            // If the complete event is not in the buffer, read data from the network.
            match this.bytes.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(bytes))) => this.buffer.extend_from_slice(&bytes),
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e.into()))),
                Poll::Ready(None) => this.ended = true,
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}
#[cfg(feature = "reqwest")]
impl<H: OverflowHandler> Drop for ChatStream<'_, H> {
    fn drop(&mut self) {
        self.add_reply();
    }
}
