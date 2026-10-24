#[cfg(feature = "reqwest")]
use super::completion::{ChatCompletion, Choice};
use super::completion::{FinishReason, Usage};
#[cfg(feature = "reqwest")]
use super::message::{AssistantMessage, FunctionCall, ToolCall, generate_call_id};
use super::message::{RawMessage, RawToolCall, with_value};
#[cfg(feature = "reqwest")]
use super::session::{OverflowHandler, Session, SlidingWindow};
#[cfg(feature = "reqwest")]
use crate::{ApiError, Error};
#[cfg(feature = "reqwest")]
use bytes::Bytes;
#[cfg(feature = "reqwest")]
use futures::{Stream, StreamExt};
use serde::{Deserialize, Deserializer, Serialize};
#[cfg(feature = "reqwest")]
use serde_json::map::Entry;
use serde_json::{Map, Value};
use std::sync::LazyLock;
#[cfg(feature = "reqwest")]
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
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
    /// All other fields with a value, e.g. `extra_content` of Gemini, which comes in the last
    /// delta. Merged into `AssistantMessage::extra`: lists are appended, like `reasoning_details`
    /// of OpenRouter, which comes in pieces, and other values keep the first.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
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
            .map(ToolCallDelta::from_raw)
            .collect();
        Ok(Self {
            content,
            refusal,
            reasoning,
            tool_calls,
            extra: with_value(raw.extra),
        })
    }
}
/// Piece of a tool call. The first piece has `id` and `name`, and `arguments` arrive in pieces.
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct ToolCallDelta {
    /// Position of the call in the reply, shared by the pieces of a call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: FunctionCallDelta,
    /// All other fields with a value, e.g. `extra_content` of Gemini. Merged into
    /// `ToolCall::extra` like `Delta::extra`.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
impl ToolCallDelta {
    fn from_raw(call: RawToolCall) -> Self {
        Self {
            index: call.index,
            id: call.id,
            function: FunctionCallDelta {
                arguments: call.function.arguments(),
                name: call.function.name,
            },
            extra: with_value(call.extra),
        }
    }
}
/// Parsed leniently, like `ToolCall`.
impl<'de> Deserialize<'de> for ToolCallDelta {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        RawToolCall::deserialize(deserializer).map(Self::from_raw)
    }
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
static EMPTY_DELTA: LazyLock<Delta> = LazyLock::new(Delta::default);
impl ChatCompletionChunk {
    /// Delta of the first choice. Empty if there is no choice.
    pub fn delta(&self) -> &Delta {
        self.choices
            .first()
            .map_or(&EMPTY_DELTA, |choice| &choice.delta)
    }
    /// Newly received content, or `""` if there is none.
    pub fn text(&self) -> &str {
        self.delta().content.as_deref().unwrap_or_default()
    }
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.choices.first()?.finish_reason.as_ref()
    }
}

/// Response assembled from chunks, its reply from deltas. Pieces of a tool call are matched by
/// `index`, as the official OpenAI SDKs do: the first `id` and `name` are kept, as some servers
/// repeat them, and `arguments` are appended. A call without `index` is complete, e.g. Gemini
/// sends calls whole.
#[cfg(feature = "reqwest")]
#[derive(Default)]
struct Reply {
    id: Option<String>,
    model: Option<String>,
    message: AssistantMessage,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    /// `index` of each call in `message.tool_calls`.
    indexes: Vec<Option<u32>>,
}
#[cfg(feature = "reqwest")]
impl Reply {
    /// Takes the first id and model, the latest usage, and the first choice. Returns false if the
    /// chunk has no first choice.
    fn add_chunk(&mut self, chunk: &ChatCompletionChunk) -> bool {
        self.id = self.id.take().or_else(|| chunk.id.clone());
        self.model = self.model.take().or_else(|| chunk.model.clone());
        if chunk.usage.is_some() {
            self.usage.clone_from(&chunk.usage);
        }
        let Some(choice) = chunk.choices.iter().find(|choice| choice.index == 0) else {
            return false;
        };
        self.apply(&choice.delta);
        if choice.finish_reason.is_some() {
            self.finish_reason.clone_from(&choice.finish_reason);
        }
        true
    }
    fn apply(&mut self, delta: &Delta) {
        fn append(text: &mut Option<String>, delta: &Option<String>) {
            if let Some(delta) = delta {
                text.get_or_insert_default().push_str(delta);
            }
        }
        /// Lists are appended, other values keep the first, as some servers repeat them.
        fn merge(fields: &mut Map<String, Value>, delta: &Map<String, Value>) {
            for (key, value) in delta {
                match fields.entry(key) {
                    Entry::Vacant(field) => {
                        field.insert(value.clone());
                    }
                    Entry::Occupied(mut field) => {
                        if let (Value::Array(items), Value::Array(more)) = (field.get_mut(), value)
                        {
                            items.extend_from_slice(more);
                        }
                    }
                }
            }
        }
        append(&mut self.message.content, &delta.content);
        merge(&mut self.message.extra, &delta.extra);
        append(&mut self.message.reasoning, &delta.reasoning);
        append(&mut self.message.refusal, &delta.refusal);
        for piece in &delta.tool_calls {
            let position = piece
                .index
                .and_then(|index| self.indexes.iter().position(|i| *i == Some(index)));
            match position {
                Some(position) => {
                    let call = &mut self.message.tool_calls[position];
                    if call.function.name.is_empty()
                        && let Some(name) = &piece.function.name
                    {
                        call.function.name = name.clone();
                    }
                    if let Some(arguments) = &piece.function.arguments {
                        call.function.arguments.push_str(arguments);
                    }
                    merge(&mut call.extra, &piece.extra);
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
                        extra: piece.extra.clone(),
                    });
                    self.indexes.push(piece.index);
                }
            }
        }
    }
    /// The response, as if it wasn't streamed. Other fields of chunks are not kept.
    fn into_completion(self) -> ChatCompletion {
        ChatCompletion {
            id: self.id,
            model: self.model,
            choices: vec![Choice {
                index: 0,
                message: self.message,
                finish_reason: self.finish_reason,
                logprobs: None,
            }],
            usage: self.usage,
            extra: Map::new(),
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

/// Adds the reply to the session with `Session::add_completion()`.
#[cfg(all(feature = "reqwest", not(target_arch = "wasm32")))]
type Adding<'a> = futures::future::BoxFuture<'a, ()>;
#[cfg(all(feature = "reqwest", target_arch = "wasm32"))]
type Adding<'a> = futures::future::LocalBoxFuture<'a, ()>;

/// Streamed reply, returned by `AI::stream()`. Each item is a chunk with the newly received part
/// of the reply, and `message()` has the reply received so far.
///
/// When the stream ends, the whole reply is added to the session and its `OverflowHandler` runs,
/// before the stream returns `None`. If the stream is dropped before that, the reply received so
/// far is added without running the handler.
#[cfg(feature = "reqwest")]
pub struct ChatStream<'a, H: OverflowHandler = SlidingWindow> {
    bytes: ByteStream,
    /// `None` once the reply is being added.
    session: Option<&'a mut Session<H>>,
    buffer: Vec<u8>,
    /// Set after `[DONE]` or end of the response.
    ended: bool,
    reply: Reply,
    /// Set once a chunk with a choice arrives, as only then there is a reply to add.
    received: bool,
    adding: Option<Adding<'a>>,
    /// The whole reply, once the stream ended. Shared with `adding`.
    completion: Option<Arc<ChatCompletion>>,
}
#[cfg(feature = "reqwest")]
impl<'a, H: OverflowHandler> ChatStream<'a, H> {
    pub(crate) fn new(bytes: ByteStream, session: &'a mut Session<H>) -> Self {
        Self {
            bytes,
            session: Some(session),
            buffer: Vec::new(),
            ended: false,
            reply: Reply::default(),
            received: false,
            adding: None,
            completion: None,
        }
    }
    /// Reply received so far, with tool calls assembled from their pieces. The whole reply once
    /// the stream ended.
    pub fn message(&self) -> &AssistantMessage {
        self.completion
            .as_ref()
            .map_or(&self.reply.message, |completion| completion.message())
    }
    /// Starts adding the reply. False if there is nothing to add, or it was added.
    fn start_adding(&mut self) -> bool {
        if !self.received {
            return false;
        }
        let Some(session) = self.session.take() else {
            return false;
        };
        let completion = Arc::new(std::mem::take(&mut self.reply).into_completion());
        self.completion = Some(Arc::clone(&completion));
        self.adding = Some(Box::pin(async move {
            session.add_completion(&completion).await;
        }));
        true
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
        self.received |= self.reply.add_chunk(&chunk);
        Some(Ok(chunk))
    }
}
#[cfg(feature = "reqwest")]
impl<H: OverflowHandler> Stream for ChatStream<'_, H> {
    type Item = Result<ChatCompletionChunk, Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(adding) = &mut this.adding {
                ready!(adding.as_mut().poll(cx));
                this.adding = None;
                return Poll::Ready(None);
            }
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
                    if this.start_adding() {
                        continue;
                    }
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
        // The session is taken once the reply is being added.
        if self.received
            && let Some(session) = &mut self.session
        {
            session.add_reply(std::mem::take(&mut self.reply.message));
        }
    }
}
