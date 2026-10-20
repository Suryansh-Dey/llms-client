use super::request::*;
#[cfg(feature = "reqwest")]
use super::sessions::{OverflowHandler, Session, SlidingWindow};
#[cfg(feature = "reqwest")]
use crate::openai::error::{ApiError, OpenAIResponseStreamError};
#[cfg(feature = "reqwest")]
use bytes::Bytes;
#[cfg(feature = "reqwest")]
use futures::Stream;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
#[cfg(feature = "reqwest")]
use std::{
    pin::Pin,
    task::{Context, Poll},
};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural stop point of the model or provided stop sequence.
    Stop,
    /// The maximum number of tokens as specified in the request was reached.
    Length,
    /// The model called a tool.
    ToolCalls,
    /// Content was omitted due to a flag from content filters.
    ContentFilter,
    /// Deprecated in favour of `ToolCalls`.
    FunctionCall,
    /// Any reason not in OpenAI specification, sent by some servers.
    #[serde(other)]
    Other,
}

/// Assistant message as received from the API. `delta` of a streamed chunk has the same shape.
#[derive(Deserialize, Default)]
struct ApiReply {
    #[serde(default)]
    content: Option<Value>,
    /// Used by DeepSeek, vLLM, Qwen..
    #[serde(default)]
    reasoning_content: Option<Value>,
    /// Used by OpenRouter, Ollama, Groq..
    #[serde(default)]
    reasoning: Option<Value>,
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ApiReplyToolCall>>,
}
#[derive(Deserialize)]
struct ApiReplyToolCall {
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: ApiReplyFunction,
}
#[derive(Deserialize, Default)]
struct ApiReplyFunction {
    #[serde(default)]
    name: Option<String>,
    /// JSON encoded string as per specification, but some servers send an object.
    #[serde(default)]
    arguments: Option<Value>,
}
impl From<ApiReply> for Chat {
    fn from(reply: ApiReply) -> Self {
        let mut parts: Vec<Part> = Vec::new();
        let reasoning = [reply.reasoning_content, reply.reasoning]
            .into_iter()
            .flatten()
            .find_map(|r| r.as_str().filter(|r| !r.is_empty()).map(str::to_string));
        if let Some(reasoning) = reasoning {
            parts.push(Part::new_thought(reasoning));
        }
        match reply.content {
            Some(Value::String(text)) if !text.is_empty() => parts.push(text.into()),
            // Some servers send content parts in replies too.
            Some(Value::Array(content)) => {
                for part in content {
                    match (part["type"].as_str(), part["text"].as_str()) {
                        (Some("thinking" | "reasoning"), _) => {
                            let thought = match &part["thinking"] {
                                Value::String(thought) => thought.clone(),
                                Value::Array(thoughts) => {
                                    thoughts.iter().filter_map(|t| t["text"].as_str()).collect()
                                }
                                _ => part["text"].as_str().unwrap_or_default().to_string(),
                            };
                            parts.push(Part::new_thought(thought));
                        }
                        (_, Some(text)) => parts.push(text.into()),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        if let Some(refusal) = reply.refusal.filter(|r| !r.is_empty()) {
            parts.push(PartType::Refusal(refusal).into());
        }
        for call in reply.tool_calls.into_iter().flatten() {
            let arguments = match call.function.arguments {
                Some(Value::String(arguments)) => arguments,
                None | Some(Value::Null) => String::new(),
                Some(arguments) => arguments.to_string(),
            };
            parts.push(
                FunctionCall::new_delta(call.id, call.function.name, arguments, call.index).into(),
            );
        }
        Chat::new(Role::Assistant, parts)
    }
}
fn deserialize_reply<'de, D>(deserializer: D) -> Result<Chat, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<ApiReply>::deserialize(deserializer)?
        .unwrap_or_default()
        .into())
}
fn serialize_reply<S>(chat: &Chat, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    ApiMessage::from_assistant(chat, true).serialize(serializer)
}
fn empty_reply() -> Chat {
    Chat::EMPTY_REPLY
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Choice {
    #[serde(default)]
    pub index: u32,
    /// `delta` in a streamed chunk.
    #[serde(
        alias = "delta",
        default = "empty_reply",
        deserialize_with = "deserialize_reply",
        serialize_with = "serialize_reply"
    )]
    pub message: Chat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
}

/// Response of `/chat/completions`. Each chunk of a streamed response is one too.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OpenAIResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Empty in the last chunk of a stream having only `usage`.
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
    /// All other fields, e.g. `created`, `system_fingerprint` or provider specific ones like
    /// `citations`.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
impl OpenAIResponse {
    #[cfg(feature = "reqwest")]
    pub(crate) fn from_value(value: Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(value)
    }
    /// Reply of the first choice. Empty if there is no choice, e.g. in a chunk having only `usage`.
    pub fn get_chat(&self) -> &Chat {
        const EMPTY: &Chat = &Chat::EMPTY_REPLY;
        self.choices.first().map_or(EMPTY, |choice| &choice.message)
    }
    pub fn get_finish_reason(&self) -> Option<&FinishReason> {
        self.choices.first()?.finish_reason.as_ref()
    }
    pub fn get_json<T>(&self) -> Result<T, serde_json::Error>
    where
        T: serde::de::DeserializeOwned,
    {
        Self::parse_json(self.get_chat().parts())
    }
    /// Parses text (excluding thoughts) as JSON. Markdown code fence around it is ignored.
    pub fn parse_json<T>(parts: &[Part]) -> Result<T, serde_json::Error>
    where
        T: serde::de::DeserializeOwned,
    {
        let text = Chat::extract_text(parts, "", Some(false));
        let mut json = text.trim();
        if let Some(fenced) = json.strip_prefix("```") {
            json = fenced.trim_start_matches("json");
            json = json.strip_suffix("```").unwrap_or(json).trim();
        }
        serde_json::from_str::<T>(json)
    }
}

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

#[cfg(feature = "reqwest")]
pin_project_lite::pin_project! {
    pub struct ResponseStream<F,T,H = SlidingWindow>
        where F:FnMut(&Session<H>, OpenAIResponse) -> T{
        #[pin]
        response_stream:Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send + 'static>,
        session: Session<H>,
        data_extractor: F,
        buffer: Vec<u8>,
        // Set after `[DONE]` or end of the response stream.
        ended: bool,
    }
}
#[cfg(feature = "reqwest")]
impl<F, T, H: OverflowHandler> Stream for ResponseStream<F, T, H>
where
    F: FnMut(&Session<H>, OpenAIResponse) -> T,
{
    type Item = Result<T, OpenAIResponseStreamError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            if let Some((end_of_event, delimiter_length)) = find_event_end(this.buffer) {
                let event = this
                    .buffer
                    .drain(..end_of_event + delimiter_length)
                    .collect::<Vec<u8>>();
                let event = String::from_utf8_lossy(&event[..end_of_event]);
                // Lines other than `data:` are comments (keep alive), `event:`, `id:` etc.
                let data = event
                    .lines()
                    .filter_map(|line| line.strip_prefix("data:"))
                    .map(str::trim)
                    .collect::<Vec<&str>>()
                    .join("\n");

                if data.is_empty() {
                    continue;
                }
                if data == "[DONE]" {
                    *this.ended = true;
                    this.buffer.clear();
                    return Poll::Ready(None);
                }
                let value: Value = match serde_json::from_str(&data) {
                    Ok(value) => value,
                    Err(e) => {
                        return Poll::Ready(Some(Err(
                            OpenAIResponseStreamError::InvalidResponseFormat(format!(
                                "JSON parsing error [{e}]: {data}"
                            )),
                        )));
                    }
                };
                if value.get("error").is_some() && value.get("choices").is_none() {
                    let error = ApiError::from_value(&value);
                    return Poll::Ready(Some(Err(OpenAIResponseStreamError::ApiError(error))));
                }
                let response = match OpenAIResponse::from_value(value) {
                    Ok(response) => response,
                    Err(e) => {
                        return Poll::Ready(Some(Err(
                            OpenAIResponseStreamError::InvalidResponseFormat(format!(
                                "Unexpected response format [{e}]: {data}"
                            )),
                        )));
                    }
                };

                // Update the session and return the data.
                this.session.update(&response);
                let data = (this.data_extractor)(this.session, response);
                return Poll::Ready(Some(Ok(data)));
            }
            if *this.ended {
                if this.buffer.iter().all(u8::is_ascii_whitespace) {
                    this.buffer.clear();
                    return Poll::Ready(None);
                }
                // The last event may not be followed by a blank line.
                this.buffer.extend_from_slice(b"\n\n");
                continue;
            }

            // If the complete event is not in the buffer, read data from the network.
            match this.response_stream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    this.buffer.extend_from_slice(&bytes);
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
                Poll::Ready(None) => {
                    *this.ended = true;
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(e.into())));
                }
            }
        }
    }
}
#[cfg(feature = "reqwest")]
impl<F, T, H> ResponseStream<F, T, H>
where
    F: FnMut(&Session<H>, OpenAIResponse) -> T,
{
    pub(crate) fn new(
        response_stream: Box<
            dyn Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send + 'static,
        >,
        session: Session<H>,
        data_extractor: F,
    ) -> Self {
        Self {
            response_stream,
            session,
            data_extractor,
            buffer: Vec::new(),
            ended: false,
        }
    }
    pub fn get_session(&self) -> &Session<H> {
        &self.session
    }
    pub fn get_session_owned(self) -> Session<H> {
        self.session
    }
}
#[cfg(feature = "reqwest")]
pub type OpenAIResponseStream<H = SlidingWindow> =
    ResponseStream<fn(&Session<H>, OpenAIResponse) -> OpenAIResponse, OpenAIResponse, H>;
