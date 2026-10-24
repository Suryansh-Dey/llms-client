//! Request body of `/chat/completions`.
use super::message::{AssistantMessage, Message, ToolCall};
use super::params::{Tool, ToolChoice};
use serde::Serialize;
use serde::ser::{SerializeMap, Serializer};
use serde_json::{Map, Value};
use std::collections::VecDeque;

/// Keys of the request body set by `AI` itself, so they are ignored in its parameters.
const RESERVED_KEYS: [&str; 5] = ["model", "messages", "tools", "tool_choice", "stream"];

pub(crate) struct ChatRequest<'a> {
    pub model: &'a str,
    pub system_prompt: Option<&'a str>,
    pub messages: &'a VecDeque<Message>,
    pub tools: &'a [Tool],
    pub tool_choice: Option<&'a ToolChoice>,
    /// Entries are added to the top level of the body.
    pub params: &'a Map<String, Value>,
    pub stream: bool,
    /// See `AI::send_reasoning()`.
    pub send_reasoning: bool,
}
impl Serialize for ChatRequest<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("model", self.model)?;
        map.serialize_entry("messages", &Messages(self))?;
        if !self.tools.is_empty() {
            map.serialize_entry("tools", self.tools)?;
        }
        if let Some(tool_choice) = self.tool_choice {
            map.serialize_entry("tool_choice", tool_choice)?;
        }
        if self.stream {
            map.serialize_entry("stream", &true)?;
        }
        for (key, value) in self.params {
            if RESERVED_KEYS.contains(&key.as_str())
                || (key == "stream_options" && !self.stream)
                || value.is_null()
            {
                continue;
            }
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

/// The system prompt followed by the history.
struct Messages<'a>(&'a ChatRequest<'a>);
impl Serialize for Messages<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let request = self.0;
        let system = request.system_prompt.map(|content| WireMessage::System {
            role: "system",
            content,
        });
        let history = request.messages.iter().map(|message| match message {
            Message::Assistant(reply) => {
                WireMessage::Assistant(AssistantView::new(reply, request.send_reasoning))
            }
            message => WireMessage::Other(message),
        });
        serializer.collect_seq(system.into_iter().chain(history))
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum WireMessage<'a> {
    System {
        role: &'static str,
        content: &'a str,
    },
    Assistant(AssistantView<'a>),
    Other(&'a Message),
}

/// `AssistantMessage` as sent to the API.
#[derive(Serialize)]
struct AssistantView<'a> {
    role: &'static str,
    /// `null` only with tool calls, as some servers reject it otherwise.
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refusal: Option<&'a str>,
    /// Not in the OpenAI specification. See `AI::send_reasoning()`.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    tool_calls: &'a [ToolCall],
    #[serde(flatten)]
    extra: &'a Map<String, Value>,
}
impl<'a> AssistantView<'a> {
    fn new(reply: &'a AssistantMessage, send_reasoning: bool) -> Self {
        let content = match reply.content.as_deref() {
            None if reply.tool_calls.is_empty() => Some(""),
            content => content,
        };
        Self {
            role: "assistant",
            content,
            refusal: reply.refusal.as_deref(),
            reasoning_content: reply.reasoning.as_deref().filter(|_| send_reasoning),
            tool_calls: &reply.tool_calls,
            extra: &reply.extra,
        }
    }
}
