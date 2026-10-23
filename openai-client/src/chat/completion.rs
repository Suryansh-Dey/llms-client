use super::message::AssistantMessage;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::LazyLock;

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

/// Token usage of a request.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    /// All other fields, e.g. `prompt_tokens_details` and `completion_tokens_details`.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Choice {
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub message: AssistantMessage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
}

/// Response of `/chat/completions`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChatCompletion {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// All other fields, e.g. `created`, `system_fingerprint` or provider specific ones like
    /// `citations`.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
static EMPTY_MESSAGE: LazyLock<AssistantMessage> = LazyLock::new(AssistantMessage::default);
impl ChatCompletion {
    /// Reply of the first choice. Empty if there is no choice.
    pub fn message(&self) -> &AssistantMessage {
        self.choices
            .first()
            .map_or(&EMPTY_MESSAGE, |choice| &choice.message)
    }
    /// Content of the reply, or `""` if there is none.
    pub fn text(&self) -> &str {
        self.message().text()
    }
    /// Parses the content of the reply as JSON. A markdown code fence around it is ignored.
    pub fn json<T: DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        self.message().json()
    }
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.choices.first()?.finish_reason.as_ref()
    }
}
