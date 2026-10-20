use derive_getters::{Dissolve, Getters};
use derive_new::new;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

use crate::openai::types::request::{
    FunctionCall, ImageDetail, InlineData, Part, PartType, Role, SystemInstruction,
};

#[derive(Dissolve, Serialize, Deserialize, new, Getters, Debug, Clone)]
pub struct Chat {
    role: Role,
    parts: Vec<Part>,
}
impl Chat {
    pub(crate) const EMPTY_REPLY: Chat = Chat {
        role: Role::Assistant,
        parts: Vec::new(),
    };
    pub fn parts_mut(&mut self) -> &mut Vec<Part> {
        &mut self.parts
    }
    pub fn parts_owned(self) -> Vec<Part> {
        self.parts
    }
    ///`seperator` used to concatenate all text parts. TL;DR use "\n" as seperator.
    ///Don't contain thoughts
    pub fn get_text_no_think(&self, seperator: impl AsRef<str>) -> String {
        Self::extract_text(&self.parts, seperator, Some(false))
    }
    ///`seperator` used to concatenate all text parts. TL;DR use "\n" as seperator.
    pub fn get_thoughts(&self, seperator: impl AsRef<str>) -> String {
        Self::extract_text(&self.parts, seperator, Some(true))
    }
    pub fn extract_text_all(parts: &[Part], seperator: impl AsRef<str>) -> String {
        Self::extract_text(parts, seperator, None)
    }
    pub(crate) fn extract_text(
        parts: &[Part],
        seperator: impl AsRef<str>,
        thought: Option<bool>,
    ) -> String {
        parts
            .iter()
            .filter_map(|part| match part.data() {
                PartType::Text(text) if thought.is_none_or(|t| t == part.is_thought()) => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect::<Vec<&str>>()
            .join(seperator.as_ref())
    }
    ///`seperator` used to concatenate all text parts. TL;DR use "\n" as seperator.
    ///Includes all text including thoughts
    pub fn get_text_all(&self, seperator: impl AsRef<str>) -> String {
        Self::extract_text_all(&self.parts(), seperator)
    }
    pub fn is_thinking(&self) -> bool {
        self.parts().iter().any(|p| p.is_thought())
    }
    /// Returns the refusal message if the model refused to answer.
    pub fn get_refusal(&self) -> Option<&str> {
        self.parts().iter().find_map(|p| match p.data() {
            PartType::Refusal(refusal) => Some(refusal.as_str()),
            _ => None,
        })
    }
    pub fn has_function_call(&self) -> bool {
        self.parts()
            .iter()
            .any(|p| matches!(p.data(), PartType::FunctionCall(_)))
    }
    pub fn get_function_calls(&self) -> impl Iterator<Item = &FunctionCall> {
        self.parts().iter().filter_map(|v| match v.data() {
            PartType::FunctionCall(call) => Some(call),
            _ => None,
        })
    }
}

#[derive(Serialize)]
struct ApiImageUrl<'a> {
    url: Cow<'a, str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<&'a ImageDetail>,
}
#[derive(Serialize)]
struct ApiInputAudio<'a> {
    data: &'a str,
    format: &'a str,
}
#[derive(Serialize)]
struct ApiFile {
    filename: String,
    file_data: String,
}
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiContentPart<'a> {
    Text { text: Cow<'a, str> },
    ImageUrl { image_url: ApiImageUrl<'a> },
    InputAudio { input_audio: ApiInputAudio<'a> },
    File { file: ApiFile },
}
impl<'a> ApiContentPart<'a> {
    /// `None` for parts which can't be in message content.
    fn from_part(part: &'a Part) -> Option<Self> {
        if part.is_thought() {
            return None;
        }
        Some(match part.data() {
            PartType::Text(text) => Self::Text {
                text: Cow::Borrowed(text),
            },
            PartType::InlineData(inline_data) => Self::from_inline_data(inline_data),
            PartType::ImageUrl(image_url) => Self::ImageUrl {
                image_url: ApiImageUrl {
                    url: Cow::Borrowed(image_url.url()),
                    detail: image_url.detail().as_ref(),
                },
            },
            PartType::Refusal(_) | PartType::FunctionCall(_) | PartType::FunctionResponse(_) => {
                return None;
            }
        })
    }
    fn from_inline_data(inline_data: &'a InlineData) -> Self {
        let mime_type = inline_data.mime_type();
        match mime_type.type_() {
            mime::IMAGE => Self::ImageUrl {
                image_url: ApiImageUrl {
                    url: Cow::Owned(inline_data.data_url()),
                    detail: None,
                },
            },
            mime::AUDIO => Self::InputAudio {
                input_audio: ApiInputAudio {
                    data: inline_data.data(),
                    format: match mime_type.subtype().as_str() {
                        "mpeg" | "mp3" => "mp3",
                        "wav" | "x-wav" | "wave" | "vnd.wave" => "wav",
                        other => other,
                    },
                },
            },
            _ => Self::File {
                file: ApiFile {
                    filename: format!("file.{}", mime_type.subtype()),
                    file_data: inline_data.data_url(),
                },
            },
        }
    }
    fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. })
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum ApiContent<'a> {
    Text(String),
    Parts(Vec<ApiContentPart<'a>>),
}
impl<'a> ApiContent<'a> {
    /// Text only content is sent as a string, as some servers (e.g. DeepSeek) only accept that.
    fn new(parts: Vec<ApiContentPart<'a>>) -> Self {
        if parts.iter().all(ApiContentPart::is_text) {
            Self::Text(
                parts
                    .iter()
                    .filter_map(|part| match part {
                        ApiContentPart::Text { text } => Some(text.as_ref()),
                        _ => None,
                    })
                    .collect(),
            )
        } else {
            Self::Parts(parts)
        }
    }
}

#[derive(Serialize)]
struct ApiFunction<'a> {
    name: &'a str,
    arguments: &'a str,
}
#[derive(Serialize)]
struct ApiToolCall<'a> {
    id: &'a str,
    r#type: &'static str,
    function: ApiFunction<'a>,
}

/// Borrowed view of a `Chat` as sent to the API. One `Chat` can be many messages, as every
/// function response is its own `tool` message.
#[derive(Serialize)]
pub(crate) struct ApiMessage<'a> {
    role: &'static str,
    /// `null` for an assistant message with only tool calls.
    content: Option<ApiContent<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ApiToolCall<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
}
impl<'a> ApiMessage<'a> {
    fn new(role: &'static str, content: ApiContent<'a>) -> Self {
        Self {
            role,
            content: Some(content),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
    pub(crate) fn from_assistant(chat: &'a Chat, send_reasoning: bool) -> Self {
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut tool_calls = Vec::new();
        for part in chat.parts() {
            match part.data() {
                PartType::Text(thought) if part.is_thought() => reasoning.push_str(thought),
                // Refusal is sent as text as many servers don't know the `refusal` field.
                PartType::Text(t) | PartType::Refusal(t) => text.push_str(t),
                PartType::FunctionCall(call) => tool_calls.push(ApiToolCall {
                    id: call.id(),
                    r#type: "function",
                    function: ApiFunction {
                        name: call.name(),
                        arguments: match call.arguments().trim() {
                            "" => "{}",
                            arguments => arguments,
                        },
                    },
                }),
                _ => {}
            }
        }
        Self {
            role: "assistant",
            content: (!text.is_empty() || tool_calls.is_empty()).then_some(ApiContent::Text(text)),
            reasoning_content: (send_reasoning && !reasoning.is_empty()).then_some(reasoning),
            tool_calls,
            tool_call_id: None,
        }
    }
}

/// Converts a session into messages as per OpenAI specification.
///
/// Reasoning is sent as `reasoning_content` only if `send_reasoning`, and only for the replies after
/// the last user prompt i.e. in the ongoing tool calling loop (as required by DeepSeek).
pub(crate) fn to_api_messages<'a>(
    sys_prompt: Option<&'a SystemInstruction>,
    chats: &[&'a Chat],
    send_reasoning: bool,
) -> Vec<ApiMessage<'a>> {
    let mut messages = Vec::with_capacity(chats.len() + 1);
    if let Some(sys_prompt) = sys_prompt {
        let text = Chat::extract_text(sys_prompt.parts(), "\n", Some(false));
        messages.push(ApiMessage::new("system", ApiContent::Text(text)));
    }
    let last_user_prompt = chats.iter().rposition(|chat| *chat.role() == Role::User);
    // Non text parts of function responses, sent in the next user message.
    let mut attachments: Vec<ApiContentPart> = Vec::new();
    for (i, chat) in chats.iter().enumerate() {
        if *chat.role() != Role::User && !attachments.is_empty() {
            let content = ApiContent::new(std::mem::take(&mut attachments));
            messages.push(ApiMessage::new("user", content));
        }
        match chat.role() {
            Role::User => {
                let mut parts = std::mem::take(&mut attachments);
                parts.extend(chat.parts().iter().filter_map(ApiContentPart::from_part));
                messages.push(ApiMessage::new("user", ApiContent::new(parts)));
            }
            Role::Assistant => {
                let send_reasoning = send_reasoning && last_user_prompt.is_none_or(|u| i > u);
                messages.push(ApiMessage::from_assistant(chat, send_reasoning));
            }
            Role::Tool => {
                for part in chat.parts() {
                    let PartType::FunctionResponse(response) = part.data() else {
                        continue;
                    };
                    let mut text = String::new();
                    let mut media = Vec::new();
                    for part in response.parts() {
                        match ApiContentPart::from_part(part) {
                            Some(ApiContentPart::Text { text: t }) => text.push_str(&t),
                            Some(other) => media.push(other),
                            None => {}
                        }
                    }
                    if !media.is_empty() {
                        if text.is_empty() {
                            text = "Output is attached in the next message.".into();
                        }
                        attachments.push(ApiContentPart::Text {
                            text: Cow::Owned(format!(
                                "Attachments of `{}` function output (call id {}):",
                                response.name(),
                                response.id()
                            )),
                        });
                        attachments.extend(media);
                    }
                    let mut message = ApiMessage::new("tool", ApiContent::Text(text));
                    message.tool_call_id = Some(response.id());
                    messages.push(message);
                }
            }
        }
    }
    if !attachments.is_empty() {
        messages.push(ApiMessage::new("user", ApiContent::new(attachments)));
    }
    messages
}
