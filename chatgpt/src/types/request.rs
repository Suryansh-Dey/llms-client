use derive_new::new;
use getset::Getters;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, new, Getters)]
pub struct ChatMessage {
    #[get = "pub"]
    pub role: Role,
    #[get = "pub"]
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[get = "pub"]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[get = "pub"]
    pub tool_call_id: Option<String>,
}
impl ChatMessage {
    pub fn append_content(&mut self, text: impl AsRef<str>) {
        self.content.push_str(text.as_ref());
    }
}

/// Wrapper for an OpenAI tool definition. The shape is user-controlled and
/// passed through to the API unchanged.
pub type ToolDefinition = Value;


