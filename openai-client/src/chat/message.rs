//! Messages of a conversation, as defined by the Chat Completions API. They are sent as they are.
use base64::{Engine, engine::general_purpose::STANDARD};
use mime::Mime;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};

/// A message of a conversation. The system prompt isn't one, as `AI` holds it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    /// Result of a tool call.
    Tool(ToolMessage),
}
impl Message {
    /// Text of the message: the text parts of a user message joined by `"\n"`, the content of an
    /// assistant message (without reasoning), or the content of a tool message.
    pub fn text(&self) -> String {
        match self {
            Self::User(message) => message.text(),
            Self::Assistant(message) => message.text().to_string(),
            Self::Tool(message) => message.content.clone(),
        }
    }
}

/// Message from the user.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct UserMessage {
    pub content: Content,
}
impl UserMessage {
    pub fn new(content: impl Into<Content>) -> Self {
        Self {
            content: content.into(),
        }
    }
    /// Text parts joined by `"\n"`.
    pub fn text(&self) -> String {
        self.content.text()
    }
}

/// Content of a user message: a string, or a list of parts like text, images, audio and files.
///
/// # Example
/// ```
/// use open_client::chat::message::{ContentPart, ImageUrl, UserMessage};
/// let message = UserMessage::new(vec![
///     ContentPart::from("What is in this image?"),
///     ImageUrl::new("https://example.com/photo.jpg").into(),
/// ]);
/// ```
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}
impl Content {
    /// Text parts joined by `"\n"`.
    pub fn text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<&str>>()
                .join("\n"),
        }
    }
}

/// Part of the content of a user message.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
    InputAudio { input_audio: InputAudio },
    File { file: File },
}

macro_rules! impl_from {
    ($target:ty { $($source:ty => |$value:ident| $body:expr),* $(,)? }) => {
        $(impl From<$source> for $target {
            fn from($value: $source) -> Self {
                $body
            }
        })*
    };
}
impl_from!(ContentPart {
    &str => |text| Self::Text { text: text.into() },
    String => |text| Self::Text { text },
    ImageUrl => |image_url| Self::ImageUrl { image_url },
    InputAudio => |input_audio| Self::InputAudio { input_audio },
    File => |file| Self::File { file },
});
impl_from!(Content {
    &str => |text| Self::Text(text.into()),
    String => |text| Self::Text(text),
    ContentPart => |part| Self::Parts(vec![part]),
    ImageUrl => |part| Self::Parts(vec![part.into()]),
    InputAudio => |part| Self::Parts(vec![part.into()]),
    File => |part| Self::Parts(vec![part.into()]),
    Vec<ContentPart> => |parts| Self::Parts(parts),
});
impl_from!(UserMessage {
    &str => |content| Self::new(content),
    String => |content| Self::new(content),
    ContentPart => |content| Self::new(content),
    ImageUrl => |content| Self::new(content),
    InputAudio => |content| Self::new(content),
    File => |content| Self::new(content),
    Vec<ContentPart> => |content| Self::new(content),
    Content => |content| Self { content },
});
impl_from!(Message {
    UserMessage => |message| Self::User(message),
    AssistantMessage => |message| Self::Assistant(message),
    ToolMessage => |message| Self::Tool(message),
});

/// `data:` URL with `bytes` in base64. Parameters of `mime` are left out, as servers reject them,
/// e.g. `qs=0.001` of a `Content-Type` header.
fn data_url(mime: &Mime, bytes: &[u8]) -> String {
    format!(
        "data:{};base64,{}",
        mime.essence_str(),
        STANDARD.encode(bytes)
    )
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}
/// An image the API server reads from `url`: an `https://` URL it can reach (e.g. public or
/// pre-signed), or a `data:` URL with the image in base64.
///
/// OpenAI accepts only images here. Some servers accept other files too, e.g. Gemini reads PDFs
/// here and rejects [`File`].
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ImageUrl {
    pub url: String,
    /// Resolution the model sees the image at, where supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}
impl ImageUrl {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            detail: None,
        }
    }
    pub fn with_detail(mut self, detail: ImageDetail) -> Self {
        self.detail = Some(detail);
        self
    }
    /// The image in a base64 `data:` URL.
    pub fn from_bytes(mime: Mime, bytes: impl AsRef<[u8]>) -> Self {
        Self::new(data_url(&mime, bytes.as_ref()))
    }
    /// Reads a local file into a base64 `data:` URL.
    #[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
    pub async fn from_path(
        path: impl AsRef<std::path::Path>,
        mime: Mime,
    ) -> Result<Self, std::io::Error> {
        Ok(Self::from_bytes(mime, tokio::fs::read(path).await?))
    }
    /// Downloads `url` into a base64 `data:` URL, for servers that can't fetch URLs. The mime type
    /// is taken from the `Content-Type` header.
    #[cfg(feature = "reqwest")]
    pub async fn download(url: &str) -> Result<Self, crate::Error> {
        let (mime, bytes) = crate::http::download(url).await?;
        Ok(Self::from_bytes(mime, bytes))
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    Wav,
    Mp3,
}
/// Audio in base64.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct InputAudio {
    /// Base64 encoded audio.
    pub data: String,
    pub format: AudioFormat,
}
impl InputAudio {
    pub fn from_bytes(format: AudioFormat, bytes: impl AsRef<[u8]>) -> Self {
        Self {
            data: STANDARD.encode(bytes),
            format,
        }
    }
    /// Reads a local file.
    #[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
    pub async fn from_path(
        path: impl AsRef<std::path::Path>,
        format: AudioFormat,
    ) -> Result<Self, std::io::Error> {
        Ok(Self::from_bytes(format, tokio::fs::read(path).await?))
    }
    /// Downloads `url`. The format is taken from the `Content-Type` header.
    #[cfg(feature = "reqwest")]
    pub async fn download(url: &str) -> Result<Self, crate::Error> {
        let (mime, bytes) = crate::http::download(url).await?;
        let format = match mime.essence_str() {
            "audio/wav" | "audio/x-wav" | "audio/wave" | "audio/vnd.wave" => AudioFormat::Wav,
            "audio/mpeg" | "audio/mp3" => AudioFormat::Mp3,
            other => {
                return Err(crate::Error::InvalidResponse(format!(
                    "Download is {other}, not WAV or MP3 audio"
                )));
            }
        };
        Ok(Self::from_bytes(format, bytes))
    }
}

/// A file such as a PDF: `file_data` in base64 along with its `filename`, or the `file_id` of a
/// file uploaded to the server.
///
/// OpenAI reads PDFs here. Gemini rejects files, but reads PDFs as [`ImageUrl`].
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct File {
    /// `data:` URL with the file in base64.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}
impl File {
    /// A file uploaded to the server.
    pub fn from_id(file_id: impl Into<String>) -> Self {
        Self {
            file_id: Some(file_id.into()),
            ..Self::default()
        }
    }
    pub fn from_bytes(filename: impl Into<String>, mime: Mime, bytes: impl AsRef<[u8]>) -> Self {
        Self {
            file_data: Some(data_url(&mime, bytes.as_ref())),
            filename: Some(filename.into()),
            file_id: None,
        }
    }
    /// Reads a local file. Its name is sent as `filename`.
    #[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
    pub async fn from_path(
        path: impl AsRef<std::path::Path>,
        mime: Mime,
    ) -> Result<Self, std::io::Error> {
        let path = path.as_ref();
        let filename = path
            .file_name()
            .map_or("file".into(), |name| name.to_string_lossy());
        Ok(Self::from_bytes(
            filename,
            mime,
            tokio::fs::read(path).await?,
        ))
    }
    /// Downloads `url`. The mime type is taken from the `Content-Type` header, and the filename
    /// from the URL.
    #[cfg(feature = "reqwest")]
    pub async fn download(url: &str) -> Result<Self, crate::Error> {
        let (mime, bytes) = crate::http::download(url).await?;
        let filename = url
            .split(['?', '#'])
            .next()
            .and_then(|path| path.rsplit('/').next())
            .filter(|name| !name.is_empty())
            .unwrap_or("file");
        Ok(Self::from_bytes(filename, mime, bytes))
    }
}

/// Reply of the model.
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct AssistantMessage {
    /// `None` when the model only calls tools.
    pub content: Option<String>,
    /// Why the model refused to answer, if it did.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    /// Thoughts of a reasoning model. Not in the OpenAI specification: servers send it as
    /// `reasoning_content` (DeepSeek, vLLM, Qwen..) or `reasoning` (OpenRouter, Ollama, Groq..).
    /// Only sent back to the API with `AI::send_reasoning(true)`.
    #[serde(rename = "reasoning_content", skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}
impl AssistantMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            ..Self::default()
        }
    }
    /// Content, or `""` if there is none.
    pub fn text(&self) -> &str {
        self.content.as_deref().unwrap_or_default()
    }
    /// Parses the content as JSON. A markdown code fence around it is ignored.
    pub fn json<T: DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        let mut json = self.text().trim();
        if let Some(fenced) = json.strip_prefix("```") {
            json = fenced.strip_prefix("json").unwrap_or(fenced);
            json = json.strip_suffix("```").unwrap_or(json).trim();
        }
        serde_json::from_str(json)
    }
}
impl_from!(AssistantMessage {
    &str => |content| Self::new(content),
    String => |content| Self::new(content),
    ToolCall => |call| Self { tool_calls: vec![call], ..Self::default() },
    Vec<ToolCall> => |tool_calls| Self { tool_calls, ..Self::default() },
});
/// Parsed leniently, as servers differ. See `RawMessage`.
impl<'de> Deserialize<'de> for AssistantMessage {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Option::<RawMessage>::deserialize(deserializer)?
            .unwrap_or_default()
            .into())
    }
}

/// Used for calls without an id, as tool messages refer to calls by id.
static GENERATED_CALL_ID: AtomicU64 = AtomicU64::new(0);
pub(crate) fn generate_call_id() -> String {
    format!("call_{}", GENERATED_CALL_ID.fetch_add(1, Ordering::Relaxed))
}

/// Call of a function by the model. Sent as a tool call of `"type": "function"`.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub function: FunctionCall,
}
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    /// JSON encoded arguments, exactly as generated by the model. Use `ToolCall::args()` to parse
    /// them.
    pub arguments: String,
}
impl ToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.to_string(),
            },
        }
    }
    /// Parses the arguments. Empty arguments are parsed as `{}`.
    pub fn args<T: DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        match self.function.arguments.trim() {
            "" => serde_json::from_str("{}"),
            arguments => serde_json::from_str(arguments),
        }
    }
}
impl Serialize for ToolCall {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct ApiToolCall<'a> {
            id: &'a str,
            r#type: &'static str,
            function: FunctionCall,
        }
        // Some servers reject empty arguments.
        let arguments = match self.function.arguments.trim() {
            "" => "{}".to_string(),
            _ => self.function.arguments.clone(),
        };
        ApiToolCall {
            id: &self.id,
            r#type: "function",
            function: FunctionCall {
                name: self.function.name.clone(),
                arguments,
            },
        }
        .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for ToolCall {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(RawToolCall::deserialize(deserializer)?.into_tool_call())
    }
}

/// Result of a tool call, sent back to the model.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolMessage {
    /// `id` of the `ToolCall` this answers.
    pub tool_call_id: String,
    pub content: String,
}
impl ToolMessage {
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }
}

/// Assistant message as received, which `delta` of a streamed chunk shares. Servers differ from
/// the specification in reasoning, content parts and arguments, so they are read as `Value`.
#[derive(Deserialize, Default)]
pub(crate) struct RawMessage {
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
    pub(crate) tool_calls: Option<Vec<RawToolCall>>,
}
impl RawMessage {
    /// `(content, reasoning, refusal)`, each `None` if empty. Content may be a list of parts,
    /// including thinking parts.
    pub(crate) fn texts(&self) -> (Option<String>, Option<String>, Option<String>) {
        let mut reasoning = [&self.reasoning_content, &self.reasoning]
            .into_iter()
            .flatten()
            .find_map(|r| r.as_str().filter(|r| !r.is_empty()).map(str::to_string));
        let content = match &self.content {
            Some(Value::String(text)) => Some(text.clone()),
            // Some servers send content parts in replies too.
            Some(Value::Array(parts)) => {
                let mut text = String::new();
                for part in parts {
                    match (part["type"].as_str(), part["text"].as_str()) {
                        (Some("thinking" | "reasoning"), _) => {
                            let thought = match &part["thinking"] {
                                Value::String(thought) => thought.clone(),
                                Value::Array(thoughts) => {
                                    thoughts.iter().filter_map(|t| t["text"].as_str()).collect()
                                }
                                _ => part["text"].as_str().unwrap_or_default().to_string(),
                            };
                            reasoning.get_or_insert_default().push_str(&thought);
                        }
                        (_, Some(part)) => text.push_str(part),
                        _ => {}
                    }
                }
                Some(text)
            }
            _ => None,
        };
        let non_empty = |text: Option<String>| text.filter(|text| !text.is_empty());
        (
            non_empty(content),
            non_empty(reasoning),
            non_empty(self.refusal.clone()),
        )
    }
}
impl From<RawMessage> for AssistantMessage {
    fn from(raw: RawMessage) -> Self {
        let (content, reasoning, refusal) = raw.texts();
        Self {
            content,
            refusal,
            reasoning,
            tool_calls: raw
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(RawToolCall::into_tool_call)
                .collect(),
        }
    }
}
#[derive(Deserialize)]
pub(crate) struct RawToolCall {
    /// Position of the call in a streamed reply.
    #[serde(default)]
    pub(crate) index: Option<u32>,
    #[serde(default)]
    pub(crate) id: Option<String>,
    #[serde(default)]
    pub(crate) function: RawFunction,
}
impl RawToolCall {
    /// A missing id is generated.
    fn into_tool_call(self) -> ToolCall {
        ToolCall {
            id: self
                .id
                .filter(|id| !id.is_empty())
                .unwrap_or_else(generate_call_id),
            function: FunctionCall {
                arguments: self.function.arguments().unwrap_or_default(),
                name: self.function.name.unwrap_or_default(),
            },
        }
    }
}
#[derive(Deserialize, Default)]
pub(crate) struct RawFunction {
    #[serde(default)]
    pub(crate) name: Option<String>,
    /// JSON encoded string as per specification, but some servers send an object.
    #[serde(default)]
    arguments: Option<Value>,
}
impl RawFunction {
    pub(crate) fn arguments(&self) -> Option<String> {
        match &self.arguments {
            Some(Value::String(arguments)) => Some(arguments.clone()),
            None | Some(Value::Null) => None,
            Some(arguments) => Some(arguments.to_string()),
        }
    }
}
