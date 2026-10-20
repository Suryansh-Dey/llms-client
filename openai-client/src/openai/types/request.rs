#[cfg(feature = "reqwest")]
use base64::{Engine, engine::general_purpose::STANDARD};
use derive_getters::{Dissolve, Getters};
use mime::{FromStrError, Mime};
#[cfg(feature = "reqwest")]
use reqwest::header::{HeaderMap, ToStrError};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value, json};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
mod api;
mod chat;
pub(crate) use api::ApiMessage;
#[cfg(feature = "reqwest")]
use api::to_api_messages;
pub use chat::Chat;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Role {
    User,
    Assistant,
    /// Result of a function (tool) call.
    Tool,
}

fn deserialize_mime<'de, D>(deserializer: D) -> Result<Mime, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Mime::from_str(&s).map_err(serde::de::Error::custom)
}
fn serialize_mime<S>(mime: &Mime, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(mime.as_ref())
}
/// Base64 encoded file. Sent as
/// - `image_url` with a data URL for `image/*`
/// - `input_audio` for `audio/*`
/// - `file` with `file_data` for anything else, e.g. PDF
#[derive(Dissolve, Serialize, Deserialize, Clone, Getters, Debug)]
pub struct InlineData {
    #[serde(
        deserialize_with = "deserialize_mime",
        serialize_with = "serialize_mime"
    )]
    mime_type: Mime,
    ///Base64 encoded string.
    data: String,
}

#[derive(thiserror::Error, Debug)]
pub enum InlineDataError {
    #[error(transparent)]
    #[cfg(feature = "reqwest")]
    RequestFailed(reqwest::Error),
    #[error("Checker function returned false")]
    ///Checker function returned false
    CheckerFalse,
    #[error("Content-Type was missing in response headers")]
    ///Content-Type was missing in response headers
    ContentTypeMissing,
    #[error(transparent)]
    #[cfg(feature = "reqwest")]
    ContentTypeParseFailed(ToStrError),
    #[error("Failed to parse mime type: {0}")]
    ///Failed to parse mime type
    InvalidMimeType(FromStrError),
}

impl InlineData {
    /// Creates a new InlineData.
    /// `data` must be a base64 encoded string.
    pub fn new(mime_type: Mime, data: String) -> Self {
        Self { mime_type, data }
    }

    /// Downloads `url`, with the mime type from its `Content-Type` header. The body is only
    /// downloaded if `checker` returns true for the response headers.
    #[cfg(feature = "reqwest")]
    pub async fn from_url_with_check<F: FnOnce(&HeaderMap) -> bool>(
        url: &str,
        checker: F,
    ) -> Result<Self, InlineDataError> {
        let response = reqwest::get(url)
            .await
            .map_err(InlineDataError::RequestFailed)?;
        if !checker(response.headers()) {
            return Err(InlineDataError::CheckerFalse);
        }

        let mime_type = response
            .headers()
            .get("Content-Type")
            .ok_or(InlineDataError::ContentTypeMissing)?
            .to_str()
            .map_err(InlineDataError::ContentTypeParseFailed)?;
        let mime_type = Mime::from_str(mime_type).map_err(InlineDataError::InvalidMimeType)?;

        let body = response
            .bytes()
            .await
            .map_err(InlineDataError::RequestFailed)?;

        Ok(InlineData::new(mime_type, STANDARD.encode(body)))
    }

    /// Downloads `url`, with the mime type from its `Content-Type` header.
    #[cfg(feature = "reqwest")]
    pub async fn from_url(url: &str) -> Result<Self, InlineDataError> {
        Self::from_url_with_check(url, |_| true).await
    }

    #[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
    pub async fn from_path(file_path: &str, mime_type: Mime) -> Result<Self, std::io::Error> {
        let data = tokio::fs::read(file_path).await?;
        Ok(InlineData::new(mime_type, STANDARD.encode(data)))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}
/// Image by URL, fetched by the API server. Use `InlineData` for local files or servers that can't
/// fetch URLs.
#[derive(Dissolve, Serialize, Deserialize, Clone, Getters, Debug)]
pub struct ImageUrl {
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<ImageDetail>,
}
impl ImageUrl {
    /// An image the API server fetches. `url` is an `https://` URL reachable by the server
    /// (e.g. public or pre-signed), or a `data:` URL with base64 content. OpenAI accepts
    /// only images; some servers, e.g. Gemini, also accept other files such as PDF.
    /// `detail` sets the resolution the model sees, where supported.
    pub fn new(url: String, detail: Option<ImageDetail>) -> Self {
        Self { url, detail }
    }
}
impl From<String> for ImageUrl {
    fn from(url: String) -> Self {
        Self { url, detail: None }
    }
}
impl From<&str> for ImageUrl {
    fn from(url: &str) -> Self {
        url.to_string().into()
    }
}

/// Used for calls without an id, as `tool_call_id` must match a call.
static GENERATED_CALL_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Serialize, Deserialize, Clone, Getters, Debug)]
pub struct FunctionCall {
    id: String,
    name: String,
    /// JSON encoded arguments, exactly as generated by the model. Use `args()` to parse them.
    arguments: String,
    /// Position of the call in a streamed reply. Deltas of the same call share it.
    #[serde(skip)]
    #[getter(skip)]
    index: Option<u32>,
}
impl FunctionCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, args: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: args.to_string(),
            index: None,
        }
    }
    pub(crate) fn new_delta(
        id: Option<String>,
        name: Option<String>,
        arguments: String,
        index: Option<u32>,
    ) -> Self {
        let name = name.unwrap_or_default();
        let id = match id {
            Some(id) if !id.is_empty() => id,
            // A delta without name continues an earlier one, so it shouldn't get an id.
            _ if name.is_empty() => String::new(),
            _ => format!("call_{}", GENERATED_CALL_ID.fetch_add(1, Ordering::Relaxed)),
        };
        Self {
            id,
            name,
            arguments,
            index,
        }
    }
    /// Returns `(id, name, arguments)`.
    pub fn dissolve(self) -> (String, String, String) {
        (self.id, self.name, self.arguments)
    }
    /// Parses `arguments`. Empty arguments are parsed as `{}`.
    pub fn args(&self) -> Result<Value, serde_json::Error> {
        if self.arguments.trim().is_empty() {
            return Ok(json!({}));
        }
        serde_json::from_str(&self.arguments)
    }
    /// Merges a streamed delta of the same call (same `index`) into `self`, as the official
    /// OpenAI SDKs do. `id` and `name` come in the first delta, and some servers repeat them in
    /// the later ones, so only the first is kept. `arguments` arrive in pieces.
    fn append(&mut self, delta: FunctionCall) {
        if self.id.is_empty() {
            self.id = delta.id;
        }
        if self.name.is_empty() {
            self.name = delta.name;
        }
        self.arguments.push_str(&delta.arguments);
    }
}

/// Output of a function call. Sent as a `tool` message. Non text parts like images can't be in a
/// `tool` message as per OpenAI specification, so they are sent in a `user` message right after it.
#[derive(Dissolve, Serialize, Deserialize, Clone, Getters, Debug)]
pub struct FunctionResponse {
    /// `id` of the `FunctionCall` this responds to.
    id: String,
    name: String,
    parts: Vec<Part>,
}
impl FunctionResponse {
    pub fn new(id: String, name: String, parts: Vec<Part>) -> Self {
        Self { id, name, parts }
    }
}

/// Converts a value into the content of a `FunctionResponse`.
///
/// Implemented for anything `Serialize`, which is sent as text: strings as they are, everything
/// else as JSON. Use `MultimodalResponse` to respond with images, audio or files.
pub trait IntoFunctionResponse {
    fn into_function_response(self) -> Result<Vec<Part>, serde_json::Error>;
}
impl<T: Serialize> IntoFunctionResponse for T {
    fn into_function_response(self) -> Result<Vec<Part>, serde_json::Error> {
        let text = match serde_json::to_value(self)? {
            Value::String(text) => text,
            value => value.to_string(),
        };
        Ok(vec![text.into()])
    }
}
/// Function response with any parts, e.g. a screenshot along with its description.
/// # Example
/// ```no_run
/// # use open_client::{mime, openai::types::{request::*, sessions::Session}};
/// # async fn run(session: &mut Session, call: FunctionCall) -> Result<(), Box<dyn std::error::Error>> {
/// session.add_function_response(call.id(), MultimodalResponse::new(vec![
///     "Screenshot of the page".into(),
///     InlineData::from_path("screenshot.png", mime::IMAGE_PNG).await?.into(),
/// ]))?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MultimodalResponse {
    pub parts: Vec<Part>,
}
impl MultimodalResponse {
    pub fn new(parts: Vec<Part>) -> Self {
        Self { parts }
    }
}
impl From<Vec<Part>> for MultimodalResponse {
    fn from(parts: Vec<Part>) -> Self {
        Self::new(parts)
    }
}
impl IntoFunctionResponse for MultimodalResponse {
    fn into_function_response(self) -> Result<Vec<Part>, serde_json::Error> {
        Ok(self.parts)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub enum PartType {
    Text(String),
    ///Image, audio or document like PDF
    InlineData(InlineData),
    ImageUrl(ImageUrl),
    /// Model refused to answer. Only in replies.
    Refusal(String),
    FunctionCall(FunctionCall),
    FunctionResponse(FunctionResponse),
}
#[derive(Dissolve, Serialize, Deserialize, Clone, Getters, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(flatten)]
    data: PartType,
    /// Set for reasoning text, received as `reasoning_content` or `reasoning`.
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<bool>,
}
impl Part {
    pub fn is_thought(&self) -> bool {
        self.thought == Some(true)
    }
    pub fn new(data: PartType) -> Self {
        Self {
            data,
            thought: None,
        }
    }
    pub fn new_thought(text: impl Into<String>) -> Self {
        Self {
            data: PartType::Text(text.into()),
            thought: Some(true),
        }
    }
    pub fn data_mut(&mut self) -> &mut PartType {
        &mut self.data
    }
    pub fn data_owned(self) -> PartType {
        self.data
    }
}
impl From<PartType> for Part {
    fn from(value: PartType) -> Self {
        Self::new(value)
    }
}
impl From<String> for Part {
    fn from(value: String) -> Self {
        Self::new(PartType::Text(value))
    }
}
impl From<&str> for Part {
    fn from(value: &str) -> Self {
        Self::new(PartType::Text(value.into()))
    }
}
impl From<InlineData> for Part {
    fn from(value: InlineData) -> Self {
        Self::new(PartType::InlineData(value))
    }
}
impl From<ImageUrl> for Part {
    fn from(value: ImageUrl) -> Self {
        Self::new(PartType::ImageUrl(value))
    }
}
impl From<FunctionCall> for Part {
    fn from(value: FunctionCall) -> Self {
        Self::new(PartType::FunctionCall(value))
    }
}
impl From<FunctionResponse> for Part {
    fn from(value: FunctionResponse) -> Self {
        Self::new(PartType::FunctionResponse(value))
    }
}

/// Value of `reasoning_effort`. Supported values depend on the model.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

/// Sent as the first message, with `system` role.
#[derive(Dissolve, Serialize, Deserialize, Getters, Debug, Clone)]
pub struct SystemInstruction {
    parts: Vec<Part>,
}
impl SystemInstruction {
    pub fn new(parts: Vec<Part>) -> Self {
        Self { parts }
    }
}
impl From<String> for SystemInstruction {
    fn from(prompt: String) -> Self {
        Self {
            parts: vec![prompt.into()],
        }
    }
}
impl<'a> From<&'a str> for SystemInstruction {
    fn from(prompt: &'a str) -> Self {
        Self {
            parts: vec![prompt.into()],
        }
    }
}

/// Controls which (if any) tool is called by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// Model will not call any tool.
    None,
    /// Default model behavior. Model decides whether to call tools or respond with text.
    Auto,
    /// Model must call one or more tools.
    Required,
    /// Model must call the function with this name.
    Function(String),
}
impl Serialize for ToolChoice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::None => serializer.serialize_str("none"),
            Self::Auto => serializer.serialize_str("auto"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function(name) => {
                json!({"type": "function", "function": {"name": name}}).serialize(serializer)
            }
        }
    }
}
impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match value.as_str() {
            Some("none") => Ok(Self::None),
            Some("auto") => Ok(Self::Auto),
            Some("required") => Ok(Self::Required),
            _ => value["function"]["name"]
                .as_str()
                .map(|name| Self::Function(name.to_string()))
                .ok_or_else(|| serde::de::Error::custom(format!("invalid tool_choice: {value}"))),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Tool {
    /// Recommended: write `#[openai_function]` above the function and pass
    /// `Tool::Function(function_name::openai_schema())`
    /// OR
    /// It must be a [function definition](https://platform.openai.com/docs/api-reference/chat/create#chat_create-tools)
    /// i.e. `{"name": .., "description": .., "parameters": ..}`
    Function(Value),
    /// Sent as it is. For provider specific tools, e.g. `json!({"type": "web_search"})`
    Custom(Value),
}
impl Tool {
    /// Name of the function, if it's a `Tool::Function`.
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Function(function) => function["name"].as_str(),
            Self::Custom(_) => None,
        }
    }
}
impl Serialize for Tool {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Function(function) => {
                let mut map = Map::new();
                map.insert("type".into(), "function".into());
                map.insert("function".into(), function.clone());
                map.serialize(serializer)
            }
            Self::Custom(tool) => tool.serialize(serializer),
        }
    }
}
impl<'de> Deserialize<'de> for Tool {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut value = Value::deserialize(deserializer)?;
        Ok(match value["type"].as_str() {
            Some("function") if value["function"].is_object() => {
                Self::Function(value["function"].take())
            }
            _ => Self::Custom(value),
        })
    }
}

/// Keys of the request body managed by `OpenAI`, so they are ignored in `generation_config`.
#[cfg(feature = "reqwest")]
const RESERVED_KEYS: [&str; 5] = ["model", "messages", "tools", "tool_choice", "stream"];

/// Request body of `/chat/completions`.
#[cfg(feature = "reqwest")]
pub(crate) struct OpenAIRequestBody<'a> {
    pub model: &'a str,
    pub sys_prompt: Option<&'a SystemInstruction>,
    pub messages: &'a [&'a Chat],
    pub tools: Option<&'a [Tool]>,
    pub tool_choice: Option<&'a ToolChoice>,
    /// Entries are added to the top level of the body.
    pub generation_config: Option<&'a Value>,
    pub stream: bool,
    /// See `OpenAI::set_send_reasoning()`.
    pub send_reasoning: bool,
}
#[cfg(feature = "reqwest")]
impl Serialize for OpenAIRequestBody<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("model", self.model)?;
        map.serialize_entry(
            "messages",
            &to_api_messages(self.sys_prompt, self.messages, self.send_reasoning),
        )?;
        if let Some(tools) = self.tools.filter(|tools| !tools.is_empty()) {
            map.serialize_entry("tools", tools)?;
        }
        if let Some(tool_choice) = self.tool_choice {
            map.serialize_entry("tool_choice", tool_choice)?;
        }
        if self.stream {
            map.serialize_entry("stream", &true)?;
        }
        if let Some(Value::Object(config)) = self.generation_config {
            for (key, value) in config {
                if RESERVED_KEYS.contains(&key.as_str())
                    || (key == "stream_options" && !self.stream)
                    || value.is_null()
                {
                    continue;
                }
                map.serialize_entry(key, value)?;
            }
        }
        map.end()
    }
}

/// Concatenates `updator` into `updating`. Text is appended to the last text part, and a streamed
/// function call delta is merged into the call with the same `index`. A call without `index` is a
/// complete call, e.g. Gemini sends each streamed call whole and without `index`.
pub(crate) fn concatenate_parts(updating: &mut Vec<Part>, updator: impl IntoIterator<Item = Part>) {
    for updator_part in updator {
        let thought = updator_part.thought;
        match updator_part.data {
            PartType::Text(updator_text) => {
                if let Some(updating_last) = updating.last_mut()
                    && updating_last.is_thought() == (thought == Some(true))
                    && let PartType::Text(updating_text) = &mut updating_last.data
                {
                    updating_text.push_str(&updator_text);
                } else {
                    updating.push(Part {
                        data: PartType::Text(updator_text),
                        thought,
                    });
                }
            }
            PartType::FunctionCall(delta) if delta.index.is_some() => {
                let call = updating.iter_mut().find_map(|part| match &mut part.data {
                    PartType::FunctionCall(call) if call.index == delta.index => Some(call),
                    _ => None,
                });
                match call {
                    Some(call) => call.append(delta),
                    None => updating.push(delta.into()),
                }
            }
            data => updating.push(Part { data, thought }),
        }
    }
}
