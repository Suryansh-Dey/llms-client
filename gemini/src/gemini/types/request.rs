use std::str::FromStr;
use base64::{Engine, engine::general_purpose::STANDARD};
use derive_new::new;
use getset::Getters;
use mime::{FromStrError, Mime};
#[cfg(feature = "reqwest")]
use reqwest::header::{HeaderMap, ToStrError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Role {
    User,
    Model,
    Function,
}

#[derive(Serialize, Deserialize, Clone, Getters, Debug)]
pub struct InlineData {
    #[get = "pub"]
    mime_type: String,
    #[get = "pub"]
    ///Base64 encoded string.
    data: String,
}

#[derive(Debug)]
pub enum InlineDataError {
    #[cfg(feature = "reqwest")]
    RequestFailed(reqwest::Error),
    CheckerFalse,
    ContentTypeMissing,
    #[cfg(feature = "reqwest")]
    ContentTypeParseFailed(ToStrError),
    InvalidMimeType(FromStrError),
}

impl InlineData {
    /// Creates a new InlineData.
    /// `data` must be a base64 encoded string.
    pub fn new(mime_type: Mime, data: String) -> Self {
        Self {
            mime_type: mime_type.to_string(),
            data,
        }
    }

    #[cfg(feature = "reqwest")]
    pub async fn from_url_with_check<F: FnOnce(&HeaderMap) -> bool>(
        url: &str,
        checker: F,
    ) -> Result<Self, InlineDataError> {
        let response = reqwest::get(url)
            .await
            .map_err(|e| InlineDataError::RequestFailed(e))?;
        if !checker(response.headers()) {
            return Err(InlineDataError::CheckerFalse);
        }

        let mime_type = response
            .headers()
            .get("Content-Type")
            .ok_or(InlineDataError::ContentTypeMissing)?
            .to_str()
            .map_err(|e| InlineDataError::ContentTypeParseFailed(e))?;
        let mime_type =
            Mime::from_str(mime_type).map_err(|e| InlineDataError::InvalidMimeType(e))?;

        let body = response
            .bytes()
            .await
            .map_err(|e| InlineDataError::RequestFailed(e))?;

        Ok(InlineData::new(mime_type, STANDARD.encode(body)))
    }

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

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Language {
    ///Unspecified language. This value should not be used.
    LanguageUnspecified,
    ///Python >= 3.10, with numpy and simpy available.
    Python,
}

#[derive(Serialize, Deserialize, Clone, new, Getters, Debug)]
pub struct ExecutableCode {
    #[get = "pub"]
    language: Language,
    #[get = "pub"]
    code: String,
}

#[derive(Serialize, Deserialize, Clone, new, Getters, Debug)]
pub struct FunctionCall {
    #[get = "pub"]
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[get = "pub"]
    args: Option<Value>,
}

#[derive(Serialize, Deserialize, Clone, new, Getters, Debug)]
pub struct FunctionResponse {
    #[get = "pub"]
    name: String,
    #[get = "pub"]
    response: Value,
}

#[derive(Serialize, Deserialize, Clone, new, Getters, Debug)]
pub struct FileData {
    #[serde(skip_serializing_if = "Option::is_none", alias = "mimeType")]
    #[get = "pub"]
    mime_type: Option<String>,
    #[serde(alias = "fileUri")]
    #[get = "pub"]
    file_uri: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Outcome {
    /// Unspecified status. This value should not be used.
    OutcomeUnspecified,
    /// Code execution completed successfully.
    OutcomeOk,
    /// Code execution finished but with a failure. `stderr` should contain the reason.
    OutcomeFailed,
    /// Code execution ran for too long, and was cancelled.
    /// There may or may not be a partial output present.
    OutcomeDeadlineExceeded,
}

#[derive(Serialize, Deserialize, Clone, new, Getters, Debug)]
pub struct CodeExecutionResult {
    #[get = "pub"]
    outcome: Outcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[get = "pub"]
    output: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub enum PartType {
    Text(String),
    ///Image or document like PDF
    InlineData(InlineData),
    ExecutableCode(ExecutableCode),
    CodeExecutionResult(CodeExecutionResult),
    FunctionCall(FunctionCall),
    FunctionResponse(FunctionResponse),
    ///For Audio file URL. Not allowed for images or PDFs, use InlineData instead.
    FileData(FileData),
}
#[derive(Serialize, Deserialize, Clone, Getters, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[get = "pub"]
    #[serde(flatten)]
    data: PartType,
    #[get = "pub"]
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<bool>,
    #[get = "pub"]
    #[serde(skip_serializing_if = "Option::is_none")]
    thought_signature: Option<String>,
}
impl Part {
    pub fn is_thought(&self) -> bool {
        self.thought_signature.is_some() || self.thought == Some(true)
    }
    pub fn new(data: PartType) -> Self {
        Self {
            data,
            thought: None,
            thought_signature: None,
        }
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
impl From<ExecutableCode> for Part {
    fn from(value: ExecutableCode) -> Self {
        Self::new(PartType::ExecutableCode(value))
    }
}
impl From<CodeExecutionResult> for Part {
    fn from(value: CodeExecutionResult) -> Self {
        Self::new(PartType::CodeExecutionResult(value))
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
impl From<FileData> for Part {
    fn from(value: FileData) -> Self {
        Self::new(PartType::FileData(value))
    }
}

#[derive(Serialize, Deserialize, new, Getters, Debug, Clone)]
pub struct Chat {
    #[get = "pub"]
    role: Role,
    #[get = "pub"]
    parts: Vec<Part>,
}
impl Chat {
    pub(super) fn parts_mut(&mut self) -> &mut Vec<Part> {
        &mut self.parts
    }
    ///`seperator` used to concatenate all text parts. TL;DR use "\n" as seperator.
    ///Don't contain thoughts
    pub fn get_text_no_think(&self, seperator: impl AsRef<str>) -> String {
        let parts = self.parts();
        let final_text = parts
            .iter()
            .filter_map(|part| {
                if let PartType::Text(text) = part.data() {
                    if !part.is_thought() {
                        Some(text.as_str())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<&str>>()
            .join(seperator.as_ref());

        final_text
    }
    ///`seperator` used to concatenate all text parts. TL;DR use "\n" as seperator.
    pub fn get_thoughts(&self, seperator: impl AsRef<str>) -> String {
        let parts = self.parts();
        let thoughts = parts
            .iter()
            .filter_map(|part| {
                if let PartType::Text(text) = part.data() {
                    if part.is_thought() {
                        Some(text.as_str())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<&str>>()
            .join(seperator.as_ref());

        thoughts
    }
    pub fn extract_text_all(parts: &[Part], seperator: impl AsRef<str>) -> String {
        parts
            .iter()
            .filter_map(|part| {
                if let PartType::Text(text_part) = part.data() {
                    // Just return the text, without checking the `thought` flag
                    Some(text_part.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<&str>>()
            .join(seperator.as_ref())
    }
    ///`seperator` used to concatenate all text parts. TL;DR use "" as seperator.
    ///Includes all text including thoughts
    pub fn get_text_all(&self, seperator: impl AsRef<str>) -> String {
        Self::extract_text_all(&self.parts(), seperator)
    }
    pub fn is_thinking(&self) -> bool {
        self.parts.iter().any(|p| p.is_thought())
    }
}

#[derive(Serialize, Deserialize, Clone, Getters, Debug, Default)]
pub struct ThinkingConfig {
    /// Indicates whether to include thoughts in the response. If true, thoughts
    /// are returned only if the model supports thought and thoughts are available.
    #[get = "pub"]
    include_thoughts: bool,
    /// Indicates the thinking budget in tokens.
    #[get = "pub"]
    thinking_budget: i32,
}
impl ThinkingConfig {
    /// Read [here](https://ai.google.dev/gemini-api/docs/thinking#set-budget) for allowed range of
    /// `thinking_budget`
    pub fn new(include_thoughts: bool, thinking_budget: u32) -> Self {
        Self {
            include_thoughts,
            thinking_budget: thinking_budget as i32,
        }
    }
    pub fn new_disable_thinking() -> Self {
        Self {
            include_thoughts: false,
            thinking_budget: 0,
        }
    }
    pub fn new_dynamic_thinking(include_thoughts: bool) -> Self {
        Self {
            include_thoughts,
            thinking_budget: -1,
        }
    }
}

#[derive(Serialize, Deserialize, Getters, new, Debug, Clone)]
pub struct SystemInstruction {
    #[get = "pub"]
    parts: Vec<Part>,
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

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmCategory {
    HarmCategoryHarassment,
    HarmCategoryHateSpeech,
    HarmCategorySexuallyExplicit,
    HarmCategoryDangerousContent,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BlockThreshold {
    BlockNone,
    BlockOnlyHigh,
    BlockMediumAndAbove,
    BlockLowAndAbove,
}
#[derive(Serialize, Deserialize, new, Getters, Debug, Clone)]
pub struct SafetySetting {
    #[get = "pub"]
    category: HarmCategory,
    #[get = "pub"]
    threshold: BlockThreshold,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    /// Configuration for function calling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_calling_config: Option<FunctionCallingConfig>,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    /// The mode in which function calling should execute.
    /// Can be "AUTO", "ANY", or "NONE".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<FunctionCallingMode>,

    /// Optional: Only provide this if mode is "ANY".
    /// Restricts the model to only call specific functions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionCallingMode {
    /// Default model behavior. Model decides whether to predict a
    /// function call or a natural language response.
    Auto,
    /// Model is constrained to always predict a function call.
    Any,
    /// Model will not predict any function call.
    None,
}

#[derive(Serialize, new)]
#[serde(rename_all = "camelCase")]
pub struct GeminiRequestBody<'a> {
    system_instruction: Option<&'a SystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [Tool]>,
    contents: &'a [&'a Chat],
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_settings: Option<&'a [SafetySetting]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<&'a ToolConfig>,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub enum Tool {
    /// Generally it can be `Tool::GoogleSearch(json!({}))`
    GoogleSearch(Value),
    /// Recommended: write `#[gemini_function]` above the function and pass
    /// `vec![function_name::gemini_schema(), ..]`
    /// OR
    /// It must be of form `vec![`[functionDeclaration](https://ai.google.dev/gemini-api/docs/function-calling?example=meeting)`, ..]`
    FunctionDeclarations(Vec<Value>),
    /// Generally it can be `Tool::CodeExecution(json!({}))`,
    CodeExecution(Value),
}

pub fn concatenate_parts(updating: &mut Vec<Part>, updator: &[Part]) {
    for updator_part in updator {
        if let Some(updating_last) = updating.last_mut() {
            match &updator_part.data {
                PartType::Text(updator_text) => {
                    if updating_last.is_thought() == updator_part.is_thought() {
                        if let PartType::Text(ref mut updating_text) = updating_last.data {
                            updating_text.push_str(&updator_text);
                            continue;
                        }
                    }
                }
                PartType::InlineData(updator_data) => {
                    if let PartType::InlineData(ref mut updating_data) = updating_last.data {
                        updating_data.data.push_str(&updator_data.data());
                        continue;
                    }
                }
                PartType::ExecutableCode(updator_data) => {
                    if let PartType::ExecutableCode(ref mut updating_data) = updating_last.data {
                        updating_data.code.push_str(&updator_data.code());
                        continue;
                    }
                }
                PartType::CodeExecutionResult(updator_data) => {
                    if let PartType::CodeExecutionResult(ref mut updating_data) = updating_last.data
                    {
                        if let Some(ref mut updating_output) = updating_data.output {
                            if let Some(updator_output) = updator_data.output() {
                                updating_output.push_str(updator_output);
                            }
                        } else {
                            updating_data.output = updator_data.output.clone();
                        }
                        continue;
                    }
                }
                _ => {}
            }
        }
        updating.push(updator_part.clone());
    }
}
