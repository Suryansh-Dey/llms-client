use base64::{Engine, engine::general_purpose::STANDARD};
use derive_new::new;
use getset::Getters;
use mime::Mime;
use reqwest::header::{HeaderMap, ToStrError};
use serde::{Deserialize, Serialize};
use serde::ser::{SerializeMap, Serializer};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum Role {
    user,
    model,
}

#[derive(Serialize, Deserialize, Clone, new, Getters, Debug)]
pub struct InlineData {
    #[get = "pub"]
    mime_type: String,
    #[get = "pub"]
    data: String,
}
#[derive(Debug)]
pub enum InlineDataError {
    RequestFailed(reqwest::Error),
    CheckerFalse,
    ContentTypeMissing,
    ContentTypeParseFailed(ToStrError),
}
impl InlineData {
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
            .map_err(|e| InlineDataError::ContentTypeParseFailed(e))?
            .to_string();
        let body = response
            .bytes()
            .await
            .map_err(|e| InlineDataError::RequestFailed(e))?;
        Ok(InlineData::new(mime_type, STANDARD.encode(body)))
    }
    pub async fn from_url(url: &str) -> Result<Self, InlineDataError> {
        Self::from_url_with_check(url, |_| true).await
    }
    pub async fn from_path(file_path: &str, mime_type: Mime) -> Result<Self, std::io::Error> {
        let data = tokio::fs::read(file_path).await?;
        Ok(InlineData::new(
            mime_type.to_string(),
            STANDARD.encode(data),
        ))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[allow(non_camel_case_types)]
pub enum Language {
    ///Unspecified language. This value should not be used.
    LANGUAGE_UNSPECIFIED,
    ///Python >= 3.10, with numpy and simpy available.
    PYTHON,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    args: Option<Value>,
}

#[derive(Serialize, Deserialize, Clone, new, Getters, Debug)]
pub struct FunctionResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    name: String,
    response: Value,
}

#[derive(Serialize, Deserialize, Clone, new, Getters, Debug)]
#[allow(non_snake_case)]
pub struct FileData {
    #[serde(skip_serializing_if = "Option::is_none", alias = "mimeType")]
    mime_type: Option<String>,
    #[serde(alias = "fileUri")]
    file_uri: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[allow(non_camel_case_types)]
pub enum Outcome {
    /// Unspecified status. This value should not be used.
    OUTCOME_UNSPECIFIED,
    /// Code execution completed successfully.
    OUTCOME_OK,
    /// Code execution finished but with a failure. `stderr` should contain the reason.
    OUTCOME_FAILED,
    /// Code execution ran for too long, and was cancelled.
    /// There may or may not be a partial output present.
    OUTCOME_DEADLINE_EXCEEDED,
}

#[derive(Serialize, Deserialize, Clone, new, Getters, Debug)]
pub struct CodeExecuteResult {
    #[get = "pub"]
    outcome: Outcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[get = "pub"]
    output: Option<String>,
}

#[derive(Serialize, Clone, new, Getters, Debug)]
pub struct TextPart {
    #[get = "pub"]
    text: String,
    #[get = "pub"]
    thought: bool,
}
impl From<String> for TextPart {
    /// Creates a TextPart from a String, where `thought` is always `false`.
    fn from(text: String) -> Self {
        TextPart::new(text, false)
    }
}
impl<'a> From<&'a str> for TextPart {
    /// Creates a TextPart from &str, where `thought` is always `false`.
    fn from(text: &'a str) -> Self {
        TextPart::new(text.to_string(), false)
    }
}

#[derive(Clone, Debug)]
#[allow(non_camel_case_types)]
pub enum Part {
    text(TextPart),
    ///Image or document
    inline_data(InlineData),
    executable_code(ExecutableCode),
    code_execution_result(CodeExecuteResult),
    functionCall(FunctionCall),
    functionResponse(FunctionResponse),
    ///For Audio file URL. Not allowed for images or PDFs, use InlineData instead.
    file_data(FileData),
}

impl serde::Serialize for Part {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Part::text(text_part) => {
                if *text_part.thought() {
                    // If it's a "thought", serialize as an object with two fields
                    let mut map = serializer.serialize_map(Some(2))?;
                    map.serialize_entry("text", text_part.text())?;
                    map.serialize_entry("thought", text_part.thought())?;
                    map.end()
                } else {
                    // If it's a regular text, we use a special serde method
                    // that will create exactly {"text": "..."}
                    serializer.serialize_newtype_variant("Part", 0, "text", text_part.text())
                }
            }

            // Standard handling for all other variants
            Part::inline_data(data) => {
                serializer.serialize_newtype_variant("Part", 1, "inlineData", data)
            }
            Part::executable_code(code) => {
                serializer.serialize_newtype_variant("Part", 2, "executableCode", code)
            }
            Part::code_execution_result(result) => {
                serializer.serialize_newtype_variant("Part", 3, "codeExecutionResult", result)
            }
            Part::functionCall(call) => {
                serializer.serialize_newtype_variant("Part", 4, "functionCall", call)
            }
            Part::functionResponse(response) => {
                serializer.serialize_newtype_variant("Part", 5, "functionResponse", response)
            }
            Part::file_data(data) => {
                serializer.serialize_newtype_variant("Part", 6, "fileData", data)
            }
        }
    }
}

impl<'de> serde::Deserialize<'de> for Part {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)] // small hack
        struct PartHelper {
            text: Option<String>,
            #[serde(default)]
            thought: bool,
            #[serde(alias = "inlineData")]
            inline_data: Option<InlineData>,
            #[serde(alias = "executableCode")]
            executable_code: Option<ExecutableCode>,
            #[serde(alias = "codeExecutionResult")]
            code_execution_result: Option<CodeExecuteResult>,
            #[serde(alias = "functionCall")]
            function_call: Option<FunctionCall>,
            #[serde(alias = "functionResponse")]
            function_response: Option<FunctionResponse>,
            #[serde(alias = "fileData")]
            file_data: Option<FileData>,
        }

        let helper = PartHelper::deserialize(deserializer)?;

        // We check the variants in order of their uniqueness
        if let Some(data) = helper.inline_data {
            Ok(Part::inline_data(data))
        } else if let Some(code) = helper.executable_code {
            Ok(Part::executable_code(code))
        } else if let Some(result) = helper.code_execution_result {
            Ok(Part::code_execution_result(result))
        } else if let Some(call) = helper.function_call {
            Ok(Part::functionCall(call))
        } else if let Some(resp) = helper.function_response {
            Ok(Part::functionResponse(resp))
        } else if let Some(data) = helper.file_data {
            Ok(Part::file_data(data))
        } else if let Some(text) = helper.text {
            // Special case: create a TextPart with the text and the `thought` flag
            let text_part = TextPart::new(text, helper.thought);
            Ok(Part::text(text_part))
        } else {
            Err(serde::de::Error::custom("Unknown Part variant in JSON"))
        }
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
}

#[derive(Serialize, Deserialize, Clone, new, Getters, Debug, Default)]
#[allow(non_snake_case)]
pub struct ThinkingConfig {
    /// Optional. Indicates whether to include thoughts in the response. If true, thoughts
	/// are returned only if the model supports thought and thoughts are available.
    #[get = "pub"]
    pub include_thoughts: bool,
    /// Optional. Indicates the thinking budget in tokens.
    #[get = "pub"]
    pub thinking_budget: i32,
}

#[derive(Serialize, Deserialize, new, Debug, Clone)]
pub struct SystemInstruction {
    parts: Vec<Part>,
}
impl SystemInstruction {
    pub fn from_str(prompt: impl Into<TextPart>) -> Self {
        Self {
            parts: vec![Part::text(prompt.into())],
        }
    }
}
#[allow(non_snake_case)]
#[derive(Serialize, new)]
pub struct GeminiRequestBody<'a> {
    system_instruction: Option<&'a SystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [Tool]>,
    contents: &'a [&'a Chat],
    #[serde(skip_serializing_if = "Option::is_none")]
    generationConfig: Option<&'a Value>,
}

#[derive(Serialize, Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum Tool {
    /// Generally it can be `Tool::google_search(json!({}))`
    google_search(Value),
    /// It is of form `Tool::function_calling(`[functionDeclaration](https://ai.google.dev/gemini-api/docs/function-calling?example=meeting)`)`
    functionDeclarations(Vec<Value>),
    /// Generally it can be `Tool::code_execution(json!({}))`,
    code_execution(Value),
}

pub fn concatenate_parts(updating: &mut Vec<Part>, updator: &[Part]) {
    for updator_part in updator {
        match updator_part {
            Part::text(updator_text_part) => {
                if let Some(Part::text(updating_text_part)) = updating.last_mut() {
                    if *updating_text_part.thought() == *updator_text_part.thought() {
                        updating_text_part.text.push_str(updator_text_part.text());
                        continue;
                    }
                    continue;
                }
            }
            Part::inline_data(updator_data) => {
                if let Some(Part::inline_data(updating_data)) = updating.last_mut() {
                    updating_data.data.push_str(&updator_data.data());
                    continue;
                }
            }
            Part::executable_code(updator_data) => {
                if let Some(Part::executable_code(updating_data)) = updating.last_mut() {
                    updating_data.code.push_str(&updator_data.code());
                    continue;
                }
            }
            Part::code_execution_result(updator_data) => {
                if let Some(Part::code_execution_result(updating_data)) = updating.last_mut() {
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
            _ => {
                updating.push(updator_part.clone());
                continue;
            }
        }
        updating.push(updator_part.clone());
    }
}
