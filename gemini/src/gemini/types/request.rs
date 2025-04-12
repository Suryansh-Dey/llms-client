use derive_new::new;
use getset::Getters;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum Role {
    user,
    developer,
    model,
}

#[derive(Serialize, Deserialize, Clone, new, Getters)]
pub struct InlineData {
    #[get = "pub"]
    mime_type: String,
    #[get = "pub"]
    data: String,
}

#[derive(Serialize, Deserialize, Clone)]
#[allow(non_camel_case_types)]
pub enum Language {
    ///Unspecified language. This value should not be used.
    LANGUAGE_UNSPECIFIED,
    ///Python >= 3.10, with numpy and simpy available.
    PYTHON,
}

#[derive(Serialize, Deserialize, Clone, new, Getters)]
pub struct ExecutableCode {
    #[get = "pub"]
    language: Language,
    #[get = "pub"]
    code: String,
}

#[derive(Serialize, Deserialize, Clone, new, Getters)]
pub struct FunctionCall {
    id: Option<String>,
    name: String,
    args: Option<Value>,
}

#[derive(Serialize, Deserialize, Clone, new, Getters)]
pub struct FunctionResponse {
    id: Option<String>,
    name: String,
    response: Value,
}

#[derive(Serialize, Deserialize, Clone, new, Getters)]
#[allow(non_snake_case)]
pub struct FileData {
    mimeType: Option<String>,
    fileUrl: String,
}

#[derive(Serialize, Deserialize, Clone)]
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

#[derive(Serialize, Deserialize, Clone, new, Getters)]
pub struct CodeExecuteResult {
    #[get = "pub"]
    outcome: Outcome,
    #[get = "pub"]
    output: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
#[allow(non_camel_case_types)]
pub enum Part {
    text(String),
    #[serde(alias = "inlineData")]
    inline_data(InlineData),
    #[serde(alias = "executableCode")]
    executable_code(ExecutableCode),
    #[serde(alias = "codeExecutionResult")]
    code_execution_result(CodeExecuteResult),
    functionCall(FunctionCall),
    functionResponse(FunctionResponse),
    fileData(FileData),
}

#[derive(Serialize, Deserialize, new, Getters)]
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

#[derive(Serialize, new)]
pub struct SystemInstruction<'a> {
    parts: &'a [Part],
}

#[derive(Serialize, new)]
pub struct GeminiRequestBody<'a> {
    system_instruction: Option<&'a SystemInstruction<'a>>,
    tools: Option<&'a [Tool]>,
    contents: &'a [&'a Chat],
    generation_config: Option<&'a Value>,
}

#[derive(Serialize)]
#[allow(non_camel_case_types)]
pub enum Tool {
    /// Generally it can be `Tool::google_search(json!({}))`
    google_search(Value),
    /// It is of form `Tool::function_calling(`[functionDeclaration](https://ai.google.dev/gemini-api/docs/function-calling?example=meeting)`)`
    functionDeclarations(Vec<Value>),
    /// Generally it can be `Tool::code_execution(json!({}))`,
    code_execution(Value),
}

pub(super) fn concatinate_parts(updating: &mut Vec<Part>, updator: &[Part]) {
    for updator_part in updator {
        match updator_part {
            Part::text(updator_text) => {
                if let Some(Part::text(updating_text)) =
                    updating.iter_mut().find(|e| matches!(e, Part::text(_)))
                {
                    updating_text.push_str(updator_text);
                    continue;
                }
            }
            Part::inline_data(updator_data) => {
                if let Some(Part::inline_data(updating_data)) = updating
                    .iter_mut()
                    .find(|e| matches!(e, Part::inline_data(_)))
                {
                    updating_data.data.push_str(&updator_data.data());
                    continue;
                }
            }
            Part::executable_code(updator_data) => {
                if let Some(Part::executable_code(updating_data)) = updating
                    .iter_mut()
                    .find(|e| matches!(e, Part::executable_code(_)))
                {
                    updating_data.code.push_str(&updator_data.code());
                    continue;
                }
            }
            Part::code_execution_result(updator_data) => {
                if let Some(Part::code_execution_result(updating_data)) = updating
                    .iter_mut()
                    .find(|e| matches!(e, Part::code_execution_result(_)))
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
            _ => {
                updating.push(updator_part.clone());
                continue;
            }
        }
        updating.push(updator_part.clone());
    }
}
