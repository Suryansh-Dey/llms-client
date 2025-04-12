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

#[derive(Serialize, Deserialize, Clone, new, Getters)]
pub struct ExecutableCode {
    #[get = "pub"]
    language: String,
    #[get = "pub"]
    code: String,
}

#[derive(Serialize, Deserialize, Clone, new, Getters)]
pub struct CodeExecuteResult {
    #[get = "pub"]
    outcome: String,
    #[get = "pub"]
    output: String,
}

#[derive(Serialize, Deserialize, Clone)]
#[allow(non_camel_case_types)]
pub enum Part {
    text(String),
    inline_data(InlineData),
    executable_code(ExecutableCode),
    code_execute_result(CodeExecuteResult),
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
pub struct GeminiBody<'a> {
    system_instruction: Option<&'a SystemInstruction<'a>>,
    tools: Option<&'a [Value]>,
    contents: &'a [&'a Chat],
    generation_config: Option<&'a Value>,
}

pub(super) fn concatinate_parts(
    updating: &mut Vec<Part>,
    updator: &[Part],
) -> Result<(), Box<dyn std::error::Error>> {
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
            Part::code_execute_result(updator_data) => {
                if let Some(Part::code_execute_result(updating_data)) = updating
                    .iter_mut()
                    .find(|e| matches!(e, Part::code_execute_result(_)))
                {
                    updating_data.output.push_str(&updator_data.output());
                    continue;
                }
            }
        }
        updating.push(updator_part.clone());
    }
    Ok(())
}
