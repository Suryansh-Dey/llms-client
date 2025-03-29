use derive_new::new;
use serde::Serialize;
use serde_json::Value;

#[derive(Serialize)]
#[allow(non_camel_case_types)]
pub enum Role {
    user,
    developer,
    model,
}

#[derive(Serialize, new)]
pub struct InlineData<'a> {
    mime_type: &'a str,
    data: &'a str,
}

#[derive(Serialize)]
#[allow(non_camel_case_types)]
pub enum Part<'a> {
    text(&'a str),
    inline_data(InlineData<'a>),
}

#[derive(Serialize, new)]
pub struct Chat<'a> {
    role: Role,
    parts: &'a [Part<'a>],
}

#[derive(Serialize, new)]
pub struct GeminiBody<'a> {
    contents: &'a [Chat<'a>],
    system_instruction: Option<&'a [Part<'a>]>,
    generation_config: Option<&'a Value>,
}
