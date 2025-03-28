use serde::Serialize;
use derive_new::new;

#[derive(Serialize)]
#[allow(non_camel_case_types)]
pub enum Role {
    user,
    developer,
    assistant,
}

#[derive(Serialize)]
#[derive(new)]
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

#[derive(Serialize)]
#[derive(new)]
pub struct Chat<'a> {
    role: Role,
    parts: &'a [Part<'a>],
}

#[derive(Serialize)]
#[derive(new)]
pub struct GeminiBody<'a> {
    contents: &'a [Chat<'a>],
}
