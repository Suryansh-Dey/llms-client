use derive_new::new;
use serde::Serialize;

#[derive(Serialize)]
#[allow(non_camel_case_types)]
pub enum Role {
    user,
    developer,
    assistant,
}

#[derive(Serialize, new)]
pub struct Chat<'a> {
    role: Role,
    content: &'a str,
}

#[derive(Serialize, new)]
pub struct OpenAiBody<'a> {
    model: &'a str,
    messages: &'a [Chat<'a>],
}
