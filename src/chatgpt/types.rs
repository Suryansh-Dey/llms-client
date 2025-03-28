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
pub struct Chat<'a> {
    role: Role,
    content: &'a str,
}

#[derive(Serialize)]
#[derive(new)]
pub struct OpenAiBody<'a> {
    model: &'a str,
    messages: &'a [Chat<'a>],
}
