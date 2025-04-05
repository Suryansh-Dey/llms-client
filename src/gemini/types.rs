use super::ask::GeminiResponse;
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
pub struct InlineData {
    mime_type: String,
    data: String,
}

#[derive(Serialize)]
#[allow(non_camel_case_types)]
pub enum Part {
    text(String),
    inline_data(InlineData),
}

#[derive(Serialize, new)]
pub struct Chat {
    role: Role,
    parts: Vec<Part>,
}

#[derive(Serialize, new)]
pub struct SystemInstruction<'a> {
    parts: &'a [Part],
}

#[derive(Serialize, new)]
pub struct GeminiBody<'a> {
    system_instruction: Option<&'a SystemInstruction<'a>>,
    contents: &'a [Chat],
    generation_config: Option<&'a Value>,
}

#[derive(Serialize)]
pub struct Session {
    history: Vec<Chat>,
    history_limit: usize,
    chat_no: usize,
}
impl Session {
    pub fn new(history_limit: usize) -> Self {
        Self {
            history: Vec::new(),
            history_limit,
            chat_no: 0,
        }
    }
    pub fn get_history(& self) -> &Vec<Chat> {
        &self.history
    }
    pub fn get_history_mut(& mut self) -> & mut Vec<Chat> {
        &mut self.history
    }
    pub fn update(& mut self, reply: GeminiResponse)->Result<(), Value> {
        self.get_history_mut().push(Chat::new(
            Role::model,
            vec![Part::text(reply.get_as_string().map_err(|value| value.clone())?.to_string())],
        ));
        Ok(())
    }
}
