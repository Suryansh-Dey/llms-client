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
    pub fn get_history(&self) -> &Vec<Chat> {
        &self.history
    }
    pub fn get_parts_mut(&mut self, chat_previous_no: usize) -> Option<&mut Chat> {
        let history_length = self.get_history().len();
        self.history.get_mut(history_length - chat_previous_no)
    }
    pub fn ask(&mut self, parts: Vec<Part>) -> &Self {
        self.history.push(Chat::new(Role::user, parts));
        self.chat_no += 1;
        self
    }
    pub fn ask_string(&mut self, prompt: String) -> &Self {
        self.history
            .push(Chat::new(Role::user, vec![Part::text(prompt)]));
        self.chat_no += 1;
        self
    }
    pub fn update(&mut self, response: GeminiResponse) -> Result<(), Value> {
        let history = &mut self.history;
        let reply = response.get_as_string().map_err(|value| value.clone())?;
        if let Some(chat) = history.last_mut() {
            if let Role::model = chat.role {
                if let Some(Part::text(data)) = chat.parts.last_mut() {
                    data.push_str(reply);
                } else {
                    chat.parts.push(Part::text(reply.to_string()));
                }
            }
        } else {
            history.push(Chat::new(Role::model, vec![Part::text(reply.to_string())]));
        }
        Ok(())
    }
}
