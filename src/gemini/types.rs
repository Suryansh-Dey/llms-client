use std::collections::VecDeque;

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
    contents: &'a [&'a Chat],
    generation_config: Option<&'a Value>,
}

#[derive(Serialize)]
pub struct Session {
    history: VecDeque<Chat>,
    history_limit: usize,
    chat_no: usize,
}
impl Session {
    pub fn new(history_limit: usize) -> Self {
        Self {
            history: VecDeque::new(),
            history_limit,
            chat_no: 0,
        }
    }
    pub fn get_history(&self) -> Vec<&Chat> {
        let (left, right) = self.history.as_slices();
        left.iter().chain(right.iter()).collect()
    }
    pub fn get_history_length(&self) -> usize {
        self.history.len()
    }
    pub fn get_chat_mut(&mut self, chat_previous_no: usize) -> Option<&mut Chat> {
        let history_length = self.get_history_length();
        self.history.get_mut(history_length - chat_previous_no)
    }
    fn add_chat(&mut self, chat: Chat) -> &mut Self {
        self.history.push_back(chat);
        self.chat_no += 1;
        if self.get_history_length() > self.history_limit {
            self.history.pop_front();
        }
        self
    }
    pub fn ask(&mut self, parts: Vec<Part>) -> &Self {
        self.add_chat(Chat::new(Role::user, parts))
    }
    pub fn ask_string(&mut self, prompt: String) -> &Self {
        self.add_chat(Chat::new(Role::user, vec![Part::text(prompt)]))
    }
    pub fn update(&mut self, response: GeminiResponse) -> Result<(), Value> {
        let history = &mut self.history;
        let reply = response.get_as_string().map_err(|value| value.clone())?;
        if let Some(chat) = history.back_mut() {
            if let Role::model = chat.role {
                if let Some(Part::text(data)) = chat.parts.last_mut() {
                    data.push_str(reply);
                } else {
                    chat.parts.push(Part::text(reply.to_string()));
                }
            }
        } else {
            history.push_back(Chat::new(Role::model, vec![Part::text(reply.to_string())]));
        }
        Ok(())
    }
}
