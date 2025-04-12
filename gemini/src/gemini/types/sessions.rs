use super::request::*;
use super::response::GeminiResponse;
use serde::Serialize;
use std::collections::VecDeque;

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
    pub fn get_history_as_vecdeque(&self) -> &VecDeque<Chat> {
        &self.history
    }
    pub(super) fn get_history_as_vecdeque_mut(&mut self) -> &mut VecDeque<Chat> {
        &mut self.history
    }
    pub fn get_chat_no(&self) -> usize {
        self.chat_no
    }
    pub fn get_history(&self) -> Vec<&Chat> {
        let (left, right) = self.history.as_slices();
        left.iter().chain(right.iter()).collect()
    }
    pub fn get_history_length(&self) -> usize {
        self.history.len()
    }
    pub fn get_parts_mut(&mut self, chat_previous_no: usize) -> Option<&mut Vec<Part>> {
        let history_length = self.get_history_length();
        self.history
            .get_mut(history_length - chat_previous_no)
            .map(|chat| chat.parts_mut())
    }
    fn add_chat(&mut self, chat: Chat) -> &mut Self {
        self.history.push_back(chat);
        self.chat_no += 1;
        if self.get_history_length() > self.history_limit {
            self.history.pop_front();
        }
        self
    }
    /// `parts` should follow [gemini doc](https://ai.google.dev/gemini-api/docs/text-generation#image-input)
    pub fn ask(&mut self, parts: Vec<Part>) -> &mut Self {
        self.add_chat(Chat::new(Role::user, parts))
    }
    pub fn ask_string(&mut self, prompt: String) -> &mut Self {
        self.add_chat(Chat::new(Role::user, vec![Part::text(prompt)]))
    }
    pub(crate) fn update(&mut self, response: &GeminiResponse) {
        let history = &mut self.history;
        let reply_parts = response.get_parts();

        if let Some(chat) = history.back_mut() {
            if let Role::model = chat.role() {
                concatinate_parts(chat.parts_mut(), reply_parts);
            } else {
                history.push_back(Chat::new(Role::model, reply_parts.clone()));
            }
        } else {
            panic!("Cannot update an empty session");
        }
    }
    pub fn last_reply(&self) -> Option<&Vec<Part>> {
        if let Some(reply) = self.get_history_as_vecdeque().back() {
            Some(&reply.parts())
        } else {
            None
        }
    }
    pub(super) fn last_reply_mut(&mut self) -> Option<&mut Vec<Part>> {
        if let Some(reply) = self.get_history_as_vecdeque_mut().back_mut() {
            Some(reply.parts_mut())
        } else {
            None
        }
    }
    ///`seperator` used to concatinate all text parts. TL;DR use "\n" as seperator.
    pub fn last_reply_text(&self, seperator: &str) -> Option<String> {
        let parts = self.last_reply();
        if let Some(parts) = parts {
            let mut concatinated_string = String::new();
            for part in parts {
                if let Part::text(text) = part {
                    concatinated_string.push_str(text);
                    concatinated_string.push_str(seperator);
                }
            }
            Some(concatinated_string)
        } else {
            None
        }
    }
    /// If last message is a question from user then only that is removed else the model reply and
    /// the user's question (just before model reply)
    pub fn forget_last_conversation(&mut self) {
        self.history.pop_back();
        if let Some(chat) = self.history.back() {
            if let Role::user = chat.role() {
                self.history.pop_back();
            }
        }
    }
}
