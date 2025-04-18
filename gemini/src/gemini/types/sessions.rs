use super::request::*;
use super::response::GeminiResponse;
use serde::Serialize;
use std::collections::VecDeque;
use std::mem::discriminant;

#[derive(Debug, Clone, Default, Serialize)]
pub struct Session {
    history: VecDeque<Chat>,
    history_limit: usize,
    chat_no: usize,
}
impl Session {
    /// `history_limit`: Total number of chat of user and model allowed.  
    /// ## Example
    /// new(2) will allow only 1 question and 1 reply to be stored.
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
    /// Count of all the chats of user and model. Divide by 2 to get No. of question reply pairs.
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
        if let Some(last_chat) = self.get_history_as_vecdeque_mut().back_mut() {
            if discriminant(last_chat.role()) == discriminant(chat.role()) {
                concatenate_parts(last_chat.parts_mut(), &chat.parts());
                return self;
            }
        }

        self.history.push_back(chat);
        self.chat_no += 1;
        if self.get_history_length() > self.history_limit {
            self.history.pop_front();
        }
        self
    }
    /// If ask is called more than once without passing through `gemini.ask(&mut session)`, the
    /// parts is concatenated with the previous parts.
    pub fn ask(&mut self, parts: Vec<Part>) -> &mut Self {
        self.add_chat(Chat::new(Role::user, parts))
    }
    /// If ask_string is called more than once without passing through `gemini.ask(&mut session)`, the
    /// prompt string is concatenated with the previous prompt.
    pub fn ask_string(&mut self, prompt: String) -> &mut Self {
        self.add_chat(Chat::new(Role::user, vec![Part::text(prompt)]))
    }
    pub fn reply(&mut self, parts: Vec<Part>) -> &mut Self {
        self.add_chat(Chat::new(Role::model, parts))
    }
    pub fn reply_string(&mut self, prompt: String) -> &mut Self {
        self.add_chat(Chat::new(Role::model, vec![Part::text(prompt)]))
    }
    /// ## Panics
    /// If called on empty session
    pub(crate) fn update(&mut self, response: &GeminiResponse) {
        let history = &mut self.history;
        let reply_parts = response.get_parts();

        if let Some(chat) = history.back_mut() {
            if let Role::model = chat.role() {
                concatenate_parts(chat.parts_mut(), reply_parts);
            } else {
                self.add_chat(Chat::new(Role::model, reply_parts.clone()));
            }
        } else {
            panic!("Cannot update an empty session");
        }
    }
    pub fn get_last_message(&self) -> Option<&Vec<Part>> {
        if let Some(reply) = self.get_history_as_vecdeque().back() {
            Some(reply.parts())
        } else {
            None
        }
    }
    pub fn get_last_message_mut(&mut self) -> Option<&mut Vec<Part>> {
        if let Some(reply) = self.get_history_as_vecdeque_mut().back_mut() {
            Some(reply.parts_mut())
        } else {
            None
        }
    }
    ///`seperator` used to concatenate all text parts. TL;DR use "\n" as seperator.
    pub fn get_last_message_text(&self, seperator: &str) -> Option<String> {
        let parts = self.get_last_message();
        if let Some(parts) = parts {
            let mut concatenated_string = String::new();
            for part in parts {
                if let Part::text(text) = part {
                    concatenated_string.push_str(text);
                    concatenated_string.push_str(seperator);
                }
            }
            Some(concatenated_string)
        } else {
            None
        }
    }
    /// If last message is a question from user then only that is removed else the model reply and
    /// the user's question (just before model reply)
    pub fn forget_last_conversation(&mut self) -> &mut Self {
        self.history.pop_back();
        if let Some(chat) = self.history.back() {
            if let Role::user = chat.role() {
                self.history.pop_back();
            }
        }
        self
    }
}
