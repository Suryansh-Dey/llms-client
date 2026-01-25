use super::request::*;
use super::response::GeminiResponse;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::VecDeque;
use std::{usize, vec};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Session {
    history: VecDeque<Chat>,
    history_limit: usize,
    chat_no: usize,
    remember_reply: bool,
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
            remember_reply: true,
        }
    }
    pub fn set_remember_reply(mut self, remember: bool) -> Self {
        self.remember_reply = remember;
        self
    }
    pub fn get_history_limit(&self) -> usize {
        self.history_limit
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
    ///`chat_previous_no` is ith last message.
    ///# Example
    ///- session.get_parts_mut(1) return last message
    ///- session.get_parts_mut(2) return 2nd last message
    pub fn get_parts_mut(&mut self, chat_previous_no: usize) -> Option<&mut Vec<Part>> {
        let chat_no = self.get_history_length().checked_sub(chat_previous_no)?;
        Some(self.history[chat_no].parts_mut())
    }
    ///`chat_previous_no` is ith last message.
    ///# Example
    ///- session.get_parts(1) return last message
    ///- session.get_parts(2) return 2nd last message
    pub fn get_parts(&self, chat_previous_no: usize) -> Option<&Vec<Part>> {
        let chat_no = self.get_history_length().checked_sub(chat_previous_no)?;
        Some(self.history[chat_no].parts())
    }
    /// `chat_no` follows 0-indexing
    pub fn get_parts_no_mut(&mut self, chat_no: usize) -> Option<&mut Vec<Part>> {
        self.get_history_as_vecdeque_mut()
            .get_mut(chat_no)
            .map(|chat| chat.parts_mut())
    }
    /// `chat_no` follows 0-indexing
    pub fn get_parts_no(&self, chat_no: usize) -> Option<&Vec<Part>> {
        self.get_history_as_vecdeque()
            .get(chat_no)
            .map(|chat| chat.parts())
    }
    pub fn get_remember_reply(&self) -> bool {
        self.remember_reply
    }
    fn add_chat(&mut self, chat: Chat) -> &mut Self {
        if let Some(last_chat) = self.get_history_as_vecdeque_mut().back_mut() {
            if last_chat.role() == chat.role() {
                concatenate_parts(last_chat.parts_mut(), &chat.parts());
                return self;
            }
        }

        self.history.push_back(chat);
        self.chat_no += 1;
        if self.get_history_length() > self.get_history_limit() {
            self.history.pop_front();
        }
        self
    }
    /// If ask is called more than once without passing through `gemini.ask(&mut session)`
    /// or `session.reply("ok")`, the parts is concatenated with the previous parts.
    pub fn ask(&mut self, parts: Vec<Part>) -> &mut Self {
        self.add_chat(Chat::new(Role::user, parts))
    }
    /// If ask_string is called more than once without passing through `gemini.ask(&mut session)`
    /// or `session.reply("opportunist")`, the prompt string is concatenated with the previous prompt.
    pub fn ask_string(&mut self, prompt: impl Into<TextPart>) -> &mut Self {
        self.add_chat(Chat::new(Role::user, vec![Part::text(prompt.into())]))
    }
    pub fn reply(&mut self, parts: Vec<Part>) -> &mut Self {
        self.add_chat(Chat::new(Role::model, parts))
    }
    pub fn reply_string(&mut self, prompt: impl Into<TextPart>) -> &mut Self {
        self.add_chat(Chat::new(Role::model, vec![Part::text(prompt.into())]))
    }
    pub fn add_function_response<T: Serialize>(
        &mut self,
        name: impl Into<String>,
        response: T,
    ) -> Result<&mut Self, serde_json::Error> {
        let res_value = serde_json::to_value(response)?;
        let final_res = if res_value.is_object() {
            res_value
        } else {
            json!({ "result": res_value })
        };

        let part = Part::functionResponse(FunctionResponse::new(name.into(), final_res));

        Ok(self.add_chat(Chat::new(Role::function, vec![part])))
    }
    pub(crate) fn update<'b>(&mut self, response: &'b GeminiResponse) -> Option<&'b Vec<Part>> {
        if self.get_remember_reply() {
            let reply_parts = response.get_chat().parts();
            self.add_chat(Chat::new(Role::model, reply_parts.clone()));
            Some(reply_parts)
        } else {
            if let Some(chat) = self.history.back() {
                if let Role::user = chat.role() {
                    self.history.pop_back();
                }
            }
            None
        }
    }
    ///Use get_last_chat instead
    #[deprecated]
    pub fn get_last_message(&self) -> Option<&Vec<Part>> {
        if let Some(reply) = self.get_history_as_vecdeque().back() {
            Some(reply.parts())
        } else {
            None
        }
    }
    ///Use get_last_chat_mut instead
    #[deprecated]
    pub fn get_last_message_mut(&mut self) -> Option<&mut Vec<Part>> {
        if let Some(reply) = self.get_history_as_vecdeque_mut().back_mut() {
            Some(reply.parts_mut())
        } else {
            None
        }
    }
    pub fn get_last_chat(&self) -> Option<&Chat> {
        self.get_history_as_vecdeque().back()
    }
    pub fn get_last_chat_mut(&mut self) -> Option<&mut Chat> {
        self.get_history_as_vecdeque_mut().back_mut()
    }
    /// Instead use
    /// ```ignore
    /// session.get_last_chat()
    /// .map(|chat| chat.get_text_no_think(seperator))
    /// ```
    ///`seperator` used to concatenate all text parts. TL;DR use "\n" as seperator.
    #[deprecated]
    pub fn get_last_message_text(&self, seperator: impl AsRef<str>) -> Option<String> {
        self.get_last_chat()
            .map(|chat| chat.get_text_no_think(seperator))
    }
    ///Instead use
    ///```ignore
    /// session.get_last_chat()
    /// .map(|chat| chat.get_thoughts(seperator))
    ///```
    ///`seperator` used to concatenate all text parts. TL;DR use "\n" as seperator.
    #[deprecated]
    pub fn get_last_message_thoughts(&self, seperator: impl AsRef<str>) -> Option<String> {
        self.get_last_chat()
            .map(|chat| chat.get_thoughts(seperator))
    }
    /// If last message is a question from user then only that is removed else the model reply and
    /// the user's question (just before model reply)
    /// # Returns
    /// Popped items as (last_chat, second_last_chat)
    pub fn forget_last_conversation(&mut self) -> (Option<Chat>, Option<Chat>) {
        let last = self.history.pop_back();
        if let Some(chat) = self.history.back() {
            if let Role::user = chat.role() {
                return (last, self.history.pop_back());
            }
        }
        (last, None)
    }
}
