use super::request::*;
use super::response::GeminiResponse;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::VecDeque;
use std::{usize, vec};

#[derive(thiserror::Error, Debug)]
pub enum AddFunctionResponseError {
    #[error("Error while parsing: {0}")]
    ///Error while parsing
    InvalidResponseFormat(serde_json::Error),
    #[error("FunctionResponse cannot be added after User prompt")]
    ///FunctionResponse cannot be added after User prompt
    FunctionResponseAfterUser,
}

/// Manages the conversation history and configuration for a Gemini session.
///
/// A `Session` tracks the sequence of `Chat` messages (user prompts and model replies)
/// and enforces a history limit to manage token usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Session {
    history: VecDeque<Chat>,
    history_limit: usize,
    chat_no: usize,
    remember_reply: bool,
}
impl Session {
    /// Creates a new `Session` with a specified history limit.
    ///
    /// # Arguments
    /// * `history_limit` - The maximum number of individual messages (user and model) to keep in history.
    ///
    /// # Example
    /// `Session::new(2)` will store only the last question and its reply.
    pub fn new(history_limit: usize) -> Self {
        Self {
            history: VecDeque::new(),
            history_limit,
            chat_no: 0,
            remember_reply: true,
        }
    }
    ///Set to false to stop automatic context storing
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
    /// Not recommended to use. Use `session.reply()`, `session.ask()`,
    /// `session.add_function_response()` instead.
    fn add_chat(&mut self, chat: Chat) -> Result<&mut Self, &'static str> {
        if let Some(last_chat) = self.get_history_as_vecdeque_mut().back_mut() {
            if last_chat.role() == chat.role() {
                concatenate_parts(last_chat.parts_mut(), &chat.parts());
                return Ok(self);
            } else if *last_chat.role() == Role::User && *chat.role() == Role::Function {
                return Err("Role::Function not allowed after Role::User");
            }
        } else if *chat.role() == Role::Function {
            return Err("Role::Function cannot be first");
        }

        self.history.push_back(chat);
        self.chat_no += 1;
        if self.get_history_length() > self.get_history_limit() {
            self.history.pop_front();
            while let Some(front_chat) = self.history.front() {
                match front_chat.role() {
                    Role::Function => self.history.pop_front(),
                    _ => break,
                };
            }
        }
        Ok(self)
    }
    /// If `ask` is called more than once without passing through `gemini.ask(&mut session)`
    /// or `session.reply("ok")`, the parts is concatenated with the previous parts.
    pub fn ask_parts(&mut self, parts: Vec<Part>) -> &mut Self {
        self.add_chat(Chat::new(Role::User, parts)).unwrap()
    }
    /// If `ask_part` is called more than once without passing through `gemini.ask(&mut session)`
    /// or `session.reply("ok")`, the parts is concatenated with the previous parts.
    pub fn ask(&mut self, part: impl Into<Part>) -> &mut Self {
        self.ask_parts(vec![part.into()])
    }
    /// Appends a user prompt to the session history.
    ///
    /// If called multiple times without an intervening model response, the prompts are concatenated.
    pub fn reply_parts(&mut self, parts: Vec<Part>) -> &mut Self {
        self.add_chat(Chat::new(Role::Model, parts)).unwrap()
    }
    /// Appends a user prompt to the session history.
    ///
    /// If called multiple times without an intervening model response, the prompts are concatenated.
    pub fn reply(&mut self, part: impl Into<Part>) -> &mut Self {
        self.reply_parts(vec![part.into()])
    }
    /// Adds a function response to the session history.
    ///
    /// This is typically used after the model has requested a function call.
    /// The response will be formatted as a `Role::Function` chat.
    ///
    /// # Errors
    /// Returns `AddFunctionResponseError::FunctionResponseAfterUser` if trying to add a response when a user prompt is expected.
    pub fn add_function_response<T: Serialize>(
        &mut self,
        name: impl Into<String>,
        response: T,
    ) -> Result<&mut Self, AddFunctionResponseError> {
        let res_value = serde_json::to_value(response)
            .map_err(|e| AddFunctionResponseError::InvalidResponseFormat(e))?;
        let final_res = if res_value.is_object() {
            res_value
        } else {
            json!({ "result": res_value })
        };

        let part = FunctionResponse::new(name.into(), final_res).into();
        self.add_chat(Chat::new(Role::Function, vec![part]))
            .map_err(|_| AddFunctionResponseError::FunctionResponseAfterUser)?;
        Ok(self)
    }
    pub(crate) fn update<'b>(&mut self, response: &'b GeminiResponse) -> Option<&'b Vec<Part>> {
        if self.get_remember_reply() {
            let reply_parts = response.get_chat().parts();
            self.reply_parts(reply_parts.clone());
            Some(reply_parts)
        } else {
            if let Some(chat) = self.history.back() {
                if let Role::User = chat.role() {
                    self.history.pop_back();
                }
            }
            None
        }
    }
    pub fn get_last_chat(&self) -> Option<&Chat> {
        self.get_history_as_vecdeque().back()
    }
    pub fn get_last_chat_mut(&mut self) -> Option<&mut Chat> {
        self.get_history_as_vecdeque_mut().back_mut()
    }
    /// If last message is a question from user then only that is removed else the model reply and
    /// the user's question (just before model reply)
    /// # Returns
    /// Popped items as (last_chat, second_last_chat)
    pub fn forget_last_conversation(&mut self) -> (Option<Chat>, Option<Chat>) {
        let last = self.history.pop_back();
        if let Some(chat) = self.history.back() {
            if let Role::User = chat.role() {
                return (last, self.history.pop_back());
            }
        }
        (last, None)
    }
}
