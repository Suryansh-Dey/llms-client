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
    #[error("FunctionResponse must be after model prompt and not first in session")]
    ///FunctionResponse must be after model prompt and not first in session
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
    /// * `history_limit` - The maximum number of `Chat` to keep in history.
    /// * Uses sliding window to maintain length.
    /// * Might pop many elements after one new insertion to maintain valid history according to
    /// API.
    ///
    /// # Example
    /// `Session::new(2)` can store only the last question and its reply.
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
    /// Count of all the chats of any role.
    pub fn get_chat_no(&self) -> usize {
        self.chat_no
    }
    pub fn get_history(&self) -> Vec<&Chat> {
        self.history.iter().collect()
    }
    pub fn get_history_owned(self) -> VecDeque<Chat> {
        self.history
    }
    pub fn get_history_length(&self) -> usize {
        self.history.len()
    }
    ///Use `get_previous_chat_mut().parts_mut()` instead
    #[deprecated]
    pub fn get_parts_mut(&mut self, chat_previous_no: usize) -> Option<&mut Vec<Part>> {
        let chat_no = self.get_history_length().checked_sub(chat_previous_no)?;
        Some(self.history[chat_no].parts_mut())
    }
    ///Use `get_previous_chat()` instead
    #[deprecated]
    pub fn get_parts(&self, chat_previous_no: usize) -> Option<&Vec<Part>> {
        let chat_no = self.get_history_length().checked_sub(chat_previous_no)?;
        Some(self.history[chat_no].parts())
    }
    ///`chat_previous_no` is ith last message.
    ///# Example
    ///- session.get_previous_chat(1) return last message
    ///- session.get_previous_chat(2) return 2nd last message
    pub fn get_previous_chat(&self, chat_previous_no: usize) -> Option<&Chat> {
        let chat_no = self.get_history_length().checked_sub(chat_previous_no)?;
        self.history.get(chat_no)
    }
    ///`chat_previous_no` is ith last message.
    ///# Example
    ///- session.get_previous_chat_mut(1) return last message
    ///- session.get_previous_chat_mut(2) return 2nd last message
    pub fn get_previous_chat_mut(&mut self, chat_previous_no: usize) -> Option<&mut Chat> {
        let chat_no = self.get_history_length().checked_sub(chat_previous_no)?;
        self.history.get_mut(chat_no)
    }
    /// Confusing to use. Use get_history_as_vecdeque_mut() instead
    #[deprecated]
    pub fn get_parts_no_mut(&mut self, chat_no: usize) -> Option<&mut Vec<Part>> {
        self.get_history_as_vecdeque_mut()
            .get_mut(chat_no)
            .map(|chat| chat.parts_mut())
    }
    /// Confusing to use. Use get_history_as_vecdeque() instead
    #[deprecated]
    pub fn get_parts_no(&self, chat_no: usize) -> Option<&Vec<Part>> {
        self.get_history_as_vecdeque()
            .get(chat_no)
            .map(|chat| chat.parts())
    }
    pub fn get_remember_reply(&self) -> bool {
        self.remember_reply
    }
    pub fn add_chat(&mut self, chat: Chat) -> Result<&mut Self, &'static str> {
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
                    Role::Model if front_chat.has_function_call() => self.history.pop_front(),
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
    /// Appends a Model prompt to the session history.
    ///
    /// If called multiple times without an intervening model response, the prompts are concatenated.
    pub fn reply_parts(&mut self, parts: Vec<Part>) -> &mut Self {
        self.add_chat(Chat::new(Role::Model, parts)).unwrap()
    }
    /// Appends a Model prompt to the session history.
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
    /// Returns `AddFunctionResponseError::FunctionResponseAfterUser` when called after user prompt.
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
    /// If last Chat is of `Role::User` or `Role::Function` then only that is removed else the model reply and
    /// the user prompt (just before model reply) are removed.
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
    pub fn remove_last_chat(&mut self) -> Option<Chat> {
        self.history.pop_back()
    }
}
