use super::request::*;
use super::response::GeminiResponse;
use derive_getters::{Dissolve, Getters};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::VecDeque;

#[derive(thiserror::Error, Debug)]
pub enum AddFunctionResponseError {
    #[error("Error while parsing: {0}")]
    ///Error while parsing
    InvalidResponseFormat(serde_json::Error),
    #[error("FunctionResponse must be after model prompt and not first in session")]
    ///FunctionResponse must be after model prompt and not first in session
    FunctionResponseAfterUser,
}

/// Strategy to keep `Session` history in check, e.g. by estimated token count.
/// Set it using `Session::set_overflow_handler()`.
pub trait OverflowHandler {
    /// Called every time a chat is added to the session, including when it is concatenated into
    /// the last chat. Forget chats from the beginning using `history.forget_oldest_chat()`.
    ///
    /// Returns true if default handler (sliding window trimming by `history_limit`) should also be run.
    fn handle(&mut self, history: &mut HistoryTrimmer<'_>) -> bool;
}
/// Allows choosing the handler at runtime with `Session<Box<dyn OverflowHandler>>`.
impl<H: OverflowHandler + ?Sized> OverflowHandler for Box<H> {
    fn handle(&mut self, history: &mut HistoryTrimmer<'_>) -> bool {
        (**self).handle(history)
    }
}

/// Default `OverflowHandler`. Only keeps the last `history_limit` chats.
///
/// Serialized as `null`. Goes through `Option<()>` so a missing field also deserializes to it,
/// letting sessions saved before overflow handlers existed load as `Session`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "Option<()>")]
pub struct SlidingWindow;
impl From<Option<()>> for SlidingWindow {
    fn from(_: Option<()>) -> Self {
        SlidingWindow
    }
}
impl OverflowHandler for SlidingWindow {
    fn handle(&mut self, _: &mut HistoryTrimmer<'_>) -> bool {
        true
    }
}

/// History can't start with a function call or response, as the prompt they belong to is gone.
fn is_invalid_start(chat: &Chat) -> bool {
    match chat.role() {
        Role::Function => true,
        Role::Model => chat.has_function_call(),
        Role::User => false,
    }
}

fn forget_oldest_chat(history: &mut VecDeque<Chat>) -> Option<Chat> {
    let count = 1 + history
        .iter()
        .skip(1)
        .take_while(|chat| is_invalid_start(chat))
        .count();
    if count >= history.len() {
        return None;
    }
    let oldest_chat = history.pop_front();
    history.drain(..count - 1);
    oldest_chat
}

/// Session history as seen by `OverflowHandler::handle()`. It can be read but only shrunk from the
/// beginning, so a handler can't leave the session in a state the API rejects.
pub struct HistoryTrimmer<'a> {
    history: &'a mut VecDeque<Chat>,
    history_limit: usize,
    chat_no: usize,
}
impl HistoryTrimmer<'_> {
    pub fn get_history_as_vecdeque(&self) -> &VecDeque<Chat> {
        self.history
    }
    pub fn get_history_length(&self) -> usize {
        self.history.len()
    }
    pub fn get_history_limit(&self) -> usize {
        self.history_limit
    }
    /// Count of all the chats of any role.
    pub fn get_chat_no(&self) -> usize {
        self.chat_no
    }
    /// Remove the oldest chat in the window.
    /// # WARNING
    /// Might delete more than 1 chat to keep session in valid state.
    ///
    /// Returns `None` without removing anything if the latest chat would be removed too.
    pub fn forget_oldest_chat(&mut self) -> Option<Chat> {
        forget_oldest_chat(self.history)
    }
}

/// Manages the conversation history and configuration for a Gemini session.
///
/// A `Session` tracks the sequence of `Chat` messages (user prompts and model replies)
/// and enforces a history limit to manage token usage.
/// Use `set_overflow_handler()` for custom strategies, e.g. based on token count.
///
/// # Serde
/// The overflow handler is serialized along with the history, so its state is restored too.
/// The default `SlidingWindow` is serialized as `null`. A session must be deserialized with the
/// same handler type it was serialized with.
#[derive(Debug, Clone, Getters, Dissolve, Serialize, Deserialize)]
pub struct Session<H = SlidingWindow> {
    history: VecDeque<Chat>,
    history_limit: usize,
    chat_no: usize,
    remember_reply: bool,
    overflow_handler: H,
}
impl Default for Session {
    fn default() -> Self {
        Session::new(0)
    }
}
impl Session {
    /// Creates a new `Session` with a specified history limit.
    ///
    /// # Arguments
    /// * `history_limit` - The maximum number of `Chat`(including all `Role`) to keep in history.
    /// * Uses sliding window to maintain length.
    /// * Might pop many elements after one new insertion OR Might keep more elements than `history_limit`, to maintain valid history according to
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
            overflow_handler: SlidingWindow,
        }
    }
}
impl<H: OverflowHandler> Session<H> {
    ///Set to false to stop automatic context storing
    pub fn set_remember_reply(mut self, remember: bool) -> Self {
        self.remember_reply = remember;
        self
    }
    /// Takes out the overflow handler, leaving the session with the default `SlidingWindow`.
    pub fn remove_overflow_handler(self) -> (Session, H) {
        let Self {
            history,
            history_limit,
            chat_no,
            remember_reply,
            overflow_handler,
        } = self;
        let session = Session {
            history,
            history_limit,
            chat_no,
            remember_reply,
            overflow_handler: SlidingWindow,
        };
        (session, overflow_handler)
    }
    /// Sets a custom `OverflowHandler` to control how history is trimmed, replacing the current one.
    /// See `OverflowHandler::handle()` for when it is called.
    /// Use `remove_overflow_handler()` first to keep the current handler.
    pub fn set_overflow_handler<T: OverflowHandler>(self, overflow_handler: T) -> Session<T> {
        let Self {
            history,
            history_limit,
            chat_no,
            remember_reply,
            overflow_handler: _,
        } = self;
        Session {
            history,
            history_limit,
            chat_no,
            remember_reply,
            overflow_handler,
        }
    }
    pub fn overflow_handler_mut(&mut self) -> &mut H {
        &mut self.overflow_handler
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
    /// Remove the oldest chat in the window.
    /// # WARNING
    /// Might delete more than 1 chat to keep session in valid state.
    ///
    /// Returns `None` without removing anything if the latest chat would be removed too.
    pub fn forget_oldest_chat(&mut self) -> Option<Chat> {
        forget_oldest_chat(&mut self.history)
    }
    pub fn add_chat(&mut self, chat: Chat) -> Result<&mut Self, &'static str> {
        match (self.history.back_mut(), chat.role()) {
            (Some(last_chat), role) if last_chat.role() == role => {
                concatenate_parts(last_chat.parts_mut(), chat.parts())
            }
            (Some(last_chat), Role::Function) if *last_chat.role() == Role::User => {
                return Err("Role::Function not allowed after Role::User");
            }
            (None, Role::Function) => return Err("Role::Function cannot be first"),
            _ => {
                self.history.push_back(chat);
                self.chat_no += 1;
            }
        }
        self.handle_overflow();
        Ok(self)
    }
    fn handle_overflow(&mut self) {
        let mut history = HistoryTrimmer {
            history: &mut self.history,
            history_limit: self.history_limit,
            chat_no: self.chat_no,
        };
        let run_default = self.overflow_handler.handle(&mut history);
        while run_default
            && self.get_history_length() > self.get_history_limit()
            && self.forget_oldest_chat().is_some()
        {}
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
    /// Returns Chat by absolute chat number. Eg. `get_chat_by_no(3)` will try to return 3rd (including all `Role`) `Some(&Chat)` asked since session created. If it went out of history window returns `None`.
    pub fn get_chat_by_no(&self, absolute_chat_no: usize) -> Option<&Chat> {
        let chat_previous_no = usize::checked_sub(self.get_chat_no() + 1, absolute_chat_no)?;
        self.get_previous_chat(chat_previous_no)
    }
    /// Returns Chat by absolute chat number. Eg. `get_chat_by_no_mut(3)` will try to return 3rd (including all `Role`) `Some(&mut Chat)` asked since session created. If it went out of history window returns `None`.
    pub fn get_chat_by_no_mut(&mut self, absolute_chat_no: usize) -> Option<&mut Chat> {
        let chat_previous_no = usize::checked_sub(self.get_chat_no() + 1, absolute_chat_no)?;
        self.get_previous_chat_mut(chat_previous_no)
    }
}
