//! Conversation history sent with each request, kept in check by an [`OverflowHandler`].
use super::message::{AssistantMessage, Message, ToolCall, ToolMessage, UserMessage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;

#[derive(thiserror::Error, Debug)]
pub enum SessionError {
    /// The last assistant message has no unanswered tool call with this id.
    #[error("No unanswered tool call with id `{0}` in the last assistant message")]
    NoPendingCall(String),
    /// The tool result couldn't be serialized to JSON.
    #[error("Tool result can't be serialized: {0}")]
    InvalidResult(serde_json::Error),
}

/// Strategy to keep the history in check, e.g. a `SlidingWindow` of messages, or a token budget.
///
/// To combine strategies, e.g. a token budget with a cap on messages, hold a `SlidingWindow` in
/// your handler and call its `handle()` from yours.
pub trait OverflowHandler {
    /// Called every time a message is added to the session. Forget messages from the beginning
    /// using `history.forget_oldest()`, which keeps the history valid.
    fn handle(&mut self, history: &mut HistoryTrimmer<'_>);
}
/// Allows choosing the handler at runtime with `Session<Box<dyn OverflowHandler>>`.
impl<H: OverflowHandler + ?Sized> OverflowHandler for Box<H> {
    fn handle(&mut self, history: &mut HistoryTrimmer<'_>) {
        (**self).handle(history)
    }
}

/// Default `OverflowHandler`. Keeps the last `limit` messages of any role. More are forgotten at
/// once if needed to keep the history valid, i.e. tool calls along with their results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlidingWindow {
    pub limit: usize,
}
impl SlidingWindow {
    pub fn new(limit: usize) -> Self {
        Self { limit }
    }
}
/// A window of 2 messages, i.e. the last prompt and its reply.
impl Default for SlidingWindow {
    fn default() -> Self {
        Self::new(2)
    }
}
impl OverflowHandler for SlidingWindow {
    fn handle(&mut self, history: &mut HistoryTrimmer<'_>) {
        while history.len() > self.limit && history.forget_oldest().is_some() {}
    }
}

/// History can't start with tool calls or their results, as the prompt they belong to is gone.
fn is_invalid_start(message: &Message) -> bool {
    match message {
        Message::Tool(_) => true,
        Message::Assistant(reply) => !reply.tool_calls.is_empty(),
        Message::User(_) => false,
    }
}
fn forget_oldest(history: &mut VecDeque<Message>) -> Option<Message> {
    let count = 1 + history
        .iter()
        .skip(1)
        .take_while(|message| is_invalid_start(message))
        .count();
    if count >= history.len() {
        return None;
    }
    let oldest = history.pop_front();
    history.drain(..count - 1);
    oldest
}

/// History as seen by `OverflowHandler::handle()`. It can be read but only shrunk from the
/// beginning, so a handler can't leave the session in a state the API rejects.
pub struct HistoryTrimmer<'a> {
    history: &'a mut VecDeque<Message>,
    message_number: usize,
}
impl HistoryTrimmer<'_> {
    pub fn history(&self) -> &VecDeque<Message> {
        self.history
    }
    pub fn len(&self) -> usize {
        self.history.len()
    }
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
    /// Number of the last message, see `Session::message_number()`.
    pub fn message_number(&self) -> usize {
        self.message_number
    }
    /// Removes the oldest message, and the tool calls and results right after it, as history
    /// can't start with them.
    ///
    /// Returns `None` without removing anything if the latest message would be removed too.
    pub fn forget_oldest(&mut self) -> Option<Message> {
        forget_oldest(self.history)
    }
}

/// Conversation history, sent with each request. Replies are added to it.
///
/// Its `OverflowHandler` decides what is forgotten, e.g. a `SlidingWindow` of messages or a token
/// budget.
///
/// # Serde
/// The overflow handler is serialized along with the history, so its state is restored too. A
/// session must be deserialized with the same handler type it was serialized with.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session<H = SlidingWindow> {
    history: VecDeque<Message>,
    message_number: usize,
    remember_reply: bool,
    overflow_handler: H,
}
/// A session keeping the last prompt and its reply, see `SlidingWindow::default()`.
impl Default for Session {
    fn default() -> Self {
        Self::new(SlidingWindow::default())
    }
}
impl<H: OverflowHandler> Session<H> {
    /// A session whose history is kept in check by `overflow_handler`.
    ///
    /// # Example
    /// `Session::new(SlidingWindow::new(10))` keeps the last 10 messages.
    pub fn new(overflow_handler: H) -> Self {
        Self {
            history: VecDeque::new(),
            message_number: 0,
            remember_reply: true,
            overflow_handler,
        }
    }
}
impl<H> Session<H> {
    pub fn history(&self) -> &VecDeque<Message> {
        &self.history
    }
    pub fn into_history(self) -> VecDeque<Message> {
        self.history
    }
    pub fn last(&self) -> Option<&Message> {
        self.history.back()
    }
    /// # Warning
    /// Edits the API rejects are possible, e.g. removing a tool call which has a result.
    pub fn last_mut(&mut self) -> Option<&mut Message> {
        self.history.back_mut()
    }
    /// `n`th last message.
    ///
    /// # Example
    /// - `session.previous(1)` returns the last message
    /// - `session.previous(2)` returns the second last message
    pub fn previous(&self, n: usize) -> Option<&Message> {
        self.history.get(self.history.len().checked_sub(n)?)
    }
    /// `n`th last message, like `previous()`.
    ///
    /// # Warning
    /// Edits the API rejects are possible, e.g. removing a tool call which has a result.
    pub fn previous_mut(&mut self, n: usize) -> Option<&mut Message> {
        let index = self.history.len().checked_sub(n)?;
        self.history.get_mut(index)
    }
    /// Message by its number, counting from 1 every message added since the session was created.
    /// E.g. `by_number(3)` is the third message. `None` if it was forgotten, or isn't added yet.
    pub fn by_number(&self, number: usize) -> Option<&Message> {
        self.previous((self.message_number + 1).checked_sub(number)?)
    }
    /// Message by its number, like `by_number()`.
    ///
    /// # Warning
    /// Edits the API rejects are possible, e.g. removing a tool call which has a result.
    pub fn by_number_mut(&mut self, number: usize) -> Option<&mut Message> {
        self.previous_mut((self.message_number + 1).checked_sub(number)?)
    }
    pub fn len(&self) -> usize {
        self.history.len()
    }
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
    /// Number of the last message, see `by_number()`. Unlike `len()`, messages forgotten from the
    /// beginning still count, while ones removed from the end (`pop()`, `forget_last_turn()`)
    /// don't.
    pub fn message_number(&self) -> usize {
        self.message_number
    }
    /// Removes the oldest message, and the tool calls and results right after it, as history
    /// can't start with them.
    ///
    /// Returns `None` without removing anything if the latest message would be removed too.
    pub fn forget_oldest(&mut self) -> Option<Message> {
        forget_oldest(&mut self.history)
    }
    /// Removes the last message.
    pub fn pop(&mut self) -> Option<Message> {
        let message = self.history.pop_back()?;
        self.message_number = self.message_number.saturating_sub(1);
        Some(message)
    }
    /// Removes the last turn: the last prompt (consecutive user messages) and everything after
    /// it, e.g. the reply, or tool calls and their results. Removes the whole history if no
    /// prompt is left in it.
    ///
    /// Returns the removed messages, oldest first.
    pub fn forget_last_turn(&mut self) -> Vec<Message> {
        let is_prompt = |message: &Message| matches!(message, Message::User(_));
        let start = match self.history.iter().rposition(is_prompt) {
            Some(last_prompt) => self
                .history
                .range(..last_prompt)
                .rposition(|message| !is_prompt(message))
                .map_or(0, |reply| reply + 1),
            None => 0,
        };
        let removed = self.history.split_off(start);
        self.message_number = self.message_number.saturating_sub(removed.len());
        removed.into()
    }
    pub fn remember_reply(&self) -> bool {
        self.remember_reply
    }
    /// If false, replies are not added, and the prompt is removed once its reply arrives. For
    /// one-off requests, e.g. structured output. Default is true.
    pub fn set_remember_reply(&mut self, remember_reply: bool) {
        self.remember_reply = remember_reply;
    }
    /// Builder form of `set_remember_reply()`, e.g.
    /// `Session::default().with_remember_reply(false)`.
    pub fn with_remember_reply(mut self, remember_reply: bool) -> Self {
        self.remember_reply = remember_reply;
        self
    }
    pub fn overflow_handler(&self) -> &H {
        &self.overflow_handler
    }
    pub fn overflow_handler_mut(&mut self) -> &mut H {
        &mut self.overflow_handler
    }
    /// Replaces the `OverflowHandler`. It first runs when the next message is added, see
    /// `OverflowHandler::handle()`.
    pub fn with_overflow_handler<T: OverflowHandler>(self, overflow_handler: T) -> Session<T> {
        Session {
            history: self.history,
            message_number: self.message_number,
            remember_reply: self.remember_reply,
            overflow_handler,
        }
    }
    /// Tool calls of the last assistant message which have no result yet.
    pub fn pending_tool_calls(&self) -> Vec<&ToolCall> {
        let mut answered = Vec::new();
        for message in self.history.iter().rev() {
            match message {
                Message::Tool(result) => answered.push(result.tool_call_id.as_str()),
                Message::Assistant(reply) => {
                    return reply
                        .tool_calls
                        .iter()
                        .filter(|call| !answered.contains(&call.id.as_str()))
                        .collect();
                }
                Message::User(_) => break,
            }
        }
        Vec::new()
    }
}
impl<H: OverflowHandler> Session<H> {
    /// Adds a user message, e.g. `session.ask("Hi")`, or one with images:
    /// ```
    /// # use open_client::chat::{message::{ContentPart, ImageUrl}, session::Session};
    /// # let mut session = Session::default();
    /// session.ask(vec![
    ///     ContentPart::from("What is in this image?"),
    ///     ImageUrl::new("https://example.com/photo.jpg").into(),
    /// ]);
    /// ```
    pub fn ask(&mut self, message: impl Into<UserMessage>) -> &mut Self {
        self.add(Message::User(message.into()));
        self
    }
    /// Adds an assistant message, e.g. for examples of replies.
    pub fn reply(&mut self, message: impl Into<AssistantMessage>) -> &mut Self {
        self.add(Message::Assistant(message.into()));
        self
    }
    /// Adds any message.
    ///
    /// # Errors
    /// `SessionError::NoPendingCall` for a tool message that doesn't answer a pending tool call.
    pub fn push(&mut self, message: impl Into<Message>) -> Result<&mut Self, SessionError> {
        let message = message.into();
        if let Message::Tool(result) = &message
            && !self
                .pending_tool_calls()
                .iter()
                .any(|call| call.id == result.tool_call_id)
        {
            return Err(SessionError::NoPendingCall(result.tool_call_id.clone()));
        }
        self.add(message);
        Ok(self)
    }
    /// Adds the result of the pending tool call with id `call_id`.
    ///
    /// `result` can be anything `Serialize`: strings are sent as they are, everything else as
    /// JSON.
    ///
    /// # Errors
    /// - `SessionError::NoPendingCall` if no unanswered tool call has this id.
    /// - `SessionError::InvalidResult` if `result` can't be serialized.
    pub fn add_tool_result(
        &mut self,
        call_id: impl Into<String>,
        result: impl Serialize,
    ) -> Result<&mut Self, SessionError> {
        let content = match serde_json::to_value(result).map_err(SessionError::InvalidResult)? {
            Value::String(content) => content,
            value => value.to_string(),
        };
        self.push(ToolMessage::new(call_id, content))
    }
    /// Adds a reply of the API, as `remember_reply` says. If not remembered, its prompt is removed
    /// too, i.e. the user messages at the end.
    #[cfg(feature = "reqwest")]
    pub(crate) fn add_reply(&mut self, reply: AssistantMessage) {
        if self.remember_reply {
            self.reply(reply);
        } else if let Some(Message::User(_)) = self.history.back() {
            self.forget_last_turn();
        }
    }
    fn add(&mut self, message: Message) {
        self.history.push_back(message);
        self.message_number += 1;
        self.overflow_handler.handle(&mut HistoryTrimmer {
            history: &mut self.history,
            message_number: self.message_number,
        });
    }
}
