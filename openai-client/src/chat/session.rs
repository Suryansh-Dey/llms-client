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

/// Strategy to keep the history in check, e.g. by estimated token count. Set it using
/// `Session::with_overflow_handler()`.
pub trait OverflowHandler {
    /// Called every time a message is added to the session. Forget messages from the beginning
    /// using `history.forget_oldest()`.
    ///
    /// Returns true if the default handler (sliding window of `limit` messages) should also be run.
    fn handle(&mut self, history: &mut HistoryTrimmer<'_>) -> bool;
}
/// Allows choosing the handler at runtime with `Session<Box<dyn OverflowHandler>>`.
impl<H: OverflowHandler + ?Sized> OverflowHandler for Box<H> {
    fn handle(&mut self, history: &mut HistoryTrimmer<'_>) -> bool {
        (**self).handle(history)
    }
}

/// Default `OverflowHandler`. Only keeps the last `limit` messages. Serialized as `null`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlidingWindow;
impl OverflowHandler for SlidingWindow {
    fn handle(&mut self, _: &mut HistoryTrimmer<'_>) -> bool {
        true
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
    limit: usize,
    message_count: usize,
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
    pub fn limit(&self) -> usize {
        self.limit
    }
    /// Count of messages added since the session was created, including forgotten ones.
    pub fn message_count(&self) -> usize {
        self.message_count
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
/// Only the last `limit` messages are kept, or what a custom `OverflowHandler` decides, e.g. by
/// token count.
///
/// # Serde
/// The overflow handler is serialized along with the history, so its state is restored too. The
/// default `SlidingWindow` is serialized as `null`. A session must be deserialized with the same
/// handler type it was serialized with.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session<H = SlidingWindow> {
    history: VecDeque<Message>,
    limit: usize,
    message_count: usize,
    remember_reply: bool,
    overflow_handler: H,
}
impl Session {
    /// A session keeping the last `limit` messages of any role. More are forgotten at once if
    /// needed to keep the history valid, i.e. tool calls along with their results.
    ///
    /// # Example
    /// `Session::new(2)` keeps only the last question and its reply.
    pub fn new(limit: usize) -> Self {
        Self {
            history: VecDeque::new(),
            limit,
            message_count: 0,
            remember_reply: true,
            overflow_handler: SlidingWindow,
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
    pub fn len(&self) -> usize {
        self.history.len()
    }
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
    pub fn limit(&self) -> usize {
        self.limit
    }
    /// Count of messages added since the session was created, including forgotten ones.
    pub fn message_count(&self) -> usize {
        self.message_count
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
        self.history.pop_back()
    }
    pub fn remember_reply(&self) -> bool {
        self.remember_reply
    }
    /// If false, replies are not added, and the prompt is removed once its reply arrives. For
    /// one-off requests, e.g. structured output. Default is true.
    pub fn set_remember_reply(&mut self, remember_reply: bool) {
        self.remember_reply = remember_reply;
    }
    pub fn overflow_handler(&self) -> &H {
        &self.overflow_handler
    }
    pub fn overflow_handler_mut(&mut self) -> &mut H {
        &mut self.overflow_handler
    }
    /// Sets a custom `OverflowHandler`, replacing the current one. See `OverflowHandler::handle()`
    /// for when it is called.
    pub fn with_overflow_handler<T: OverflowHandler>(self, overflow_handler: T) -> Session<T> {
        Session {
            history: self.history,
            limit: self.limit,
            message_count: self.message_count,
            remember_reply: self.remember_reply,
            overflow_handler,
        }
    }
    /// Takes out the overflow handler, leaving the session with the default `SlidingWindow`.
    pub fn take_overflow_handler(self) -> (Session, H) {
        let session = Session {
            history: self.history,
            limit: self.limit,
            message_count: self.message_count,
            remember_reply: self.remember_reply,
            overflow_handler: SlidingWindow,
        };
        (session, self.overflow_handler)
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
    /// # let mut session = Session::new(10);
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
    /// Adds a reply of the API, as `remember_reply` says.
    #[cfg(feature = "reqwest")]
    pub(crate) fn add_reply(&mut self, reply: AssistantMessage) {
        if self.remember_reply {
            self.reply(reply);
        } else if let Some(Message::User(_)) = self.history.back() {
            self.history.pop_back();
        }
    }
    fn add(&mut self, message: Message) {
        self.history.push_back(message);
        self.message_count += 1;
        let mut history = HistoryTrimmer {
            history: &mut self.history,
            limit: self.limit,
            message_count: self.message_count,
        };
        let run_default = self.overflow_handler.handle(&mut history);
        while run_default && self.history.len() > self.limit && self.forget_oldest().is_some() {}
    }
}
