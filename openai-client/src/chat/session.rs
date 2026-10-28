//! Conversation history sent with each request, kept in check by an [`OverflowHandler`].
#[cfg(feature = "reqwest")]
use super::AI;
use super::completion::ChatCompletion;
use super::message::{AssistantMessage, Message, ToolCall, ToolMessage, UserMessage};
#[cfg(feature = "reqwest")]
use super::message::{Content, ContentPart};
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

/// `Send`, except on wasm32 where futures of HTTP requests aren't. Required of
/// `OverflowHandler`s and their futures, so requests and streams can be sent between threads.
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSend: Send {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + ?Sized> MaybeSend for T {}
/// `Send`, except on wasm32 where futures of HTTP requests aren't. Required of
/// `OverflowHandler`s and their futures, so requests and streams can be sent between threads.
#[cfg(target_arch = "wasm32")]
pub trait MaybeSend {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> MaybeSend for T {}

/// Strategy to keep the history in check, e.g. the default `Compactor`, which has the model
/// summarize it, or a token budget using the `usage` of responses.
///
/// It is implemented with an `async fn`, e.g.
/// ```
/// # use open_client::chat::ChatCompletion;
/// # use open_client::chat::session::{HistoryTrimmer, OverflowHandler};
/// struct TokenBudget {
///     max_tokens: u64,
/// }
/// impl OverflowHandler for TokenBudget {
///     async fn handle(&mut self, history: &mut HistoryTrimmer<'_>, response: &ChatCompletion) {
///         let tokens = response.usage.as_ref().map_or(0, |usage| usage.total_tokens);
///         if tokens > self.max_tokens {
///             // Keeps the last prompt and its reply
///             while history.len() > 2 && history.forget_oldest().is_some() {}
///         }
///     }
/// }
/// ```
///
/// To combine strategies, hold one handler in another and call its `handle()` from yours.
pub trait OverflowHandler: MaybeSend {
    /// Called after each reply is added to the session, with the response it came in, see
    /// `Session::add_completion()`. Forget messages from the beginning using
    /// `history.forget_oldest()`, which keeps the history valid.
    ///
    /// If the request or stream is dropped while this awaits, e.g. a request for a summary, it
    /// stops there. So change the history after the last `.await`.
    fn handle(
        &mut self,
        history: &mut HistoryTrimmer<'_>,
        response: &ChatCompletion,
    ) -> impl Future<Output = ()> + MaybeSend;
}

/// Default `OverflowHandler`. Once a reply takes the conversation past `limit` tokens, the model
/// summarizes everything before the last prompt, which is then forgotten, and the summary is put
/// at the start of that prompt. So the last turn is kept whole, and the next summary takes this
/// one in.
///
/// Servers with prompt caching serve most of each request from cache: between summaries, messages
/// are only added, so each request starts with the one before it, and the summary is asked for
/// with the same system prompt, tools and history. Forgetting the oldest messages after every
/// reply would change the start of the history each time instead.
///
/// - The size is `usage.total_tokens` of the reply, so a response without `usage` is never
///   compacted. `AI` asks for it in streams too, see `AI::include_usage()`.
/// - The summary is asked for after the reply arrived, so a streamed reply is all received first,
///   and the stream ends once the summary is in.
/// - It needs the client that got the reply, so only replies of `AI::ask()` and `AI::stream()` are
///   compacted, not ones added with `Session::add_completion()`.
/// - If the request for the summary fails, nothing is forgotten, and it is asked for again after
///   the next reply.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Compactor {
    /// Tokens above which the history is summarized. Half the model's context window leaves room
    /// for the request for the summary, and for the turns after it.
    pub limit: u64,
    /// Asks for the summary. Sent as a user message after the history it summarizes, with tool
    /// calls turned off. Not saved along with the session while it is the default.
    #[serde(default = "default_prompt", skip_serializing_if = "is_default_prompt")]
    pub prompt: String,
}
const DEFAULT_PROMPT: &str = "Summarize our conversation so far, so that it can go on from your \
summary alone. Keep what I asked and why, what I told you about myself and my situation, the \
facts, figures and sources found for me, what was decided, and what is still open, leaving out \
what didn't come up. If our conversation starts with a summary, keep what still matters from it \
too. Write only the summary.";
fn default_prompt() -> String {
    DEFAULT_PROMPT.into()
}
fn is_default_prompt(prompt: &str) -> bool {
    prompt == DEFAULT_PROMPT
}
impl Compactor {
    /// Summarizes the history above `limit` tokens, with the default prompt.
    pub fn new(limit: u64) -> Self {
        Self {
            limit,
            prompt: default_prompt(),
        }
    }
}
/// Summarizes the history above 64,000 tokens, half of a 128K context window.
impl Default for Compactor {
    fn default() -> Self {
        Self::new(64_000)
    }
}
impl OverflowHandler for Compactor {
    async fn handle(&mut self, history: &mut HistoryTrimmer<'_>, response: &ChatCompletion) {
        #[cfg(feature = "reqwest")]
        self.compact(history, response).await;
        #[cfg(not(feature = "reqwest"))]
        let _ = (history, response);
    }
}
#[cfg(feature = "reqwest")]
impl Compactor {
    async fn compact(&self, history: &mut HistoryTrimmer<'_>, response: &ChatCompletion) {
        let Some(ai) = history.ai else {
            return;
        };
        let tokens = response
            .usage
            .as_ref()
            .map_or(0, |usage| usage.total_tokens);
        // Nothing is before the last turn if it starts the history.
        let start = last_turn_start(history.history());
        if tokens <= self.limit || start == 0 {
            return;
        }
        let mut earlier: VecDeque<Message> = history.history().range(..start).cloned().collect();
        earlier.push_back(Message::User(UserMessage::new(self.prompt.clone())));
        let summary = match ai.ask_without_tools(&earlier).await {
            Ok(completion) if completion.message().tool_calls.is_empty() => {
                completion.text().trim().to_owned()
            }
            _ => return,
        };
        if summary.is_empty() {
            return;
        }
        // Changed only after the last `.await`, see `OverflowHandler::handle()`.
        let kept = history.len() - start;
        while history.len() > kept && history.forget_oldest().is_some() {}
        if let Some(Message::User(prompt)) = history.get_mut(0) {
            let summary = format!(
                "<summary_of_earlier_conversation>\n{summary}\n</summary_of_earlier_conversation>"
            );
            match &mut prompt.content {
                Content::Text(text) => text.insert_str(0, &format!("{summary}\n\n")),
                Content::Parts(parts) => parts.insert(0, ContentPart::Text { text: summary }),
            }
        }
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
/// Position of the last prompt: the first of the user messages last in a row. 0 if there is none.
fn last_turn_start(history: &VecDeque<Message>) -> usize {
    let is_prompt = |message: &Message| matches!(message, Message::User(_));
    match history.iter().rposition(is_prompt) {
        Some(last_prompt) => history
            .range(..last_prompt)
            .rposition(|message| !is_prompt(message))
            .map_or(0, |reply| reply + 1),
        None => 0,
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

/// History as seen by `OverflowHandler::handle()`. It can be read and edited, but only shrunk
/// from the beginning, so messages keep their numbers.
pub struct HistoryTrimmer<'a> {
    history: &'a mut VecDeque<Message>,
    message_number: usize,
    /// Client the reply came from, to ask for a summary. `None` for `Session::add_completion()`.
    #[cfg(feature = "reqwest")]
    ai: Option<&'a AI>,
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
    /// Message at `index`, counting from the oldest, e.g. to put a summary of the forgotten
    /// messages into the prompt now first.
    ///
    /// # Warning
    /// Edits the API rejects are possible, e.g. removing a tool call which has a result.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Message> {
        self.history.get_mut(index)
    }
}

/// Conversation history, sent with each request. Replies are added to it.
///
/// After each reply, its `OverflowHandler` decides what is forgotten, e.g. the default
/// `Compactor`, which has the model summarize it.
///
/// # Serde
/// The overflow handler is serialized along with the history, so its state is restored too. A
/// session must be deserialized with the same handler type it was serialized with.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session<H = Compactor> {
    history: VecDeque<Message>,
    message_number: usize,
    remember_reply: bool,
    overflow_handler: H,
}
/// A session kept in check by `Compactor::default()`.
impl Default for Session {
    fn default() -> Self {
        Self::new(Compactor::default())
    }
}
impl<H: OverflowHandler> Session<H> {
    /// A session whose history is kept in check by `overflow_handler`.
    ///
    /// # Example
    /// `Session::new(Compactor::new(32_000))` has the model summarize the history above 32,000
    /// tokens.
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
        let removed = self.history.split_off(last_turn_start(&self.history));
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
    /// Replaces the `OverflowHandler`. It first runs after the next reply, see
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
    }
}
impl<H: OverflowHandler> Session<H> {
    /// Adds the reply of `response` as `remember_reply` says, then runs the `OverflowHandler`.
    ///
    /// `AI::ask()` and `AI::stream()` call it, so it is only needed for responses received
    /// another way. The handler gets no client then, so `Compactor` doesn't summarize.
    pub async fn add_completion(&mut self, response: &ChatCompletion) {
        self.add_reply(response.message().clone());
        let mut history = HistoryTrimmer {
            history: &mut self.history,
            message_number: self.message_number,
            #[cfg(feature = "reqwest")]
            ai: None,
        };
        self.overflow_handler.handle(&mut history, response).await;
    }
    /// `add_completion()` of a reply from `ai`, which the handler can ask for a summary.
    #[cfg(feature = "reqwest")]
    pub(crate) async fn add_completion_from(&mut self, ai: &AI, response: &ChatCompletion) {
        self.add_reply(response.message().clone());
        let mut history = HistoryTrimmer {
            history: &mut self.history,
            message_number: self.message_number,
            ai: Some(ai),
        };
        self.overflow_handler.handle(&mut history, response).await;
    }
}
