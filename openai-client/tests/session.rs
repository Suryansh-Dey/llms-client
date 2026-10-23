//! `Session` history: messages, tool calls and results, overflow handling and serde.
use open_client::chat::message::{AssistantMessage, Message, ToolCall, ToolMessage};
use open_client::chat::session::{
    HistoryTrimmer, OverflowHandler, Session, SessionError, SlidingWindow,
};
use open_client::chat::{ChatCompletion, Choice, Usage};
use serde::{Deserialize, Serialize};
use serde_json::json;

fn roles<H>(session: &Session<H>) -> Vec<&'static str> {
    session
        .history()
        .iter()
        .map(|message| match message {
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::Tool(_) => "tool",
        })
        .collect()
}
fn pending_ids(session: &Session) -> Vec<&str> {
    session
        .pending_tool_calls()
        .into_iter()
        .map(|call| call.id.as_str())
        .collect()
}
/// Response with `reply`, using `total_tokens`.
fn response(reply: impl Into<AssistantMessage>, total_tokens: u64) -> ChatCompletion {
    ChatCompletion {
        id: None,
        model: None,
        choices: vec![Choice {
            index: 0,
            message: reply.into(),
            finish_reason: None,
            logprobs: None,
        }],
        usage: Some(Usage {
            total_tokens,
            ..Default::default()
        }),
        extra: Default::default(),
    }
}
/// Asks for the weather of two cities, and replies with a call for each.
fn weather_calls() -> Session {
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("Weather in Paris and London?");
    session.reply(vec![
        ToolCall::new("call_paris", "get_weather", json!({"city": "Paris"})),
        ToolCall::new("call_london", "get_weather", json!({"city": "London"})),
    ]);
    session
}

#[test]
fn every_message_is_kept() {
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("Hello").ask("world");
    session.reply("Hi");
    assert_eq!(roles(&session), ["user", "user", "assistant"]);
    assert_eq!(session.message_number(), 3);
    assert_eq!(session.last().unwrap().text(), "Hi");
    assert_eq!(session.pop().unwrap().text(), "Hi");
    assert_eq!(session.len(), 2);
}

#[test]
fn tool_results_are_matched_by_id() {
    let mut session = Session::new(SlidingWindow::new(10));
    assert!(matches!(
        session.add_tool_result("call_paris", "Sunny"),
        Err(SessionError::NoPendingCall(id)) if id == "call_paris"
    ));
    session.ask("Hi");
    assert!(
        session
            .push(ToolMessage::new("call_paris", "Sunny"))
            .is_err()
    );
    assert_eq!(roles(&session), ["user"]);

    let mut session = weather_calls();
    // Two calls of the same function are told apart by id
    assert_eq!(pending_ids(&session), ["call_paris", "call_london"]);
    assert!(matches!(
        session.add_tool_result("call_rome", "Sunny"),
        Err(SessionError::NoPendingCall(id)) if id == "call_rome"
    ));

    session.add_tool_result("call_london", "Rainy").unwrap();
    assert_eq!(pending_ids(&session), ["call_paris"]);
    assert!(matches!(
        session.add_tool_result("call_london", "Rainy again"),
        Err(SessionError::NoPendingCall(_))
    ));
    // Anything `Serialize` is sent as JSON text, strings as they are
    session
        .add_tool_result("call_paris", json!({"sky": "clear", "celsius": 21}))
        .unwrap();
    assert!(pending_ids(&session).is_empty());

    // One message per result, in the order they were added
    assert_eq!(roles(&session), ["user", "assistant", "tool", "tool"]);
    let results: Vec<_> = session.history().iter().skip(2).collect();
    assert_eq!(
        results,
        [
            &Message::Tool(ToolMessage::new("call_london", "Rainy")),
            &Message::Tool(ToolMessage::new(
                "call_paris",
                r#"{"celsius":21,"sky":"clear"}"#
            )),
        ]
    );
    session.reply("Sunny in Paris, rainy in London");
    assert!(pending_ids(&session).is_empty());
}

#[tokio::test]
async fn sliding_window_forgets_calls_with_their_results() {
    let mut session = weather_calls();
    session.add_tool_result("call_paris", "Sunny").unwrap();
    session.add_tool_result("call_london", "Rainy").unwrap();
    session.reply("Sunny in Paris, rainy in London");
    session.ask("And tomorrow?");
    assert_eq!(session.len(), 6);

    // Forgetting the first prompt would start the history with tool calls, which the API
    // rejects, so the calls and their results are forgotten with it.
    assert_eq!(
        session.forget_oldest().unwrap().text(),
        "Weather in Paris and London?"
    );
    assert_eq!(roles(&session), ["assistant", "user"]);

    // With a limit, the same happens after each reply
    let mut session = Session::new(SlidingWindow::new(2));
    session.ask("Weather in Paris?");
    let call = ToolCall::new("call_paris", "get_weather", json!({"city": "Paris"}));
    session.add_completion(&response(call, 0)).await;
    session.add_tool_result("call_paris", "Sunny").unwrap();
    assert_eq!(session.len(), 3, "only replies are followed by trimming");
    session.add_completion(&response("Sunny", 0)).await;
    assert_eq!(roles(&session), ["assistant"]);
    session.ask("Thanks");
    assert_eq!(roles(&session), ["assistant", "user"]);
    assert_eq!(session.message_number(), 5);
}

#[tokio::test]
async fn latest_message_is_never_forgotten() {
    let mut session = Session::new(SlidingWindow::new(0));
    session.ask("Hi");
    session.add_completion(&response("Hello", 0)).await;
    assert_eq!(roles(&session), ["assistant"]);
    assert!(session.forget_oldest().is_none());
}

/// Keeps at most `max_chars` characters of text, counting what it forgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CharBudget {
    max_chars: usize,
    forgotten: usize,
    /// Also caps the count of messages, if set.
    window: Option<SlidingWindow>,
}
impl OverflowHandler for CharBudget {
    async fn handle(&mut self, history: &mut HistoryTrimmer<'_>, response: &ChatCompletion) {
        let chars = |history: &HistoryTrimmer| {
            history
                .history()
                .iter()
                .map(|message| message.text().len())
                .sum::<usize>()
        };
        while chars(history) > self.max_chars && history.forget_oldest().is_some() {
            self.forgotten += 1;
        }
        if let Some(window) = &mut self.window {
            window.handle(history, response).await;
        }
    }
}

#[tokio::test]
async fn custom_overflow_handler() {
    let budget = CharBudget {
        max_chars: 10,
        forgotten: 0,
        window: None,
    };
    // Only the handler decides what is forgotten
    let mut session = Session::new(budget.clone());
    session.ask("12345");
    session.add_completion(&response("12345", 0)).await;
    assert_eq!(session.len(), 2);
    session.ask("1");
    session.add_completion(&response("1", 0)).await;
    assert_eq!(roles(&session), ["assistant", "user", "assistant"]);
    assert_eq!(session.overflow_handler().forgotten, 1);

    // Strategies combine by calling each other
    let mut session = Session::new(CharBudget {
        window: Some(SlidingWindow::new(1)),
        ..budget
    });
    session.ask("1");
    session.add_completion(&response("2", 0)).await;
    assert_eq!(session.len(), 1);

    // Handler state is saved along with the session
    let mut session = Session::new(CharBudget {
        max_chars: 1,
        forgotten: 7,
        window: None,
    });
    session.ask("Hi");
    let saved = serde_json::to_string(&session).unwrap();
    let restored: Session<CharBudget> = serde_json::from_str(&saved).unwrap();
    assert_eq!(restored.overflow_handler().forgotten, 7);
    assert_eq!(restored.len(), 1);
}

/// Records the history length and tokens it is called with.
#[derive(Default)]
struct Recorder {
    calls: Vec<(usize, u64)>,
}
impl OverflowHandler for Recorder {
    async fn handle(&mut self, history: &mut HistoryTrimmer<'_>, response: &ChatCompletion) {
        let tokens = response
            .usage
            .as_ref()
            .map_or(0, |usage| usage.total_tokens);
        self.calls.push((history.len(), tokens));
    }
}

#[tokio::test]
async fn handler_runs_after_each_reply() {
    let mut session = Session::new(Recorder::default());
    // Not when messages are added
    session.ask("Hi").reply("Hello").ask("Bye");
    assert!(session.overflow_handler().calls.is_empty());
    // After the reply is added, so it sees the reply
    session.add_completion(&response("Bye", 42)).await;
    assert_eq!(session.overflow_handler().calls, [(4, 42)]);
    assert_eq!(session.last().unwrap().text(), "Bye");
    // Also if the reply isn't remembered, after its prompt is removed
    session.set_remember_reply(false);
    session.ask("One-off");
    session.add_completion(&response("Done", 7)).await;
    assert_eq!(session.overflow_handler().calls, [(4, 42), (4, 7)]);
}

/// Above 4 messages, puts a summary of all but the last turn into its prompt.
struct Summarizer;
impl OverflowHandler for Summarizer {
    async fn handle(&mut self, history: &mut HistoryTrimmer<'_>, _: &ChatCompletion) {
        if history.len() <= 4 {
            return;
        }
        // A request for the summary would be awaited here, before changing the history.
        tokio::task::yield_now().await;
        let mut forgotten = Vec::new();
        while history.len() > 2
            && let Some(message) = history.forget_oldest()
        {
            forgotten.push(message.text());
        }
        if let Some(Message::User(prompt)) = history.get_mut(0) {
            let summary = forgotten.join(", ");
            prompt.content = format!("Summary: {summary}\n\n{}", prompt.text()).into();
        }
    }
}

#[tokio::test]
async fn handler_can_summarize_into_first_prompt() {
    let mut session = Session::new(Summarizer);
    session
        .ask("Q1")
        .reply("A1")
        .ask("Q2")
        .reply("A2")
        .ask("Q3");
    session.add_completion(&response("A3", 0)).await;
    assert_eq!(roles(&session), ["user", "assistant"]);
    let summary = "Summary: Q1, A1, Q2, A2\n\nQ3";
    assert_eq!(session.history()[0].text(), summary);
    // Messages keep their numbers
    assert_eq!(session.message_number(), 6);
    assert_eq!(session.by_number(5).unwrap().text(), summary);
    assert!(session.by_number(4).is_none());
}

#[tokio::test]
async fn replacing_overflow_handler() {
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("1").reply("2").ask("3");
    // The history is kept, and the new handler runs after the next reply
    let mut session = session.with_overflow_handler(SlidingWindow::new(2));
    assert_eq!(session.len(), 3);
    session.add_completion(&response("4", 0)).await;
    assert_eq!(roles(&session), ["user", "assistant"]);
    assert_eq!(session.message_number(), 4);
}

#[test]
fn serde_round_trip() {
    let mut session = weather_calls();
    session.add_tool_result("call_paris", "Sunny").unwrap();
    let saved = serde_json::to_value(&session).unwrap();
    assert_eq!(saved["overflow_handler"], json!({"limit": 10}));
    // Messages are saved in the shape of the API
    assert_eq!(
        saved["history"][0],
        json!({"role": "user", "content": "Weather in Paris and London?"})
    );

    let restored: Session = serde_json::from_value(saved).unwrap();
    assert_eq!(roles(&restored), ["user", "assistant", "tool"]);
    assert_eq!(pending_ids(&restored), ["call_london"]);
    assert_eq!(restored.message_number(), 3);
    assert_eq!(restored.history(), session.history());
}

#[test]
fn assistant_message_helpers() {
    let reply = AssistantMessage::from("```json\n{\"a\": 1}\n```");
    assert_eq!(reply.json::<serde_json::Value>().unwrap(), json!({"a": 1}));
    assert_eq!(AssistantMessage::default().text(), "");
}

#[test]
fn messages_by_position_and_number() {
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("first question");
    session.reply("first answer");
    session.ask("second question");
    session.forget_oldest();
    assert_eq!(session.previous(1).unwrap().text(), "second question");
    assert_eq!(session.previous(2).unwrap().text(), "first answer");
    assert!(session.previous(3).is_none(), "forgotten");
    assert!(session.previous(0).is_none());

    assert!(session.by_number(1).is_none(), "forgotten");
    assert_eq!(session.by_number(2).unwrap().text(), "first answer");
    assert_eq!(session.by_number(3).unwrap().text(), "second question");
    assert!(session.by_number(4).is_none());
    assert!(session.by_number(0).is_none());

    // Numbers stay right after removing from the end
    session.pop();
    session.ask("third question");
    assert_eq!(session.message_number(), 3);
    assert_eq!(session.by_number(3).unwrap().text(), "third question");
}

#[test]
fn messages_can_be_edited() {
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("Hi").reply("Hello");
    if let Some(Message::Assistant(reply)) = session.last_mut() {
        reply.content = Some("Hey".into());
    }
    if let Some(Message::User(prompt)) = session.previous_mut(2) {
        prompt.content = "Hi there".into();
    }
    assert_eq!(session.by_number(1).unwrap().text(), "Hi there");
    if let Some(Message::Assistant(reply)) = session.by_number_mut(2) {
        reply.content.as_mut().unwrap().push('!');
    }
    assert_eq!(session.last().unwrap().text(), "Hey!");
}

#[test]
fn forget_last_turn() {
    // A prompt of many messages, and its tool calling loop
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("Hi").reply("Hello");
    session.ask("Weather in Paris?").ask("And London?");
    session.reply(vec![
        ToolCall::new("call_paris", "get_weather", json!({"city": "Paris"})),
        ToolCall::new("call_london", "get_weather", json!({"city": "London"})),
    ]);
    session.add_tool_result("call_paris", "Sunny").unwrap();
    session.add_tool_result("call_london", "Rainy").unwrap();
    session.reply("Sunny in Paris, rainy in London");

    let removed = session.forget_last_turn();
    assert_eq!(removed.len(), 6);
    assert_eq!(removed[0].text(), "Weather in Paris?");
    assert_eq!(roles(&session), ["user", "assistant"]);
    assert_eq!(session.message_number(), 2);

    // A prompt without reply yet
    session.ask("Bye");
    assert_eq!(session.forget_last_turn().len(), 1);
    assert_eq!(roles(&session), ["user", "assistant"]);

    // With no prompt left, the whole history is the last turn
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("Hi").reply("Hello");
    session.forget_oldest();
    assert_eq!(roles(&session), ["assistant"]);
    assert_eq!(session.forget_last_turn().len(), 1);
    assert!(session.is_empty());
    assert!(session.forget_last_turn().is_empty());
}

#[test]
fn defaults() {
    let mut session = Session::default();
    assert_eq!(*session.overflow_handler(), SlidingWindow::new(usize::MAX));
    assert!(session.remember_reply());
    session.ask("Hi").reply("Hello").ask("Bye");
    assert_eq!(
        roles(&session),
        ["user", "assistant", "user"],
        "all are kept"
    );

    let session = Session::default().with_remember_reply(false);
    assert!(!session.remember_reply());
}
