//! `Session` history: roles, function calls and responses, overflow handling and serde.
use open_client::mime;
use open_client::openai::types::request::{
    Chat, FunctionCall, InlineData, MultimodalResponse, Part, PartType, Role,
};
use open_client::openai::types::sessions::{
    AddFunctionResponseError, HistoryTrimmer, OverflowHandler, Session,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

fn roles<H: OverflowHandler>(session: &Session<H>) -> Vec<Role> {
    session
        .get_history()
        .iter()
        .map(|chat| chat.role().clone())
        .collect()
}
fn pending_ids(session: &Session) -> Vec<String> {
    session
        .get_pending_function_calls()
        .iter()
        .map(|call| call.id().clone())
        .collect()
}
/// `(id, name, text)` of every function response in the last chat.
fn responses(session: &Session) -> Vec<(String, String, String)> {
    session
        .get_last_chat()
        .unwrap()
        .parts()
        .iter()
        .filter_map(|part| match part.data() {
            PartType::FunctionResponse(r) => Some((
                r.id().clone(),
                r.name().clone(),
                Chat::extract_text_all(r.parts(), ""),
            )),
            _ => None,
        })
        .collect()
}
/// Asks for the weather of two cities, and replies with a call for each.
fn weather_calls() -> Session {
    let mut session = Session::new(10);
    session.ask("Weather in Paris and London?");
    session.reply_parts(vec![
        FunctionCall::new("call_paris", "get_weather", json!({"city": "Paris"})).into(),
        FunctionCall::new("call_london", "get_weather", json!({"city": "London"})).into(),
    ]);
    session
}

#[test]
fn chats_of_same_role_are_concatenated() {
    let mut session = Session::new(10);
    session.ask("Hello").ask(" world");
    session.reply_parts(vec![Part::new_thought("Greet "), "Hi".into()]);
    session.reply_parts(vec![Part::new_thought("back"), " there".into()]);

    assert_eq!(roles(&session), [Role::User, Role::Assistant]);
    assert_eq!(session.get_chat_no(), 2);
    let reply = session.get_last_chat().unwrap();
    assert_eq!(reply.get_text_no_think(""), "Hi there");
    assert_eq!(reply.get_thoughts(""), "Greet back");
    assert_eq!(session.get_history()[0].get_text_all(""), "Hello world");
}

#[test]
fn manually_added_calls_are_never_merged() {
    let mut session = Session::new(10);
    session.ask("Hi");
    session.reply(FunctionCall::new("a", "f", json!({"x": 1})));
    session.reply(FunctionCall::new("a", "f", json!({"x": 2})));
    let calls: Vec<_> = session
        .get_last_chat()
        .unwrap()
        .get_function_calls()
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1].args().unwrap(), json!({"x": 2}));
}

#[test]
fn tool_chat_must_follow_assistant_reply() {
    let tool_chat = || Chat::new(Role::Tool, Vec::new());
    let mut session = Session::new(10);
    assert!(session.add_chat(tool_chat()).is_err());
    session.ask("Hi");
    assert!(session.add_chat(tool_chat()).is_err());
    assert_eq!(roles(&session), [Role::User]);
}

#[test]
fn function_responses_are_matched_by_id() {
    let mut session = Session::new(10);
    assert!(matches!(
        session.add_function_response("call_paris", "Sunny"),
        Err(AddFunctionResponseError::FunctionResponseAfterUser)
    ));
    let mut session = weather_calls();
    // Two calls of the same function are told apart by id
    assert_eq!(pending_ids(&session), ["call_paris", "call_london"]);
    assert!(matches!(
        session.add_function_response("call_rome", "Sunny"),
        Err(AddFunctionResponseError::NoPendingCall(id)) if id == "call_rome"
    ));

    session
        .add_function_response("call_london", "Rainy")
        .unwrap();
    assert_eq!(pending_ids(&session), ["call_paris"]);
    assert!(matches!(
        session.add_function_response("call_london", "Rainy again"),
        Err(AddFunctionResponseError::NoPendingCall(_))
    ));
    // Anything `Serialize` is sent as JSON text, strings as they are
    session
        .add_function_response("call_paris", json!({"sky": "clear", "celsius": 21}))
        .unwrap();
    assert!(pending_ids(&session).is_empty());

    // Responses are in one chat, in the order they were added, with names of their calls
    assert_eq!(roles(&session), [Role::User, Role::Assistant, Role::Tool]);
    assert_eq!(
        responses(&session),
        [
            ("call_london".into(), "get_weather".into(), "Rainy".into()),
            (
                "call_paris".into(),
                "get_weather".into(),
                r#"{"celsius":21,"sky":"clear"}"#.into()
            ),
        ]
    );
    session.reply("Sunny in Paris, rainy in London");
    assert!(pending_ids(&session).is_empty());
}

#[test]
fn multimodal_function_response() {
    let mut session = weather_calls();
    let photo = InlineData::new(mime::IMAGE_PNG, "iVBORw0KGgo=".into());
    session
        .add_function_response(
            "call_paris",
            MultimodalResponse::new(vec!["Photo of the sky".into(), photo.into()]),
        )
        .unwrap();
    let PartType::FunctionResponse(response) = session.get_last_chat().unwrap().parts()[0].data()
    else {
        panic!("Expected a function response");
    };
    assert_eq!(
        Chat::extract_text_all(response.parts(), ""),
        "Photo of the sky"
    );
    assert!(matches!(
        response.parts()[1].data(),
        PartType::InlineData(data) if *data.mime_type() == mime::IMAGE_PNG
    ));
}

#[test]
fn sliding_window_forgets_calls_with_their_responses() {
    let mut session = weather_calls();
    session
        .add_function_response("call_paris", "Sunny")
        .unwrap();
    session
        .add_function_response("call_london", "Rainy")
        .unwrap();
    session.reply("Sunny in Paris, rainy in London");
    session.ask("And tomorrow?");
    assert_eq!(session.get_history_length(), 5);

    // Forgetting the first prompt would start the history with a function call, which the API
    // rejects, so the call and its responses are forgotten with it.
    assert_eq!(
        session.forget_oldest_chat().unwrap().get_text_all(""),
        "Weather in Paris and London?"
    );
    assert_eq!(roles(&session), [Role::Assistant, Role::User]);

    // With a limit, the same happens automatically
    let mut session = Session::new(2);
    session.ask("Weather in Paris?");
    session.reply(FunctionCall::new(
        "call_paris",
        "get_weather",
        json!({"city": "Paris"}),
    ));
    session
        .add_function_response("call_paris", "Sunny")
        .unwrap();
    session.reply("Sunny");
    session.ask("Thanks");
    assert_eq!(roles(&session), [Role::Assistant, Role::User]);
    assert_eq!(session.get_chat_no(), 5);
}

#[test]
fn latest_chat_is_never_forgotten() {
    let mut session = Session::new(0);
    session.ask("Hi");
    assert_eq!(session.get_history_length(), 1);
    assert!(session.forget_oldest_chat().is_none());
}

#[test]
fn chats_by_absolute_number() {
    let mut session = Session::new(2);
    session.ask("first question");
    session.reply("first answer");
    session.ask("second question");
    assert!(session.get_chat_by_no(1).is_none(), "out of the window");
    assert_eq!(
        session.get_chat_by_no(2).unwrap().get_text_all(""),
        "first answer"
    );
    assert_eq!(
        session.get_chat_by_no(3).unwrap().get_text_all(""),
        "second question"
    );
    assert!(session.get_chat_by_no(4).is_none());
}

/// Keeps at most `max_chars` characters of text, counting what it forgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CharBudget {
    max_chars: usize,
    forgotten: usize,
    run_default: bool,
}
impl OverflowHandler for CharBudget {
    fn handle(&mut self, history: &mut HistoryTrimmer<'_>) -> bool {
        let chars = |history: &HistoryTrimmer| {
            history
                .get_history_as_vecdeque()
                .iter()
                .map(|chat| chat.get_text_all("").len())
                .sum::<usize>()
        };
        while chars(history) > self.max_chars && history.forget_oldest_chat().is_some() {
            self.forgotten += 1;
        }
        self.run_default
    }
}

#[test]
fn custom_overflow_handler() {
    let budget = CharBudget {
        max_chars: 10,
        forgotten: 0,
        run_default: false,
    };
    // `history_limit` is ignored when the handler returns false
    let mut session = Session::new(1).set_overflow_handler(budget.clone());
    session.ask("12345");
    session.reply("12345");
    assert_eq!(session.get_history_length(), 2);
    // Also called when a chat is concatenated
    session.reply("1");
    assert_eq!(session.get_history_length(), 1);
    assert_eq!(session.overflow_handler().forgotten, 1);

    // Returning true also runs the default sliding window
    let mut session = Session::new(1).set_overflow_handler(CharBudget {
        run_default: true,
        ..budget
    });
    session.ask("1");
    session.reply("2");
    assert_eq!(session.get_history_length(), 1);

    // Handler state is saved along with the session
    let mut session = Session::new(10).set_overflow_handler(CharBudget {
        max_chars: 1,
        forgotten: 7,
        run_default: false,
    });
    session.ask("Hi");
    let saved = serde_json::to_string(&session).unwrap();
    let restored: Session<CharBudget> = serde_json::from_str(&saved).unwrap();
    assert_eq!(restored.overflow_handler().forgotten, 7);
    assert_eq!(restored.get_history_length(), 1);

    // Taking the handler out restores the default
    let (session, handler) = restored.remove_overflow_handler();
    assert_eq!(handler.max_chars, 1);
    assert_eq!(session.get_history_limit(), 10);
}

#[test]
fn handler_chosen_at_runtime() {
    let handler: Box<dyn OverflowHandler> = Box::new(CharBudget {
        max_chars: 3,
        forgotten: 0,
        run_default: false,
    });
    let mut session = Session::new(10).set_overflow_handler(handler);
    session.ask("abc");
    session.reply("def");
    assert_eq!(session.get_history_length(), 1);
}

#[test]
fn serde_round_trip() {
    let mut session = weather_calls();
    session
        .add_function_response("call_paris", "Sunny")
        .unwrap();
    let saved = serde_json::to_value(&session).unwrap();
    // The default handler has no state
    assert_eq!(saved["overflow_handler"], json!(null));

    let restored: Session = serde_json::from_value(saved).unwrap();
    assert_eq!(roles(&restored), [Role::User, Role::Assistant, Role::Tool]);
    assert_eq!(pending_ids(&restored), ["call_london"]);
    assert_eq!(restored.get_chat_no(), 3);
}
