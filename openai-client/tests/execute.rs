//! `execute_function_calls!` and `execute_function_calls_with_callback!` on a session whose last
//! reply has tool calls, as the model would send them.
use open_client::chat::message::{Message, ToolCall};
use open_client::chat::session::Session;
use open_client::{execute_function_calls, execute_function_calls_with_callback, openai_function};
use serde::Serialize;
use serde_json::{Value, json};

#[openai_function]
/// Converts a temperature to Fahrenheit.
fn to_fahrenheit(celsius: f64) -> f64 {
    celsius * 9.0 / 5.0 + 32.0
}

#[openai_function]
/// Looks up the stock of a product.
async fn stock(product: String) -> Result<u32, String> {
    match product.as_str() {
        "apple" => Ok(12),
        _ => Err(format!("Unknown product {product}")),
    }
}

#[openai_function]
/// Takes a photo with the camera.
async fn take_photo() -> String {
    "photo.png".into()
}

/// A session whose last reply calls these functions.
fn session_with_calls(calls: Vec<ToolCall>) -> Session {
    let mut session = Session::default();
    session.ask("Help me");
    session.reply(calls);
    session
}

/// `(call id, content)` of every tool message.
fn results(session: &Session) -> Vec<(String, String)> {
    session
        .history()
        .iter()
        .filter_map(|message| match message {
            Message::Tool(result) => Some((result.tool_call_id.clone(), result.content.clone())),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn executes_all_pending_calls() {
    let mut session = session_with_calls(vec![
        ToolCall::new("call_1", "to_fahrenheit", json!({"celsius": 100})),
        ToolCall::new("call_2", "stock", json!({"product": "apple"})),
        // Same function twice, e.g. in parallel calls
        ToolCall::new("call_3", "to_fahrenheit", json!({"celsius": 0})),
    ]);

    // One result per listed function, the last one if it was called many times. Functions not
    // called get `None`.
    let results = execute_function_calls!(session, stock, to_fahrenheit, take_photo);
    assert_eq!(results, [Some(Ok(json!(12))), Some(Ok(json!(32.0))), None]);

    let mut results = self::results(&session);
    results.sort();
    assert_eq!(
        results,
        [
            ("call_1".into(), "212.0".into()),
            ("call_2".into(), "12".into()),
            ("call_3".into(), "32.0".into())
        ]
    );
    assert!(session.pending_tool_calls().is_empty());
}

#[tokio::test]
async fn errors_are_sent_to_the_model() {
    let mut session = session_with_calls(vec![
        ToolCall::new("call_1", "stock", json!({"product": "pear"})),
        ToolCall::new("call_2", "to_fahrenheit", json!({"celsius": "hot"})),
    ]);
    let results = execute_function_calls!(session, stock, to_fahrenheit);
    assert_eq!(results[0], Some(Err("Unknown product pear".into())));
    assert!(matches!(&results[1], Some(Err(e)) if e.starts_with("invalid type")));

    let results = self::results(&session);
    assert_eq!(
        results[0],
        (
            "call_1".into(),
            r#"{"Error":"Unknown product pear"}"#.into()
        )
    );
    assert!(results[1].1.starts_with(r#"{"Error":"invalid type"#));
}

#[tokio::test]
async fn invalid_json_arguments() {
    let call = serde_json::from_value::<ToolCall>(json!({
        "id": "call_1",
        "type": "function",
        "function": {"name": "to_fahrenheit", "arguments": "{\"celsius\": 10"}
    }))
    .unwrap();
    let mut session = session_with_calls(vec![call]);
    let results = execute_function_calls!(session, to_fahrenheit);
    assert!(matches!(&results[0], Some(Err(e)) if e.starts_with("Invalid JSON arguments")));
}

#[tokio::test]
async fn calls_answered_manually_are_skipped() {
    let mut session = session_with_calls(vec![
        ToolCall::new("call_1", "stock", json!({"product": "apple"})),
        ToolCall::new("call_2", "take_photo", json!({})),
    ]);
    session.add_tool_result("call_1", 99).unwrap();
    let results = execute_function_calls!(session, stock, take_photo);
    assert_eq!(results[0], None);
    assert_eq!(
        self::results(&session),
        [
            ("call_1".into(), "99".into()),
            ("call_2".into(), "photo.png".into())
        ]
    );
}

#[tokio::test]
async fn nothing_to_execute() {
    let mut session = Session::default();
    session.ask("Hi");
    session.reply("Hello! How can I help?");
    let results = execute_function_calls!(session, stock);
    assert_eq!(results, [None]);
    assert_eq!(session.len(), 2);
}

#[tokio::test]
async fn callback_shapes_the_result() {
    let mut session = session_with_calls(vec![
        ToolCall::new("call_1", "stock", json!({"product": "pear"})),
        ToolCall::new("call_2", "to_fahrenheit", json!({"celsius": 20})),
    ]);
    let results = execute_function_calls_with_callback!(
        session,
        |name: String, result: Result<Value, String>| match result {
            Ok(value) => format!("{name} returned {value}"),
            Err(e) => format!("{name} failed: {e}. Ask the user for another product."),
        },
        stock,
        to_fahrenheit
    );
    // Results are the function outputs, not what the callback returned
    assert_eq!(results[1], Some(Ok(json!(68.0))));
    let mut results = self::results(&session);
    results.sort();
    assert_eq!(
        results,
        [
            (
                "call_1".into(),
                "stock failed: Unknown product pear. Ask the user for another product.".into()
            ),
            ("call_2".into(), "to_fahrenheit returned 68.0".into())
        ]
    );
}

#[tokio::test]
async fn callback_can_return_any_serialize() {
    #[derive(Serialize)]
    struct Photo {
        file: String,
        saved: bool,
    }
    let mut session = session_with_calls(vec![ToolCall::new("call_1", "take_photo", json!({}))]);
    execute_function_calls_with_callback!(
        session,
        |_name, result: Result<Value, String>| Photo {
            file: result.unwrap().as_str().unwrap().to_string(),
            saved: true,
        },
        take_photo
    );
    assert_eq!(
        results(&session),
        [(
            "call_1".into(),
            r#"{"file":"photo.png","saved":true}"#.into()
        )]
    );
}
