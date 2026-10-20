//! `execute_function_calls!` and `execute_function_calls_with_callback!` on a session whose last
//! reply has function calls, as the model would send them.
use open_client::mime;
use open_client::openai::types::request::{
    Chat, FunctionCall, InlineData, MultimodalResponse, PartType, Role,
};
use open_client::openai::types::sessions::Session;
use open_client::openai::utils::{
    execute_function_calls, execute_function_calls_with_callback, openai_function,
};
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
fn session_with_calls(calls: Vec<FunctionCall>) -> Session {
    let mut session = Session::new(10);
    session.ask("Help me");
    session.reply_parts(calls.into_iter().map(Into::into).collect());
    session
}

/// `(id, text)` of every function response in the last chat.
fn responses(session: &Session) -> Vec<(String, String)> {
    let chat = session.get_last_chat().unwrap();
    assert_eq!(*chat.role(), Role::Tool);
    chat.parts()
        .iter()
        .filter_map(|part| match part.data() {
            PartType::FunctionResponse(r) => {
                Some((r.id().clone(), Chat::extract_text_all(r.parts(), "")))
            }
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn executes_all_pending_calls() {
    let mut session = session_with_calls(vec![
        FunctionCall::new("call_1", "to_fahrenheit", json!({"celsius": 100})),
        FunctionCall::new("call_2", "stock", json!({"product": "apple"})),
        // Same function twice, e.g. in parallel calls
        FunctionCall::new("call_3", "to_fahrenheit", json!({"celsius": 0})),
    ]);

    // One result per listed function, the last one if it was called many times. Functions not
    // called get `None`.
    let results = execute_function_calls!(session, stock, to_fahrenheit, take_photo);
    assert_eq!(results, [Some(Ok(json!(12))), Some(Ok(json!(32.0))), None]);

    let mut responses = responses(&session);
    responses.sort();
    assert_eq!(
        responses,
        [
            ("call_1".into(), "212.0".into()),
            ("call_2".into(), "12".into()),
            ("call_3".into(), "32.0".into())
        ]
    );
    assert!(session.get_pending_function_calls().is_empty());
}

#[tokio::test]
async fn errors_are_sent_to_the_model() {
    let mut session = session_with_calls(vec![
        FunctionCall::new("call_1", "stock", json!({"product": "pear"})),
        FunctionCall::new("call_2", "to_fahrenheit", json!({"celsius": "hot"})),
    ]);
    let results = execute_function_calls!(session, stock, to_fahrenheit);
    assert_eq!(results[0], Some(Err("Unknown product pear".into())));
    assert!(matches!(&results[1], Some(Err(e)) if e.starts_with("invalid type")));

    let responses = responses(&session);
    assert_eq!(
        responses[0],
        (
            "call_1".into(),
            r#"{"Error":"Unknown product pear"}"#.into()
        )
    );
    assert!(responses[1].1.starts_with(r#"{"Error":"invalid type"#));
}

#[tokio::test]
async fn invalid_json_arguments() {
    let call = serde_json::from_value::<FunctionCall>(json!({
        "id": "call_1", "name": "to_fahrenheit", "arguments": "{\"celsius\": 10"
    }))
    .unwrap();
    let mut session = session_with_calls(vec![call]);
    let results = execute_function_calls!(session, to_fahrenheit);
    assert!(matches!(&results[0], Some(Err(e)) if e.starts_with("Invalid JSON arguments")));
}

#[tokio::test]
async fn calls_answered_manually_are_skipped() {
    let mut session = session_with_calls(vec![
        FunctionCall::new("call_1", "stock", json!({"product": "apple"})),
        FunctionCall::new("call_2", "take_photo", json!({})),
    ]);
    session.add_function_response("call_1", 99).unwrap();
    let results = execute_function_calls!(session, stock, take_photo);
    assert_eq!(results[0], None);
    assert_eq!(
        responses(&session),
        [
            ("call_1".into(), "99".into()),
            ("call_2".into(), "photo.png".into())
        ]
    );
}

#[tokio::test]
async fn nothing_to_execute() {
    let mut session = Session::new(10);
    session.ask("Hi");
    session.reply("Hello! How can I help?");
    let results = execute_function_calls!(session, stock);
    assert_eq!(results, [None]);
    assert_eq!(session.get_history_length(), 2);
}

#[tokio::test]
async fn callback_shapes_the_response() {
    let mut session = session_with_calls(vec![
        FunctionCall::new("call_1", "stock", json!({"product": "pear"})),
        FunctionCall::new("call_2", "to_fahrenheit", json!({"celsius": 20})),
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
    let mut responses = responses(&session);
    responses.sort();
    assert_eq!(
        responses,
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
async fn callback_can_respond_with_images() {
    let mut session =
        session_with_calls(vec![FunctionCall::new("call_1", "take_photo", json!({}))]);
    execute_function_calls_with_callback!(
        session,
        |_name, result: Result<Value, String>| {
            let file = result.unwrap().as_str().unwrap().to_string();
            MultimodalResponse::new(vec![
                format!("Saved as {file}").into(),
                InlineData::new(mime::IMAGE_PNG, "iVBORw0KGgo=".into()).into(),
            ])
        },
        take_photo
    );
    let chat = session.get_last_chat().unwrap();
    let PartType::FunctionResponse(response) = chat.parts()[0].data() else {
        panic!("Expected a function response");
    };
    assert_eq!(
        Chat::extract_text_all(response.parts(), ""),
        "Saved as photo.png"
    );
    assert!(matches!(
        response.parts()[1].data(),
        PartType::InlineData(_)
    ));
}
