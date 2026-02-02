use gemini_client_api::gemini::types::request::{FunctionCall, PartType};
use gemini_client_api::gemini::{
    types::sessions::Session,
    utils::{GeminiSchema, execute_function_calls_with_callback, gemini_function},
};
use serde_json::json;

#[gemini_function]
/// Add two numbers
async fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}

#[gemini_function]
/// A function that fails
async fn fail_fn() -> Result<String, String> {
    Err("Simulated failure".into())
}

#[tokio::test]
async fn test_callback_success() {
    let mut session = Session::new(10);
    // Simulate a model response requesting add_numbers
    let parts =
        vec![FunctionCall::new("add_numbers".to_string(), Some(json!({"a": 10, "b": 20}))).into()];
    session.reply_parts(parts);

    let results = execute_function_calls_with_callback!(
        session,
        |res| {
            match res {
                Ok(val) => val,
                Err(e) => json!({"error": e}),
            }
        },
        add_numbers
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0], Some(Ok(json!(30))));

    // Verify session
    let history = session.get_history();
    let last_chat = history.last().unwrap();
    if let PartType::FunctionResponse(resp) = last_chat.parts()[0].data() {
        assert_eq!(resp.response(), &json!({"result": 30}));
    } else {
        panic!("Expected FunctionResponse");
    }
}

#[tokio::test]
async fn test_callback_failure() {
    let mut session = Session::new(10);
    // Simulate a model response requesting fail_fn
    let parts = vec![FunctionCall::new("fail_fn".to_string(), Some(json!({}))).into()];
    session.reply_parts(parts);

    let results = execute_function_calls_with_callback!(
        session,
        |res| {
            match res {
                Ok(val) => val,
                Err(e) => json!({"error_msg": e}),
            }
        },
        fail_fn
    );

    assert_eq!(results.len(), 1);
    // The macro returns the original result, which should be an error
    assert!(results[0].as_ref().unwrap().is_err());

    // Verify session - it should have the recovered value from callback
    let history = session.get_history();
    let last_chat = history.last().unwrap();
    if let PartType::FunctionResponse(resp) = last_chat.parts()[0].data() {
        assert_eq!(resp.response(), &json!({"error_msg": "Simulated failure"}));
    } else {
        panic!("Expected FunctionResponse");
    }
}
