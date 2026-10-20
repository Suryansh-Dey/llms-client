use open_client::openai::types::request::{
    Chat, FunctionCall, InlineData, MultimodalResponse, PartType,
};
use open_client::openai::{
    types::sessions::Session,
    utils::{execute_function_calls_with_callback, openai_function},
};
use serde_json::json;

#[openai_function]
/// Add two numbers
async fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}

#[openai_function]
/// A function that fails
async fn fail_fn() -> Result<String, String> {
    Err("Simulated failure".into())
}

fn response_text(session: &Session) -> String {
    match session.get_last_chat().unwrap().parts()[0].data() {
        PartType::FunctionResponse(resp) => Chat::extract_text_all(resp.parts(), ""),
        _ => panic!("Expected FunctionResponse"),
    }
}

#[tokio::test]
async fn test_callback_success() {
    let mut session = Session::new(10);
    // Simulate a model response requesting add_numbers
    let parts = vec![FunctionCall::new("call_1", "add_numbers", json!({"a": 10, "b": 20})).into()];
    session.reply_parts(parts);

    let results = execute_function_calls_with_callback!(
        session,
        |_name, res| {
            match res {
                Ok(val) => val,
                Err(e) => json!({"error": e}),
            }
        },
        add_numbers
    );

    assert_eq!(results.len(), 1);
    assert_eq!(results[0], Some(Ok(json!(30))));
    assert_eq!(response_text(&session), "30");
}

#[tokio::test]
async fn test_callback_failure() {
    let mut session = Session::new(10);
    // Simulate a model response requesting fail_fn
    let parts = vec![FunctionCall::new("call_1", "fail_fn", json!({})).into()];
    session.reply_parts(parts);

    let results = execute_function_calls_with_callback!(
        session,
        |_name, res| {
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
    assert_eq!(
        response_text(&session),
        r#"{"error_msg":"Simulated failure"}"#
    );
}

#[tokio::test]
async fn test_callback_multimodal() {
    let mut session = Session::new(10);
    let parts = vec![FunctionCall::new("call_1", "add_numbers", json!({"a": 1, "b": 2})).into()];
    session.reply_parts(parts);

    // Callback can return anything accepted by `add_function_response`
    execute_function_calls_with_callback!(
        session,
        |_name, res: Result<serde_json::Value, String>| {
            MultimodalResponse::new(vec![
                format!("Sum is {}, here is a chart", res.unwrap()).into(),
                InlineData::new(mime::IMAGE_PNG, "aW1n".into()).into(),
            ])
        },
        add_numbers
    );
    match session.get_last_chat().unwrap().parts()[0].data() {
        PartType::FunctionResponse(resp) => {
            assert_eq!(resp.parts().len(), 2);
            assert!(matches!(resp.parts()[1].data(), PartType::InlineData(_)));
        }
        _ => panic!("Expected FunctionResponse"),
    }
}
