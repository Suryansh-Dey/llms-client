use gemini_client_api::gemini::types::request::{FunctionCall, Part, Role};
use gemini_client_api::gemini::{
    types::sessions::Session,
    utils::{GeminiSchema, execute_function_calls, gemini_function},
};
use serde_json::json;

#[gemini_function]
/// Add two numbers
async fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}

#[gemini_function]
/// Greet a person
async fn greet(name: String) -> String {
    format!("Hello, {}!", name)
}

#[tokio::test]
async fn execute_function_calls_test() {
    let mut session = Session::new(10);

    // Simulate a model response with function calls
    let parts = vec![
        Part::functionCall(FunctionCall::new(
            "add_numbers".to_string(),
            Some(json!({"a": 10, "b": 20})),
        )),
        Part::functionCall(FunctionCall::new(
            "greet".to_string(),
            Some(json!({"name": "Gemini"})),
        )),
    ];
    session.reply(parts);

    execute_function_calls!(session, add_numbers, greet);
    let history = session.get_history();

    assert_eq!(history.len(), 2);
    let last_chat = history[1];
    assert_eq!(*last_chat.role(), Role::function);
    assert_eq!(last_chat.parts().len(), 2);

    // Check specific values
    let mut results = Vec::new();
    for part in last_chat.parts() {
        if let Part::functionResponse(resp) = part {
            results.push((resp.name().clone(), resp.response().clone()));
        }
    }
    results.sort_by(|a, b| a.0.cmp(&b.0));

    assert_eq!(results[0].0, "add_numbers");
    assert_eq!(results[0].1, json!(30));
    assert_eq!(results[1].0, "greet");
    assert_eq!(results[1].1, json!("Hello, Gemini!"));
}
