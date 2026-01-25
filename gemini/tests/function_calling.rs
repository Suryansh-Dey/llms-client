use gemini_client_api::gemini::ask::Gemini;
use gemini_client_api::gemini::types::request::{FunctionCall, Part, Role, Tool};
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

#[gemini_function]
/// A function that might fail
async fn fail_fn() -> Result<String, String> {
    Err("Simulated failure".into())
}

#[gemini_function]
/// A function that returns a direct value
fn sync_fn(x: i32) -> i32 {
    x * 2
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

    let results = execute_function_calls!(session, add_numbers, greet);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0], Ok(json!(30)));
    assert_eq!(results[1], Ok(json!("Hello, Gemini!")));

    let history = session.get_history();

    assert_eq!(history.len(), 2);
    let last_chat = history[1];
    assert_eq!(*last_chat.role(), Role::function);
    assert_eq!(last_chat.parts().len(), 2);

    // Check specific values in session
    let mut session_results = Vec::new();
    for part in last_chat.parts() {
        if let Part::functionResponse(resp) = part {
            session_results.push((resp.name().clone(), resp.response().clone()));
        }
    }
    session_results.sort_by(|a, b| a.0.cmp(&b.0));

    assert_eq!(session_results[0].0, "add_numbers");
    assert_eq!(session_results[0].1, json!({"result": 30}));
    assert_eq!(session_results[1].0, "greet");
    assert_eq!(session_results[1].1, json!({"result": "Hello, Gemini!"}));
}

#[tokio::test]
async fn test_failure_no_session_update() {
    let mut session = Session::new(10);
    let parts = vec![Part::functionCall(FunctionCall::new(
        "fail_fn".to_string(),
        Some(json!({})),
    ))];
    session.reply(parts);

    let results = execute_function_calls!(session, fail_fn);
    assert_eq!(results.len(), 1);
    assert!(results[0].is_err());

    let history = session.get_history();
    // Only model reply should be there.
    // No function response should be added because it failed.
    assert_eq!(history.len(), 1);
    assert_eq!(*history[0].role(), Role::model);
}

#[tokio::test]
async fn test_non_result_always_success() {
    let mut session = Session::new(10);
    let parts = vec![Part::functionCall(FunctionCall::new(
        "sync_fn".to_string(),
        Some(json!({"x": 21})),
    ))];
    session.reply(parts);

    let results = execute_function_calls!(session, sync_fn);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], Ok(json!(42)));

    let history = session.get_history();
    assert_eq!(history.len(), 2);
    assert_eq!(*history[1].role(), Role::function);
}
#[gemini_function]
///Lists file in my current dir
async fn list_files() -> String {
    r#" Cargo.lock
Cargo.toml
gemini-proc-macros
README.md
src
target
tests"#
        .into()
}
#[tokio::test]
async fn ask_with_function_calls() {
    let mut session = Session::new(10);
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.5-flash",
        None,
    )
    .set_tools(vec![Tool::functionDeclarations(vec![
        list_files::gemini_schema(),
    ])]);
    session.ask_string("What files I have in this directory");
    ai.ask(&mut session).await.unwrap();
    execute_function_calls!(session, list_files);
    ai.ask(&mut session).await.unwrap();
    println!("{:?}", session.get_history());
}
