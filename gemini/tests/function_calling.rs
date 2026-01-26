use gemini_client_api::gemini::ask::Gemini;
use gemini_client_api::gemini::types::request::{FunctionCall, PartType, Role, Tool};
use gemini_client_api::gemini::{
    types::sessions::Session,
    utils::{GeminiSchema, execute_function_calls, gemini_function},
};
use serde_json::json;
use std::error::Error;

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
        FunctionCall::new("add_numbers".to_string(), Some(json!({"a": 10, "b": 20}))).into(),
        FunctionCall::new("greet".to_string(), Some(json!({"name": "Gemini"}))).into(),
    ];
    session.reply(parts);

    let results = execute_function_calls!(session, add_numbers, greet);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0], Ok(json!(30)));
    assert_eq!(results[1], Ok(json!("Hello, Gemini!")));

    let history = session.get_history();

    assert_eq!(history.len(), 2);
    let last_chat = history[1];
    assert_eq!(*last_chat.role(), Role::Function);
    assert_eq!(last_chat.parts().len(), 2);

    // Check specific values in session
    let mut session_results = Vec::new();
    for part in last_chat.parts() {
        if let PartType::FunctionResponse(resp) = part.data() {
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
    let parts = vec![FunctionCall::new("fail_fn".to_string(), Some(json!({}))).into()];
    session.reply(parts);

    let results = execute_function_calls!(session, fail_fn);
    assert_eq!(results.len(), 1);
    assert!(results[0].is_err());

    let history = session.get_history();
    // Only model reply should be there.
    // No function response should be added because it failed.
    assert_eq!(history.len(), 1);
    assert_eq!(*history[0].role(), Role::Model);
}

#[tokio::test]
async fn test_non_result_always_success() {
    let mut session = Session::new(10);
    let parts = vec![FunctionCall::new("sync_fn".to_string(), Some(json!({"x": 21}))).into()];
    session.reply(parts);

    let results = execute_function_calls!(session, sync_fn);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], Ok(json!(42)));

    let history = session.get_history();
    assert_eq!(history.len(), 2);
    assert_eq!(*history[1].role(), Role::Function);
}
#[gemini_function]
///Lists files in my dir
async fn list_files(
    ///Path to the dir
    path: String,
) -> Result<String, Box<dyn Error>> {
    Ok(std::fs::read_dir(path)?
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect::<Vec<String>>()
        .join(", "))
}

#[tokio::test]
async fn ask_with_function_calls() {
    let mut session = Session::new(10);
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.5-flash",
        None,
    )
    .set_tools(vec![Tool::FunctionDeclarations(vec![
        list_files::gemini_schema(),
    ])]);
    session.ask_string("What files I have in current directory");
    let response = ai.ask(&mut session).await.unwrap(); //Received a function call
    let result = execute_function_calls!(session, list_files); //doesn't update session if Error
    println!("function output: {:?}", result);
    if result.len() != 0 {
        //If any function call at all happened
        let response = ai.ask(&mut session).await.unwrap(); //Providing output of the function call and continue
        println!("{:?}", response.get_chat().parts());
    } else {
        println!("{:?}", response.get_chat().parts());
    }
}
