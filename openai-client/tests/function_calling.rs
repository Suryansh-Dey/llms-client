use open_client::openai::types::request::{Chat, FunctionCall, PartType, Role, Tool};
use open_client::openai::{
    types::sessions::Session,
    utils::{OpenAISchema, execute_function_calls, openai_function},
};
use serde_json::json;
use std::error::Error;

#[openai_function]
/// Add two numbers
async fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}

#[openai_function]
/// Greet a person
async fn greet(name: String) -> String {
    format!("Hello, {}!", name)
}

#[openai_function]
/// A function that might fail
async fn fail_fn() -> Result<String, String> {
    Err("Simulated failure".into())
}

#[openai_function]
/// A function that returns a direct value
fn sync_fn(x: i32) -> i32 {
    x * 2
}

fn responses(chat: &Chat) -> Vec<(String, String, String)> {
    chat.parts()
        .iter()
        .filter_map(|part| match part.data() {
            PartType::FunctionResponse(resp) => Some((
                resp.id().clone(),
                resp.name().clone(),
                Chat::extract_text_all(resp.parts(), ""),
            )),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn execute_function_calls_test() {
    let mut session = Session::new(10);

    // Simulate a model response with function calls
    let parts = vec![
        FunctionCall::new("call_1", "add_numbers", json!({"a": 10, "b": 20})).into(),
        FunctionCall::new("call_2", "greet", json!({"name": "Ferris"})).into(),
    ];
    session.reply_parts(parts);

    let results = execute_function_calls!(session, add_numbers, greet);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0], Some(Ok(json!(30))));
    assert_eq!(results[1], Some(Ok(json!("Hello, Ferris!"))));

    let history = session.get_history();
    assert_eq!(history.len(), 2);
    let last_chat = history[1];
    assert_eq!(*last_chat.role(), Role::Tool);

    let mut session_results = responses(last_chat);
    session_results.sort();
    assert_eq!(
        session_results,
        [
            ("call_1".into(), "add_numbers".into(), "30".into()),
            ("call_2".into(), "greet".into(), "Hello, Ferris!".into())
        ]
    );
}

#[tokio::test]
async fn same_function_called_twice() {
    let mut session = Session::new(10);
    session.reply_parts(vec![
        FunctionCall::new("call_1", "sync_fn", json!({"x": 1})).into(),
        FunctionCall::new("call_2", "sync_fn", json!({"x": 2})).into(),
    ]);
    execute_function_calls!(session, sync_fn);
    let mut session_results = responses(session.get_last_chat().unwrap());
    session_results.sort();
    assert_eq!(
        session_results,
        [
            ("call_1".into(), "sync_fn".into(), "2".into()),
            ("call_2".into(), "sync_fn".into(), "4".into())
        ]
    );
}

#[tokio::test]
async fn skips_calls_responded_manually() {
    let mut session = Session::new(10);
    session.reply_parts(vec![
        FunctionCall::new("call_1", "greet", json!({"name": "A"})).into(),
        FunctionCall::new("call_2", "greet", json!({"name": "B"})).into(),
    ]);
    session.add_function_response("call_1", "Hi A").unwrap();
    execute_function_calls!(session, greet);
    assert_eq!(
        responses(session.get_last_chat().unwrap()),
        [
            ("call_1".into(), "greet".into(), "Hi A".into()),
            ("call_2".into(), "greet".into(), "Hello, B!".into())
        ]
    );
    assert!(session.get_pending_function_calls().is_empty());
}

#[tokio::test]
async fn invalid_arguments_are_reported() {
    let mut session = Session::new(10);
    session.reply_parts(vec![
        FunctionCall::new("call_1", "add_numbers", json!({"a": "ten"})).into(),
    ]);
    let results = execute_function_calls!(session, add_numbers);
    assert!(results[0].as_ref().unwrap().is_err());
    assert!(
        responses(session.get_last_chat().unwrap())[0]
            .2
            .contains("Error")
    );
}

#[tokio::test]
async fn test_failure_err_session_update() {
    let mut session = Session::new(10);
    let parts = vec![FunctionCall::new("call_1", "fail_fn", json!({})).into()];
    session.reply_parts(parts);

    let results = execute_function_calls!(session, fail_fn);
    assert_eq!(results.len(), 1);
    assert!(results[0].as_ref().unwrap().is_err());

    let history = session.get_history();
    // Error is sent as the function response
    assert_eq!(history.len(), 2);
    assert_eq!(*history[0].role(), Role::Assistant);
    assert_eq!(*history[1].role(), Role::Tool);
    assert_eq!(
        responses(history[1])[0].2,
        r#"{"Error":"Simulated failure"}"#
    );
}

#[tokio::test]
async fn test_non_result_always_success() {
    let mut session = Session::new(10);
    let parts = vec![FunctionCall::new("call_1", "sync_fn", json!({"x": 21})).into()];
    session.reply_parts(parts);

    let results = execute_function_calls!(session, sync_fn);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], Some(Ok(json!(42))));

    let history = session.get_history();
    assert_eq!(history.len(), 2);
    assert_eq!(*history[1].role(), Role::Tool);
}

#[openai_function]
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

mod live {
    use super::*;
    use open_client::openai::ask::{DEFAULT_BASE_URL, OpenAI};

    #[tokio::test]
    async fn ask_with_function_calls() {
        let mut session = Session::new(10);
        let ai = OpenAI::new(
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not found"),
            std::env::var("OPENAI_MODEL").unwrap_or("gpt-5-mini".into()),
            None,
        )
        .set_base_url(std::env::var("OPENAI_BASE_URL").unwrap_or(DEFAULT_BASE_URL.into()))
        .set_tool(Tool::Function(list_files::openai_schema()));
        session.ask("What files I have in current directory");
        let response = ai.ask(&mut session).await.unwrap(); //Received a function call
        let result = execute_function_calls!(session, list_files);
        println!("function output: {:?}", result);
        assert!(result[0].is_some(), "Model should call list_files");
        //Providing output of the function call and continue
        let response_after = ai.ask(&mut session).await.unwrap();
        println!("{:?}", response.get_chat().parts());
        println!("{}", response_after.get_chat().get_text_no_think(""));
        assert!(
            response_after
                .get_chat()
                .get_text_no_think("")
                .contains("Cargo.toml")
        );
    }
}
