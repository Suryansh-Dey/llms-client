//! Tests against a real API. Configure with environment variables:
//! - `OPENAI_API_KEY`
//! - `OPENAI_BASE_URL` (default: OpenAI)
//! - `OPENAI_MODEL` (default: `gpt-5-mini`)
//! - `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
use crate::openai::ask::{DEFAULT_BASE_URL, OpenAI};
use crate::openai::embed::OpenAIEmbedding;
use crate::openai::error::OpenAIResponseError;
use crate::openai::types::request::{InlineData, MultimodalResponse, ReasoningEffort, Tool};
use crate::openai::types::sessions::Session;
use futures::StreamExt;
use serde_json::{Value, json};

fn env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or(default.to_string())
}
pub(crate) fn ai(sys_prompt: Option<&str>) -> OpenAI {
    OpenAI::new(
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not found"),
        env("OPENAI_MODEL", "gpt-5-mini"),
        sys_prompt.map(Into::into),
    )
    .set_base_url(env("OPENAI_BASE_URL", DEFAULT_BASE_URL))
}

#[tokio::test]
async fn ask_string() {
    let mut session = Session::new(6);
    let response = ai(None).ask(session.ask("Hi")).await.unwrap();
    println!("{}", response.get_chat().get_text_all(""));
    assert!(!response.get_chat().get_text_no_think("").is_empty());
    assert_eq!(session.get_history_length(), 2);
}

#[tokio::test]
async fn ask_string_for_json() {
    let mut session = Session::new(6).set_remember_reply(false);
    let response = ai(Some("Classify the given words"))
        .set_json_mode(json!({
            "type": "object",
            "properties": {
                "positive":{
                    "type":"array",
                    "items":{"type":"string"}
                },
                "negative":{
                    "type":"array",
                    "items":{"type":"string"}
                }
            },
            "required":["positive", "negative"],
            "additionalProperties": false
        }))
        .ask(session.ask(r#"["Joy", "Success", "Love", "Fear", "Failure", "Hate"]"#))
        .await
        .unwrap();

    let json: Value = response.get_json().unwrap();
    println!("{}", json);
    assert!(json["positive"].as_array().unwrap().len() >= 2);
    assert_eq!(session.get_history_length(), 0);
}

#[tokio::test]
async fn ask_streamed() {
    let mut session = Session::new(6);
    session.ask("Can you explain me something in one line?");
    let ai = ai(None);
    ai.ask(&mut session).await.unwrap();
    session.ask("machine learning");
    let mut response_stream = ai
        .ask_as_stream_with_extractor(session, |session, _| {
            session.get_last_chat().unwrap().get_text_no_think("")
        })
        .await
        .unwrap();
    let mut last = String::new();
    while let Some(response) = response_stream.next().await {
        last = response.unwrap();
        println!("{last}");
    }
    assert!(!last.is_empty());
    assert_eq!(response_stream.get_session().get_history_length(), 4);
}

#[tokio::test]
async fn ask_streamed_with_tools() {
    let mut session = Session::new(6);
    session.ask("What is the weather in London and in Paris? Call the function for both cities.");
    let ai = ai(None).set_tool(Tool::Function(json!({
        "name": "get_weather",
        "description": "Get the weather of a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        }
    })));
    let mut response_stream = ai.ask_as_stream(session).await.unwrap();
    while let Some(response) = response_stream.next().await {
        response.unwrap();
    }
    let mut session = response_stream.get_session_owned();
    let calls: Vec<(String, Value)> = session
        .get_last_chat()
        .unwrap()
        .get_function_calls()
        .map(|call| (call.id().clone(), call.args().unwrap()))
        .collect();
    println!("Calls: {calls:?}");
    assert_eq!(calls.len(), 2);
    for (id, args) in calls {
        let city = args["city"].as_str().unwrap().to_string();
        session
            .add_function_response(id, format!("Sunny, 25 degrees in {city}"))
            .unwrap();
    }
    let response = ai.ask(&mut session).await.unwrap();
    println!("{}", response.get_chat().get_text_no_think(""));
    assert!(response.get_chat().get_text_no_think("").contains("25"));
}

#[tokio::test]
async fn reasoning_effort() {
    let mut session = Session::new(4);
    let ai = ai(None).set_reasoning_effort(ReasoningEffort::Low);
    session.ask("How many 'r's are in strawberry? Answer in one word.");
    let response = ai.ask(&mut session).await.unwrap();
    println!("Thoughts: {}", response.get_chat().get_thoughts(""));
    assert!(response.get_chat().get_text_no_think("").len() > 1);
}

#[tokio::test]
async fn status_not_ok() {
    let mut session = Session::new(6);
    let response = ai(None)
        .set_api_key("wrong_api_key")
        .ask(session.ask("Hi"))
        .await;
    match response {
        Err(OpenAIResponseError::StatusNotOk(e)) => {
            println!("{e}");
            assert!(e.status.is_client_error());
            assert!(!e.error.message.is_empty());
        }
        other => panic!("Expected invalid api key error, got {other:?}"),
    }
}

#[tokio::test]
async fn embedding() {
    let embedder = OpenAIEmbedding::new(
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not found"),
        env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    .set_base_url(env("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    .set_dimensions(256);
    let response = embedder
        .embed_texts(vec!["I love Rust".into(), "Rust is my favourite".into()])
        .await
        .unwrap();
    assert_eq!(response.embeddings().len(), 2);
    assert_eq!(response.embedding().dimension(), 256);
}

#[tokio::test]
async fn multimodal_function_response() {
    let mut session = Session::new(10);
    let ai = ai(None).set_tool(Tool::Function(json!({
        "name": "take_screenshot",
        "description": "Takes a screenshot of the user's screen",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": false}
    })));
    session.ask("Take a screenshot and tell me what kind of chart is on my screen.");
    ai.ask(&mut session).await.unwrap();
    let call_id = session.get_pending_function_calls()[0].id().clone();
    let screenshot = InlineData::from_path("tests/lda.png", mime::IMAGE_PNG)
        .await
        .unwrap();
    session
        .add_function_response(
            call_id,
            MultimodalResponse::new(vec!["Screenshot taken".into(), screenshot.into()]),
        )
        .unwrap();
    let response = ai.ask(&mut session).await.unwrap();
    let reply = response.get_chat().get_text_no_think("").to_lowercase();
    println!("{reply}");
    assert!(reply.contains("scatter"));
}
