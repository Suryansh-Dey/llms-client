use crate::gemini::ask::Gemini;
use crate::gemini::types::request::{SystemInstruction, ThinkingConfig, Tool};
use crate::gemini::types::sessions::{self, Session};
use futures::StreamExt;
use serde_json::{Value, json};

#[tokio::test]
async fn ask_string() {
    let mut session = Session::new(6);
    let response = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.0-flash",
        None,
    )
    .ask(session.ask_string("Hi"))
    .await
    .unwrap();
    println!("{}", response.get_text(""));
}

#[tokio::test]
async fn ask_string_for_json() {
    let mut session = Session::new(6).set_remember_reply(false);
    let response = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.0-flash-lite",
        Some(SystemInstruction::from_str("Classify the given words")),
    )
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
        "required":["positive", "negative"]
    }))
    .ask(session.ask_string(r#"["Joy", "Success", "Love", "Hope", "Confidence", "Peace", "Victory", "Harmony", "Inspiration", "Gratitude", "Prosperity", "Strength", "Freedom", "Comfort", "Brilliance" "Fear", "Failure", "Hate", "Doubt", "Pain", "Suffering", "Loss", "Anxiety", "Despair", "Betrayal", "Weakness", "Chaos", "Misery", "Frustration", "Darkness"]"#))
    .await
    .unwrap();

    let json: Value = response.get_json().unwrap();
    println!("{}", json);
}

#[tokio::test]
async fn ask_streamed() {
    let mut session = Session::new(6);
    session.ask_string("Can you explain me something in one line?");
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.5-flash",
        None,
    );
    ai.ask(&mut session).await.unwrap();
    session.ask_string("machine learning");
    let mut response_stream = ai
        .ask_as_stream_with_extractor(session, |session, _| {
            session.get_last_message_text("").unwrap()
        })
        .await
        .unwrap();
    while let Some(response) = response_stream.next().await {
        println!("{}", response.unwrap());
    }
}

#[tokio::test]
async fn ask_streamed_with_tools() {
    let mut session = Session::new(6);
    session.ask_string("find sum of first 100 prime number using code");
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.0-flash",
        None,
    )
    .set_tools(vec![Tool::code_execution(json!({}))]);
    let mut response_stream = ai.ask_as_stream(session).await.unwrap();
    while let Some(response) = response_stream.next().await {
        println!("{}", response.unwrap().get_text(""));
    }
    println!(
        "Complete reply: {:#?}",
        json!(response_stream.get_session().get_last_message().unwrap())
    );
}

#[tokio::test]
async fn ask_thinking() {
    let mut session = Session::new(4);
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.5-flash",
        None,
    )
    .set_thinking_config(ThinkingConfig::new(true, 1024));
    session.ask_string("How to calculate width of a binary tree?");
    let response = ai.ask(&mut session).await.unwrap();
    println!("{}", response.get_text(""));
}
