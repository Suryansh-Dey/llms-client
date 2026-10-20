//! Tests against a real API, ignored by default. Run with
//! ```sh
//! OPENAI_API_KEY=.. OPENAI_BASE_URL=.. OPENAI_MODEL=.. OPENAI_EMBEDDING_MODEL=.. cargo test -- --ignored
//! ```
//! `OPENAI_BASE_URL` defaults to OpenAI, `OPENAI_MODEL` to `gpt-5-mini` and
//! `OPENAI_EMBEDDING_MODEL` to `text-embedding-3-small`.
use futures::StreamExt;
use open_client::mime;
use open_client::openai::ask::OpenAI;
use open_client::openai::embed::OpenAIEmbedding;
use open_client::openai::error::OpenAIResponseError;
use open_client::openai::types::request::{InlineData, MultimodalResponse, ReasoningEffort, Tool};
use open_client::openai::types::sessions::Session;
use open_client::openai::utils::{
    OpenAISchema, execute_function_calls, openai_function, openai_schema,
};
use serde::Deserialize;

const IMAGE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lda.png");

fn ai(sys_prompt: Option<&str>) -> OpenAI {
    let model = std::env::var("OPENAI_MODEL").unwrap_or("gpt-5-mini".into());
    OpenAI::from_env(model, sys_prompt.map(Into::into)).unwrap()
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn conversation_keeps_history() {
    let ai = ai(Some("Answer in one short sentence."));
    let mut session = Session::new(10);
    ai.ask(session.ask("My name is Ferris.")).await.unwrap();
    let response = ai.ask(session.ask("What is my name?")).await.unwrap();
    assert!(response.get_chat().get_text_no_think("").contains("Ferris"));
    assert_eq!(session.get_history_length(), 4);
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn json_mode() {
    #[openai_schema]
    #[derive(Deserialize)]
    struct Sentiments {
        positive: Vec<String>,
        negative: Vec<String>,
    }
    let ai = ai(Some("Classify the given words")).set_json_mode(Sentiments::openai_schema());
    let mut session = Session::new(2).set_remember_reply(false);
    let response = ai.ask(session.ask("Joy, Fear, Hope, Anger")).await.unwrap();
    let sentiments: Sentiments = response.get_json().unwrap();
    assert!(sentiments.positive.contains(&"Joy".to_string()));
    assert!(sentiments.negative.contains(&"Fear".to_string()));
    assert_eq!(session.get_history_length(), 0);
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn streamed_text() {
    let mut session = Session::new(10);
    session.ask("Count from 1 to 20, separated by spaces.");
    let mut stream = ai(None)
        .ask_as_stream_with_extractor(session, |session, _| {
            session.get_last_chat().unwrap().get_text_no_think("")
        })
        .await
        .unwrap();
    let mut reply = String::new();
    while let Some(text) = stream.next().await {
        reply = text.unwrap();
    }
    assert!(reply.contains("20"));
    assert_eq!(stream.get_session().get_history_length(), 2);
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn streamed_parallel_calls() {
    // Only its schema is used, the calls are answered below.
    #[openai_function]
    /// Returns the weather of a city.
    #[allow(dead_code)]
    fn get_weather(city: String) -> String {
        city
    }
    let ai = ai(None).set_tool(Tool::Function(get_weather::openai_schema()));
    let mut session = Session::new(10);
    session.ask("What is the weather in Paris and in London? Call the function for both.");
    let mut stream = ai.ask_as_stream(session).await.unwrap();
    while let Some(chunk) = stream.next().await {
        chunk.unwrap();
    }
    let mut session = stream.get_session_owned();

    // Arguments streamed in pieces are assembled into complete calls
    let calls: Vec<(String, String)> = session
        .get_pending_function_calls()
        .iter()
        .map(|call| {
            let (city,) = get_weather::parse_arguments(&call.args().unwrap()).unwrap();
            (call.id().clone(), city)
        })
        .collect();
    assert_eq!(calls.len(), 2, "{calls:?}");
    for (id, city) in calls {
        let weather = if city.contains("Paris") { 21 } else { 14 };
        session
            .add_function_response(id, format!("{weather} degrees Celsius"))
            .unwrap();
    }
    let reply = ai.ask(&mut session).await.unwrap();
    let reply = reply.get_chat().get_text_no_think("");
    assert!(reply.contains("21") && reply.contains("14"), "{reply}");
}

#[openai_function]
/// Returns the secret number.
fn secret_number() -> u32 {
    7319
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn function_calling_loop() {
    let ai = ai(None).set_tool(Tool::Function(secret_number::openai_schema()));
    let mut session = Session::new(10);
    let mut response = ai
        .ask(session.ask("What is the secret number plus one?"))
        .await
        .unwrap();
    while !session.get_pending_function_calls().is_empty() {
        let results = execute_function_calls!(session, secret_number);
        assert!(results[0].is_some());
        response = ai.ask(&mut session).await.unwrap();
    }
    assert!(response.get_chat().get_text_no_think("").contains("7320"));
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn image_in_prompt() {
    let chart = InlineData::from_path(IMAGE, mime::IMAGE_PNG).await.unwrap();
    let mut session = Session::new(2);
    session.ask("What kind of chart is this?").ask(chart);
    let response = ai(None).ask(&mut session).await.unwrap();
    let reply = response.get_chat().get_text_no_think("").to_lowercase();
    assert!(reply.contains("scatter"), "{reply}");
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn image_in_function_response() {
    #[openai_function]
    /// Takes a screenshot of the user's screen.
    #[allow(dead_code)]
    fn take_screenshot() {}

    let ai = ai(None).set_tool(Tool::Function(take_screenshot::openai_schema()));
    let mut session = Session::new(10);
    ai.ask(session.ask("Take a screenshot. What kind of chart is on my screen?"))
        .await
        .unwrap();
    let call_id = session.get_pending_function_calls()[0].id().clone();
    let screenshot = InlineData::from_path(IMAGE, mime::IMAGE_PNG).await.unwrap();
    session
        .add_function_response(
            call_id,
            MultimodalResponse::new(vec!["Screenshot taken".into(), screenshot.into()]),
        )
        .unwrap();
    let response = ai.ask(&mut session).await.unwrap();
    let reply = response.get_chat().get_text_no_think("").to_lowercase();
    assert!(reply.contains("scatter"), "{reply}");
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn reasoning_effort() {
    let ai = ai(None).set_reasoning_effort(ReasoningEffort::Low);
    let mut session = Session::new(2);
    let response = ai
        .ask(session.ask("How many letters r are in strawberry? Answer with a digit."))
        .await
        .unwrap();
    assert!(response.get_chat().get_text_no_think("").contains('3'));
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn wrong_api_key() {
    let ai = ai(None).set_api_key("wrong_api_key");
    match ai.ask(Session::new(2).ask("Hi")).await {
        Err(OpenAIResponseError::StatusNotOk(e)) => {
            assert!(e.status.is_client_error());
            assert!(!e.error.message.is_empty());
        }
        other => panic!("Expected StatusNotOk, got {other:?}"),
    }
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn embeddings() {
    let model = std::env::var("OPENAI_EMBEDDING_MODEL").unwrap_or("text-embedding-3-small".into());
    let embedder = OpenAIEmbedding::from_env(model)
        .unwrap()
        .set_dimensions(256);
    let response = embedder
        .embed_texts(vec![
            "I love programming in Rust".into(),
            "Rust is my favourite programming language".into(),
            "It is going to rain tomorrow".into(),
        ])
        .await
        .unwrap();
    let embeddings = response.embeddings();
    assert!(embeddings.iter().all(|e| e.dimension() == 256));
    let similarity = |a: usize, b: usize| -> f32 {
        let (a, b) = (embeddings[a].values(), embeddings[b].values());
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / norm(a) / norm(b)
    };
    assert!(similarity(0, 1) > similarity(0, 2));
}
