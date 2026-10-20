use crate::openai::{
    ask::OpenAI,
    embed::OpenAIEmbedding,
    error::{ApiError, OpenAIResponseError},
    types::sessions::Session,
};
use serde_json::json;

/// Stand-in for a real key. No error or `Debug` output may ever contain it.
const SECRET: &str = "SECRET_KEY_THAT_MUST_NOT_LEAK";

#[test]
fn api_error_formats() {
    let openai = ApiError::from_body(
        r#"{"error": {"message": "Bad key", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}}"#,
    );
    assert_eq!(openai.message, "Bad key");
    assert_eq!(openai.r#type.as_deref(), Some("invalid_request_error"));
    assert_eq!(openai.param, None);
    assert_eq!(openai.code, Some(json!("invalid_api_key")));

    // Ollama
    assert_eq!(
        ApiError::from_body(r#"{"error": "model not found"}"#).message,
        "model not found"
    );
    // vLLM
    let vllm = ApiError::from_body(r#"{"object": "error", "message": "Too long", "code": 400}"#);
    assert_eq!(vllm.message, "Too long");
    assert_eq!(vllm.code, Some(json!(400)));
    // FastAPI validation error
    let fastapi =
        ApiError::from_body(r#"{"detail": [{"loc": ["body"], "msg": "field required"}]}"#);
    assert!(fastapi.message.contains("field required"));
    // Gemini
    let gemini = ApiError::from_body(
        r#"[{"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}}]"#,
    );
    assert_eq!(gemini.message, "API key not valid");
    // Not JSON
    assert_eq!(
        ApiError::from_body(" Bad Gateway \n").message,
        "Bad Gateway"
    );
}

#[tokio::test]
async fn api_key_is_not_in_error() {
    // The request never reaches the API, so `reqwest` reports the request URL itself
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_nanos(1))
        .build()
        .unwrap();
    let mut session = Session::new(6);
    let error = OpenAI::new_with_client(SECRET, "gpt-5-mini", None, client)
        .set_base_url(format!("https://127.0.0.1:1/{SECRET}"))
        .ask(session.ask("Hi"))
        .await
        .expect_err("a 1ns timeout should fail");
    assert!(matches!(error, OpenAIResponseError::ReqwestError(_)));
    assert!(!format!("{error}").contains(SECRET));
    assert!(!format!("{error:?}").contains(SECRET));
}

#[tokio::test]
async fn reqwest_error_url_is_stripped() {
    let error = reqwest::Client::new()
        .get(format!("https://127.0.0.1:1/?key={SECRET}"))
        .send()
        .await
        .expect_err("connecting to port 1 should fail");
    // Precondition: this is why the URL has to be stripped
    assert!(
        format!("{error}").contains(SECRET),
        "reqwest is expected to print the request URL"
    );

    let error = OpenAIResponseError::from(error);
    assert!(!format!("{error}").contains(SECRET));
    assert!(!format!("{error:?}").contains(SECRET));
}

#[test]
fn api_key_is_not_in_debug_output() {
    let ai = OpenAI::new(SECRET, "gpt-5-mini", None);
    assert!(!format!("{ai:?}").contains(SECRET));
    assert!(format!("{ai:?}").contains("gpt-5-mini"));

    let embedding = OpenAIEmbedding::new(SECRET, "text-embedding-3-small");
    assert!(!format!("{embedding:?}").contains(SECRET));
    assert!(format!("{embedding:?}").contains("text-embedding-3-small"));
}
