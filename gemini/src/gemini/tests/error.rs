use crate::gemini::{
    ask::Gemini,
    embed::GeminiEmbedding,
    error::{GeminiResponseError, Status},
    types::sessions::Session,
};

/// Stand-in for a real key. No error or `Debug` output may ever contain it.
const SECRET: &str = "SECRET_KEY_THAT_MUST_NOT_LEAK";

#[tokio::test]
async fn status_not_ok_test() {
    let mut session = Session::new(6);
    let response = Gemini::new("wrong_api_key", "gemini-2.5-flash", None)
        .ask(session.ask("Hi"))
        .await;
    match response {
        Err(GeminiResponseError::StatusNotOk(e)) => {
            assert!(e.error.status == Status::InvalidArgument)
        }
        _ => panic!("Expected invalid api key error"),
    }
}

#[tokio::test]
async fn api_key_is_not_in_error() {
    // The API rejects the key, so the error is built from the response body
    let mut session = Session::new(6);
    let error = Gemini::new(SECRET, "gemini-2.5-flash", None)
        .ask(session.ask("Hi"))
        .await
        .expect_err("a bogus api key should be rejected");
    assert!(!format!("{error}").contains(SECRET));
    assert!(!format!("{error:?}").contains(SECRET));

    // The request never reaches the API, so `reqwest` reports the request URL itself
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_nanos(1))
        .build()
        .unwrap();
    let mut session = Session::new(6);
    let error = Gemini::new_with_client(SECRET, "gemini-2.5-flash", None, client)
        .ask(session.ask("Hi"))
        .await
        .expect_err("a 1ns timeout should fail");
    assert!(matches!(error, GeminiResponseError::ReqwestError(_)));
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

    let error = GeminiResponseError::from(error);
    assert!(!format!("{error}").contains(SECRET));
    assert!(!format!("{error:?}").contains(SECRET));
}

#[test]
fn api_key_is_not_in_debug_output() {
    let ai = Gemini::new(SECRET, "gemini-2.5-flash", None);
    assert!(!format!("{ai:?}").contains(SECRET));
    assert!(format!("{ai:?}").contains("gemini-2.5-flash"));

    let embedding = GeminiEmbedding::new(SECRET, "gemini-embedding-001");
    assert!(!format!("{embedding:?}").contains(SECRET));
    assert!(format!("{embedding:?}").contains("gemini-embedding-001"));
}
