use crate::gemini::{ask::Gemini, error::GeminiResponseError, types::sessions::Session};

#[tokio::test]
async fn status_not_ok_test() {
    let mut session = Session::new(6);
    let response = Gemini::new("wrong_api_key", "gemini-2.5-flash", None)
        .ask(session.ask("Hi"))
        .await;
    match response {
        Err(GeminiResponseError::StatusNotOk(e)) => assert!(e.error.code == 400),
        _ => panic!("Expected invalid api key error"),
    }
}
