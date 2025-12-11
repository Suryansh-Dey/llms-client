use crate::ask::ChatGpt;
use crate::types::sessions::Session;
use serde_json::json;

#[tokio::test]
async fn ask_plain_text() {
    let mut session = Session::new(6);
    session.ask("Say hi in one short sentence.");
    let response = ChatGpt::new(
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not found"),
        "gpt-4o-mini",
        None,
    )
    .ask(&mut session)
    .await
    .unwrap();
    println!("response: {}", response.text());
}

#[tokio::test]
async fn ask_structured_json() {
    let mut session = Session::new(4).set_remember_reply(false);
    session.ask("Give me a user profile.");

    let response = ChatGpt::new(
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not found"),
        "gpt-4o-mini",
        None,
    )
    .set_json_mode(json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "location": {"type": "string"}
        },
        "required": ["name", "age", "location"]
    }))
    .ask(&mut session)
    .await
    .unwrap();

    println!("{}", response.text());
}
