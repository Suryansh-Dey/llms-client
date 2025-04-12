use crate::gemini::ask::Gemini;
use crate::gemini::types::request::{Part, SystemInstruction, Tool};
use crate::gemini::types::sessions::Session;
use futures::StreamExt;
use serde_json::{Value, json};

#[actix_web::test]
async fn ask_string() {
    let mut session = Session::new(6);
    let response = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-1.5-flash".to_string(),
        None,
    )
    .ask(session.ask_string("Hi".to_string()))
    .await
    .unwrap();
    println!("{}", response.get_text(""));
}

#[actix_web::test]
async fn ask_string_for_json() {
    let mut session = Session::new(6);
    let response = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-1.5-flash".to_string(),
        Some(SystemInstruction::new(&[Part::text("Calssify the given words".to_string())])),
    )
    .set_json_mode(json!({
        "type": "object",
        "properties": {
            "positive":{
                "type":"array",
                "items":{"type":"string"}
            },
            "negetive":{
                "type":"array",
                "items":{"type":"string"}
            }
        }
    }))
    .ask(session.ask_string("[\"Joy\", \"Success\", \"Love\", \"Hope\", \"Confidence\", \"Peace\", \"Victory\", \"Harmony\", \"Inspiration\", \"Gratitude\", \"Prosperity\", \"Strength\", \"Freedom\", \"Comfort\", \"Brilliance\" \"Fear\", \"Failure\", \"Hate\", \"Doubt\", \"Pain\", \"Suffering\", \"Loss\", \"Anxiety\", \"Despair\", \"Betrayal\", \"Weakness\", \"Chaos\", \"Misery\", \"Frustration\", \"Darkness\"]
".to_string()))
    .await
    .unwrap();

    let json: Value = response.get_json().unwrap();
    println!("{}", json);
}

#[actix_web::test]
async fn ask_streamed() {
    let mut session = Session::new(6);
    session.ask_string("How are you".to_string());
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-1.5-flash".to_string(),
        None,
    );
    let mut response_stream = ai.ask_as_stream(&mut session).await.unwrap();
    while let Some(response) = response_stream.next().await {
        println!("{}", response.unwrap().get_text(""));
    }
    println!(
        "Complete reply: {}",
        session.get_last_reply_text("").unwrap()
    );
}

#[actix_web::test]
async fn ask_with_tools() {
    let mut session = Session::new(6);
    session.ask_string("find sum of first 100 prime number using code".to_string());
    let mut ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.0-flash".to_string(),
        None,
    );
    ai.set_tools(Some(vec![Tool::code_execution(json!({}))]));
    let mut response_stream = ai.ask_as_stream(&mut session).await.unwrap();
    while let Some(response) = response_stream.next().await {
        if let Ok(response) = response {
            println!("{}", response.get_text(""));
        }
    }
    println!(
        "Complete reply: {:#?}",
        json!(session.get_last_reply().unwrap())
    );
}
