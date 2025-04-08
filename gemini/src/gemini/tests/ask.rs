use crate::gemini::ask::{Gemini, GeminiResponseStream};
use crate::gemini::types::{Part, Session, SystemInstruction};
use futures::StreamExt;
use serde_json::json;

#[actix_web::test]
async fn ask_string() {
    let mut session = Session::new(5);
    let response = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-1.5-flash".to_string(),
        None,
    )
    .ask(session.ask_string("Hi".to_string()))
    .await
    .unwrap();
    println!("{}", response);
}

#[actix_web::test]
async fn ask_string_for_json() {
    let mut session = Session::new(5);
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
    println!("{}", GeminiResponseStream::parse_json(response).unwrap());
}

#[actix_web::test]
async fn ask_streamed() {
    let mut session = Session::new(5);
    session.ask_string("How are you".to_string());
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-1.5-flash".to_string(),
        None,
    );
    let mut response_stream = ai.ask_as_stream(&mut session).await.unwrap();
    while let Some(response) = response_stream.next().await {
        println!("{}", response.unwrap());
    }
    println!("Last reply: {}", session.last_reply().unwrap());
}
