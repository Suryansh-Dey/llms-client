# Basic usage
```rust
use crate::gemini::ask::Gemini;
use crate::gemini::types::request::{Part, SystemInstruction, Tool};
use crate::gemini::types::sessions::Session;
use gemini_client_api::gemini::utils::MarkdownToParts;
use futures::StreamExt;
use serde_json::json;

async fn see_markdown() {
    let mut session = Session::new(6);
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-1.5-flash".to_string(),
        None,
    );

    let response1 = ai.ask(session.ask_string("Hi, can you tell me which one of two bowls has more healty item?".to_string())).await.unwrap();
    println!("{}", response1.get_text(""));

    let parser = MarkdownToParts::new("Here is their image ![but fire](https://th.bing.com/th?id=ORMS.0ba175d4898e31ae84dc62d9cd09ec84&pid=Wdp&w=612&h=304&qlt=90&c=1&rs=1&dpr=1.5&p=0). Thanks by the way").await;
    let default_mime_type = String::from("image/png");
    let parts = parser.process(default_mime_type);

    let response2 = ai.ask(session.ask(parts))
    .await
    .unwrap();

    println!("{}", response2.get_text(""));
}

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
    println!("{}", response.get_text(""));
}

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
    println!("Complete reply: {}", session.last_reply_text("").unwrap());
}

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
    println!("Complete reply: {:#?}", json!(session.last_reply().unwrap()));
}
```
