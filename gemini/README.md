**Main function should use `#[actix_web::main]` instead of `#[tokio::main]`.** Don't worry, your all tokio functions will work the same!
# Basic usage
```rust
use gemini_client_api::gemini::{
    ask::Gemini,
    types::request::{SystemInstruction, Tool},
    types::sessions::Session,
    utils::MarkdownToParts,
};
use futures::StreamExt;
use serde_json::json;

async fn see_markdown() {
    let mut session = Session::new(6);
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.0-flash",
        None,
    );

    let response1 = ai.ask(session.ask_string("Hi, can you tell me which one of two bowls has more healty item?")).await.unwrap();
    println!("{}", response1.get_text("")); //Question and reply both automatically gets stored in `session` for context.

    let parser = MarkdownToParts::new("Here is their ![image](https://th.bing.com/th?id=ORMS.0ba175d4898e31ae84dc62d9cd09ec84&pid=Wdp&w=612&h=304&qlt=90&c=1&rs=1&dpr=1.5&p=0). Thanks by the way", |_|"image/png".to_string()).await;
    //Can even read from file path of files on your device!
    let parts = parser.process();

    let response2 = ai.ask(session.ask(parts))
    .await
    .unwrap();

    println!("{}", response2.get_text(""));
}

async fn ask_string_for_json() {
    let mut session = Session::new(6);
    session.set_remember_reply(false);
    let response = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.0-flash",
        Some(SystemInstruction::from_str("Calssify the given words")),
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
"))
    .await
    .unwrap();
    println!("{}", response.get_text(""));
}

async fn ask_streamed() {
    let mut session = Session::new(6);
    session.ask_string("How are you");
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.5-pro-exp-03-25",
        None,
    );
    let mut response_stream = ai.ask_as_stream(&mut session).await.unwrap();
    while let Some(response) = response_stream.next().await {
        println!("{}", response.unwrap().get_text(""));
    }
    println!("Complete reply: {}", session.get_last_message_text("").unwrap());
}

async fn ask_streamed_with_tools() {
    let mut session = Session::new(6);
    session.ask_string("find sum of first 100 prime number using code");
    let mut ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.0-flash",
        None,
    );
    ai.set_tools(Some(vec![Tool::code_execution(json!({}))]));
    let mut response_stream = ai.ask_as_stream(&mut session).await.unwrap();
    while let Some(response) = response_stream.next().await {
        if let Ok(response) = response {
            println!("{}", response.get_text(""));
        }
    }
    println!("Complete reply: {:#?}", json!(session.get_last_message().unwrap()));
}
```
