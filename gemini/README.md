# Overview
A Rust library to use Google's Gemini API. It is extremely flexible and modular to integrate with any framework.  
For example, since Actix supports stream of `Result<Bytes, Error>` for response streaming, you can get it directly instead of making a wrapper stream around a response stream of futures, which is a pain.

### Features
- Automatic context management
- Inbuilt markdown to parts parser enables AI to see markdown images or files, even if they are from your device storage!
- Vision to see images
- Code execution by Gemini
- File reading like PDF or any document, even audio files like MP3
- Function call support
- Thinking and Safety setting

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

    let parts = MarkdownToParts::new("Here is their ![image](https://th.bing.com/th?id=ORMS.0ba175d4898e31ae84dc62d9cd09ec84&pid=Wdp&w=612&h=304&qlt=90&c=1&rs=1&dpr=1.5&p=0). Thanks by the way", |_|mime::IMAGE_PNG)
        .await.process();
    //Can even read from file path of files on your device!

    let response2 = ai.ask(session.ask(parts))
    .await.unwrap();

    println!("{}", response2.get_text(""));
}

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

async fn ask_streamed() {
    let mut session = Session::new(6);
    session.ask_string("How are you");
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.5-flash",
        None,
    );
    let mut response_stream = ai
        .ask_as_stream(session).await.unwrap();
    while let Some(response) = response_stream.next().await {
        println!("{}", response.unwrap().get_text(""));
    }
}

async fn ask_streamed_with_tools() {
    let mut session = Session::new(6);
    session.ask_string("find sum of first 100 prime number using code");
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.0-flash",
        None,
    )
    .set_tools(vec![Tool::code_execution(json!({}))]);
    let mut response_stream = ai
        .ask_as_stream(session).await.unwrap();
    while let Some(response) = response_stream.next().await {
        if let Ok(response) = response {
            println!("{}", response.get_text(""));
        }
    }
    println!(
        "Complete reply: {:#?}",
        json!(response_stream.get_session().get_last_message().unwrap())
    );
}

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
    println!("{}", response.get_text_no_think(""));
}
```
