# Gemini
## Installation
```bash
cargo add gemini-client-api

```
## Overview
A Rust library to use Google's Gemini API with macro super powers! It is extremely flexible and modular to integrate with any framework.  
For example, since Actix supports stream of `Result<Bytes, Error>` for response streaming, you can get it directly instead of making a wrapper stream around a response stream of futures, which is a pain.

### Features
- Automatic context management
- Automatic function calling. Trust me!
- Automatic JSON schema generation
- Inbuilt markdown to parts parser enables AI to see markdown images or files, even if they are from your device storage!
- Vision to see images
- Code execution by Gemini
- File reading like PDF or any document, even audio files like MP3
- Function call support
- Thinking and Safety setting

## Basic usage
```rust
use gemini_client_api::gemini::{
    ask::Gemini,
    types::request::{Tool, ThinkingConfig},
    types::sessions::Session,
    utils::MarkdownToParts,
};
use futures::StreamExt;
use serde::Deserialize;
use std::error::Error;
use serde_json::json;

async fn see_markdown() {
    let mut session = Session::new(6);
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.0-flash",
        None,
    );
    let response1 = ai.ask(session.ask_string("Hi, can you tell me which one of two bowls has more healty item?")).await.unwrap();
    //Question and reply both automatically gets stored in `session` for context.
    println!("{}", response1.get_chat().get_text_no_think(""));

    let parts = MarkdownToParts::new("Here is their ![image](https://th.bing.com/th?id=ORMS.0ba175d4898e31ae84dc62d9cd09ec84&pid=Wdp&w=612&h=304&qlt=90&c=1&rs=1&dpr=1.5&p=0). Thanks by the way", |_|mime::IMAGE_PNG)
        .await.process();
    //Can even read from file path of files on your device!

    let response2 = ai.ask(session.ask(parts))
    .await.unwrap();

    println!("{}", response2.get_chat().get_text_no_think(""));
}

async fn ask_string_for_json_with_struct() {
    #[derive(Debug, Deserialize)]
    #[gemini_schema]
    struct Schema {
        positive: Vec<String>,
        negative: Vec<String>,
    }
    let mut session = Session::new(6).set_remember_reply(false);
    let response = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.5-flash",
        Some("Classify the given words".into()),
    )
    .set_json_mode(Schema::gemini_schema())
    .ask(session.ask_string(r#"["Joy", "Success", "Love", "Hope", "Confidence", "Peace", "Victory", "Harmony", "Inspiration", "Gratitude", "Prosperity", "Strength", "Freedom", "Comfort", "Brilliance", "Fear", "Failure", "Hate", "Doubt", "Pain", "Suffering", "Loss", "Anxiety", "Despair", "Betrayal", "Weakness", "Chaos", "Misery", "Frustration", "Darkness"]"#))
    .await
    .unwrap();

    let json: Schema = response.get_json().unwrap();
    println!(
        "positives:{:?}\nnegatives:{:?}",
        json.positive, json.negative
    )
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
        println!("{}", response.unwrap().get_chat().get_text_no_think(""));
    }
}

async fn ask_streamed_with_tools() {
    let mut session = Session::new(6);
    session.ask_string("find sum of first 100 prime number using code");
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.5-flash",
        None,
    )
    .set_tools(vec![Tool::code_execution(json!({}))]);
    let mut response_stream = ai
        .ask_as_stream(session).await.unwrap();
    while let Some(response) = response_stream.next().await {
        if let Ok(response) = response {
            println!("{}", response.get_chat().get_text_no_think(""));
        }
    }
    println!(
        "Complete reply: {:#?}",
        json!(response_stream.get_session().get_last_chat().unwrap())
    );
}

use gemini_client_api::gemini::utils::{GeminiSchema, execute_function_calls, gemini_function};
///Lists files in my dir
async fn list_files(
    ///Path to the dir
    path: String,
) -> Result<String, Box<dyn Error>> {
    Ok(std::fs::read_dir(path)?
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect::<Vec<String>>()
        .join(", "))
}

async fn ask_with_function_calls() {
    let mut session = Session::new(10);
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-2.5-flash",
        None,
    )
    .set_tools(vec![Tool::functionDeclarations(vec![
        list_files::gemini_schema(),
    ])]);
    session.ask_string("What files I have in current directory");
    let response = ai.ask(&mut session).await.unwrap(); //Received a function call
    let result = execute_function_calls!(session, list_files); //don't update session if Error
    println!("function output: {:?}", result);
    if result.len() != 0 {
        //If any function call at all happened
        let response = ai.ask(&mut session).await.unwrap(); //Providing output of the function call and continue
        println!("{:?}", response.get_chat().parts());
    } else {
        println!("{:?}", response.get_chat().parts());
    }
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
    println!("{}", response.get_chat().get_text_no_think(""));
}
```
# TODO
1. Do the same for chatGPT
