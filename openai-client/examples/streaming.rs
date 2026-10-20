use futures::StreamExt;
use open_client::openai::ask::OpenAI;
use open_client::openai::types::sessions::Session;
use std::env;
use std::io::{Write, stdout};

#[tokio::main]
async fn main() {
    let mut session = Session::new(10);
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let mut ai = OpenAI::new(api_key, "gpt-5-mini", None);
    // Optional: receive token usage in the last chunk
    ai.set_generation_config()["stream_options"] = serde_json::json!({"include_usage": true});

    let prompt = "Write a poem about crab-like robots on Mars.";
    println!("User: {}\n", prompt);
    print!("AI: ");
    stdout().flush().unwrap();
    session.ask(prompt);

    // Start a streaming request
    let mut response_stream = ai.ask_as_stream(session).await.unwrap();

    while let Some(chunk_result) = response_stream.next().await {
        match chunk_result {
            Ok(response) => {
                // Get the text from the current chunk
                let text = response.get_chat().get_text_no_think("");
                print!("{text}");
                stdout().flush().unwrap();
                if let Some(usage) = response.usage {
                    println!("\n\nUsage: {usage}");
                }
            }
            Err(e) => {
                eprintln!("\nError receiving chunk: {e}",);
                break;
            }
        }
    }

    println!("\n\n--- Stream Complete ---");
    // Note: The session passed to ask_as_stream is updated as you exhaust the stream.
    session = response_stream.get_session_owned();
    println!("Updated session: {session:?}")
}
#[tokio::test]
async fn with_extractor() {
    let mut session = Session::new(10);
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let ai = OpenAI::new(api_key, "gpt-5-mini", None);

    println!("--- Streaming Example ---");
    let prompt = "Write a poem about crab-like robots on Mars.";
    println!("User: {prompt}\n");
    println!("AI: ");
    session.ask(prompt);

    // Start a streaming request
    let mut response_stream = ai
        .ask_as_stream_with_extractor(session, |session, _response| {
            session.get_last_chat().unwrap().get_text_no_think("\n")
        })
        .await
        .unwrap();

    while let Some(chunk_result) = response_stream.next().await {
        match chunk_result {
            Ok(response) => {
                // Get the text from the current chunk
                println!("Complete poem: {response}\n");
            }
            Err(e) => {
                eprintln!("\nError receiving chunk: {e}",);
                break;
            }
        }
    }

    println!("\n\n--- Stream Complete ---");
    // Note: The session passed to ask_as_stream is updated as you exhaust the stream.
    session = response_stream.get_session_owned();
    println!("Updated session: {session:?}")
}
