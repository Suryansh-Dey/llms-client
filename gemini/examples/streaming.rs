use futures::StreamExt;
use gemini_client_api::gemini::ask::Gemini;
use gemini_client_api::gemini::types::sessions::Session;
use std::env;
use std::io::{Write, stdout};

#[tokio::main]
async fn main() {
    let mut session = Session::new(10);
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let ai = Gemini::new(api_key, "gemini-2.5-flash", None);

    println!("--- Streaming Example ---");
    let prompt = "Write a poem about crab-like robots on Mars.";
    println!("User: {}\n", prompt);
    print!("Gemini: ");
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
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let ai = Gemini::new(api_key, "gemini-2.5-flash", None);

    println!("--- Streaming Example ---");
    let prompt = "Write a poem about crab-like robots on Mars.";
    println!("User: {prompt}\n");
    println!("Gemini: ");
    session.ask(prompt);

    // Start a streaming request
    let mut response_stream = ai
        .ask_as_stream_with_extractor(session, |session, _gemini_response| {
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
