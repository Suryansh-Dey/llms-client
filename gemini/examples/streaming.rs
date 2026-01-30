use futures::StreamExt;
use gemini_client_api::gemini::ask::Gemini;
use gemini_client_api::gemini::types::sessions::Session;
use std::env;
use std::io::{stdout, Write};

#[tokio::main]
async fn main() {
    let mut session = Session::new(10);
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let ai = Gemini::new(api_key, "gemini-2.5-flash", None);

    println!("--- Streaming Example ---");
    let prompt = "Write a short poem about crab-like robots on Mars.";
    println!("User: {}\n", prompt);
    print!("Gemini: ");
    stdout().flush().unwrap();

    // Start a streaming request
    let mut response_stream = ai.ask_as_stream(session.ask(prompt).clone()).await.unwrap();

    while let Some(chunk_result) = response_stream.next().await {
        match chunk_result {
            Ok(response) => {
                // Get the text from the current chunk
                let text = response.get_chat().get_text_no_think("");
                print!("{}", text);
                stdout().flush().unwrap();
            }
            Err(e) => {
                eprintln!("\nError receiving chunk: {:?}", e);
                break;
            }
        }
    }

    println!("\n\n--- Stream Complete ---");
    // Note: The session passed to ask_as_stream is updated as you exhaust the stream.
}
