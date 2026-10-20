//! Prints the reply while it is generated.
use open_client::futures::StreamExt;
use open_client::openai::ask::OpenAI;
use open_client::openai::types::sessions::Session;
use open_client::serde_json::json;
use std::io::{Write, stdout};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ai = OpenAI::from_env("gpt-5-mini", None)?;
    // Optional: token usage is sent too. OpenAI sends it in the last chunk, others in every chunk.
    ai.set_generation_config()["stream_options"] = json!({"include_usage": true});

    let mut session = Session::new(10);
    session.ask("Write a short poem about the borrow checker.");

    // The session is moved into the stream, which adds the reply to it as chunks arrive. It is
    // handed back with the error if the request fails.
    let mut stream = ai.ask_as_stream(session).await.map_err(|(_, e)| e)?;
    // Each chunk has only the newly generated text. See `ask_as_stream_with_extractor()` to get
    // something else for each chunk, e.g. the whole reply so far.
    let mut usage = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        print!("{}", chunk.get_chat().get_text_no_think(""));
        stdout().flush()?;
        usage = chunk.usage.or(usage);
    }

    if let Some(usage) = usage {
        println!("\n\nUsage: {usage}");
    }
    let session = stream.get_session_owned();
    println!("Chats in session: {}", session.get_history_length());
    Ok(())
}
