//! Prints the reply while it is generated.
use open_client::chat::AI;
use open_client::chat::session::Session;
use open_client::futures::StreamExt;
use std::io::{Write, stdout};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Optional: token usage is sent too. OpenAI sends it in the last chunk, others in every chunk.
    let ai = AI::from_env("gpt-5-mini", None)?.include_usage(true);
    let mut session = Session::new(10);
    session.ask("Write a short poem about the borrow checker.");

    // The stream borrows the session, and adds the reply to it once the stream ends.
    let mut stream = ai.stream(&mut session).await?;
    // Each chunk has only the newly generated text. `stream.message()` has the reply so far.
    let mut usage = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        print!("{}", chunk.text());
        stdout().flush()?;
        usage = chunk.usage.or(usage);
    }
    if let Some(usage) = usage {
        println!(
            "\n\nTokens: {} in, {} out",
            usage.prompt_tokens, usage.completion_tokens
        );
    }
    drop(stream);
    println!("Messages in session: {}", session.len());
    Ok(())
}
