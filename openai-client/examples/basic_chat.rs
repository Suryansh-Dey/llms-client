//! A conversation. The session keeps the history, so follow-up questions have context.
//!
//! Works with any OpenAI compatible API: set `OPENAI_API_KEY`, and `OPENAI_BASE_URL` for APIs
//! other than OpenAI (then also change the model). See `providers.rs` to set them in code.
use open_client::chat::AI;
use open_client::chat::session::Session;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = AI::from_env(
        "gpt-5-mini",
        Some("You are a travel guide. Answer in 2 sentences.".into()),
    )?;
    // Keeps the last 10 messages, prompts and replies.
    let mut session = Session::new(10);

    for prompt in [
        "Suggest one city in Japan to visit in autumn.",
        // Only makes sense with the history
        "What food should I try there?",
        "Summarise your suggestions in one line.",
    ] {
        println!("User: {prompt}");
        let completion = ai.ask(session.ask(prompt)).await?;
        println!("AI: {}\n", completion.text());
    }
    println!("Messages in history: {}", session.len());
    Ok(())
}
