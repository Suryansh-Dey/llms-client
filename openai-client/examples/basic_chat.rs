//! A conversation. The session keeps the history, so follow-up questions have context.
//!
//! Works with any OpenAI compatible API: set `OPENAI_API_KEY`, and `OPENAI_BASE_URL` for APIs
//! other than OpenAI (then also change the model). See `providers.rs` to set them in code.
use open_client::openai::ask::OpenAI;
use open_client::openai::types::sessions::Session;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = OpenAI::from_env(
        "gpt-5-mini",
        Some("You are a travel guide. Answer in 2 sentences.".into()),
    )?;
    // Keeps the last 10 chats, prompts and replies.
    let mut session = Session::new(10);

    for prompt in [
        "Suggest one city in Japan to visit in autumn.",
        // Only makes sense with the history
        "What food should I try there?",
        "Summarise your suggestions in one line.",
    ] {
        println!("User: {prompt}");
        let response = ai.ask(session.ask(prompt)).await?;
        // Thoughts, if the model returns any, are left out
        println!("AI: {}\n", response.get_chat().get_text_no_think(""));
    }
    println!("Chats in history: {}", session.get_history_length());
    Ok(())
}
