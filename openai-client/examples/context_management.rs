//! Keeping a long conversation within a token budget, with a custom `OverflowHandler`.
//!
//! The handler estimates tokens from characters. It calibrates the estimate with the prompt size
//! the API reports in `usage`, and forgets the oldest messages when the history grows over budget.
use open_client::chat::AI;
use open_client::chat::message::Message;
use open_client::chat::session::{HistoryTrimmer, OverflowHandler, Session, SlidingWindow};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

fn characters(history: &VecDeque<Message>) -> usize {
    history.iter().map(|message| message.text().len()).sum()
}

#[derive(Debug, Serialize, Deserialize)]
struct TokenBudget {
    max_tokens: usize,
    /// Starts at the common estimate of 4, then learnt from `usage`.
    characters_per_token: f64,
    /// Also caps the count of messages.
    window: SlidingWindow,
}
impl TokenBudget {
    fn tokens(&self, history: &VecDeque<Message>) -> usize {
        (characters(history) as f64 / self.characters_per_token) as usize
    }
}
impl OverflowHandler for TokenBudget {
    fn handle(&mut self, history: &mut HistoryTrimmer<'_>) {
        // Might forget more than one message, e.g. tool calls along with their results.
        // Stops (returns None) instead of forgetting the latest message.
        while self.tokens(history.history()) > self.max_tokens && history.forget_oldest().is_some()
        {
        }
        // Strategies combine by calling each other.
        self.window.handle(history);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = AI::from_env("gpt-5-mini", Some("Answer in 2 sentences.".into()))?;
    let system_prompt_characters = ai.get_system_prompt().map_or(0, str::len);
    let mut session = Session::new(TokenBudget {
        max_tokens: 200,
        characters_per_token: 4.0,
        window: SlidingWindow::new(50),
    });

    for prompt in [
        "What is ownership in Rust?",
        "How is borrowing related to it?",
        "What are lifetimes?",
        "Why does Rust need all three?",
        "Which of my questions do you still remember?",
    ] {
        session.ask(prompt);
        let sent = system_prompt_characters + characters(session.history());
        let completion = ai.ask(&mut session).await?;
        println!("User: {prompt}\nAI: {}", completion.text());

        if let Some(usage) = &completion.usage {
            // Rough, as `prompt_tokens` also counts the formatting of messages.
            session.overflow_handler_mut().characters_per_token =
                sent as f64 / usage.prompt_tokens as f64;
        }
        let budget = session.overflow_handler();
        println!(
            "[{} messages, ~{} of {} tokens, {:.1} characters per token]\n",
            session.len(),
            budget.tokens(session.history()),
            budget.max_tokens,
            budget.characters_per_token
        );
    }

    // The handler is saved along with the session, so what it learnt survives a restart.
    let saved = open_client::serde_json::to_string(&session)?;
    let restored: Session<TokenBudget> = open_client::serde_json::from_str(&saved)?;
    println!(
        "Restored {} messages, {:.1} characters per token",
        restored.len(),
        restored.overflow_handler().characters_per_token
    );
    Ok(())
}
