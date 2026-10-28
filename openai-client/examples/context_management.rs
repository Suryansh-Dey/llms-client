//! Keeping a long conversation within a token budget, with a custom `OverflowHandler` instead of
//! the default `Compactor`, which has the model summarize it.
//!
//! After each reply, the handler learns the tokens per character from the `usage` of the
//! response, and forgets the oldest messages while the history is estimated over budget.
use open_client::chat::message::Message;
use open_client::chat::session::{HistoryTrimmer, OverflowHandler, Session};
use open_client::chat::{AI, ChatCompletion};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

fn characters(history: &VecDeque<Message>) -> usize {
    history.iter().map(|message| message.text().len()).sum()
}

#[derive(Debug, Serialize, Deserialize)]
struct TokenBudget {
    max_tokens: u64,
    /// Starts at the common estimate of 1/4, then learnt from `usage`.
    tokens_per_character: f64,
    /// Estimated tokens of the history, after forgetting.
    tokens: u64,
}
impl OverflowHandler for TokenBudget {
    async fn handle(&mut self, history: &mut HistoryTrimmer<'_>, response: &ChatCompletion) {
        // `prompt_tokens` counts what was sent: the history before the reply, and the system
        // prompt. Rough, as it also counts the formatting of messages. `completion_tokens` isn't
        // used, as it counts reasoning too, which isn't sent back.
        let sent = characters(history.history()).saturating_sub(response.text().len());
        if let Some(usage) = &response.usage
            && sent > 0
        {
            self.tokens_per_character = usage.prompt_tokens as f64 / sent as f64;
        }
        let estimate = |history: &HistoryTrimmer| {
            (characters(history.history()) as f64 * self.tokens_per_character) as u64
        };
        // Might forget more than one message, e.g. tool calls along with their results.
        // Stops (returns None) instead of forgetting the latest message.
        while estimate(history) > self.max_tokens && history.forget_oldest().is_some() {}
        self.tokens = estimate(history);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = AI::from_env("gpt-5-mini", Some("Answer in 2 sentences.".into()))?;
    let mut session = Session::new(TokenBudget {
        max_tokens: 200,
        tokens_per_character: 0.25,
        tokens: 0,
    });

    for prompt in [
        "What is ownership in Rust?",
        "How is borrowing related to it?",
        "What are lifetimes?",
        "Why does Rust need all three?",
        "Which of my questions do you still remember?",
    ] {
        let completion = ai.ask(session.ask(prompt)).await?;
        println!("User: {prompt}\nAI: {}", completion.text());
        let budget = session.overflow_handler();
        println!(
            "[{} messages, ~{} of {} tokens, {:.2} tokens per character]\n",
            session.len(),
            budget.tokens,
            budget.max_tokens,
            budget.tokens_per_character
        );
    }

    // The handler is saved along with the session, so what it learnt survives a restart.
    let saved = open_client::serde_json::to_string(&session)?;
    let restored: Session<TokenBudget> = open_client::serde_json::from_str(&saved)?;
    println!(
        "Restored {} messages, {:.2} tokens per character",
        restored.len(),
        restored.overflow_handler().tokens_per_character
    );
    Ok(())
}
