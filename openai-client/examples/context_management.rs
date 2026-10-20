//! Keeping a long conversation within a token budget, with a custom `OverflowHandler`.
//!
//! The handler estimates tokens from characters. It calibrates the estimate with the prompt size
//! the API reports in `usage`, and forgets the oldest chats when the history grows over budget.
use open_client::openai::ask::OpenAI;
use open_client::openai::types::request::Chat;
use open_client::openai::types::sessions::{HistoryTrimmer, OverflowHandler, Session};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

fn characters(history: &VecDeque<Chat>) -> usize {
    history.iter().map(|chat| chat.get_text_all("").len()).sum()
}

#[derive(Debug, Serialize, Deserialize)]
struct TokenBudget {
    max_tokens: usize,
    /// Starts at the common estimate of 4, then learnt from `usage`.
    characters_per_token: f64,
}
impl TokenBudget {
    fn tokens(&self, history: &VecDeque<Chat>) -> usize {
        (characters(history) as f64 / self.characters_per_token) as usize
    }
}
impl OverflowHandler for TokenBudget {
    fn handle(&mut self, history: &mut HistoryTrimmer<'_>) -> bool {
        // Might forget more than one chat, e.g. a function call along with its responses.
        // Stops (returns None) instead of forgetting the latest chat.
        while self.tokens(history.get_history_as_vecdeque()) > self.max_tokens
            && history.forget_oldest_chat().is_some()
        {}
        // Also run the default sliding window, so `history_limit` stays a cap on chats.
        true
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = OpenAI::from_env("gpt-5-mini", Some("Answer in 2 sentences.".into()))?;
    let sys_prompt = ai.sys_prompt().as_ref().unwrap();
    let sys_prompt_characters = Chat::extract_text_all(sys_prompt.parts(), "").len();
    let mut session = Session::new(50).set_overflow_handler(TokenBudget {
        max_tokens: 200,
        characters_per_token: 4.0,
    });

    for prompt in [
        "What is ownership in Rust?",
        "How is borrowing related to it?",
        "What are lifetimes?",
        "Why does Rust need all three?",
        "Which of my questions do you still remember?",
    ] {
        session.ask(prompt);
        let sent = sys_prompt_characters + characters(session.get_history_as_vecdeque());
        let response = ai.ask(&mut session).await?;
        println!(
            "User: {prompt}\nAI: {}",
            response.get_chat().get_text_no_think("")
        );

        let prompt_tokens = response
            .usage
            .as_ref()
            .and_then(|u| u["prompt_tokens"].as_u64());
        if let Some(prompt_tokens) = prompt_tokens {
            // Rough, as `prompt_tokens` also counts the formatting of messages.
            session.overflow_handler_mut().characters_per_token =
                sent as f64 / prompt_tokens as f64;
        }
        let budget = session.overflow_handler();
        println!(
            "[{} chats, ~{} of {} tokens, {:.1} characters per token]\n",
            session.get_history_length(),
            budget.tokens(session.get_history_as_vecdeque()),
            budget.max_tokens,
            budget.characters_per_token
        );
    }

    // The handler is saved along with the session, so what it learnt survives a restart.
    let saved = open_client::serde_json::to_string(&session)?;
    let restored: Session<TokenBudget> = open_client::serde_json::from_str(&saved)?;
    println!(
        "Restored {} chats, {:.1} characters per token",
        restored.get_history_length(),
        restored.overflow_handler().characters_per_token
    );
    Ok(())
}
