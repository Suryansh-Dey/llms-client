use open_client::openai::ask::OpenAI;
use open_client::openai::types::request::Chat;
use open_client::openai::types::sessions::{HistoryTrimmer, OverflowHandler, Session};
use serde::{Deserialize, Serialize};
use std::env;

/// Rough estimate: ~4 characters per token. Only text parts are counted.
fn estimate_tokens<'a>(chats: impl IntoIterator<Item = &'a Chat>) -> usize {
    chats
        .into_iter()
        .map(|chat| chat.get_text_no_think("").len() / 4)
        .sum()
}

/// Keeps the estimated token count of the history under `max_tokens` by forgetting the oldest chats.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct TokenBudget {
    max_tokens: usize,
    /// State kept by the handler. It is saved and restored along with the session.
    forgotten_chats: usize,
}
impl OverflowHandler for TokenBudget {
    fn handle(&mut self, history: &mut HistoryTrimmer<'_>) -> bool {
        while estimate_tokens(history.get_history_as_vecdeque()) > self.max_tokens {
            let length = history.get_history_length();
            // Might forget more than one chat to keep the history valid, e.g. a function call is
            // forgotten together with its response. Returns None instead of forgetting the latest chat.
            if history.forget_oldest_chat().is_none() {
                break;
            }
            self.forgotten_chats += length - history.get_history_length();
        }
        // Also run the default trimming, so `history_limit` stays a hard cap on the number of chats.
        true
    }
}

#[tokio::main]
async fn main() {
    // 1. At most 20 chats, and at most ~300 tokens of text.
    let mut session = Session::new(20).set_overflow_handler(TokenBudget {
        max_tokens: 300,
        forgotten_chats: 0,
    });

    // 2. Create the client
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let ai = OpenAI::new(
        api_key,
        "gpt-5-mini",
        Some("You are a helpful tutor. Answer in 2-3 sentences.".into()),
    );

    // 3. Have a conversation. Old chats are forgotten as the budget fills up.
    let prompts = [
        "What is ownership in Rust?",
        "How is borrowing related to it?",
        "What are lifetimes?",
        "Give me a one line summary of everything we discussed.",
    ];
    for prompt in prompts {
        session.ask(prompt);
        let response = ai.ask(&mut session).await.unwrap();
        println!("User: {prompt}");
        println!("AI: {}", response.get_chat().get_text_no_think(""));
        println!(
            "[chats in history: {}, estimated tokens: {}, forgotten chats: {}]\n",
            session.get_history_length(),
            estimate_tokens(session.get_history_as_vecdeque()),
            session.overflow_handler().forgotten_chats
        );
    }

    // 4. The handler is saved along with the session, so its state survives a restart.
    let saved = serde_json::to_string(&session).unwrap();
    let restored: Session<TokenBudget> = serde_json::from_str(&saved).unwrap();
    println!(
        "Restored session with {} chats, {} forgotten so far",
        restored.get_history_length(),
        restored.overflow_handler().forgotten_chats
    );
}
