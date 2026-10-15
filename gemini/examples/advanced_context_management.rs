use gemini_client_api::gemini::ask::Gemini;
use gemini_client_api::gemini::types::sessions::{OverflowHandler, Session};
use std::env;

/// Keeps the estimated token count of the history under `max_tokens` by forgetting the oldest chats.
#[derive(Clone, Debug)]
struct TokenBudget {
    max_tokens: usize,
}
impl TokenBudget {
    /// Rough estimate: ~4 characters per token. Only text parts are counted.
    fn estimate_tokens(session: &Session) -> usize {
        session
            .get_history()
            .iter()
            .map(|chat| chat.get_text_no_think("").len() / 4)
            .sum()
    }
}
impl OverflowHandler for TokenBudget {
    fn handle(&mut self, session: &mut Session) -> bool {
        // Never forget the latest chat, it is the prompt about to be sent (or the reply just received).
        while Self::estimate_tokens(session) > self.max_tokens && session.get_history_length() > 1 {
            // Might forget more than one chat to keep the history valid,
            // e.g. a function call is forgotten together with its response.
            session.forget_oldest_chat();
        }
        // Also run the default trimming, so `history_limit` stays a hard cap on the number of chats.
        true
    }
}

#[tokio::main]
async fn main() {
    // 1. At most 20 chats, and at most ~300 tokens of text.
    let mut session = Session::new(20);
    session.set_overflow_handler(TokenBudget { max_tokens: 300 });

    // 2. Create the Gemini client
    // Get your API key from https://aistudio.google.com/app/apikey
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let ai = Gemini::new(
        api_key,
        "gemini-2.5-flash",
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
        println!("Gemini: {}", response.get_chat().get_text_no_think(""));
        println!(
            "[chats in history: {}, estimated tokens: {}]\n",
            session.get_history_length(),
            TokenBudget::estimate_tokens(&session)
        );
    }

    // 4. The handler is not serialized. Set it again after loading a saved session.
    let saved = serde_json::to_string(&session).unwrap();
    let mut restored: Session = serde_json::from_str(&saved).unwrap();
    restored.set_overflow_handler(TokenBudget { max_tokens: 300 });
    println!(
        "Restored session with {} chats",
        restored.get_history_length()
    );
}
