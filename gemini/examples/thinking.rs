use gemini_client_api::gemini::ask::Gemini;
use gemini_client_api::gemini::types::request::{ThinkingConfig, ThinkingLevel};
use gemini_client_api::gemini::types::sessions::Session;
use std::env;

#[tokio::main]
async fn main() {
    let mut session = Session::new(4);
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    // Note: Thinking mode requires a supported model like gemini-2.5+
    let ai = Gemini::new(api_key, "gemini-3-flash-preview", None)
        .set_thinking_config(ThinkingConfig::new(true, ThinkingLevel::Low));

    let prompt = "How many 'r's are in the word strawberry?";
    println!("User: {}\n", prompt);

    let response = ai.ask(session.ask(prompt)).await.unwrap();

    // Show the "thoughts" part separately
    let thoughts = response.get_chat().get_thoughts("\n");
    if !thoughts.is_empty() {
        println!("--- Gemini's Thoughts ---\n{}\n", thoughts);
    }

    // Show the final answer
    let answer = response.get_chat().get_text_no_think("");
    println!("--- Gemini's Answer ---\n{}", answer);
}
