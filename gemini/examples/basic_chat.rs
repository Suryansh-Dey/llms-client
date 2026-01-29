use gemini_client_api::gemini::ask::Gemini;
use gemini_client_api::gemini::types::sessions::Session;
use std::env;

#[tokio::main]
async fn main() {
    // 1. Initialize the session with a history limit (e.g., 6 messages)
    let mut session = Session::new(6);

    // 2. Create the Gemini client
    // Get your API key from https://aistudio.google.com/app/apikey
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let ai = Gemini::new(api_key, "gemini-2.5-flash", None);

    println!("--- Basic Chat Example ---");

    // 3. Ask a question
    let prompt = "What are the benefits of using Rust for systems programming?";
    println!("User: {}", prompt);

    let response = ai.ask(session.ask_string(prompt)).await.unwrap();

    // 4. Print the reply
    // get_text_no_think("") extracts text and ignores "thought" parts (if any)
    let reply = response.get_chat().get_text_no_think("");
    println!("\nGemini: {}", reply);

    // 5. The session now contains the interaction
    println!("\nMessages in history: {}", session.get_history_length());
}
