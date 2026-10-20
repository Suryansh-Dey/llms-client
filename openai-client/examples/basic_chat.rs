use open_client::openai::ask::OpenAI;
use open_client::openai::types::sessions::Session;
use std::env;

#[tokio::main]
async fn main() {
    // 1. Initialize the session with a history limit (e.g., 6 messages)
    let mut session = Session::new(6);

    // 2. Create the client. See `custom_endpoint.rs` for APIs other than OpenAI.
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let ai = OpenAI::new(
        api_key,
        "gpt-5-mini",
        Some("You are a senior engineer at google".into()),
    );

    // 3. Ask a question
    let prompt = "What are the benefits of using Rust for systems programming?";
    session.ask(prompt).ask("\nKeep you answer short"); // consecutive asks gets concatenated

    println!("User: {:?}", session.get_last_chat().unwrap().parts());
    let response = ai.ask(&mut session).await.unwrap();

    // 4. Print the reply
    // get_text_no_think("") extracts text and ignores "thought" parts (if any)
    let reply = response.get_chat().get_text_no_think("");
    println!("\nAI: {}", reply);

    // 5. The session now contains the interaction
    println!("\nMessages in history: {}", session.get_history_length());
}
