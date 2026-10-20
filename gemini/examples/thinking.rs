use gemini_client_api::gemini::ask::Gemini;
use gemini_client_api::gemini::types::request::{ThinkingConfig, ThinkingLevel};
use gemini_client_api::gemini::types::sessions::Session;

#[tokio::main]
async fn main() {
    let mut session = Session::new(4);

    // Note: Thinking mode requires a supported model like gemini-2.5+
    let ai = Gemini::from_env("gemini-3-flash-preview", None)
        .expect("GEMINI_API_KEY must be set")
        .set_thinking_config(ThinkingConfig::new(true, ThinkingLevel::Low)); //For gemini-2.5 only
    //budget is allowed. Eg. `.set_thinking_config(ThinkingConfig::new(true, 1024))`

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
