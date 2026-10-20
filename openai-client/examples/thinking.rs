use open_client::openai::ask::OpenAI;
use open_client::openai::types::request::ReasoningEffort;
use open_client::openai::types::sessions::Session;
use std::env;

#[tokio::main]
async fn main() {
    let mut session = Session::new(4);

    // OpenAI doesn't return thoughts in chat completions, but many other providers do,
    // as `reasoning_content` (DeepSeek, vLLM, Qwen..) or `reasoning` (OpenRouter, Ollama, Groq..).
    let api_key = env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");
    let ai = OpenAI::new(api_key, "deepseek-reasoner", None)
        .set_base_url("https://api.deepseek.com")
        // DeepSeek needs thoughts sent back while it is calling tools. Off by default.
        .set_send_reasoning(true);

    let prompt = "How many 'r's are in the word strawberry?";
    println!("User: {}\n", prompt);

    let response = ai.ask(session.ask(prompt)).await.unwrap();

    // Show the "thoughts" part separately
    let thoughts = response.get_chat().get_thoughts("\n");
    if !thoughts.is_empty() {
        println!("--- Thoughts ---\n{}\n", thoughts);
    }

    // Show the final answer
    let answer = response.get_chat().get_text_no_think("");
    println!("--- Answer ---\n{}", answer);

    // For OpenAI reasoning models, the effort can be controlled, but thoughts are not returned.
    let openai = OpenAI::new(
        env::var("OPENAI_API_KEY").unwrap_or_default(),
        "gpt-5-mini",
        None,
    )
    .set_reasoning_effort(ReasoningEffort::Low);
    let _ = openai;
}
