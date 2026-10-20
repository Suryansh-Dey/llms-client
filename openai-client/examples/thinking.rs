//! Thoughts of reasoning models. Many providers return them, as `reasoning_content` (DeepSeek,
//! vLLM, Qwen..) or `reasoning` (OpenRouter, Ollama, Groq..). OpenAI doesn't in chat completions;
//! use `set_reasoning_effort()` there to control how long its models think.
//!
//! Run with `OPENAI_BASE_URL=https://api.deepseek.com` and a DeepSeek API key.
use open_client::openai::ask::OpenAI;
use open_client::openai::types::request::Tool;
use open_client::openai::types::sessions::Session;
use open_client::openai::utils::{OpenAISchema, execute_function_calls, openai_function};

#[openai_function]
/// Returns the population of a city, in millions.
fn population(city: String) -> Result<f64, String> {
    match city.as_str() {
        "Tokyo" => Ok(37.1),
        "Delhi" => Ok(33.8),
        _ => Err(format!("No data for {city}")),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = OpenAI::from_env("deepseek-reasoner", None)?
        .set_tool(Tool::Function(population::openai_schema()))
        // DeepSeek needs the thoughts sent back while it is calling tools. Off by default, as
        // other servers like OpenAI reject them.
        .set_send_reasoning(true);
    let mut session = Session::new(20);

    let prompt = "Which has more people, Tokyo or Delhi, and by how much?";
    println!("User: {prompt}\n");
    let mut response = ai.ask(session.ask(prompt)).await?;
    loop {
        let thoughts = response.get_chat().get_thoughts("");
        if !thoughts.is_empty() {
            println!("Thoughts: {thoughts}\n");
        }
        if session.get_pending_function_calls().is_empty() {
            break;
        }
        execute_function_calls!(session, population);
        response = ai.ask(&mut session).await?;
    }
    println!("AI: {}", response.get_chat().get_text_no_think(""));
    Ok(())
}
