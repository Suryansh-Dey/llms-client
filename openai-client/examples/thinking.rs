//! Reasoning of thinking models. Many providers return it, as `reasoning_content` (DeepSeek,
//! vLLM, Qwen..) or `reasoning` (OpenRouter, Ollama, Groq..). OpenAI doesn't in chat completions;
//! use `reasoning_effort()` there to control how long its models think.
//!
//! Run with `OPENAI_BASE_URL=https://api.deepseek.com` and a DeepSeek API key.
use open_client::chat::session::{Session, SlidingWindow};
use open_client::chat::{AI, Tool};
use open_client::{OpenAISchema, execute_function_calls, openai_function};

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
    let ai = AI::from_env("deepseek-reasoner", None)?
        .tool(Tool::Function(population::openai_schema()))
        // DeepSeek needs the reasoning sent back while it is calling tools. Off by default, as
        // other servers like OpenAI reject it.
        .send_reasoning(true);
    let mut session = Session::new(SlidingWindow::new(20));

    let prompt = "Which has more people, Tokyo or Delhi, and by how much?";
    println!("User: {prompt}\n");
    let mut completion = ai.ask(session.ask(prompt)).await?;
    loop {
        if let Some(reasoning) = &completion.message().reasoning {
            println!("Reasoning: {reasoning}\n");
        }
        if session.pending_tool_calls().is_empty() {
            break;
        }
        execute_function_calls!(session, population);
        completion = ai.ask(&mut session).await?;
    }
    println!("AI: {}", completion.text());
    Ok(())
}
