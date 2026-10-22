//! Rust functions as tools. `#[openai_function]` generates their schema from the signature and doc
//! comments, and `execute_function_calls!` runs the calls the model makes, adding the results to
//! the session. See `manual_function_calling.rs` to handle calls yourself.
use open_client::chat::session::{Session, SlidingWindow};
use open_client::chat::{AI, Tool};
use open_client::{OpenAISchema, execute_function_calls, openai_function};

#[openai_function]
/// Returns the price of a cryptocurrency in US dollars.
async fn get_price(
    /// Symbol, e.g. "BTC"
    symbol: String,
) -> Result<f64, String> {
    // Errors are sent to the model as the result, so it can react to them.
    match symbol.to_uppercase().as_str() {
        "BTC" => Ok(67_000.0),
        "ETH" => Ok(3_500.0),
        _ => Err(format!("No price for {symbol}")),
    }
}

#[openai_function]
/// Converts US dollars to another currency.
fn convert_usd(
    amount: f64,
    /// Currency code, e.g. "EUR"
    currency: String,
) -> Result<f64, String> {
    match currency.as_str() {
        "EUR" => Ok(amount * 0.92),
        "INR" => Ok(amount * 83.0),
        _ => Err(format!("Can't convert to {currency}")),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = AI::from_env("gpt-5-mini", None)?
        .tool(Tool::Function(get_price::openai_schema()))
        .tool(Tool::Function(convert_usd::openai_schema()));
    let mut session = Session::new(SlidingWindow::new(20));

    let prompt = "How much are 2 BTC and 10 ETH worth in euros?";
    println!("User: {prompt}");
    let mut completion = ai.ask(session.ask(prompt)).await?;

    // The model may call several functions at once, and call more after seeing the results.
    while !session.pending_tool_calls().is_empty() {
        for call in session.pending_tool_calls() {
            println!(
                "Model calls {}({})",
                call.function.name, call.function.arguments
            );
        }
        execute_function_calls!(session, get_price, convert_usd);
        completion = ai.ask(&mut session).await?;
    }
    println!("AI: {}", completion.text());
    Ok(())
}
