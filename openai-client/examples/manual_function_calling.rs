//! Handling tool calls yourself, e.g. to check them before acting. Each result refers to its call
//! by id.
//!
//! Tool results are text, as per the OpenAI specification. To show the model an image, e.g. a
//! screenshot, send it in a user message after the results.
use open_client::chat::message::{ContentPart, ImageUrl, ToolCall};
use open_client::chat::session::{Session, SlidingWindow};
use open_client::chat::{AI, Tool};
use open_client::{OpenAISchema, mime, openai_function};

#[openai_function]
/// Books a table at a restaurant.
fn book_table(restaurant: String, people: u32) -> String {
    format!("Booked a table for {people} at {restaurant}")
}

#[openai_function]
/// Takes a screenshot of the user's screen.
async fn take_screenshot() -> std::io::Result<ImageUrl> {
    // A real app would capture the screen.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lda.png");
    ImageUrl::from_path(path, mime::IMAGE_PNG).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ai = AI::from_env("gpt-5-mini", None)?;
    ai.tool(Tool::Function(book_table::openai_schema()))
        .tool(Tool::Function(take_screenshot::openai_schema()));
    let mut session = Session::new(SlidingWindow::new(20));

    let prompt = "Book a table for 20 at Bella Napoli, and tell me what is on my screen.";
    println!("User: {prompt}");
    let mut completion = ai.ask(session.ask(prompt)).await?;

    while !session.pending_tool_calls().is_empty() {
        let calls: Vec<ToolCall> = session.pending_tool_calls().into_iter().cloned().collect();
        let mut screenshots = Vec::new();
        for call in calls {
            println!(
                "Model calls {}({})",
                call.function.name, call.function.arguments
            );
            match call.function.name.as_str() {
                "book_table" => {
                    let (restaurant, people) = book_table::parse_arguments(&call.args()?)?;
                    // Check the call before acting on it
                    let result = if people > 10 {
                        "Not booked: groups over 10 people must call the restaurant".to_string()
                    } else {
                        book_table(restaurant, people)
                    };
                    session.add_tool_result(call.id, result)?;
                }
                "take_screenshot" => {
                    screenshots.push(take_screenshot().await?.into());
                    session.add_tool_result(call.id, "Screenshot is in the next message")?;
                }
                name => {
                    session.add_tool_result(call.id, format!("No function {name}"))?;
                }
            }
        }
        if !screenshots.is_empty() {
            let mut content = vec![ContentPart::from("Screenshot taken by `take_screenshot`:")];
            content.extend(screenshots);
            session.ask(content);
        }
        completion = ai.ask(&mut session).await?;
    }
    println!("AI: {}", completion.text());
    Ok(())
}
