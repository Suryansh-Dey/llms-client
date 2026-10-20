//! Handling function calls yourself, e.g. to check them before acting, or to respond with an
//! image. Each response refers to its call by id.
use open_client::mime;
use open_client::openai::ask::OpenAI;
use open_client::openai::types::request::{FunctionCall, InlineData, MultimodalResponse, Tool};
use open_client::openai::types::sessions::Session;
use open_client::openai::utils::{OpenAISchema, openai_function};

#[openai_function]
/// Books a table at a restaurant.
fn book_table(restaurant: String, people: u32) -> String {
    format!("Booked a table for {people} at {restaurant}")
}

#[openai_function]
/// Takes a screenshot of the user's screen.
async fn take_screenshot() -> std::io::Result<InlineData> {
    // A real app would capture the screen.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lda.png");
    InlineData::from_path(path, mime::IMAGE_PNG).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = OpenAI::from_env("gpt-5-mini", None)?
        .set_tool(Tool::Function(book_table::openai_schema()))
        .set_tool(Tool::Function(take_screenshot::openai_schema()));
    let mut session = Session::new(20);

    let prompt = "Book a table for 20 at Bella Napoli, and tell me what is on my screen.";
    println!("User: {prompt}");
    let mut response = ai.ask(session.ask(prompt)).await?;

    while !session.get_pending_function_calls().is_empty() {
        let calls: Vec<FunctionCall> = session
            .get_pending_function_calls()
            .into_iter()
            .cloned()
            .collect();
        for call in calls {
            println!("Model calls {}({})", call.name(), call.arguments());
            match call.name().as_str() {
                "book_table" => {
                    let (restaurant, people) = book_table::parse_arguments(&call.args()?)?;
                    // Check the call before acting on it
                    let result = if people > 10 {
                        "Not booked: groups over 10 people must call the restaurant".to_string()
                    } else {
                        book_table(restaurant, people)
                    };
                    session.add_function_response(call.id(), result)?;
                }
                "take_screenshot" => {
                    // Responses can have images, audio or files too
                    let screenshot = take_screenshot().await?;
                    let response =
                        MultimodalResponse::new(vec!["Screenshot".into(), screenshot.into()]);
                    session.add_function_response(call.id(), response)?;
                }
                name => {
                    session.add_function_response(call.id(), format!("No function {name}"))?;
                }
            }
        }
        response = ai.ask(&mut session).await?;
    }
    println!("AI: {}", response.get_chat().get_text_no_think(""));
    Ok(())
}
