use open_client::openai::ask::OpenAI;
use open_client::openai::types::request::Tool;
use open_client::openai::types::sessions::Session;
use open_client::openai::utils::{OpenAISchema, execute_function_calls, openai_function};
use std::env;
use std::error::Error;

/// This function will be made available to the model.
/// The doc comments are used as descriptions for the tool.
#[openai_function]
/// Returns the result of adding two numbers.
fn add_numbers(
    /// The first number.
    a: f64,
    /// The second number.
    b: f64,
) -> f64 {
    println!("[Executing Tool] adding {} + {}", a, b);
    a + b
}

#[openai_function]
/// Function to get the current temperature.
fn get_temperature(location: String) -> Result<String, &'static str> {
    println!("[Executing Tool] getting temperature for {}", location);
    Err("API is out of service")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut session = Session::new(10);
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // 1. Initialize the client and register tools
    let ai = OpenAI::new(api_key, "gpt-5-mini", None)
        .set_tool(Tool::Function(add_numbers::openai_schema()))
        .set_tool(Tool::Function(get_temperature::openai_schema()));

    let prompt = "What is 123.45 plus 678.9, and what's the weather like in London?";
    println!("User: {}\n", prompt);

    // 2. Ask the model. It might reply with one or more function calls.
    let mut response = ai.ask(session.ask(prompt)).await?;

    // 3. Loop to handle potential multiple rounds of function calls
    loop {
        if response.get_chat().has_function_call() {
            println!("Model requested function calls...");

            // 4. Use the macro to execute all requested calls and update the session
            let results = execute_function_calls!(session, add_numbers, get_temperature);

            for (idx, res) in results.iter().enumerate() {
                if let Some(r) = res {
                    println!("  Call #{} result: {:?}", idx, r);
                }
            }

            // 5. Send the results back to get the final natural language response
            response = ai.ask(&mut session).await?;
        } else {
            // No more function calls, show the final response
            println!("\nAI: {}", response.get_chat().get_text_no_think(""));
            break;
        }
    }

    Ok(())
}

#[openai_function]
/// Takes a screenshot of the user's screen.
fn take_screenshot() {}

#[tokio::test]
async fn handle_manually() -> Result<(), Box<dyn Error>> {
    use open_client::openai::types::request::{InlineData, MultimodalResponse};

    let mut session = Session::new(10);
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // 1. Initialize the client and register tools
    let ai = OpenAI::new(api_key, "gpt-5-mini", None).set_tools(vec![
        Tool::Function(add_numbers::openai_schema()),
        Tool::Function(get_temperature::openai_schema()),
        Tool::Function(take_screenshot::openai_schema()),
    ]);

    let prompt =
        "What is 123.45 plus 678.9, what's the weather like in London and what's on my screen?";
    println!("User: {}\n", prompt);

    // 2. Ask the model. It might reply with one or more function calls.
    let mut response = ai.ask(session.ask(prompt)).await?;

    // 3. Loop to handle potential multiple rounds of function calls
    loop {
        if response.get_chat().has_function_call() {
            println!("Model requested function calls...");

            // 4. Use the macro to execute one requested calls and update the session
            let _ = execute_function_calls!(session, add_numbers);

            for call in response.get_chat().get_function_calls() {
                if call.name() == "get_temperature" {
                    let (location,) = get_temperature::parse_arguments(&call.args()?)
                        .expect("Model responded with wrong argument format");

                    println!("[Executing call] getting temperature for {}", location);
                    session // Note: You must update session manually
                        .add_function_response(
                            call.id(),
                            format!("temperature of {location} is 38 degree Celsius"),
                        )?;
                } else if call.name() == "take_screenshot" {
                    // Function responses can have images, audio or files too
                    session.add_function_response(
                        call.id(),
                        MultimodalResponse::new(vec![
                            "Screenshot of the screen".into(),
                            InlineData::from_path("tests/lda.png", mime::IMAGE_PNG)
                                .await?
                                .into(),
                        ]),
                    )?;
                }
            }

            // 5. Send the results back to get the final natural language response
            response = ai.ask(&mut session).await?;
        } else {
            // No more function calls, show the final response
            println!("\nAI: {}", response.get_chat().get_text_no_think(""));
            break;
        }
    }
    Ok(())
}
