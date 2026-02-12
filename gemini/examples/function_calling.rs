use gemini_client_api::gemini::ask::Gemini;
use gemini_client_api::gemini::types::request::Tool;
use gemini_client_api::gemini::types::sessions::Session;
use gemini_client_api::gemini::utils::{GeminiSchema, execute_function_calls, gemini_function};
use std::env;
use std::error::Error;

/// This function will be made available to Gemini.
/// The doc comments are used as descriptions for the tool.
#[gemini_function]
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

#[gemini_function]
/// Function to get the current temperature.
fn get_temperature(location: String) -> Result<String, &'static str> {
    println!("[Executing Tool] getting temperature for {}", location);
    Err("API is out of service")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut session = Session::new(10);
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    // 1. Initialize Gemini and register tools
    let ai =
        Gemini::new(api_key, "gemini-2.5-flash", None).set_tools(vec![Tool::FunctionDeclarations(
            vec![
                add_numbers::gemini_schema(),
                get_temperature::gemini_schema(),
            ],
        )]);

    println!("--- Function Calling Example ---");
    let prompt = "What is 123.45 plus 678.9, and what's the weather like in London?";
    println!("User: {}\n", prompt);

    // 2. Ask Gemini. It might reply with one or more function calls.
    let mut response = ai.ask(session.ask(prompt)).await?;

    // 3. Loop to handle potential multiple rounds of function calls
    loop {
        if response.get_chat().has_function_call() {
            println!("Gemini requested function calls...");

            // 4. Use the macro to execute all requested calls and update the session
            let results = execute_function_calls!(session, add_numbers, get_temperature);

            for (idx, res) in results.iter().enumerate() {
                if let Some(r) = res {
                    println!("  Call #{} result: {:?}", idx, r);
                }
            }

            // 5. Send the results back to Gemini to get the final natural language response
            response = ai.ask(&mut session).await?;
        } else {
            // No more function calls, show the final response
            println!("\nGemini: {}", response.get_chat().get_text_no_think(""));
            break;
        }
    }

    Ok(())
}

#[tokio::test]
async fn handle_manually() {
    use gemini_client_api::gemini::types::request::PartType;
    let mut session = Session::new(10);
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    // 1. Initialize Gemini and register tools
    let ai =
        Gemini::new(api_key, "gemini-2.5-flash", None).set_tools(vec![Tool::FunctionDeclarations(
            vec![
                add_numbers::gemini_schema(),
                get_temperature::gemini_schema(),
            ],
        )]);

    println!("--- Function Calling Example ---");
    let prompt = "What is 123.45 plus 678.9, and what's the weather like in London?";
    println!("User: {}\n", prompt);

    // 2. Ask Gemini. It might reply with one or more function calls.
    let mut response = ai.ask(session.ask(prompt)).await?;

    // 3. Loop to handle potential multiple rounds of function calls
    loop {
        if response.get_chat().has_function_call() {
            println!("Gemini requested function calls...");

            // 4. Use the macro to execute all requested calls and update the session
            let _ = execute_function_calls!(session, add_numbers);

            for part in response.get_chat().parts() {
                match part.data() {
                    PartType::FunctionCall(function_call)
                        if function_call.name() == "get_temperature" =>
                    {
                        get_temperature::execute_with_closure(
                            function_call.args().as_ref().unwrap(),
                            |location| {
                                println!(
                                    "[Executing Closure] getting temperature for {}",
                                    location
                                );
                                session // Note: You must update session manually
                                    .add_function_response(
                                        "get_temperature",
                                        format!("temperature of {location} is 38 degree Celsius"),
                                    )
                                    .unwrap();
                            },
                        )
                        .unwrap()
                    }
                    _ => {}
                }
            }

            // 5. Send the results back to Gemini to get the final natural language response
            response = ai.ask(&mut session).await?;
        } else {
            // No more function calls, show the final response
            println!("\nGemini: {}", response.get_chat().get_text_no_think(""));
            break;
        }
    }
}
