use gemini_client_api::gemini::ask::Gemini;
use gemini_client_api::gemini::types::sessions::Session;
use gemini_client_api::gemini::utils::{GeminiSchema, gemini_schema};
use serde::Deserialize;
use std::env;

/// Define your desired response structure.
/// Use the `#[gemini_schema]` macro to automatically generate the JSON schema.
#[derive(Debug, Deserialize)]
#[gemini_schema]
#[allow(dead_code)]
struct MovieReview {
    /// The title of the movie.
    title: String,
    /// A score from 1 to 10.
    rating: u8,
    /// A list of main actors.
    cast: Vec<String>,
    /// A short summary of the plot.
    summary: String,
}

#[tokio::main]
async fn main() {
    let mut session = Session::new(2).set_remember_reply(false);
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    // Enable JSON mode by passing the generated schema
    let ai =
        Gemini::new(api_key, "gemini-2.5-flash", None).set_json_mode(MovieReview::gemini_schema());

    let prompt = "Give me a review for the movie Interstellar.";
    println!("User: {}", prompt);

    let response = ai.ask(session.ask(prompt)).await.unwrap();
    let review: MovieReview = response
        .get_json()
        .expect("Gemini responded with wrong structure");

    println!("Gemini structured output:\n{review:#?}");
}
