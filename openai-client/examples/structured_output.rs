use open_client::openai::ask::OpenAI;
use open_client::openai::types::sessions::Session;
use open_client::openai::utils::{OpenAISchema, openai_schema};
use serde::Deserialize;
use std::env;

/// Define your desired response structure.
/// Use the `#[openai_schema]` macro to automatically generate the JSON schema.
#[derive(Debug, Deserialize)]
#[openai_schema]
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
    /// Sequel of the movie, if any.
    sequel: Option<String>,
}

#[tokio::main]
async fn main() {
    let mut session = Session::new(2).set_remember_reply(false);
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // Enable JSON mode by passing the generated schema
    let ai = OpenAI::new(api_key, "gpt-5-mini", None).set_json_mode(MovieReview::openai_schema());

    let prompt = "Give me a review for the movie Interstellar.";
    println!("User: {}", prompt);

    let response = ai.ask(session.ask(prompt)).await.unwrap();
    let review: MovieReview = response
        .get_json()
        .expect("Model responded with wrong structure");

    println!("Structured output:\n{review:#?}");
}
