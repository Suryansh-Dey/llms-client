//! A reply in your own type. `#[openai_schema]` generates the JSON schema of a type, and JSON mode
//! makes the model follow it.
use open_client::openai::ask::OpenAI;
use open_client::openai::types::sessions::Session;
use open_client::openai::utils::{OpenAISchema, openai_schema};
use serde::Deserialize;

#[openai_schema]
#[derive(Debug, Deserialize)]
enum Difficulty {
    Easy,
    Medium,
    Hard,
}

#[openai_schema]
#[derive(Deserialize)]
struct Ingredient {
    name: String,
    /// With unit, e.g. "200 g"
    quantity: String,
}

#[openai_schema]
#[derive(Deserialize)]
/// A cooking recipe
struct Recipe {
    name: String,
    /// Total time in minutes
    minutes: u32,
    difficulty: Difficulty,
    ingredients: Vec<Ingredient>,
    steps: Vec<String>,
    /// A wine to pair with, if any
    wine: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = OpenAI::from_env("gpt-5-mini", None)?.set_json_mode(Recipe::openai_schema());
    // One-off requests don't need the reply in history.
    let mut session = Session::new(2).set_remember_reply(false);

    let response = ai
        .ask(session.ask("A quick vegetarian pasta for two"))
        .await?;
    let recipe: Recipe = response.get_json()?;

    println!(
        "{} ({} minutes, {:?})",
        recipe.name, recipe.minutes, recipe.difficulty
    );
    for ingredient in recipe.ingredients {
        println!("- {} {}", ingredient.quantity, ingredient.name);
    }
    for (i, step) in recipe.steps.iter().enumerate() {
        println!("{}. {step}", i + 1);
    }
    if let Some(wine) = recipe.wine {
        println!("Pairs with {wine}");
    }
    Ok(())
}
