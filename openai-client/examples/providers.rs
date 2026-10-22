//! Any server following the OpenAI specification works by setting its base URL. Requests go to
//! `{base_url}/chat/completions`.
//!
//! Providers whose API key isn't set are skipped. Ollama is tried at its default local address.
use open_client::chat::AI;
use open_client::chat::session::Session;
use open_client::reqwest::{
    self,
    header::{HeaderMap, HeaderValue},
};
use open_client::serde_json::json;
use std::env;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut providers = Vec::new();

    if let Ok(key) = env::var("DEEPSEEK_API_KEY") {
        let mut deepseek = AI::new(key, "deepseek-chat", None);
        deepseek.base_url("https://api.deepseek.com");
        providers.push(deepseek);
    }
    if let Ok(key) = env::var("GEMINI_API_KEY") {
        let mut gemini = AI::new(key, "gemini-2.5-flash", None);
        gemini.base_url("https://generativelanguage.googleapis.com/v1beta/openai");
        providers.push(gemini);
    }
    if let Ok(key) = env::var("OPENROUTER_API_KEY") {
        // Servers needing other headers, or a timeout, can be used with a custom client.
        let mut headers = HeaderMap::new();
        headers.insert("X-Title", HeaderValue::from_static("open-client example"));
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(60))
            .build()?;
        let mut openrouter = AI::with_client(key, "deepseek/deepseek-chat", None, client);
        openrouter
            .base_url("https://openrouter.ai/api/v1")
            // Provider specific parameters go to the top level of the request body. Here: route
            // to the cheapest provider.
            .param("provider", json!({"sort": "price"}));
        providers.push(openrouter);
    }
    // Local servers need no API key.
    let mut ollama = AI::with_no_api_key("llama3.2", None);
    ollama.base_url("http://localhost:11434/v1");
    providers.push(ollama);

    for ai in providers {
        let mut session = Session::default();
        match ai.ask(session.ask("Say hi in 3 words.")).await {
            Ok(completion) => println!(
                "{} at {}: {}",
                ai.get_model(),
                ai.get_base_url(),
                completion.text()
            ),
            Err(e) => eprintln!("{} at {} failed: {e}", ai.get_model(), ai.get_base_url()),
        }
    }
    Ok(())
}
