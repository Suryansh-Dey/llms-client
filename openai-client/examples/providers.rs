//! Any server following the OpenAI specification works by setting its base URL. Requests go to
//! `{base_url}/chat/completions`.
//!
//! Providers whose API key isn't set are skipped. Ollama is tried at its default local address.
use open_client::openai::ask::OpenAI;
use open_client::openai::types::sessions::Session;
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
        providers
            .push(OpenAI::new(key, "deepseek-chat", None).set_base_url("https://api.deepseek.com"));
    }
    if let Ok(key) = env::var("GEMINI_API_KEY") {
        providers.push(
            OpenAI::new(key, "gemini-2.5-flash", None)
                .set_base_url("https://generativelanguage.googleapis.com/v1beta/openai"),
        );
    }
    if let Ok(key) = env::var("OPENROUTER_API_KEY") {
        // Servers needing other headers, or a timeout, can be used with a custom client.
        let mut headers = HeaderMap::new();
        headers.insert("X-Title", HeaderValue::from_static("open-client example"));
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(60))
            .build()?;
        let mut openrouter = OpenAI::new_with_client(key, "deepseek/deepseek-chat", None, client)
            .set_base_url("https://openrouter.ai/api/v1");
        // Provider specific parameters are set in the generation config, which goes to the top
        // level of the request body. Here: route to the cheapest provider.
        openrouter.set_generation_config()["provider"] = json!({"sort": "price"});
        providers.push(openrouter);
    }
    // Local servers need no API key.
    providers
        .push(OpenAI::with_no_api_key("llama3.2", None).set_base_url("http://localhost:11434/v1"));

    for ai in providers {
        let mut session = Session::new(2);
        match ai.ask(session.ask("Say hi in 3 words.")).await {
            Ok(response) => println!(
                "{} at {}: {}",
                ai.model(),
                ai.base_url(),
                response.get_chat().get_text_no_think("")
            ),
            Err(e) => eprintln!("{} at {} failed: {e}", ai.model(), ai.base_url()),
        }
    }
    Ok(())
}
