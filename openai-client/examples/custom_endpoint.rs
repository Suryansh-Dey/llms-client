use open_client::openai::ask::OpenAI;
use open_client::openai::types::sessions::Session;
use std::env;
use std::time::Duration;

#[tokio::main]
async fn main() {
    // Any server following OpenAI specification works by setting its base URL.
    // Requests are sent to `{base_url}/chat/completions`.
    let deepseek = OpenAI::new(
        env::var("DEEPSEEK_API_KEY").unwrap_or_default(),
        "deepseek-chat",
        None,
    )
    .set_base_url("https://api.deepseek.com");

    // Local servers usually need no API key. Empty key sends no `Authorization` header.
    let ollama = OpenAI::new("", "llama3.2", None).set_base_url("http://localhost:11434/v1");

    // Servers needing other headers (or a timeout) can use a custom reqwest::Client.
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert("HTTP-Referer", "https://example.com".parse().unwrap());
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap();
    let openrouter = OpenAI::new_with_client(
        env::var("OPENROUTER_API_KEY").unwrap_or_default(),
        "deepseek/deepseek-chat",
        None,
        client,
    )
    .set_base_url("https://openrouter.ai/api/v1");

    // Provider specific parameters go in the generation config, at the top level of the request.
    let mut gemini = OpenAI::new(
        env::var("GEMINI_API_KEY").unwrap_or_default(),
        "gemini-2.5-flash",
        None,
    )
    .set_base_url("https://generativelanguage.googleapis.com/v1beta/openai");
    gemini.set_generation_config()["temperature"] = 0.2.into();

    for ai in [deepseek, ollama, openrouter, gemini] {
        let mut session = Session::new(2);
        match ai.ask(session.ask("Say hi in 3 words")).await {
            Ok(response) => println!(
                "{} at {}: {}",
                ai.model(),
                ai.base_url(),
                response.get_chat().get_text_no_think("")
            ),
            Err(e) => eprintln!("{} at {} failed: {e}", ai.model(), ai.base_url()),
        }
    }
}
