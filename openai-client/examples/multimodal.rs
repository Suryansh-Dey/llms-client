use open_client::openai::ask::OpenAI;
use open_client::openai::types::sessions::Session;
use open_client::openai::utils::MarkdownToParts;
use std::env;

#[tokio::test]
async fn raw_multimodal() {
    use open_client::openai::types::request::{ImageUrl, InlineData};

    let mut session = Session::new(6);
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let ai = OpenAI::new(api_key, "gpt-5-mini", None);

    session
        .ask("What is there in this image and this pdf")
        // Image URL fetched by the API server
        .ask(ImageUrl::from(
            "https://avatars.githubusercontent.com/u/140788315?v=4",
        ))
        // Downloaded and sent as base64
        .ask(InlineData::from_url("https://bitmesra.ac.in/UploadedDocuments/admingo/files/221225_List%20of%20Holiday_2026_26.pdf").await.unwrap());

    let response = ai.ask(&mut session).await.unwrap();
    println!("\nAI: {}", response.get_chat().get_text_no_think(""));
}

#[tokio::main]
async fn main() {
    let mut session = Session::new(6);
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let ai = OpenAI::new(api_key, "gpt-5-mini", None);

    // Use MarkdownToParts to easily parse a string with image/file markers
    // It supports both URLs and local file paths!
    let content = "Describe this image: ![](https://avatars.githubusercontent.com/u/140788315?v=4)";
    println!("Processing: {}", content);

    let parts = MarkdownToParts::new(content, |_| mime::IMAGE_PNG)
        .await
        .process();

    let response = ai.ask(session.ask_parts(parts)).await.unwrap();

    println!("\nAI: {}", response.get_chat().get_text_no_think(""));
}
