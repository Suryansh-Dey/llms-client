use gemini_client_api::gemini::ask::Gemini;
use gemini_client_api::gemini::types::sessions::Session;
use gemini_client_api::gemini::utils::MarkdownToParts;
use std::env;

#[tokio::test]
async fn raw_multimodal() {
    use gemini_client_api::gemini::types::request::InlineData;

    let mut session = Session::new(6);
    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let ai = Gemini::new(api_key, "gemini-2.5-flash", None);

    session.ask("What is there in this pdf")
        .ask(InlineData::from_url("https://bitmesra.ac.in/UploadedDocuments/admingo/files/221225_List%20of%20Holiday_2026_26.pdf").await.unwrap());

    let response = ai.ask(&mut session).await.unwrap();
    println!("\nGemini: {}", response.get_chat().get_text_no_think(""));
}
#[tokio::main]
async fn main() {
    let mut session = Session::new(6);
    let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let ai = Gemini::new(api_key, "gemini-2.5-flash", None);

    println!("--- Multimodal (Images/Files) Example ---");

    // Use MarkdownToParts to easily parse a string with image/file markers
    // It supports both URLs and local file paths!
    let content = "Describe this image: ![](https://avatars.githubusercontent.com/u/140788315?v=4)";
    println!("Processing: {}", content);

    let parts = MarkdownToParts::new(content, |_| mime::IMAGE_PNG)
        .await
        .process();

    let response = ai.ask(session.ask_parts(parts)).await.unwrap();

    println!("\nGemini: {}", response.get_chat().get_text_no_think(""));
}
