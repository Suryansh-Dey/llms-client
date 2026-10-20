//! Images and files in prompts.
use open_client::mime;
use open_client::openai::ask::OpenAI;
use open_client::openai::types::request::{ImageDetail, ImageUrl, InlineData};
use open_client::openai::types::sessions::Session;

const CHART: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lda.png");
const LOGO: &str = "https://www.rust-lang.org/logos/rust-logo-512x512.png";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = OpenAI::from_env("gpt-5-mini", Some("Answer in one sentence.".into()))?;
    let mut session = Session::new(10);

    // A local file, sent as base64. Besides images, audio (e.g. `audio/wav`) and other files
    // (e.g. `application/pdf`) can be sent, if the model supports them.
    let chart = InlineData::from_path(CHART, mime::IMAGE_PNG).await?;
    session.ask("What kind of chart is this?").ask(chart);
    let response = ai.ask(&mut session).await?;
    println!("AI: {}", response.get_chat().get_text_no_think(""));

    // An image URL, fetched by the API server. Use `InlineData::from_url()` to download it
    // yourself, for servers that can't fetch URLs.
    let logo = ImageUrl::new(LOGO.into(), Some(ImageDetail::Low));
    session.ask("Whose logo is this?").ask(logo);
    let response = ai.ask(&mut session).await?;
    println!("AI: {}", response.get_chat().get_text_no_think(""));

    // Text and images interleaved, in the order the model should see them
    let parts = vec![
        "Which of these is a scatter plot, the first".into(),
        ImageUrl::new(LOGO.into(), Some(ImageDetail::Low)).into(),
        "or the second?".into(),
        InlineData::from_path(CHART, mime::IMAGE_PNG).await?.into(),
    ];
    let response = ai.ask(session.ask_parts(parts)).await?;
    println!("AI: {}", response.get_chat().get_text_no_think(""));
    Ok(())
}
