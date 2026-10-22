//! Images, PDFs and audio in prompts. Each is a content part of its own type, sent as it is, so
//! what a server accepts depends on the server:
//! - OpenAI reads images as `ImageUrl` and PDFs as `File`.
//! - Gemini reads images and PDFs as `ImageUrl`, and rejects `File`.
//!
//! Pass a WAV file to also ask about audio: `cargo run --example multimodal -- speech.wav`
use open_client::chat::AI;
use open_client::chat::message::{
    AudioFormat, ContentPart, File, ImageDetail, ImageUrl, InputAudio,
};
use open_client::chat::session::{Session, SlidingWindow};
use open_client::mime;

const CHART: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lda.png");
const LOGO: &str = "https://www.rust-lang.org/logos/rust-logo-512x512.png";
const PDF: &str = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ai = AI::from_env("gpt-5-mini", Some("Answer in one sentence.".into()))?;
    let mut session = Session::new(SlidingWindow::new(10));

    // A local image, sent as a base64 `data:` URL.
    let chart = ImageUrl::from_path(CHART, mime::IMAGE_PNG).await?;
    session.ask(vec![
        ContentPart::from("What kind of chart is this?"),
        chart.into(),
    ]);
    let completion = ai.ask(&mut session).await?;
    println!("AI: {}", completion.text());

    // An image URL, fetched by the API server. Use `ImageUrl::download()` to download it
    // yourself, for servers that can't fetch URLs.
    let logo = ImageUrl::new(LOGO).with_detail(ImageDetail::Low);
    session.ask(vec![ContentPart::from("Whose logo is this?"), logo.into()]);
    let completion = ai.ask(&mut session).await?;
    println!("AI: {}", completion.text());

    // A PDF, downloaded and sent as base64.
    let pdf: ContentPart = if ai
        .get_base_url()
        .contains("generativelanguage.googleapis.com")
    {
        ImageUrl::download(PDF).await?.into()
    } else {
        File::download(PDF).await?.into()
    };
    session.ask(vec![ContentPart::from("What does this PDF say?"), pdf]);
    let completion = ai.ask(&mut session).await?;
    println!("AI: {}", completion.text());

    // Audio, for models that hear, e.g. Gemini or `gpt-4o-audio-preview`.
    if let Some(path) = std::env::args().nth(1) {
        let audio = InputAudio::from_path(path, AudioFormat::Wav).await?;
        session.ask(vec![ContentPart::from("What is said here?"), audio.into()]);
        let completion = ai.ask(&mut session).await?;
        println!("AI: {}", completion.text());
    }
    Ok(())
}
