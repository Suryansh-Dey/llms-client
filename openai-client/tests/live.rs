//! Tests against a real API, ignored by default. Run with
//! ```sh
//! OPENAI_API_KEY=.. OPENAI_BASE_URL=.. OPENAI_MODEL=.. OPENAI_EMBEDDING_MODEL=.. cargo test -- --ignored
//! ```
//! `OPENAI_BASE_URL` defaults to OpenAI, `OPENAI_MODEL` to `gpt-5-mini` and
//! `OPENAI_EMBEDDING_MODEL` to `text-embedding-3-small`.
use futures::StreamExt;
use open_client::chat::message::{ContentPart, File, ImageUrl};
use open_client::chat::session::{Session, SlidingWindow};
use open_client::chat::{AI, ReasoningEffort, ResponseFormat, Tool};
use open_client::embeddings::Embedder;
use open_client::{
    Error, OpenAISchema, execute_function_calls, mime, openai_function, openai_schema,
};
use serde::Deserialize;

const IMAGE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lda.png");

fn ai(system_prompt: Option<&str>) -> AI {
    let model = std::env::var("OPENAI_MODEL").unwrap_or("gpt-5-mini".into());
    AI::from_env(model, system_prompt.map(Into::into)).unwrap()
}

/// A one page PDF showing `text`.
fn pdf(text: &str) -> Vec<u8> {
    let content = format!("BT /F1 24 Tf 72 700 Td ({text}) Tj ET");
    let objects = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R \
         /Resources << /Font << /F1 5 0 R >> >> >>"
            .to_string(),
        format!(
            "<< /Length {} >>\nstream\n{content}\nendstream",
            content.len()
        ),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
    ];
    let mut pdf = String::from("%PDF-1.4\n");
    let mut offsets = Vec::new();
    for (i, object) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.push_str(&format!("{} 0 obj\n{object}\nendobj\n", i + 1));
    }
    let xref = pdf.len();
    pdf.push_str(&format!(
        "xref\n0 {}\n0000000000 65535 f \n",
        objects.len() + 1
    ));
    for offset in offsets {
        pdf.push_str(&format!("{offset:010} 00000 n \n"));
    }
    pdf.push_str(&format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
        objects.len() + 1
    ));
    pdf.into_bytes()
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn conversation_keeps_history() {
    let ai = ai(Some("Answer in one short sentence."));
    let mut session = Session::new(SlidingWindow::new(10));
    ai.ask(session.ask("My name is Ferris.")).await.unwrap();
    let completion = ai.ask(session.ask("What is my name?")).await.unwrap();
    assert!(completion.text().contains("Ferris"));
    assert_eq!(session.len(), 4);
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn structured_output() {
    #[openai_schema]
    #[derive(Deserialize)]
    struct Sentiments {
        positive: Vec<String>,
        negative: Vec<String>,
    }
    let mut ai = ai(Some("Classify the given words"));
    ai.response_format(ResponseFormat::json_schema(Sentiments::openai_schema()));
    let mut session = Session::default().with_remember_reply(false);
    let completion = ai.ask(session.ask("Joy, Fear, Hope, Anger")).await.unwrap();
    let sentiments: Sentiments = completion.json().unwrap();
    assert!(sentiments.positive.contains(&"Joy".to_string()));
    assert!(sentiments.negative.contains(&"Fear".to_string()));
    assert!(session.is_empty());
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn streamed_text() {
    let mut ai = ai(None);
    ai.include_usage(true);
    let mut session = Session::new(SlidingWindow::new(10));
    let mut stream = ai
        .stream(session.ask("Count from 1 to 20, separated by spaces."))
        .await
        .unwrap();
    let mut text = String::new();
    let mut usage = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        text.push_str(chunk.text());
        usage = chunk.usage.or(usage);
    }
    assert!(text.contains("20"));
    assert_eq!(stream.message().text(), text);
    assert!(usage.unwrap().completion_tokens > 0);
    drop(stream);
    assert_eq!(session.len(), 2);
    assert_eq!(session.last().unwrap().text(), text);
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn streamed_parallel_calls() {
    // Only its schema is used, the calls are answered below.
    #[openai_function]
    /// Returns the weather of a city.
    #[allow(dead_code)]
    fn get_weather(city: String) -> String {
        city
    }
    let mut ai = ai(None);
    ai.tool(Tool::Function(get_weather::openai_schema()));
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("What is the weather in Paris and in London? Call the function for both.");
    let mut stream = ai.stream(&mut session).await.unwrap();
    while let Some(chunk) = stream.next().await {
        chunk.unwrap();
    }
    drop(stream);

    // Arguments streamed in pieces are assembled into complete calls
    let calls: Vec<(String, String)> = session
        .pending_tool_calls()
        .iter()
        .map(|call| {
            let (city,) = get_weather::parse_arguments(&call.args().unwrap()).unwrap();
            (call.id.clone(), city)
        })
        .collect();
    assert_eq!(calls.len(), 2, "{calls:?}");
    for (id, city) in calls {
        let weather = if city.contains("Paris") { 21 } else { 14 };
        session
            .add_tool_result(id, format!("{weather} degrees Celsius"))
            .unwrap();
    }
    let completion = ai.ask(&mut session).await.unwrap();
    let reply = completion.text();
    assert!(reply.contains("21") && reply.contains("14"), "{reply}");
}

#[openai_function]
/// Returns the secret number.
fn secret_number() -> u32 {
    7319
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn function_calling_loop() {
    let mut ai = ai(None);
    ai.tool(Tool::Function(secret_number::openai_schema()));
    let mut session = Session::new(SlidingWindow::new(10));
    let mut completion = ai
        .ask(session.ask("What is the secret number plus one?"))
        .await
        .unwrap();
    while !session.pending_tool_calls().is_empty() {
        let results = execute_function_calls!(session, secret_number);
        assert!(results[0].is_some());
        completion = ai.ask(&mut session).await.unwrap();
    }
    assert!(completion.text().contains("7320"));
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn image_in_prompt() {
    let chart = ImageUrl::from_path(IMAGE, mime::IMAGE_PNG).await.unwrap();
    let mut session = Session::default();
    session.ask(vec![
        ContentPart::from("What kind of chart is this?"),
        chart.into(),
    ]);
    let completion = ai(None).ask(&mut session).await.unwrap();
    let reply = completion.text().to_lowercase();
    assert!(reply.contains("scatter"), "{reply}");
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn pdf_in_prompt() {
    let ai = ai(None);
    let pdf = pdf("The secret word is PINEAPPLE");
    // Gemini reads PDFs as images and rejects files. OpenAI reads them only as files.
    let pdf: ContentPart = if ai
        .get_base_url()
        .contains("generativelanguage.googleapis.com")
    {
        ImageUrl::from_bytes(mime::APPLICATION_PDF, pdf).into()
    } else {
        File::from_bytes("secret.pdf", mime::APPLICATION_PDF, pdf).into()
    };
    let mut session = Session::default();
    session.ask(vec![
        ContentPart::from("What is the secret word in this PDF?"),
        pdf,
    ]);
    let completion = ai.ask(&mut session).await.unwrap();
    assert!(completion.text().to_uppercase().contains("PINEAPPLE"));
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn image_after_tool_result() {
    #[openai_function]
    /// Takes a screenshot of the user's screen.
    #[allow(dead_code)]
    fn take_screenshot() {}

    let mut ai = ai(None);
    ai.tool(Tool::Function(take_screenshot::openai_schema()));
    let mut session = Session::new(SlidingWindow::new(10));
    ai.ask(session.ask("Take a screenshot. What kind of chart is on my screen?"))
        .await
        .unwrap();
    let call_id = session.pending_tool_calls()[0].id.clone();
    // Tool results are text, so the screenshot follows in a user message.
    session
        .add_tool_result(call_id, "Screenshot taken, sent in the next message.")
        .unwrap();
    let screenshot = ImageUrl::from_path(IMAGE, mime::IMAGE_PNG).await.unwrap();
    session.ask(screenshot);
    let completion = ai.ask(&mut session).await.unwrap();
    let reply = completion.text().to_lowercase();
    assert!(reply.contains("scatter"), "{reply}");
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn reasoning_effort() {
    let mut ai = ai(None);
    ai.reasoning_effort(ReasoningEffort::Low);
    let mut session = Session::default();
    let completion = ai
        .ask(session.ask("How many letters r are in strawberry? Answer with a digit."))
        .await
        .unwrap();
    assert!(completion.text().contains('3'));
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn wrong_api_key() {
    let mut ai = ai(None);
    ai.api_key("wrong_api_key");
    match ai.ask(Session::default().ask("Hi")).await {
        Err(Error::Api { status, error }) => {
            assert!(status.unwrap().is_client_error());
            assert!(!error.message.is_empty());
        }
        other => panic!("Expected Api error, got {other:?}"),
    }
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY"]
async fn embeddings() {
    let model = std::env::var("OPENAI_EMBEDDING_MODEL").unwrap_or("text-embedding-3-small".into());
    let mut embedder = Embedder::from_env(model).unwrap();
    embedder.dimensions(256);
    let response = embedder
        .embed(vec![
            "I love programming in Rust",
            "Rust is my favourite programming language",
            "It is going to rain tomorrow",
        ])
        .await
        .unwrap();
    let embeddings = response.into_embeddings();
    assert!(embeddings.iter().all(|e| e.len() == 256));
    let similarity = |a: usize, b: usize| -> f32 {
        let (a, b) = (&embeddings[a], &embeddings[b]);
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / norm(a) / norm(b)
    };
    assert!(similarity(0, 1) > similarity(0, 2));
}
