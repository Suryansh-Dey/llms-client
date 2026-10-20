use open_client::openai::ask::{DEFAULT_BASE_URL, OpenAI};
use open_client::openai::types::sessions::Session;
use open_client::openai::utils::MarkdownToParts;

fn ai() -> OpenAI {
    OpenAI::new(
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not found"),
        std::env::var("OPENAI_MODEL").unwrap_or("gpt-5-mini".into()),
        None,
    )
    .set_base_url(std::env::var("OPENAI_BASE_URL").unwrap_or(DEFAULT_BASE_URL.into()))
}

#[tokio::test]
async fn live_see_web_markdown() {
    let parts = MarkdownToParts::new("What can you see inside this image ![but fire](https://th.bing.com/th?id=ORMS.0ba175d4898e31ae84dc62d9cd09ec84&pid=Wdp&w=612&h=304&qlt=90&c=1&rs=1&dpr=1.5&p=0)?", |_| mime::IMAGE_PNG).await.process();

    let mut session = Session::new(6);
    let response = ai().ask(session.ask_parts(parts)).await.unwrap();

    println!("{}", response.get_chat().get_text_no_think(""));
}

#[tokio::test]
async fn live_see_fs_markdown() {
    let parts = MarkdownToParts::new(
        "What can you see inside this image ![but fire](tests/lda.png)?",
        |_| mime::IMAGE_PNG,
    )
    .await
    .process();
    let mut session = Session::new(6);
    let mut ai = ai();
    ai.set_generation_config()["temperature"] = 0.into();
    let response = ai
        .ask(session.ask_parts(parts))
        .await
        .unwrap()
        .get_chat()
        .get_text_no_think("\n");

    assert!(response.contains("scatter plot"));
    println!("{}", response);
}
