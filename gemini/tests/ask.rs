use gemini_client_api::gemini::ask::Gemini;
use gemini_client_api::gemini::types::sessions::Session;
use gemini_client_api::gemini::utils::MarkdownToParts;

#[tokio::test]
async fn see_web_markdown() {
    let parts = MarkdownToParts::new("What can you see inside this image ![but fire](https://th.bing.com/th?id=ORMS.0ba175d4898e31ae84dc62d9cd09ec84&pid=Wdp&w=612&h=304&qlt=90&c=1&rs=1&dpr=1.5&p=0)?", |_| mime::IMAGE_PNG).await.process();

    let mut session = Session::new(6);
    let response = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-1.5-flash",
        None,
    )
    .ask(session.ask(parts))
    .await
    .unwrap();

    println!("{}", response.get_text(""));
}

#[tokio::test]
async fn see_fs_markdown() {
    let parts = MarkdownToParts::new(
        "What can you see inside this image ![but fire](tests/lda.png)?",
        |_| mime::IMAGE_PNG,
    )
    .await
    .process();
    let mut session = Session::new(6);
    let response = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-1.5-flash",
        None,
    )
    .ask(session.ask(parts))
    .await
    .unwrap();

    println!("{}", response.get_text(""));
}
