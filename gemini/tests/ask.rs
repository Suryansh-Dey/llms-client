use gemini_client_api::gemini::utils::MarkdownToParts;
use gemini_client_api::gemini::types::sessions::Session;
use gemini_client_api::gemini::ask::Gemini;

#[actix_web::test]
async fn see_markdown() {
    let parser = MarkdownToParts::new("What can you see inside this image ![but fire](https://th.bing.com/th?id=ORMS.0ba175d4898e31ae84dc62d9cd09ec84&pid=Wdp&w=612&h=304&qlt=90&c=1&rs=1&dpr=1.5&p=0)?").await;
    let default_mime_type = String::from("image/png");
    let parts = parser.process(default_mime_type);

    let mut session = Session::new(6);
    let response = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"),
        "gemini-1.5-flash".to_string(),
        None,
    )
    .ask(session.ask(parts))
    .await
    .unwrap();

    println!("{}", response.get_text(""));
}
