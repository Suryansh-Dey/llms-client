use crate::gemini::types::response::GeminiResponse;
use crate::gemini::utils::MarkdownToParts;

#[actix_web::test]
async fn process_web() {
    let markdown = " water is good ![but fire](https://th.bing.com/th?id=ORMS.0ba175d4898e31ae84dc62d9cd09ec84&pid=Wdp&w=612&h=304&qlt=90&c=1&rs=1&dpr=1.5&p=0). thanks thanks";
    let parser = MarkdownToParts::new(markdown, |_| "image/png".to_string()).await;
    let parts = parser.process();
    assert_eq!(GeminiResponse::extract_text(&parts, ""), markdown);
    assert_eq!(parts.len(), 3);
}

#[actix_web::test]
async fn process_fs() {
    let markdown = " water is good ![but fire](tests/lda.png). thanks thanks";
    let parser = MarkdownToParts::new(markdown, |_| "image/png".to_string()).await;
    let parts = parser.process();
    assert_eq!(GeminiResponse::extract_text(&parts, ""), markdown);
    assert_eq!(parts.len(), 3);
}
