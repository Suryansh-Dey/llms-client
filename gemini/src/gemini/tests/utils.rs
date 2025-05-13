use crate::gemini::types::request::Part;
use crate::gemini::types::response::GeminiResponse;
use crate::gemini::utils::MarkdownToParts;
use serde_json::json;

#[tokio::test]
async fn process_web() {
    let markdown = " water is good ![but fire](https://th.bing.com/th?id=ORMS.0ba175d4898e31ae84dc62d9cd09ec84&pid=Wdp&w=612&h=304&qlt=90&c=1&rs=1&dpr=1.5&p=0). thanks thanks";
    let parser = MarkdownToParts::new(markdown, |_| mime::IMAGE_PNG).await;
    let parts = parser.process();
    assert_eq!(GeminiResponse::extract_text(&parts, ""), markdown);
    assert_eq!(parts.len(), 3);
}

#[tokio::test]
async fn process_fs() {
    let markdown = " water is good ![but fire](tests/lda.png). thanks thanks";
    let parser = MarkdownToParts::new(markdown, |_| mime::IMAGE_PNG).await;
    let parts = parser.process();
    assert_eq!(GeminiResponse::extract_text(&parts, ""), markdown);
    assert_eq!(parts.len(), 3);
}

#[tokio::test]
async fn process() {
    let markdown = " water is good ![but fire](tests/lda.png).  thanks thanks ![but fire](https://th.bing.com/th?id=ORMS.0ba175d4898e31ae84dc62d9cd09ec84&pid=Wdp&w=612&h=304&qlt=90&c=1&rs=1&dpr=1.5&p=0).";
    let parser = MarkdownToParts::new(markdown, |_| mime::IMAGE_PNG).await;
    let parts = parser.process();
    assert_eq!(GeminiResponse::extract_text(&parts, ""), markdown);
    assert_eq!(parts.len(), 5);
    assert_eq!(
        json!(parts[2]),
        json!(Part::text(".  thanks thanks ![but fire](https://th.bing.com/th?id=ORMS.0ba175d4898e31ae84dc62d9cd09ec84&pid=Wdp&w=612&h=304&qlt=90&c=1&rs=1&dpr=1.5&p=0)".to_string()))
    );
}

#[tokio::test]
async fn process_with_error() {
    let markdown = " water is good ![but fire](lda.png).  thanks thanks ![but fire](https://th.bing.com/th?id=ORMS.0ba175d4898e31ae84dc62d9cd09ec84&pid=Wdp&w=612&h=304&qlt=90&c=1&rs=1&dpr=1.5&p=0).";
    let parser = MarkdownToParts::new(markdown, |_| mime::IMAGE_PNG).await;
    let parts = parser.process();
    assert_eq!(GeminiResponse::extract_text(&parts, ""), markdown);
    assert_eq!(parts.len(), 3);
    assert_eq!(json!(parts[2]), json!(Part::text(".".to_string())));
}
