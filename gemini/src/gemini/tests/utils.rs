use crate::gemini::utils::MarkdownToParts;
use serde_json::json;

#[actix_web::test]
async fn process() {
    let parser = MarkdownToParts::new(" water is good ![but fire](https://th.bing.com/th?id=ORMS.0ba175d4898e31ae84dc62d9cd09ec84&pid=Wdp&w=612&h=304&qlt=90&c=1&rs=1&dpr=1.5&p=0). thanks thanks").await;
    let default_mime_type = String::from("image/png");
    let parts = parser.process(default_mime_type);
    for part in parts {
        println!("{:#?}", json!(part));
    }
}
