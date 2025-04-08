use actix_web::HttpMessage;
use awc::Client;
use base64::{Engine, engine::general_purpose::STANDARD};
use futures::future::join_all;
use regex::Regex;
use std::time::Duration;

const IMAGE_REQ_TIMEOUT:Duration = Duration::from_secs(10);

pub fn get_image_regex() -> Regex {
    Regex::new(r"!\[(.*?)\].?\((.*?)\)").unwrap()
}
///Returns Vector of tuple with (mime_type, base64)
pub async fn get_image_base64s(markdown: &str) -> Vec<(Option<String>, Option<String>)> {
    let client = Client::builder().timeout(IMAGE_REQ_TIMEOUT).finish();
    let image_detect = get_image_regex();
    let mut tasks: Vec<_> = Vec::new();

    for image in image_detect.captures_iter(&markdown) {
        let url = image[2].to_string();
        tasks.push(async {
            let response = client.get(url).send().await;
            let mut response = match response {
                Err(_) => return (None, None),
                Ok(data) => data,
            };
            let mime_type = response
                .mime_type()
                .ok()
                .flatten()
                .map(|mime| mime.to_string());
            let base64 = response
                .body()
                .await
                .and_then(|bytes| Ok(STANDARD.encode(bytes)))
                .ok();
            (mime_type, base64)
        });
    }
    join_all(tasks).await
}
