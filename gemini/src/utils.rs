use actix_web::HttpMessage;
use awc::Client;
use base64::{Engine, engine::general_purpose::STANDARD};
use futures::future::join_all;
use regex::Regex;
use std::time::Duration;

const REQ_TIMEOUT: Duration = Duration::from_secs(10);

pub struct MatchedFiles {
    pub index: usize,
    pub length: usize,
    pub mime_type: String,
    pub base64: Option<String>,
}
/// # Panics
/// `regex` must have a Regex with atleast 1 capture group with file URL as first capture group, else it PANICS
/// # Arguments
/// `guess_mime_type` is used to detect mimi_type of URL pointing to file system or web resource
/// with no "Content-Type" header.
pub async fn get_file_base64s(
    markdown: &str,
    regex: Regex,
    guess_mime_type: fn(url: &str) -> String,
) -> Vec<MatchedFiles> {
    let client = Client::builder().timeout(REQ_TIMEOUT).finish();
    let mut tasks: Vec<_> = Vec::new();

    for file in regex.captures_iter(&markdown) {
        let url = file[1].to_string();
        let capture = file.get(0).unwrap();
        tasks.push((async |capture: regex::Match<'_>| {
            let (mime_type, base64) = if url.starts_with("https://") || url.starts_with("http://") {
                let response = client.get(&url).send().await;
                match response {
                    Ok(mut response) => (
                        response
                            .mime_type()
                            .ok()
                            .flatten()
                            .map(|mime| mime.to_string())
                            .unwrap_or_else(|| guess_mime_type(&url)),
                        response
                            .body()
                            .await
                            .ok()
                            .map(|bytes| STANDARD.encode(bytes)),
                    ),
                    Err(_) => (guess_mime_type(&url), None),
                }
            } else {
                (
                    guess_mime_type(&url),
                    tokio::fs::read(url)
                        .await
                        .ok()
                        .map(|bytes| STANDARD.encode(&bytes)),
                )
            };
            MatchedFiles {
                index: capture.start(),
                length: capture.len(),
                mime_type,
                base64,
            }
        })(capture));
    }
    join_all(tasks).await
}
