use base64::{Engine, engine::general_purpose::STANDARD};
use futures::future::join_all;
pub use mime;
use regex::Regex;
use reqwest::Client;
pub use reqwest::header::{HeaderMap, HeaderValue};
use std::time::Duration;

const REQ_TIMEOUT: Duration = Duration::from_secs(10);

pub struct MatchedFiles {
    pub index: usize,
    pub length: usize,
    pub mime_type: Option<String>,
    pub base64: Option<String>,
}
/// # Panics
/// `regex` must have a Regex with atleast 1 capture group with file URL as first capture group, else it PANICS
/// # Arguments
/// `guess_mime_type` is used to detect mimi_type of URL pointing to file system or web resource
/// with no "Content-Type" header.
pub async fn get_file_base64s(
    markdown: impl AsRef<str>,
    regex: Regex,
    guess_mime_type: fn(url: &str) -> mime::Mime,
    decide_download: fn(headers: &HeaderMap) -> bool,
) -> Vec<MatchedFiles> {
    let client = Client::builder().timeout(REQ_TIMEOUT).build().unwrap();
    let mut tasks = Vec::new();

    for file in regex.captures_iter(markdown.as_ref()) {
        let capture = file.get(0).unwrap();
        let url = file[1].to_string();
        tasks.push((async |capture: regex::Match<'_>, url: String| {
            let (mime_type, base64) = if url.starts_with("https://") || url.starts_with("http://") {
                let response = client.get(&url).send().await;
                match response {
                    Ok(response) if (decide_download)(response.headers()) => {
                        let mime_type = response
                            .headers()
                            .get("Content-Type")
                            .map(|mime| mime.to_str().ok())
                            .flatten()
                            .map(|str| str.to_string());

                        let base64 = response
                            .bytes()
                            .await
                            .ok()
                            .map(|bytes| STANDARD.encode(bytes));
                        let mime_type = match base64 {
                            Some(_) => {
                                mime_type.or_else(|| Some(guess_mime_type(&url).to_string()))
                            }
                            None => None,
                        };
                        (mime_type, base64)
                    }
                    _ => (None, None),
                }
            } else {
                let base64 = tokio::fs::read(url.clone())
                    .await
                    .ok()
                    .map(|bytes| STANDARD.encode(&bytes));
                match base64 {
                    Some(base64) => (Some(guess_mime_type(&url).to_string()), Some(base64)),
                    None => (None, None),
                }
            };
            MatchedFiles {
                index: capture.start(),
                length: capture.len(),
                mime_type,
                base64,
            }
        })(capture, url));
    }
    join_all(tasks).await
}
