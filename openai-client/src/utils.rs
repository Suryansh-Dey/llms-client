//! Helpers of `MarkdownToParts`. Only built with the `reqwest` feature.
use mime::Mime;

#[derive(Clone)]
pub(crate) struct MatchedFiles {
    pub index: usize,
    pub length: usize,
    pub mime_type: Option<Mime>,
    pub base64: Option<String>,
}

/// Downloads (or reads from the file system) every file captured by the first capture group of
/// `regex`.
///
/// # Panics
/// `regex` must have at least 1 capture group with the file URL as first capture group.
/// # Arguments
/// `guess_mime_type` is used to detect mime type of URL pointing to file system or web resource
/// with no "Content-Type" header.
pub(crate) async fn get_file_base64s(
    markdown: impl AsRef<str>,
    regex: regex::Regex,
    guess_mime_type: fn(url: &str) -> Mime,
    decide_download: fn(headers: &reqwest::header::HeaderMap) -> bool,
    timeout: std::time::Duration,
) -> Vec<MatchedFiles> {
    use base64::{Engine, engine::general_purpose::STANDARD};
    use std::str::FromStr;

    let client = reqwest::Client::builder().timeout(timeout).build().unwrap();
    let mut tasks = Vec::new();

    for file in regex.captures_iter(markdown.as_ref()) {
        let capture = file.get(0).unwrap();
        let (index, length) = (capture.start(), capture.len());
        let url = file[1].to_string();
        let client = client.clone();
        tasks.push(async move {
            let (mime_type, base64) = if url.starts_with("https://") || url.starts_with("http://") {
                match client.get(&url).send().await {
                    Ok(response) if decide_download(response.headers()) => {
                        let mime_type = response
                            .headers()
                            .get("Content-Type")
                            .and_then(|mime| mime.to_str().ok())
                            .and_then(|mime| Mime::from_str(mime).ok());
                        let base64 = response
                            .bytes()
                            .await
                            .ok()
                            .map(|bytes| STANDARD.encode(bytes));
                        let mime_type = base64
                            .as_ref()
                            .map(|_| mime_type.unwrap_or_else(|| guess_mime_type(&url)));
                        (mime_type, base64)
                    }
                    _ => (None, None),
                }
            } else {
                match read_file(&url).await {
                    Some(bytes) => (Some(guess_mime_type(&url)), Some(STANDARD.encode(bytes))),
                    None => (None, None),
                }
            };
            MatchedFiles {
                index,
                length,
                mime_type,
                base64,
            }
        });
    }
    futures::future::join_all(tasks).await
}

#[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
async fn read_file(path: &str) -> Option<Vec<u8>> {
    tokio::fs::read(path).await.ok()
}
/// No file system without `tokio`, e.g. in WASM.
#[cfg(not(all(feature = "tokio", not(target_arch = "wasm32"))))]
async fn read_file(_path: &str) -> Option<Vec<u8>> {
    None
}
