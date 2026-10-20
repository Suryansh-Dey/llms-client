#[cfg(feature = "reqwest")]
use super::types::request::*;
#[cfg(feature = "reqwest")]
use crate::utils::MatchedFiles;
#[cfg(feature = "reqwest")]
use regex::Regex;
#[cfg(feature = "reqwest")]
use reqwest::header::HeaderMap;
#[cfg(feature = "reqwest")]
use std::time::Duration;
mod macros;
pub use macros::OpenAISchema;
pub use open_client_proc_macros::{
    execute_function_calls, execute_function_calls_with_callback, openai_function, openai_schema,
};

#[cfg(feature = "reqwest")]
const REQ_TIMEOUT: Duration = Duration::from_secs(10);

/// Builds `MarkdownToParts`. Create it with `MarkdownToParts::builder()`.
#[cfg(feature = "reqwest")]
pub struct MarkdownToPartsBuilder {
    regex: Option<Regex>,
    guess_mime_type: Option<fn(url: &str) -> mime::Mime>,
    decide_download: Option<fn(headers: &HeaderMap) -> bool>,
    timeout: Option<Duration>,
}
#[cfg(feature = "reqwest")]
impl MarkdownToPartsBuilder {
    /// Regex matching the files, with the file URL as its first capture group.
    /// Default is `(?s)!\[.*?].?\((.*?)\)`, i.e. markdown images `![alt](link)`.
    ///
    /// # Panics
    /// `.build()` panics if `regex` has no capture group.
    pub fn regex(mut self, regex: Regex) -> Self {
        self.regex = Some(regex);
        self
    }
    /// Used to detect mime type of a file path or a web resource with no "Content-Type" header.
    /// Default is `image/png`.
    pub fn guess_mime_type(mut self, guess_mime_type: fn(url: &str) -> mime::Mime) -> Self {
        self.guess_mime_type = Some(guess_mime_type);
        self
    }
    /// Decides if a web resource is downloaded, from its response headers. If it returns false,
    /// the resource is not fetched and stays as text. Default downloads everything.
    pub fn decide_download(mut self, decide_download: fn(headers: &HeaderMap) -> bool) -> Self {
        self.decide_download = Some(decide_download);
        self
    }
    /// Timeout of each download. Default is 10 seconds.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
    /// Downloads web resources and reads local files of `markdown`.
    pub async fn build(self, markdown: &str) -> MarkdownToParts<'_> {
        MarkdownToParts {
            markdown,
            base64s: crate::utils::get_file_base64s(
                markdown,
                self.regex
                    .unwrap_or_else(|| Regex::new(r"(?s)!\[.*?].?\((.*?)\)").unwrap()),
                self.guess_mime_type.unwrap_or(|_| mime::IMAGE_PNG),
                self.decide_download.unwrap_or(|_| true),
                self.timeout.unwrap_or(REQ_TIMEOUT),
            )
            .await,
        }
    }
}

/// Converts markdown to parts considering `![image](link)` means the model will see the images too.
/// `link` can be a URL or a file path.
///
/// # Example
/// ```no_run
/// # async fn run() {
/// use open_client::openai::utils::MarkdownToParts;
/// let parts = MarkdownToParts::builder()
///     .guess_mime_type(|_| open_client::mime::IMAGE_JPEG)
///     .build("What is in this image? ![photo](photo.jpg)")
///     .await
///     .process();
/// # }
/// ```
#[cfg(feature = "reqwest")]
#[derive(Clone)]
pub struct MarkdownToParts<'a> {
    markdown: &'a str,
    base64s: Vec<MatchedFiles>,
}
#[cfg(feature = "reqwest")]
impl<'a> MarkdownToParts<'a> {
    pub fn builder() -> MarkdownToPartsBuilder {
        MarkdownToPartsBuilder {
            regex: None,
            guess_mime_type: None,
            decide_download: None,
            timeout: None,
        }
    }
    /// Splits the markdown into text and the files it links. Files which couldn't be fetched stay
    /// as text.
    pub fn process(mut self) -> Vec<Part> {
        let mut parts: Vec<Part> = Vec::new();
        let mut removed_length = 0;
        for file in self.base64s {
            if let MatchedFiles {
                index,
                length,
                mime_type: Some(mime_type),
                base64: Some(base64),
            } = file
            {
                let end = index + length - removed_length;
                let text = &self.markdown[..end];
                parts.push(text.into());
                parts.push(InlineData::new(mime_type, base64).into());

                self.markdown = &self.markdown[end..];
                removed_length += end;
            }
        }
        if !self.markdown.is_empty() {
            parts.push(self.markdown.into());
        }
        parts
    }
}
