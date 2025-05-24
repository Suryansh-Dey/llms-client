use super::types::request::*;
use crate::utils::{self, MatchedFiles};
use regex::Regex;
use reqwest::header::HeaderMap;
use std::time::Duration;

const REQ_TIMEOUT: Duration = Duration::from_secs(10);

pub struct MarkdownToPartsBuilder {
    regex: Option<Regex>,
    guess_mime_type: Option<fn(url: &str) -> mime::Mime>,
    decide_download: Option<fn(headers: &HeaderMap) -> bool>,
    timeout: Option<Duration>,
}
impl MarkdownToPartsBuilder {
    ///# Panics
    ///`regex` must have a Regex with only 1 capture group with file URL as first capture
    ///group, else it PANICS when `.build()` is called.
    pub fn regex(mut self, regex: Regex) -> Self {
        self.regex = Some(regex);
        self
    }
    /// `guess_mime_type` is used to detect mimi_type of URL pointing to file system or web resource
    /// with no "Content-Type" header.
    pub fn guess_mime_type(mut self, guess_mime_type: fn(url: &str) -> mime::Mime) -> Self {
        self.guess_mime_type = Some(guess_mime_type);
        self
    }
    /// `decide_download` is used to decide if to download. If it returns false, resource will not
    /// be fetched and won't be in `parts`
    pub fn decide_download(mut self, decide_download: fn(headers: &HeaderMap) -> bool) -> Self {
        self.decide_download = Some(decide_download);
        self
    }
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
    pub async fn build<'a>(self, markdown: &'a str) -> MarkdownToParts<'a> {
        MarkdownToParts {
            markdown,
            base64s: utils::get_file_base64s(
                markdown,
                self.regex
                    .unwrap_or(Regex::new(r"(?s)!\[.*?].?\((.*?)\)").unwrap()),
                self.guess_mime_type.unwrap_or(|_| mime::IMAGE_PNG),
                |_| true,
                self.timeout.unwrap_or(REQ_TIMEOUT),
            )
            .await,
        }
    }
}
///Converts markdown to parts considering `![image](link)` means Gemini will be see the images too. `link` can be URL or file path.  
pub struct MarkdownToParts<'a> {
    markdown: &'a str,
    base64s: Vec<MatchedFiles>,
}
impl<'a> MarkdownToParts<'a> {
    pub fn builder() -> MarkdownToPartsBuilder {
        MarkdownToPartsBuilder {
            regex: None,
            guess_mime_type: None,
            decide_download: None,
            timeout: None,
        }
    }
    ///# Panics
    ///`regex` must have a Regex with only 1 capture group with file URL as first capture
    ///group, else it PANICS.
    /// # Arguments
    /// `guess_mime_type` is used to detect mimi_type of URL pointing to file system or web resource
    /// with no "Content-Type" header.
    /// `decide_download` is used to decide if to download. If it returns false, resource will not
    /// be fetched and won't be in `parts`
    /// # Example
    /// ```ignore
    /// from_regex("Your markdown string...", Regex::new(r"(?s)!\[.*?].?\((.*?)\)").unwrap(), |_| mime::IMAGE_PNG, |_| true)
    /// ```
    pub async fn from_regex_checked(
        markdown: &'a str,
        regex: Regex,
        guess_mime_type: fn(url: &str) -> mime::Mime,
        decide_download: fn(headers: &HeaderMap) -> bool,
    ) -> Self {
        Self {
            base64s: utils::get_file_base64s(
                markdown,
                regex,
                guess_mime_type,
                decide_download,
                REQ_TIMEOUT,
            )
            .await,
            markdown,
        }
    }
    ///# Panics
    /// `regex` must have a Regex with only 1 capture group with file URL as first capture group, else it PANICS.
    /// # Arguments
    /// `guess_mime_type` is used to detect mimi_type of URL pointing to file system or web resource
    /// with no "Content-Type" header.
    /// # Example
    /// ```ignore
    /// from_regex("Your markdown string...", Regex::new(r"(?s)!\[.*?].?\((.*?)\)").unwrap(), |_|
    /// mime::IMAGE_PNG)
    /// ```
    pub async fn from_regex(
        markdown: &'a str,
        regex: Regex,
        guess_mime_type: fn(url: &str) -> mime::Mime,
    ) -> Self {
        Self::from_regex_checked(markdown, regex, guess_mime_type, |_| true).await
    }
    /// # Arguments
    /// `guess_mime_type` is used to detect mimi_type of URL pointing to file system or web resource
    /// with no "Content-Type" header.
    /// `decide_download` is used to decide if to download. If it returns false, resource will not
    /// be fetched and won't be in `parts`
    /// # Example
    /// ```ignore
    /// new("Your markdown string...", |_| mime::IMAGE_PNG, |_| true)
    /// ```
    pub async fn new_checked(
        markdown: &'a str,
        guess_mime_type: fn(url: &str) -> mime::Mime,
        decide_download: fn(headers: &HeaderMap) -> bool,
    ) -> Self {
        let image_regex = Regex::new(r"(?s)!\[.*?].?\((.*?)\)").unwrap();
        Self::from_regex_checked(markdown, image_regex, guess_mime_type, decide_download).await
    }
    /// # Arguments
    /// `guess_mime_type` is used to detect mimi_type of URL pointing to file system or web resource
    /// with no "Content-Type" header.
    /// # Example
    /// ```ignore
    /// new("Your markdown string...", |_| mime::IMAGE_PNG)
    /// ```
    pub async fn new(markdown: &'a str, guess_mime_type: fn(url: &str) -> mime::Mime) -> Self {
        Self::new_checked(markdown, guess_mime_type, |_| true).await
    }
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
                parts.push(Part::text(text.to_string()));
                parts.push(Part::inline_data(InlineData::new(mime_type, base64)));

                self.markdown = &self.markdown[end..];
                removed_length += end;
            }
        }
        if self.markdown.len() != 0 {
            parts.push(Part::text(self.markdown.to_string()));
        }
        parts
    }
}
