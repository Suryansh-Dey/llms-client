use super::types::request::*;
use crate::utils::{self, MatchedFiles};
use regex::Regex;

pub struct MarkdownToParts<'a> {
    base64s: Vec<Option<MatchedFiles>>,
    markdown: &'a str,
}
impl<'a> MarkdownToParts<'a> {
    ///# Panics
    /// `regex` must have a Regex with atleast 1 capture group with file URL as first capture group, else it PANICS.
    /// # Arguments
    /// `mime_type_guess` is used to detect mimi_type of URL pointing to file system or web resource
    /// with no "Content-Type" header.
    /// # Example
    /// ```rust
    /// from_regex("Your markdown string...", Regex::new(r"(?s)!\[.*?].?\((.*?)\)").unwrap(), |_| "image/png".to_string())
    /// ```
    pub async fn from_regex(
        markdown: &'a str,
        regex: Regex,
        guess_mime_type: fn(url: &str) -> String,
    ) -> Self {
        Self {
            base64s: utils::get_file_base64s(markdown, regex, guess_mime_type).await,
            markdown,
        }
    }
    ///Converts markdown to parts considering `![image](link)` means Gemini will be see the images too. `link` can be URL or file path.  
    /// `mime_type_guess` is used to detect mimi_type of URL pointing to file system or web resource
    /// with no "Content-Type" header.
    /// # Example
    /// ```rust
    /// new("Your markdown string...", |_| "image/png".to_string())
    /// ```
    pub async fn new(markdown: &'a str, guess_mime_type: fn(url: &str) -> String) -> Self {
        let image_regex = Regex::new(r"(?s)!\[.*?].?\((.*?)\)").unwrap();
        Self {
            base64s: utils::get_file_base64s(markdown, image_regex, guess_mime_type).await,
            markdown,
        }
    }
    pub fn process(mut self) -> Vec<Part> {
        let mut parts: Vec<Part> = Vec::new();
        let mut removed_length = 0;
        for file in self.base64s {
            if let Some(file) = file {
                let end = file.index + file.length - removed_length;
                let text = &self.markdown[..end];
                parts.push(Part::text(text.to_string()));
                parts.push(Part::inline_data(InlineData::new(
                    file.mime_type,
                    file.base64,
                )));

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
