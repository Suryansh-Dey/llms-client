use super::types::request::*;
use crate::utils;

///Converts markdown to parts considering `![image](line)` means Gemini will be see the images too
pub struct MarkdownToParts<'a> {
    base64s: Vec<(Option<String>, Option<String>)>,
    markdown: &'a str,
}
impl<'a> MarkdownToParts<'a> {
    pub async fn new(markdown: &'a str) -> Self {
        Self {
            base64s: utils::get_image_base64s(markdown).await,
            markdown,
        }
    }
    pub fn process(self, default_mime_type: String) -> Vec<Part> {
        let image_detect = utils::get_image_regex();
        let mut parts: Vec<Part> = Vec::new();
        let mut i = 0;
        let mut markdown = self.markdown;
        while let Some(image) = image_detect.find(&markdown) {
            let start = image.start();
            let image_markdown = image.as_str();
            let text = &markdown[..start + image_markdown.len()];

            parts.push(Part::text(text.to_string()));
            if let (mime, Some(base64)) = &self.base64s[i] {
                parts.push(Part::inline_data(InlineData::new(
                    mime.as_ref().unwrap_or(&default_mime_type).to_string(),
                    base64.to_string(),
                )));
            }

            markdown = &markdown[start + image_markdown.len()..];
            i += 1;
        }
        if markdown.len() != 0 {
            parts.push(Part::text(markdown.to_string()));
        }
        parts
    }
}
