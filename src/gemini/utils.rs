use super::types::*;
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
    pub fn process(&'a mut self, default_mime_type: &'a String) -> Vec<Part<'a>> {
        let image_detect = utils::get_image_regex();
        let mut parts: Vec<Part> = Vec::new();
        let mut i = 0;
        while let Some(image) = image_detect.find(&self.markdown) {
            let start = image.start();
            let image_markdown = image.as_str();
            let text = &self.markdown[..start + image_markdown.len()];

            parts.push(Part::text(text));
            if let (mime, Some(base64)) = &self.base64s[i] {
                parts.push(Part::inline_data(InlineData::new(
                    mime.as_ref().unwrap_or(default_mime_type),
                    &base64,
                )));
            }

            self.markdown = &self.markdown[start + image_markdown.len()..];
            i += 1;
        }
        parts.push(Part::text(&self.markdown));
        parts
    }
}
