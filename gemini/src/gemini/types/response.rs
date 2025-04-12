use super::request::*;
use super::sessions::Session;
use actix_web::dev::{Decompress, Payload};
use awc::ClientResponse;
use derive_new::new;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    pin::Pin,
    task::{Context, Poll},
};

#[derive(Serialize, Deserialize, new)]
#[allow(non_snake_case)]
struct Candidate {
    content: Chat,
    pub finishReason: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct GeminiResponse {
    candidates: Vec<Candidate>,
    pub usageMetadata: Value,
    pub modelVersion: String,
}
impl GeminiResponse {
    pub(crate) async fn new(
        mut response: ClientResponse<Decompress<Payload>>,
    ) -> Result<GeminiResponse, awc::error::JsonPayloadError> {
        response.json().await
    }
    pub(crate) fn from(string: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(string)
    }
    pub fn get_parts(&self) -> &Vec<Part> {
        self.candidates[0].content.parts()
    }
    pub fn parse_json(text: &str) -> Result<Value, serde_json::Error> {
        let unescaped_str = text.replace("\\\"", "\"").replace("\\n", "\n");
        serde_json::from_str::<Value>(&unescaped_str)
    }
    pub fn extract_text(parts: &[Part], seperator: &str) -> String {
        let mut concatinated_string = String::new();
        for part in parts {
            if let Part::text(text) = part {
                concatinated_string.push_str(text);
                concatinated_string.push_str(seperator);
            }
        }
        concatinated_string
    }
    ///`seperator` used to concatinate all text parts. TL;DR use "" as seperator.
    pub fn get_text(&self, seperator: &str) -> String {
        Self::extract_text(&self.get_parts(), seperator)
    }
}

pin_project_lite::pin_project! {
#[derive(new)]
    pub struct GeminiResponseStream<'a>{
        #[pin]
        response_stream:ClientResponse<Decompress<Payload>>,
        session: &'a mut Session
    }
}
impl<'a> Stream for GeminiResponseStream<'a> {
    type Item = Result<GeminiResponse, Box<dyn std::error::Error>>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();

        match this.response_stream.poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                let text = String::from_utf8_lossy(&bytes);
                if text == "]" {
                    Poll::Ready(None)
                } else {
                    let response = GeminiResponse::from(text[1..].trim())?;
                    this.session.update(&response)?;
                    Poll::Ready(Some(Ok(response)))
                }
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e.into()))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}
