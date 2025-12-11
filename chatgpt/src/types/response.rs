use super::request::ChatMessage;
use crate::error::ChatGptStreamError;
use bytes::Bytes;
use derive_new::new;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use std::task::{Context, Poll};

#[derive(Debug, Clone, Serialize, Deserialize, new)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    #[serde(default, rename = "finish_reason")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatGptResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Value>,
}
impl ChatGptResponse {
    pub(crate) async fn new(response: reqwest::Response) -> Result<Self, reqwest::Error> {
        response.json().await
    }
    pub fn message(&self) -> &ChatMessage {
        &self.choices[0].message
    }
    pub fn text(&self) -> &str {
        &self.message().content
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaMessage {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceDelta {
    pub index: usize,
    pub delta: DeltaMessage,
    #[serde(default, rename = "finish_reason")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatGptStreamChunk {
    pub id: Option<String>,
    pub choices: Vec<ChoiceDelta>,
    #[serde(default)]
    pub usage: Option<Value>,
}
impl ChatGptStreamChunk {
    pub(crate) fn from_str(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }
    pub fn text_delta(&self) -> String {
        self.choices
            .iter()
            .filter_map(|choice| choice.delta.content.clone())
            .collect::<Vec<String>>()
            .join("")
    }
}

pin_project_lite::pin_project! {
    pub struct ResponseStream<F, T>
    where
        F: FnMut(&super::sessions::Session, ChatGptStreamChunk) -> T,
    {
        #[pin]
        response_stream: Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send + 'static>,
        session: super::sessions::Session,
        data_extractor: F,
        buffer: Vec<u8>,
    }
}

impl<F, T> Stream for ResponseStream<F, T>
where
    F: FnMut(&super::sessions::Session, ChatGptStreamChunk) -> T,
{
    type Item = Result<T, ChatGptStreamError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        loop {
            if let Some((pos, delim_len)) = find_delimiter(&this.buffer) {
                let message_bytes = this
                    .buffer
                    .drain(..pos + delim_len)
                    .collect::<Vec<u8>>();

                let raw = String::from_utf8_lossy(&message_bytes);
                if let Some(payload) = raw.trim().strip_prefix("data:") {
                    let payload = payload.trim();
                    if payload.is_empty() {
                        continue;
                    }
                    if payload == "[DONE]" {
                        return Poll::Ready(None);
                    }
                    let chunk = match ChatGptStreamChunk::from_str(payload) {
                        Ok(chunk) => chunk,
                        Err(e) => {
                            let err = ChatGptStreamError::InvalidResposeFormat(format!(
                                "JSON parsing error [{}]: {}",
                                e, payload
                            ));
                            return Poll::Ready(Some(Err(err)));
                        }
                    };
                    this.session.update_stream(&chunk);
                    let data = (this.data_extractor)(this.session, chunk);
                    return Poll::Ready(Some(Ok(data)));
                }
                continue;
            }

            match this.response_stream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    this.buffer.extend_from_slice(&bytes);
                }
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    if this.buffer.is_empty() {
                        return Poll::Ready(None);
                    } else {
                        let err_text = String::from_utf8_lossy(this.buffer).into_owned();
                        let err = ChatGptStreamError::InvalidResposeFormat(format!(
                            "Stream ended with incomplete data in the buffer: {}",
                            err_text
                        ));
                        this.buffer.clear();
                        return Poll::Ready(Some(Err(err)));
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    return Poll::Ready(Some(Err(ChatGptStreamError::ReqwestError(e))));
                }
            }
        }
    }
}

impl<F, T> ResponseStream<F, T>
where
    F: FnMut(&super::sessions::Session, ChatGptStreamChunk) -> T,
{
    pub(crate) fn new(
        response_stream: Box<
            dyn Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send + 'static,
        >,
        session: super::sessions::Session,
        data_extractor: F,
    ) -> Self {
        Self {
            response_stream,
            session,
            data_extractor,
            buffer: Vec::new(),
        }
    }
    pub fn get_session(&self) -> &super::sessions::Session {
        &self.session
    }
    pub fn get_session_owned(self) -> super::sessions::Session {
        self.session
    }
}

pub type ChatGptResponseStream =
    ResponseStream<fn(&super::sessions::Session, ChatGptStreamChunk) -> ChatGptStreamChunk, ChatGptStreamChunk>;

fn find_delimiter(buffer: &[u8]) -> Option<(usize, usize)> {
    buffer
        .windows(2)
        .position(|w| w == b"\n\n")
        .map(|pos| (pos, 2))
        .or_else(|| {
            buffer
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|pos| (pos, 4))
        })
}


