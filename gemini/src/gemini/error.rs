use awc::error::{JsonPayloadError, PayloadError, SendRequestError};
use derive_more::From;
use std::str::Utf8Error;

#[derive(Debug, From)]
pub enum GeminiResponseError {
    SendRequestError(SendRequestError),
    PayloadError(PayloadError),
    Utf8Error(Utf8Error),
    JsonParseError(JsonPayloadError),
    ///Contains the response string
    StatusNotOk(String),
}
impl std::fmt::Display for GeminiResponseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for GeminiResponseError {}

#[derive(Debug, From)]
pub enum GeminiResponseStreamError {
    InvalidResposeFormat(String),
    PayloadError(PayloadError),
}
impl std::fmt::Display for GeminiResponseStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for GeminiResponseStreamError {}
