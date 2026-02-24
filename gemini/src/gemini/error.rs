use reqwest::StatusCode;
use serde::{Deserialize, Deserializer};
use serde_json::Value;

#[derive(Deserialize, thiserror::Error, Debug, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Status {
    #[error("The request body is malformed.")]
    ///The request body is malformed.
    InvalidArgument,
    #[error(
        "Gemini API free tier is not available in your country. Please enable billing on your project in Google AI Studio."
    )]
    ///Gemini API free tier is not available in your country. Please enable billing on your project in Google AI Studio.
    FailedPrecondition,
    #[error("Your API key doesn't have the required permissions.")]
    ///Your API key doesn't have the required permissions.
    PermissionDenied,
    #[error("The requested resource wasn't found.")]
    ///The requested resource wasn't found.
    NotFound,
    #[error("You've exceeded the rate limit.")]
    ///You've exceeded the rate limit.
    ResourceExhausted,
    #[error("An unexpected error occurred on Google's side.")]
    ///An unexpected error occurred on Google's side.
    Internal,
    #[error("The service may be temporarily overloaded or down.")]
    ///The service may be temporarily overloaded or down.
    Unavailable,
    #[error("The service is unable to finish processing within the deadline.")]
    ///The service is unable to finish processing within the deadline.
    DeadlineExceeded,
}
fn deserialize_status_code<'de, D>(deserializer: D) -> Result<StatusCode, D::Error>
where
    D: Deserializer<'de>,
{
    let s = u16::deserialize(deserializer)?;
    StatusCode::from_u16(s).map_err(serde::de::Error::custom)
}
#[derive(Deserialize, thiserror::Error, Debug)]
#[error("{code}, {status}\nMessage: {message}\nDetails: {:#?}", details.as_ref().unwrap_or(&vec![]))]
pub struct Error {
    #[serde(deserialize_with = "deserialize_status_code")]
    pub code: StatusCode,
    pub message: String,
    pub status: Status,
    pub details: Option<Vec<Value>>,
}

#[derive(Deserialize, thiserror::Error, Debug)]
#[error("Gemini API Error: {error}")]
pub struct GeminiError {
    pub error: Error,
}
#[derive(thiserror::Error, Debug)]
pub enum GeminiResponseError {
    #[error(transparent)]
    ReqwestError(reqwest::Error),
    #[error("Response status not Ok.\n{0}")]
    StatusNotOk(GeminiError),
    #[error("Cannot Respond if last Chat has Role::Model")]
    ///Cannot Responnd if last Chat has Role::Model
    NothingToRespond,
}

#[derive(thiserror::Error, Debug)]
pub enum GeminiResponseStreamError {
    #[error(transparent)]
    ReqwestError(reqwest::Error),
    #[error("Invalid Response Format received. Response: {0}")]
    ///Invalid Response Format received. Contains response string
    InvalidResposeFormat(String),
}
