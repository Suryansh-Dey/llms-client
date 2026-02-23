use reqwest::StatusCode;
use serde::{Deserialize, Deserializer};
use serde_json::Value;

fn deserialize_status_code<'de, D>(deserializer: D) -> Result<StatusCode, D::Error>
where
    D: Deserializer<'de>,
{
    let s = u16::deserialize(deserializer)?;
    StatusCode::from_u16(s).map_err(serde::de::Error::custom)
}
#[derive(Deserialize, thiserror::Error, Debug)]
#[error("{code}, status {status} : {message}")]
pub struct Error {
    #[serde(deserialize_with = "deserialize_status_code")]
    pub code: StatusCode,
    pub message: String,
    pub status: String,
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
    #[cfg(feature = "reqwest")]
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
    #[cfg(feature = "reqwest")]
    ReqwestError(reqwest::Error),
    #[error("Invalid Response Format received. Response: {0}")]
    ///Invalid Response Format received. Contains response string
    InvalidResposeFormat(String),
}
