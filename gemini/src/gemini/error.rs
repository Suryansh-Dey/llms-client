#[derive(Debug)]
pub enum GeminiResponseError {
    ReqwestError(reqwest::Error),
    ///Contains the response string
    StatusNotOk(String),
}
impl From<reqwest::Error> for GeminiResponseError {
    fn from(err: reqwest::Error) -> Self {
        GeminiResponseError::ReqwestError(err)
    }
}
impl From<String> for GeminiResponseError {
    fn from(s: String) -> Self {
        GeminiResponseError::StatusNotOk(s)
    }
}
impl std::fmt::Display for GeminiResponseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for GeminiResponseError {}

#[derive(Debug)]
pub enum GeminiResponseStreamError {
    ReqwestError(reqwest::Error),
    ///Contains the response string
    InvalidResposeFormat(String),
}
impl From<reqwest::Error> for GeminiResponseStreamError {
    fn from(err: reqwest::Error) -> Self {
        GeminiResponseStreamError::ReqwestError(err)
    }
}
impl From<String> for GeminiResponseStreamError {
    fn from(s: String) -> Self {
        GeminiResponseStreamError::InvalidResposeFormat(s)
    }
}
impl std::fmt::Display for GeminiResponseStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for GeminiResponseStreamError {}
