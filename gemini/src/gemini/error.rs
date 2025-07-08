#[derive(Debug)]
pub enum GeminiResponseError {
    ReqwestError(reqwest::Error),
    ///Contains the response string
    StatusNotOk(String),
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
impl std::fmt::Display for GeminiResponseStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for GeminiResponseStreamError {}
