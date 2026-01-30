#[derive(thiserror::Error, Debug)]
pub enum GeminiResponseError {
    #[error(transparent)]
    #[cfg(feature = "reqwest")]
    ReqwestError(reqwest::Error),
    #[error("Response status not Ok. Response string: {0}")]
    StatusNotOk(String),
    #[error("Cannot Responnd if last Chat has Role::Model")]
    NothingToRespond,
}

#[derive(thiserror::Error, Debug)]
pub enum GeminiResponseStreamError {
    #[error(transparent)]
    #[cfg(feature = "reqwest")]
    ReqwestError(reqwest::Error),
    #[error("Invalid Response Format received. Response: {0}")]
    InvalidResposeFormat(String),
}
