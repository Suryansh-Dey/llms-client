#[derive(Debug)]
pub enum ChatGptError {
    ReqwestError(reqwest::Error),
    StatusNotOk(String),
    InvalidResponseFormat(String),
}

#[derive(Debug)]
pub enum ChatGptStreamError {
    ReqwestError(reqwest::Error),
    InvalidResposeFormat(String),
}


