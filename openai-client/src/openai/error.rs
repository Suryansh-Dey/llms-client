use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Error object sent by the API.
///
/// Servers differ in error format, so it is parsed leniently from
/// - `{"error": {"message": .., "type": .., "param": .., "code": ..}}` (OpenAI specification)
/// - `{"error": "message"}`, `{"message": ..}`, `{"detail": ..}` or a list of these
/// - any other body, kept as `message`
#[derive(Serialize, Deserialize, thiserror::Error, Debug, Clone)]
#[error("{message}")]
pub struct ApiError {
    pub message: String,
    /// Error type, e.g. `invalid_request_error`
    pub r#type: Option<String>,
    pub param: Option<String>,
    /// String or number, depending on the server.
    pub code: Option<Value>,
}
impl ApiError {
    pub(crate) fn from_body(body: &str) -> Self {
        match serde_json::from_str::<Value>(body) {
            Ok(value) => Self::from_value(&value),
            Err(_) => Self::from_message(body.trim()),
        }
    }
    pub(crate) fn from_value(value: &Value) -> Self {
        let value = match value {
            Value::Array(values) if !values.is_empty() => &values[0],
            value => value,
        };
        let error = match value.get("error") {
            Some(error) => error,
            None => value,
        };
        let message = match error {
            Value::String(message) => message.clone(),
            error => match error.get("message").or_else(|| error.get("detail")) {
                Some(Value::String(message)) => message.clone(),
                Some(detail) => detail.to_string(),
                None => error.to_string(),
            },
        };
        let field = |key| error.get(key).and_then(Value::as_str).map(str::to_string);
        Self {
            message,
            r#type: field("type"),
            param: field("param"),
            code: error.get("code").filter(|code| !code.is_null()).cloned(),
        }
    }
    fn from_message(message: &str) -> Self {
        Self {
            message: message.to_string(),
            r#type: None,
            param: None,
            code: None,
        }
    }
}

#[derive(thiserror::Error, Debug)]
#[error("API Error: {status}{}\nMessage: {error}", error.r#type.as_ref().map(|t| format!(", {t}")).unwrap_or_default())]
pub struct OpenAIError {
    pub status: StatusCode,
    pub error: ApiError,
}

#[derive(thiserror::Error, Debug)]
pub enum OpenAIResponseError {
    #[error(transparent)]
    ReqwestError(reqwest::Error),
    #[error("Response status not Ok.\n{0}")]
    StatusNotOk(OpenAIError),
    #[error("Invalid Response Format received. {0}")]
    ///Response is not as per OpenAI specification. Contains the error and response body.
    InvalidResponseFormat(String),
    #[error("Cannot Respond if last Chat has Role::Assistant")]
    ///Cannot Respond if last Chat has Role::Assistant
    NothingToRespond,
}
/// Strips the URL from the error, as `reqwest::Error` prints the request URL when displayed.
impl From<reqwest::Error> for OpenAIResponseError {
    fn from(error: reqwest::Error) -> Self {
        Self::ReqwestError(error.without_url())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum OpenAIResponseStreamError {
    #[error(transparent)]
    ReqwestError(reqwest::Error),
    #[error("Invalid Response Format received. {0}")]
    ///Invalid Response Format received. Contains the error and response string
    InvalidResponseFormat(String),
    #[error("Error event received in stream: {0}")]
    ///Error sent by the API in the middle of the stream.
    ApiError(ApiError),
}
/// Strips the URL from the error, as `reqwest::Error` prints the request URL when displayed.
impl From<reqwest::Error> for OpenAIResponseStreamError {
    fn from(error: reqwest::Error) -> Self {
        Self::ReqwestError(error.without_url())
    }
}

/// Error of `from_env()` constructors: an environment variable is missing or not valid unicode.
#[derive(thiserror::Error, Debug)]
#[error("`{name}` environment variable: {source}")]
pub struct EnvVarError {
    pub name: &'static str,
    #[source]
    pub source: std::env::VarError,
}
