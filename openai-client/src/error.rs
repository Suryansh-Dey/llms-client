#[cfg(feature = "reqwest")]
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
    #[cfg(feature = "reqwest")]
    pub(crate) fn from_body(body: &str) -> Self {
        match serde_json::from_str::<Value>(body) {
            Ok(value) => Self::from_value(&value),
            Err(_) => Self::from_message(body.trim()),
        }
    }
    #[cfg(feature = "reqwest")]
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
    #[cfg(feature = "reqwest")]
    fn from_message(message: &str) -> Self {
        Self {
            message: message.to_string(),
            r#type: None,
            param: None,
            code: None,
        }
    }
}

/// Error of a request to the API.
#[cfg(feature = "reqwest")]
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// The request couldn't be sent or its response read, e.g. no connection or a timeout.
    #[error(transparent)]
    Http(reqwest::Error),
    /// Error sent by the API. `status` is `None` for an error event in the middle of a stream.
    #[error("API error{}: {error}", api_error_context(status, error))]
    Api {
        status: Option<StatusCode>,
        error: ApiError,
    },
    /// The response isn't as per the OpenAI specification. Has the reason and the response.
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
    /// The last message of the session is from the assistant, so there is nothing to respond to.
    #[error("Nothing to respond to, as the last message is from the assistant")]
    NothingToRespond,
}
/// Strips the URL, as `reqwest::Error` prints the request URL, which may hold secrets.
#[cfg(feature = "reqwest")]
impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        Self::Http(error.without_url())
    }
}
/// ` (401 Unauthorized, invalid_request_error)`, or the parts of it that are known.
#[cfg(feature = "reqwest")]
fn api_error_context(status: &Option<StatusCode>, error: &ApiError) -> String {
    let context: Vec<String> = status
        .map(|status| status.to_string())
        .into_iter()
        .chain(error.r#type.clone())
        .collect();
    if context.is_empty() {
        String::new()
    } else {
        format!(" ({})", context.join(", "))
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
