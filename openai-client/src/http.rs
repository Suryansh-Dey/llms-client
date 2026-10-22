//! Helpers shared by the clients.
use crate::{ApiError, DEFAULT_BASE_URL, EnvVarError, Error};
use reqwest::{RequestBuilder, Response};
use serde::Deserialize;
use serde_json::Value;

/// Adds `Authorization: Bearer <api_key>` header, unless `api_key` is empty (e.g. for local servers).
pub(crate) fn authorize(request: RequestBuilder, api_key: &str) -> RequestBuilder {
    if api_key.is_empty() {
        request
    } else {
        request.bearer_auth(api_key)
    }
}
/// Reads `OPENAI_API_KEY` (required) and `OPENAI_BASE_URL` (optional, unset or empty means
/// `DEFAULT_BASE_URL`).
pub(crate) fn read_env() -> Result<(String, String), EnvVarError> {
    let var = |name| std::env::var(name).map_err(|source| EnvVarError { name, source });
    let api_key = var("OPENAI_API_KEY")?;
    let base_url = match var("OPENAI_BASE_URL") {
        Ok(base_url) if !base_url.is_empty() => base_url,
        Ok(_)
        | Err(EnvVarError {
            source: std::env::VarError::NotPresent,
            ..
        }) => DEFAULT_BASE_URL.to_string(),
        Err(e) => return Err(e),
    };
    Ok((api_key, base_url))
}
pub(crate) fn join_url(base_url: &str, path: &str) -> String {
    format!("{}/{path}", base_url.trim_end_matches('/'))
}
pub(crate) async fn error_from_response(response: Response) -> Error {
    let status = response.status();
    match response.text().await {
        Ok(body) => Error::Api {
            status: Some(status),
            error: ApiError::from_body(&body),
        },
        Err(e) => e.into(),
    }
}
/// Reads the JSON body of a response, treating error status or an error object as an error.
pub(crate) async fn read_json(response: Response) -> Result<Value, Error> {
    if !response.status().is_success() {
        return Err(error_from_response(response).await);
    }
    let status = response.status();
    let body = response.text().await?;
    let value: Value = serde_json::from_str(&body)
        .map_err(|e| Error::InvalidResponse(format!("JSON parsing error [{e}]: {body}")))?;
    // Some servers report errors with status 200
    if value.get("error").is_some_and(|e| !e.is_null()) && value.get("choices").is_none() {
        return Err(Error::Api {
            status: Some(status),
            error: ApiError::from_value(&value),
        });
    }
    Ok(value)
}
pub(crate) fn deserialize_value<'a, T: Deserialize<'a>>(value: &'a Value) -> Result<T, Error> {
    T::deserialize(value)
        .map_err(|e| Error::InvalidResponse(format!("Unexpected response format [{e}]: {value}")))
}
/// Downloads `url`, returning its mime type from `Content-Type`, and its body.
pub(crate) async fn download(url: &str) -> Result<(mime::Mime, bytes::Bytes), Error> {
    let response = reqwest::get(url).await?;
    if !response.status().is_success() {
        return Err(error_from_response(response).await);
    }
    let mime = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| {
            Error::InvalidResponse("Download has no valid Content-Type header".into())
        })?;
    Ok((mime, response.bytes().await?))
}
