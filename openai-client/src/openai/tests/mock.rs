//! Tests against a local HTTP server, checking what is sent and how replies are handled.
use crate::openai::ask::OpenAI;
use crate::openai::embed::OpenAIEmbedding;
use crate::openai::error::{OpenAIResponseError, OpenAIResponseStreamError};
use crate::openai::types::request::Role;
use crate::openai::types::sessions::Session;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

pub(crate) struct Request {
    pub line: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
}
impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

/// Serves one request with `status` and `chunks` as body, written one by one.
/// Returns base URL and the received request.
pub(crate) async fn serve(
    status: u16,
    content_type: &str,
    chunks: Vec<String>,
) -> (String, JoinHandle<Request>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let content_type = content_type.to_string();
    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buffer = Vec::new();
        let mut chunk = [0u8; 4096];
        let header_end = loop {
            let n = socket.read(&mut chunk).await.unwrap();
            buffer.extend_from_slice(&chunk[..n]);
            if let Some(p) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                break p;
            }
        };
        let head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
        let mut lines = head.lines();
        let line = lines.next().unwrap().to_string();
        let headers: Vec<(String, String)> = lines
            .filter_map(|l| l.split_once(':'))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect();
        let length: usize = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
            .map_or(0, |(_, v)| v.parse().unwrap());
        while buffer.len() < header_end + 4 + length {
            let n = socket.read(&mut chunk).await.unwrap();
            buffer.extend_from_slice(&chunk[..n]);
        }
        let body = serde_json::from_slice(&buffer[header_end + 4..]).unwrap_or(Value::Null);

        let head = format!(
            "HTTP/1.1 {status} STATUS\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n"
        );
        socket.write_all(head.as_bytes()).await.unwrap();
        for chunk in chunks {
            socket.write_all(chunk.as_bytes()).await.unwrap();
            socket.flush().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        socket.shutdown().await.ok();
        Request {
            line,
            headers,
            body,
        }
    });
    (base_url, handle)
}

fn completion(message: Value) -> String {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "mock-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "citations": ["https://example.com"]
    })
    .to_string()
}

#[tokio::test]
async fn ask_sends_request_to_base_url() {
    let (base_url, request) = serve(
        200,
        "application/json",
        vec![completion(
            json!({"role": "assistant", "content": "Hello!"}),
        )],
    )
    .await;
    let mut ai = OpenAI::new("secret", "mock-model", Some("Be brief".into()))
        .set_base_url(format!("{base_url}/"));
    ai.set_generation_config()["temperature"] = 0.5.into();
    let mut session = Session::new(6);
    let response = ai.ask(session.ask("Hi")).await.unwrap();

    assert_eq!(response.get_chat().get_text_no_think(""), "Hello!");
    assert_eq!(response.model.as_deref(), Some("mock-model"));
    assert_eq!(response.extra["citations"][0], "https://example.com");
    assert_eq!(session.get_history_length(), 2);
    assert_eq!(*session.get_last_chat().unwrap().role(), Role::Assistant);

    let request = request.await.unwrap();
    assert_eq!(request.line, "POST /v1/chat/completions HTTP/1.1");
    assert_eq!(request.header("authorization"), Some("Bearer secret"));
    assert_eq!(
        request.body,
        json!({
            "model": "mock-model",
            "messages": [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "Hi"}
            ],
            "temperature": 0.5
        })
    );
}

#[tokio::test]
async fn empty_api_key_sends_no_authorization() {
    let (base_url, request) = serve(
        200,
        "application/json",
        vec![completion(json!({"role": "assistant", "content": "ok"}))],
    )
    .await;
    let ai = OpenAI::new("", "local-model", None).set_base_url(base_url);
    ai.ask(Session::new(2).ask("Hi")).await.unwrap();
    assert_eq!(request.await.unwrap().header("authorization"), None);
}

#[tokio::test]
async fn error_status_is_parsed() {
    let body = json!({"error": {"message": "Incorrect API key", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}});
    let (base_url, _) = serve(401, "application/json", vec![body.to_string()]).await;
    let ai = OpenAI::new("wrong", "mock-model", None).set_base_url(base_url);
    match ai.ask(Session::new(2).ask("Hi")).await {
        Err(OpenAIResponseError::StatusNotOk(e)) => {
            assert_eq!(e.status, 401);
            assert_eq!(e.error.message, "Incorrect API key");
            assert_eq!(e.error.r#type.as_deref(), Some("invalid_request_error"));
            assert_eq!(e.error.code, Some(json!("invalid_api_key")));
        }
        other => panic!("Expected StatusNotOk, got {other:?}"),
    }
}

#[tokio::test]
async fn invalid_response_contains_body() {
    let (base_url, _) = serve(200, "text/html", vec!["<html>Not an API</html>".into()]).await;
    let ai = OpenAI::new("", "mock-model", None).set_base_url(base_url);
    match ai.ask(Session::new(2).ask("Hi")).await {
        Err(OpenAIResponseError::InvalidResponseFormat(e)) => assert!(e.contains("Not an API")),
        other => panic!("Expected InvalidResponseFormat, got {other:?}"),
    }
}

fn chunk(delta: Value, finish_reason: Option<&str>) -> String {
    let chunk = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    });
    format!("data: {chunk}\n\n")
}

#[tokio::test]
async fn stream_assembles_reply() {
    let chunks = vec![
        ": keep alive\n\n".to_string(),
        chunk(
            json!({"role": "assistant", "reasoning_content": "Thinking"}),
            None,
        ),
        chunk(json!({"reasoning_content": " hard"}), None),
        chunk(json!({"content": "Hel"}), None),
        // An event split across network chunks, with CRLF line endings
        format!(
            "data: {}",
            json!({"choices": [{"index": 0, "delta": {"content": "lo"}}]})
        ),
        "\r\n\r\n".to_string(),
        chunk(
            json!({"tool_calls": [{"index": 0, "id": "call_a", "type": "function", "function": {"name": "add", "arguments": ""}}]}),
            None,
        ),
        chunk(
            json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"a\": 1,"}}]}),
            None,
        ),
        chunk(
            json!({"tool_calls": [{"index": 1, "id": "call_b", "type": "function", "function": {"name": "add", "arguments": "{\"a\": 3, \"b\": 4}"}}]}),
            None,
        ),
        chunk(
            json!({"tool_calls": [{"index": 0, "function": {"arguments": " \"b\": 2}"}}]}),
            None,
        ),
        chunk(json!({}), Some("tool_calls")),
        format!(
            "data: {}\n\n",
            json!({"choices": [], "usage": {"total_tokens": 9}})
        ),
        "data: [DONE]\n\n".to_string(),
    ];
    let (base_url, request) = serve(200, "text/event-stream", chunks).await;
    let ai = OpenAI::new("key", "mock-model", None).set_base_url(base_url);
    let mut session = Session::new(10);
    session.ask("Add 1 + 2 and 3 + 4");

    let mut stream = ai.ask_as_stream(session).await.unwrap();
    let mut text = String::new();
    let mut usage = None;
    while let Some(response) = stream.next().await {
        let response = response.unwrap();
        text.push_str(&response.get_chat().get_text_no_think(""));
        usage = response.usage.or(usage);
    }
    assert_eq!(text, "Hello");
    assert_eq!(usage, Some(json!({"total_tokens": 9})));

    let session = stream.get_session_owned();
    assert_eq!(session.get_history_length(), 2);
    let reply = session.get_last_chat().unwrap();
    assert_eq!(reply.get_thoughts(""), "Thinking hard");
    assert_eq!(reply.get_text_no_think(""), "Hello");
    let calls: Vec<_> = reply
        .get_function_calls()
        .map(|call| (call.id().as_str(), call.args().unwrap()))
        .collect();
    assert_eq!(
        calls,
        [
            ("call_a", json!({"a": 1, "b": 2})),
            ("call_b", json!({"a": 3, "b": 4}))
        ]
    );
    assert_eq!(request.await.unwrap().body["stream"], true);
}

#[tokio::test]
async fn stream_error_event() {
    let chunks = vec![
        chunk(json!({"content": "Hi"}), None),
        format!(
            "data: {}\n\n",
            json!({"error": {"message": "Provider overloaded", "code": 502}})
        ),
    ];
    let (base_url, _) = serve(200, "text/event-stream", chunks).await;
    let ai = OpenAI::new("key", "mock-model", None).set_base_url(base_url);
    let mut session = Session::new(10);
    session.ask("Hi");
    let mut stream = ai.ask_as_stream(session).await.unwrap();
    assert!(stream.next().await.unwrap().is_ok());
    match stream.next().await.unwrap() {
        Err(OpenAIResponseStreamError::ApiError(e)) => {
            assert_eq!(e.message, "Provider overloaded");
            assert_eq!(e.code, Some(json!(502)));
        }
        other => panic!("Expected ApiError, got {other:?}"),
    }
    // Stream without [DONE] still ends
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn stream_status_not_ok_returns_session() {
    let (base_url, _) = serve(
        404,
        "application/json",
        vec![r#"{"error": "model 'x' not found"}"#.into()],
    )
    .await;
    let ai = OpenAI::new("", "x", None).set_base_url(base_url);
    let mut session = Session::new(10);
    session.ask("Hi");
    match ai.ask_as_stream(session).await {
        Err((session, OpenAIResponseError::StatusNotOk(e))) => {
            assert_eq!(e.error.message, "model 'x' not found");
            assert_eq!(session.get_history_length(), 1);
        }
        _ => panic!("Expected StatusNotOk"),
    }
}

#[tokio::test]
async fn embeddings() {
    let body = json!({
        "object": "list",
        // Out of order on purpose
        "data": [
            {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}
        ],
        "model": "mock-embedding",
        "usage": {"prompt_tokens": 2, "total_tokens": 2}
    });
    let (base_url, request) = serve(200, "application/json", vec![body.to_string()]).await;
    let embedder = OpenAIEmbedding::new("key", "mock-embedding")
        .set_base_url(base_url)
        .set_dimensions(2);
    let response = embedder
        .embed_texts(vec!["first".into(), "second".into()])
        .await
        .unwrap();
    assert_eq!(response.embedding().values(), &[0.1, 0.2]);
    assert_eq!(response.embeddings()[1].values(), &[0.3, 0.4]);
    assert_eq!(response.embedding().dimension(), 2);

    let request = request.await.unwrap();
    assert_eq!(request.line, "POST /v1/embeddings HTTP/1.1");
    assert_eq!(
        request.body,
        json!({"model": "mock-embedding", "input": ["first", "second"], "dimensions": 2})
    );
}
