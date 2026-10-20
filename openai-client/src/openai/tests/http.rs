//! Tests against a local HTTP server, checking what is sent and how replies are handled.
//! Parsing of streamed events is tested in `stream.rs`.
use crate::openai::ask::OpenAI;
use crate::openai::embed::OpenAIEmbedding;
use crate::openai::error::OpenAIResponseError;
use crate::openai::types::request::{PartType, Role};
use crate::openai::types::sessions::Session;
use crate::openai::utils::MarkdownToParts;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

struct Request {
    line: String,
    headers: Vec<(String, String)>,
    body: Value,
}
impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

/// Serves one request with `status` and `body`. Returns the base URL and the received request.
async fn serve(
    status: u16,
    content_type: &str,
    body: impl Into<Vec<u8>>,
) -> (String, JoinHandle<Request>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let head = format!(
        "HTTP/1.1 {status} STATUS\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n"
    );
    let body = body.into();
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
        let request_head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
        let mut lines = request_head.lines();
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
        let request_body = serde_json::from_slice(&buffer[header_end + 4..]).unwrap_or(Value::Null);

        socket.write_all(head.as_bytes()).await.unwrap();
        socket.write_all(&body).await.unwrap();
        socket.shutdown().await.ok();
        Request {
            line,
            headers,
            body: request_body,
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
        completion(json!({"role": "assistant", "content": "Hello!"})),
    )
    .await;
    // Trailing slash is ignored
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
async fn no_api_key_sends_no_authorization() {
    let (base_url, request) = serve(
        200,
        "application/json",
        completion(json!({"role": "assistant", "content": "ok"})),
    )
    .await;
    let ai = OpenAI::with_no_api_key("local-model", None).set_base_url(base_url);
    ai.ask(Session::new(2).ask("Hi")).await.unwrap();
    assert_eq!(request.await.unwrap().header("authorization"), None);
}

#[tokio::test]
async fn error_status_is_parsed() {
    let body = json!({"error": {"message": "Incorrect API key", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}});
    let (base_url, _) = serve(401, "application/json", body.to_string()).await;
    let ai = OpenAI::new("wrong", "mock-model", None).set_base_url(base_url);
    let mut session = Session::new(2);
    match ai.ask(session.ask("Hi")).await {
        Err(OpenAIResponseError::StatusNotOk(e)) => {
            assert_eq!(e.status, 401);
            assert_eq!(e.error.message, "Incorrect API key");
            assert_eq!(e.error.r#type.as_deref(), Some("invalid_request_error"));
            assert_eq!(e.error.code, Some(json!("invalid_api_key")));
        }
        other => panic!("Expected StatusNotOk, got {other:?}"),
    }
    // The prompt stays, so it can be retried
    assert_eq!(session.get_history_length(), 1);
}

#[tokio::test]
async fn error_object_with_ok_status() {
    let body = json!({"error": {"message": "Rate limit exceeded", "code": 429}});
    let (base_url, _) = serve(200, "application/json", body.to_string()).await;
    let ai = OpenAI::new("key", "mock-model", None).set_base_url(base_url);
    match ai.ask(Session::new(2).ask("Hi")).await {
        Err(OpenAIResponseError::StatusNotOk(e)) => {
            assert_eq!(e.status, 200);
            assert_eq!(e.error.message, "Rate limit exceeded");
        }
        other => panic!("Expected StatusNotOk, got {other:?}"),
    }
}

#[tokio::test]
async fn invalid_response_contains_body() {
    let (base_url, _) = serve(200, "text/html", "<html>Not an API</html>").await;
    let ai = OpenAI::new("key", "mock-model", None).set_base_url(base_url);
    match ai.ask(Session::new(2).ask("Hi")).await {
        Err(OpenAIResponseError::InvalidResponseFormat(e)) => assert!(e.contains("Not an API")),
        other => panic!("Expected InvalidResponseFormat, got {other:?}"),
    }
}

#[tokio::test]
async fn ask_as_stream() {
    let chunk = json!({"choices": [{"index": 0, "delta": {"content": "Hello"}}]});
    let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
    let (base_url, request) = serve(200, "text/event-stream", body).await;
    let ai = OpenAI::new("key", "mock-model", None).set_base_url(base_url);
    let mut session = Session::new(10);
    session.ask("Hi");

    let mut stream = ai.ask_as_stream(session).await.unwrap();
    let mut text = String::new();
    while let Some(response) = stream.next().await {
        text.push_str(&response.unwrap().get_chat().get_text_no_think(""));
    }
    assert_eq!(text, "Hello");
    assert_eq!(stream.get_session().get_history_length(), 2);
    assert_eq!(request.await.unwrap().body["stream"], true);
}

#[tokio::test]
async fn stream_status_not_ok_returns_session() {
    let (base_url, _) = serve(
        404,
        "application/json",
        r#"{"error": "model 'x' not found"}"#,
    )
    .await;
    let ai = OpenAI::new("key", "x", None).set_base_url(base_url);
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
    let (base_url, request) = serve(200, "application/json", body.to_string()).await;
    let embedder = OpenAIEmbedding::new("key", "mock-embedding")
        .set_base_url(base_url)
        .set_dimensions(2);
    let response = embedder
        .embed_texts(vec!["first".into(), "second".into()])
        .await
        .unwrap();
    assert_eq!(response.embedding().values(), &[0.1, 0.2]);
    assert_eq!(response.embeddings()[1].values(), &[0.3, 0.4]);

    let request = request.await.unwrap();
    assert_eq!(request.line, "POST /v1/embeddings HTTP/1.1");
    assert_eq!(
        request.body,
        json!({"model": "mock-embedding", "input": ["first", "second"], "dimensions": 2})
    );
}

#[tokio::test]
async fn embeddings_count_must_match_inputs() {
    let body = json!({"data": [{"index": 0, "embedding": [0.1]}]});
    let (base_url, _) = serve(200, "application/json", body.to_string()).await;
    let embedder = OpenAIEmbedding::new("key", "mock-embedding").set_base_url(base_url);
    match embedder.embed_texts(vec!["a".into(), "b".into()]).await {
        Err(OpenAIResponseError::InvalidResponseFormat(e)) => {
            assert_eq!(e, "Expected 2 embeddings, received 1")
        }
        other => panic!("Expected InvalidResponseFormat, got {other:?}"),
    }
}

#[tokio::test]
async fn markdown_image_is_downloaded() {
    let png = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lda.png")).unwrap();
    let (base_url, _) = serve(200, "image/png", png.clone()).await;
    let markdown = format!("Before ![chart]({base_url}/lda.png) after");
    let parts = MarkdownToParts::builder()
        // Not used, as the server sends Content-Type
        .guess_mime_type(|_| mime::IMAGE_JPEG)
        .build(&markdown)
        .await
        .process();

    assert_eq!(parts.len(), 3);
    assert!(
        matches!(parts[0].data(), PartType::Text(text) if text.starts_with("Before ![chart]("))
    );
    match parts[1].data() {
        PartType::InlineData(data) => {
            assert_eq!(*data.mime_type(), mime::IMAGE_PNG);
            use base64::{Engine, engine::general_purpose::STANDARD};
            assert_eq!(STANDARD.decode(data.data()).unwrap(), png);
        }
        other => panic!("Expected InlineData, got {other:?}"),
    }
    assert!(matches!(parts[2].data(), PartType::Text(text) if text == " after"));
}
