//! Tests against a local HTTP server, checking what is sent and how replies are handled.
//! Parsing of streamed events is tested in `stream.rs`.
use crate::Error;
use crate::chat::AI;
use crate::chat::message::{AudioFormat, File, ImageUrl, InputAudio, Message};
use crate::chat::session::{Session, SlidingWindow};
use crate::embeddings::Embedder;
use base64::{Engine, engine::general_purpose::STANDARD};
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
    let mut ai = AI::new("secret", "mock-model", Some("Be brief".into()));
    ai.base_url(format!("{base_url}/")).temperature(0.5);
    let mut session = Session::new(SlidingWindow::new(6));
    let completion = ai.ask(session.ask("Hi")).await.unwrap();

    assert_eq!(completion.text(), "Hello!");
    assert_eq!(completion.model.as_deref(), Some("mock-model"));
    assert_eq!(completion.usage.unwrap().total_tokens, 2);
    assert_eq!(completion.extra["citations"][0], "https://example.com");
    assert_eq!(session.len(), 2);
    assert!(matches!(session.last(), Some(Message::Assistant(_))));

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
    let mut ai = AI::with_no_api_key("local-model", None);
    ai.base_url(base_url);
    ai.ask(Session::default().ask("Hi")).await.unwrap();
    assert_eq!(request.await.unwrap().header("authorization"), None);
}

#[tokio::test]
async fn nothing_to_respond() {
    let mut ai = AI::new("key", "mock-model", None);
    ai.base_url("http://127.0.0.1:1");
    let mut session = Session::default();
    session.ask("Hi").reply("Hello");
    assert!(matches!(
        ai.ask(&mut session).await,
        Err(Error::NothingToRespond)
    ));
    assert!(matches!(
        ai.stream(&mut session).await,
        Err(Error::NothingToRespond)
    ));
}

#[tokio::test]
async fn error_status_is_parsed() {
    let body = json!({"error": {"message": "Incorrect API key", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}});
    let (base_url, _) = serve(401, "application/json", body.to_string()).await;
    let mut ai = AI::new("wrong", "mock-model", None);
    ai.base_url(base_url);
    let mut session = Session::default();
    match ai.ask(session.ask("Hi")).await {
        Err(Error::Api { status, error }) => {
            assert_eq!(status.unwrap(), 401);
            assert_eq!(error.message, "Incorrect API key");
            assert_eq!(error.r#type.as_deref(), Some("invalid_request_error"));
            assert_eq!(error.code, Some(json!("invalid_api_key")));
        }
        other => panic!("Expected Api error, got {other:?}"),
    }
    // The prompt stays, so it can be retried
    assert_eq!(session.len(), 1);
}

#[tokio::test]
async fn error_object_with_ok_status() {
    let body = json!({"error": {"message": "Rate limit exceeded", "code": 429}});
    let (base_url, _) = serve(200, "application/json", body.to_string()).await;
    let mut ai = AI::new("key", "mock-model", None);
    ai.base_url(base_url);
    match ai.ask(Session::default().ask("Hi")).await {
        Err(Error::Api { status, error }) => {
            assert_eq!(status.unwrap(), 200);
            assert_eq!(error.message, "Rate limit exceeded");
        }
        other => panic!("Expected Api error, got {other:?}"),
    }
}

#[tokio::test]
async fn invalid_response_contains_body() {
    let (base_url, _) = serve(200, "text/html", "<html>Not an API</html>").await;
    let mut ai = AI::new("key", "mock-model", None);
    ai.base_url(base_url);
    match ai.ask(Session::default().ask("Hi")).await {
        Err(Error::InvalidResponse(e)) => assert!(e.contains("Not an API")),
        other => panic!("Expected InvalidResponse, got {other:?}"),
    }
}

#[tokio::test]
async fn stream() {
    let chunk = json!({"choices": [{"index": 0, "delta": {"content": "Hello"}}]});
    let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
    let (base_url, request) = serve(200, "text/event-stream", body).await;
    let mut ai = AI::new("key", "mock-model", None);
    ai.base_url(base_url).include_usage(true);
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("Hi");

    let mut stream = ai.stream(&mut session).await.unwrap();
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        text.push_str(chunk.unwrap().text());
    }
    assert_eq!(text, "Hello");
    assert_eq!(stream.session().len(), 2);

    let request = request.await.unwrap();
    assert_eq!(request.body["stream"], true);
    assert_eq!(
        request.body["stream_options"],
        json!({"include_usage": true})
    );
}

#[tokio::test]
async fn stream_status_not_ok() {
    let (base_url, _) = serve(
        404,
        "application/json",
        r#"{"error": "model 'x' not found"}"#,
    )
    .await;
    let mut ai = AI::new("key", "x", None);
    ai.base_url(base_url);
    let mut session = Session::new(SlidingWindow::new(10));
    match ai.stream(session.ask("Hi")).await {
        Err(Error::Api { error, .. }) => assert_eq!(error.message, "model 'x' not found"),
        _ => panic!("Expected Api error"),
    }
    assert_eq!(session.len(), 1);
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
    let mut embedder = Embedder::new("key", "mock-embedding");
    embedder
        .base_url(base_url)
        .dimensions(2)
        .param("model", "ignored");
    let response = embedder.embed(vec!["first", "second"]).await.unwrap();
    assert_eq!(response.embedding(), [0.1, 0.2]);
    assert_eq!(response.data[1].embedding, [0.3, 0.4]);

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
    let mut embedder = Embedder::new("key", "mock-embedding");
    embedder.base_url(base_url);
    match embedder.embed(vec!["a", "b"]).await {
        Err(Error::InvalidResponse(e)) => assert_eq!(e, "Expected 2 embeddings, received 1"),
        other => panic!("Expected InvalidResponse, got {other:?}"),
    }
}

#[tokio::test]
async fn downloads() {
    let png = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lda.png")).unwrap();
    let (base_url, request) = serve(200, "image/png", png.clone()).await;
    let image = ImageUrl::download(&format!("{base_url}/lda.png"))
        .await
        .unwrap();
    assert_eq!(request.await.unwrap().line, "GET /v1/lda.png HTTP/1.1");
    let data = image.url.strip_prefix("data:image/png;base64,").unwrap();
    assert_eq!(STANDARD.decode(data).unwrap(), png);

    // The filename is taken from the URL, and parameters of the mime type are left out
    let (base_url, _) = serve(200, "application/pdf; qs=0.001", "%PDF").await;
    let file = File::download(&format!("{base_url}/docs/report.pdf?v=2"))
        .await
        .unwrap();
    assert_eq!(file.filename.as_deref(), Some("report.pdf"));
    assert_eq!(
        file.file_data.as_deref(),
        Some("data:application/pdf;base64,JVBERg==")
    );

    let (base_url, _) = serve(200, "audio/mpeg", "mp3").await;
    let audio = InputAudio::download(&base_url).await.unwrap();
    assert_eq!(audio.format, AudioFormat::Mp3);
    // Only WAV and MP3 are allowed in `input_audio`
    let (base_url, _) = serve(200, "audio/ogg", "ogg").await;
    assert!(matches!(
        InputAudio::download(&base_url).await,
        Err(Error::InvalidResponse(_))
    ));

    let (base_url, _) = serve(404, "text/plain", "Not Found").await;
    assert!(matches!(
        ImageUrl::download(&base_url).await,
        Err(Error::Api { .. })
    ));
}
