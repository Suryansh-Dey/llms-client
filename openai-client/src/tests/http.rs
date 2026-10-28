//! Tests against a local HTTP server, checking what is sent and how replies are handled.
//! Parsing of streamed events is tested in `stream.rs`.
use crate::Error;
use crate::chat::message::{AudioFormat, File, ImageUrl, InputAudio, Message};
use crate::chat::session::{Compactor, OverflowHandler, Session};
use crate::chat::{AI, ChatCompletion, ResponseFormat, Tool};
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
    let (base_url, requests) = serve_all(vec![(status, content_type, body.into())]).await;
    let request = tokio::spawn(async move { requests.await.unwrap().pop().unwrap() });
    (base_url, request)
}

/// Serves a request for each response, in order. Returns the base URL and the received requests.
async fn serve_all(responses: Vec<(u16, &str, Vec<u8>)>) -> (String, JoinHandle<Vec<Request>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let responses: Vec<(String, Vec<u8>)> = responses
        .into_iter()
        .map(|(status, content_type, body)| {
            let head = format!(
                "HTTP/1.1 {status} STATUS\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n"
            );
            (head, body)
        })
        .collect();
    let handle = tokio::spawn(async move {
        let mut requests = Vec::new();
        for (head, body) in responses {
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
            let request_body =
                serde_json::from_slice(&buffer[header_end + 4..]).unwrap_or(Value::Null);

            socket.write_all(head.as_bytes()).await.unwrap();
            socket.write_all(&body).await.unwrap();
            socket.shutdown().await.ok();
            requests.push(Request {
                line,
                headers,
                body: request_body,
            });
        }
        requests
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
    let mut session = Session::default();
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

/// Response with `content`, using `total_tokens`.
fn reply(content: &str, total_tokens: u64) -> (u16, &'static str, Vec<u8>) {
    let body = json!({
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": total_tokens, "completion_tokens": 0, "total_tokens": total_tokens}
    });
    (200, "application/json", body.to_string().into_bytes())
}
fn texts<H>(session: &Session<H>) -> Vec<String> {
    session.history().iter().map(Message::text).collect()
}
fn compacting_ai(base_url: String) -> AI {
    let mut ai = AI::new("key", "mock-model", Some("Be brief".into()));
    ai.base_url(base_url)
        .temperature(0.0)
        .response_format(ResponseFormat::json_schema(json!({"type": "object"})))
        .tool(Tool::Function(
            json!({"name": "search", "parameters": {"type": "object"}}),
        ));
    ai
}
const SUMMARY_1: &str =
    "<summary_of_earlier_conversation>\nSummary 1\n</summary_of_earlier_conversation>";

#[tokio::test]
async fn compactor_summarizes_all_but_last_turn() {
    let (base_url, requests) = serve_all(vec![
        reply("A1", 50),
        reply("A2", 101),
        reply(" Summary 1 ", 120),
        reply("A3", 90),
        reply("A4", 150),
        reply("Summary 2", 160),
    ])
    .await;
    let ai = compacting_ai(base_url);
    let mut session = Session::new(Compactor::new(100));
    ai.ask(session.ask("Q1")).await.unwrap();
    assert_eq!(texts(&session), ["Q1", "A1"], "within the limit");
    // Past it, all before the last prompt is summarized into that prompt
    ai.ask(session.ask("Q2")).await.unwrap();
    assert_eq!(texts(&session), [format!("{SUMMARY_1}\n\nQ2"), "A2".into()]);
    assert_eq!(session.message_number(), 4, "messages keep their numbers");
    // Until the next summary, messages are only added
    ai.ask(session.ask("Q3")).await.unwrap();
    ai.ask(session.ask("Q4")).await.unwrap();
    // The summary is summarized again, along with what came after it
    let summary_2 =
        "<summary_of_earlier_conversation>\nSummary 2\n</summary_of_earlier_conversation>";
    assert_eq!(texts(&session), [format!("{summary_2}\n\nQ4"), "A4".into()]);

    let requests = requests.await.unwrap();
    let messages = |i: usize| requests[i].body["messages"].as_array().unwrap().clone();
    // Asked for with the request of the reply, minus its turn, so it starts the same
    assert_eq!(
        messages(2),
        json!([
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": Compactor::default().prompt}
        ])
        .as_array()
        .unwrap()
        .clone()
    );
    assert!(messages(1).starts_with(&messages(2)[..3]));
    let summary_request = &requests[2].body;
    assert_eq!(summary_request["tools"], requests[1].body["tools"]);
    assert_eq!(summary_request["temperature"], 0.0);
    // With tool calls turned off, and as text
    assert_eq!(summary_request["tool_choice"], "none");
    assert!(summary_request.get("response_format").is_none());
    assert!(requests[1].body.get("tool_choice").is_none());
    // Each request starts with the one before it, until the next summary
    assert!(messages(4).starts_with(&messages(3)));
    assert!(messages(5).starts_with(&messages(4)[..messages(4).len() - 1]));
    assert_eq!(messages(5)[1]["content"], format!("{SUMMARY_1}\n\nQ2"));
}

#[tokio::test]
async fn compactor_keeps_tool_calls_of_last_turn() {
    let call = json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]
    });
    let calling =
        json!({"choices": [{"index": 0, "message": call}], "usage": {"total_tokens": 200}});
    let (base_url, _) = serve_all(vec![
        (200, "application/json", calling.to_string().into_bytes()),
        reply("Summary 1", 120),
    ])
    .await;
    let ai = compacting_ai(base_url);
    let mut session = Session::new(Compactor::new(100));
    session.ask("Q1").reply("A1").ask("Q2");
    // Compacted after a reply with calls too, which stay unanswered
    ai.ask(&mut session).await.unwrap();
    assert_eq!(
        texts(&session),
        [format!("{SUMMARY_1}\n\nQ2"), String::new()]
    );
    session.add_tool_result("call_1", "Found").unwrap();
}

#[tokio::test]
async fn compactor_keeps_history_without_summary() {
    let (base_url, requests) = serve_all(vec![
        reply("A1", 500),
        reply("A2", 500),
        (
            500,
            "application/json",
            br#"{"error": {"message": "overloaded"}}"#.to_vec(),
        ),
        reply("A3", 500),
        reply("", 10),
    ])
    .await;
    let ai = compacting_ai(base_url);
    let mut session = Session::new(Compactor::new(100));
    // Nothing is before the only turn
    ai.ask(session.ask("Q1")).await.unwrap();
    // A failed or empty summary changes nothing, and is asked for again after the next reply
    ai.ask(session.ask("Q2")).await.unwrap();
    assert_eq!(texts(&session), ["Q1", "A1", "Q2", "A2"]);
    ai.ask(session.ask("Q3")).await.unwrap();
    assert_eq!(texts(&session), ["Q1", "A1", "Q2", "A2", "Q3", "A3"]);
    assert_eq!(requests.await.unwrap().len(), 5);

    // Nor without a client
    let mut session = Session::new(Compactor::new(100));
    session.ask("Q1").reply("A1").ask("Q2");
    let completion: ChatCompletion = serde_json::from_slice(&reply("A2", 500).2).unwrap();
    session.add_completion(&completion).await;
    assert_eq!(session.len(), 4);
}

#[tokio::test]
async fn compactor_summarizes_before_stream_ends() {
    let chunk = json!({"choices": [{"index": 0, "delta": {"content": "A2"}}]});
    let usage = json!({"choices": [], "usage": {"total_tokens": 150}});
    let body = format!("data: {chunk}\n\ndata: {usage}\n\ndata: [DONE]\n\n");
    let (base_url, requests) = serve_all(vec![
        (200, "text/event-stream", body.into_bytes()),
        reply("Summary 1", 120),
    ])
    .await;
    let ai = compacting_ai(base_url);
    let mut session = Session::new(Compactor::new(100));
    session.ask("Q1").reply("A1").ask("Q2");
    let mut stream = ai.stream(&mut session).await.unwrap();
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        text.push_str(chunk.unwrap().text());
    }
    drop(stream);
    assert_eq!(text, "A2");
    assert_eq!(texts(&session), [format!("{SUMMARY_1}\n\nQ2"), "A2".into()]);
    // Streams ask for usage by default
    let requests = requests.await.unwrap();
    assert_eq!(
        requests[0].body["stream_options"],
        json!({"include_usage": true})
    );
}

/// Requests and streams can be spawned whatever the handler, as it is `Send`.
#[allow(dead_code)]
async fn requests_and_streams_are_send<H: OverflowHandler>(ai: &AI, session: &mut Session<H>) {
    fn is_send<T: Send>(_: &T) {}
    is_send(&ai.ask(session));
    is_send(&ai.stream(session));
    is_send(&ai.stream(session).await.unwrap());
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
    let mut session = Session::default();
    session.ask("Hi");

    let mut stream = ai.stream(&mut session).await.unwrap();
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        text.push_str(chunk.unwrap().text());
    }
    assert_eq!(text, "Hello");
    drop(stream);
    assert_eq!(session.len(), 2);

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
    let mut session = Session::default();
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
