//! `ChatStream` fed with in-memory chunks, so events can be split at exact byte positions.
use crate::Error;
use crate::chat::message::{AssistantMessage, Message};
use crate::chat::session::{HistoryTrimmer, OverflowHandler, Session, SlidingWindow};
use crate::chat::{ChatCompletion, ChatCompletionChunk, ChatStream, FinishReason};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

fn session() -> Session {
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("Hi");
    session
}
fn chat_stream<H: OverflowHandler>(
    session: &mut Session<H>,
    chunks: Vec<Vec<u8>>,
) -> ChatStream<'_, H> {
    let chunks = futures::stream::iter(chunks.into_iter().map(|chunk| Ok(Bytes::from(chunk))));
    ChatStream::new(Box::new(chunks), session)
}
/// Reads the whole stream.
async fn read<H: OverflowHandler>(
    mut stream: ChatStream<'_, H>,
) -> Vec<Result<ChatCompletionChunk, Error>> {
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }
    items
}
fn reply<H>(session: &Session<H>) -> &AssistantMessage {
    match session.last() {
        Some(Message::Assistant(reply)) => reply,
        other => panic!("Expected an assistant message, got {other:?}"),
    }
}

fn event(delta: Value) -> String {
    format!(
        "data: {}\n\n",
        json!({"choices": [{"index": 0, "delta": delta}]})
    )
}

#[tokio::test]
async fn events_split_at_any_byte() {
    let body = [
        ": keep alive\r\n\r\n",
        // CRLF line endings, and a multi byte character to split
        "data: {\"choices\": [{\"index\": 0, \"delta\": {\"role\": \"assistant\", \"content\": \"Hé\"}}]}\r\n\r\n",
        // Other fields than `data:` are ignored
        "event: message\nid: 2\n",
        "data: {\"choices\": [{\"index\": 0, \"delta\": {\"content\": \"llo\"}}]}\n\n",
        // `data:` of many lines is joined
        "data: {\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"stop\"}],\n",
        "data: \"usage\": {\"total_tokens\": 3}}\n\n",
        "data: [DONE]\n\n",
    ]
    .concat()
    .into_bytes();

    let mut splits: Vec<Vec<Vec<u8>>> = (0..=body.len())
        .map(|at| vec![body[..at].to_vec(), body[at..].to_vec()])
        .collect();
    splits.push(body.iter().map(|byte| vec![*byte]).collect());

    for chunks in splits {
        let mut session = session();
        let items = read(chat_stream(&mut session, chunks)).await;
        let items: Vec<ChatCompletionChunk> = items.into_iter().map(Result::unwrap).collect();
        let texts: Vec<&str> = items.iter().map(ChatCompletionChunk::text).collect();
        assert_eq!(texts, ["Hé", "llo", ""]);
        assert_eq!(items[2].usage.as_ref().unwrap().total_tokens, 3);
        assert_eq!(session.len(), 2);
        assert_eq!(reply(&session).text(), "Héllo");
    }
}

#[tokio::test]
async fn last_event_without_blank_line_or_done() {
    let body = format!(
        "data: {}",
        json!({"choices": [{"index": 0, "delta": {"content": "Hi"}}]})
    );
    let mut session = session();
    let items = read(chat_stream(&mut session, vec![body.into_bytes()])).await;
    assert_eq!(items.len(), 1);
    assert_eq!(reply(&session).text(), "Hi");
}

#[tokio::test]
async fn reply_so_far_and_reply_on_drop() {
    let body = [
        event(json!({"content": "Hel"})),
        event(json!({"content": "lo"})),
    ]
    .concat();
    let mut session = session();
    let mut stream = chat_stream(&mut session, vec![body.into_bytes()]);
    assert_eq!(stream.message().text(), "");
    stream.next().await.unwrap().unwrap();
    assert_eq!(stream.message().text(), "Hel");
    // Not added until the stream ends or is dropped
    assert_eq!(stream.session().unwrap().len(), 1);
    drop(stream);
    assert_eq!(session.len(), 2);
    assert_eq!(reply(&session).text(), "Hel");

    // Nothing is added if nothing was received
    let mut session = self::session();
    drop(chat_stream(&mut session, vec![]));
    assert_eq!(session.len(), 1);
}

/// Keeps the responses it is called with, after waiting like a request would.
#[derive(Default)]
struct Recorder(Vec<ChatCompletion>);
impl OverflowHandler for Recorder {
    async fn handle(&mut self, _: &mut HistoryTrimmer<'_>, response: &ChatCompletion) {
        tokio::task::yield_now().await;
        self.0.push(response.clone());
    }
}

#[tokio::test]
async fn overflow_handler_runs_when_stream_ends() {
    let chunk = |choices: Value, usage: Value| {
        let chunk = json!({"id": "chatcmpl-1", "model": "m", "choices": choices, "usage": usage});
        format!("data: {chunk}\n\n")
    };
    let body = [
        chunk(
            json!([{"index": 0, "delta": {"content": "Hel"}}]),
            Value::Null,
        ),
        chunk(
            json!([{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]),
            Value::Null,
        ),
        chunk(json!([]), json!({"total_tokens": 7})),
        "data: [DONE]\n\n".into(),
    ]
    .concat();
    let mut session = Session::new(Recorder::default());
    session.ask("Hi");
    let mut stream = chat_stream(&mut session, vec![body.into_bytes()]);
    for _ in 0..3 {
        stream.next().await.unwrap().unwrap();
    }
    // The reply is added, and the session is with the handler while it waits
    assert!(futures::poll!(stream.next()).is_pending());
    assert!(stream.session().is_none());
    assert_eq!(stream.message().text(), "Hello");
    assert!(stream.next().await.is_none());
    assert_eq!(stream.session().unwrap().len(), 2);
    assert_eq!(stream.message().text(), "Hello");
    drop(stream);

    // It gets the response assembled from the chunks
    let responses = &session.overflow_handler().0;
    assert_eq!(responses.len(), 1);
    let response = &responses[0];
    assert_eq!(response.id.as_deref(), Some("chatcmpl-1"));
    assert_eq!(response.model.as_deref(), Some("m"));
    assert_eq!(response.text(), "Hello");
    assert_eq!(response.finish_reason(), Some(&FinishReason::Stop));
    assert_eq!(response.usage.as_ref().unwrap().total_tokens, 7);

    // Not if the stream is dropped before it ends, or while the handler waits
    for polls in [1, 2] {
        session.ask("Again");
        let mut stream = chat_stream(
            &mut session,
            vec![event(json!({"content": "Hi"})).into_bytes()],
        );
        for _ in 0..polls {
            let _ = futures::poll!(stream.next());
        }
        drop(stream);
        assert_eq!(reply(&session).text(), "Hi");
    }
    assert_eq!(session.len(), 6);
    assert_eq!(session.overflow_handler().0.len(), 1);
}

#[tokio::test]
async fn remember_reply_false_forgets_prompt() {
    let mut session = session();
    // The whole prompt, even of many messages
    session.ask("there");
    session.set_remember_reply(false);
    let mut stream = chat_stream(
        &mut session,
        vec![event(json!({"content": "Hi"})).into_bytes()],
    );
    while stream.next().await.is_some() {}
    // Only the stream has the reply
    assert_eq!(stream.message().text(), "Hi");
    drop(stream);
    assert!(session.is_empty());
}

fn calls(session: &Session) -> Vec<(String, String, Value)> {
    reply(session)
        .tool_calls
        .iter()
        .map(|call| {
            let args = call.args().unwrap();
            (call.id.clone(), call.function.name.clone(), args)
        })
        .collect()
}

#[tokio::test]
async fn parallel_calls_are_merged_by_index() {
    let body = [
        event(json!({"role": "assistant", "content": null})),
        event(json!({"tool_calls": [
            {"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": ""}}
        ]})),
        event(json!({"tool_calls": [
            {"index": 1, "id": "call_b", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":"}}
        ]})),
        // Some servers repeat the id and name in every delta
        event(json!({"tool_calls": [
            {"index": 0, "id": "call_a", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
        ]})),
        event(json!({"tool_calls": [{"index": 1, "function": {"arguments": " \"London\"}"}}]})),
        "data: [DONE]\n\n".to_string(),
    ]
    .concat();
    let mut session = session();
    let items = read(chat_stream(&mut session, vec![body.into_bytes()])).await;
    assert!(items.iter().all(Result::is_ok));
    assert_eq!(
        calls(&session),
        [
            (
                "call_a".into(),
                "get_weather".into(),
                json!({"city": "Paris"})
            ),
            (
                "call_b".into(),
                "get_weather".into(),
                json!({"city": "London"})
            )
        ]
    );
    assert_eq!(reply(&session).content, None);
    // Complete calls can be answered
    session.add_tool_result("call_b", "Rainy").unwrap();
    session.add_tool_result("call_a", "Sunny").unwrap();
    assert!(session.pending_tool_calls().is_empty());
    assert!(matches!(session.last(), Some(Message::Tool(_))));
}

#[tokio::test]
async fn calls_without_index_are_complete() {
    // Like Gemini, which sends each call whole, without `index`
    let body = [
        event(json!({"tool_calls": [
            {"id": "a", "function": {"name": "f", "arguments": "{\"x\": 1}"}},
            {"id": "b", "function": {"name": "f", "arguments": "{\"x\": 2}"}}
        ]})),
        event(json!({"tool_calls": [{"function": {"name": "g", "arguments": "{}"}}]})),
        event(json!({"tool_calls": [{"function": {"name": "g", "arguments": "{}"}}]})),
    ]
    .concat();
    let mut session = session();
    read(chat_stream(&mut session, vec![body.into_bytes()])).await;
    let calls = calls(&session);
    assert_eq!(calls.len(), 4);
    assert_eq!(calls[0], ("a".into(), "f".into(), json!({"x": 1})));
    assert_eq!(calls[1], ("b".into(), "f".into(), json!({"x": 2})));
    // Ids are generated for calls without one, as results refer to calls by id
    assert!(!calls[2].0.is_empty());
    assert_ne!(calls[2].0, calls[3].0);
}

#[tokio::test]
async fn reasoning_is_merged() {
    let body = [
        event(json!({"reasoning_content": "Let me "})),
        event(json!({"reasoning": "think"})),
        event(json!({"content": "Done"})),
    ]
    .concat();
    let mut session = session();
    read(chat_stream(&mut session, vec![body.into_bytes()])).await;
    assert_eq!(reply(&session).reasoning.as_deref(), Some("Let me think"));
    assert_eq!(reply(&session).text(), "Done");
}

#[tokio::test]
async fn error_event_and_invalid_event() {
    let body = [
        event(json!({"content": "Hi"})),
        format!(
            "data: {}\n\n",
            json!({"error": {"message": "Provider overloaded", "type": "server_error", "code": 502}})
        ),
        "data: {not json\n\n".to_string(),
    ]
    .concat();
    let mut session = session();
    let items = read(chat_stream(&mut session, vec![body.into_bytes()])).await;
    assert_eq!(items.len(), 3);
    assert!(items[0].is_ok());
    match &items[1] {
        Err(Error::Api {
            status: None,
            error,
        }) => {
            assert_eq!(error.message, "Provider overloaded");
            assert_eq!(error.code, Some(json!(502)));
        }
        other => panic!("Expected Api error, got {other:?}"),
    }
    match &items[2] {
        Err(Error::InvalidResponse(e)) => assert!(e.contains("{not json")),
        other => panic!("Expected InvalidResponse, got {other:?}"),
    }
    // What was received before the error is kept
    assert_eq!(reply(&session).text(), "Hi");
}
