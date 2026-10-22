//! `ChatStream` fed with in-memory chunks, so events can be split at exact byte positions.
use crate::Error;
use crate::chat::message::{AssistantMessage, Message};
use crate::chat::session::Session;
use crate::chat::{ChatCompletionChunk, ChatStream};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

fn session() -> Session {
    let mut session = Session::new(10);
    session.ask("Hi");
    session
}
fn chat_stream(session: &mut Session, chunks: Vec<Vec<u8>>) -> ChatStream<'_> {
    let chunks = futures::stream::iter(chunks.into_iter().map(|chunk| Ok(Bytes::from(chunk))));
    ChatStream::new(Box::new(chunks), session)
}
/// Reads the whole stream.
async fn read(mut stream: ChatStream<'_>) -> Vec<Result<ChatCompletionChunk, Error>> {
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }
    items
}
fn reply(session: &Session) -> &AssistantMessage {
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
    assert_eq!(stream.session().len(), 1);
    drop(stream);
    assert_eq!(session.len(), 2);
    assert_eq!(reply(&session).text(), "Hel");

    // Nothing is added if nothing was received
    let mut session = self::session();
    drop(chat_stream(&mut session, vec![]));
    assert_eq!(session.len(), 1);
}

#[tokio::test]
async fn remember_reply_false_forgets_prompt() {
    let mut session = session();
    // The whole prompt, even of many messages
    session.ask("there");
    session.set_remember_reply(false);
    read(chat_stream(
        &mut session,
        vec![event(json!({"content": "Hi"})).into_bytes()],
    ))
    .await;
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
