//! `ResponseStream` fed with in-memory chunks, so events can be split at exact byte positions.
use crate::openai::error::OpenAIResponseStreamError;
use crate::openai::types::request::Role;
use crate::openai::types::response::{OpenAIResponse, OpenAIResponseStream, ResponseStream};
use crate::openai::types::sessions::Session;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

fn response_stream(chunks: Vec<Vec<u8>>) -> OpenAIResponseStream {
    let mut session = Session::new(10);
    session.ask("Hi");
    let chunks = futures::stream::iter(chunks.into_iter().map(|chunk| Ok(Bytes::from(chunk))));
    ResponseStream::new(
        Box::new(chunks),
        session,
        (|_, response| response) as fn(&Session, OpenAIResponse) -> OpenAIResponse,
    )
}

/// Reads the whole stream. Returns the items and the session.
async fn read(
    mut stream: OpenAIResponseStream,
) -> (
    Vec<Result<OpenAIResponse, OpenAIResponseStreamError>>,
    Session,
) {
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }
    (items, stream.get_session_owned())
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
        let (items, session) = read(response_stream(chunks)).await;
        let items: Vec<OpenAIResponse> = items.into_iter().map(Result::unwrap).collect();
        let texts: Vec<String> = items
            .iter()
            .map(|item| item.get_chat().get_text_no_think(""))
            .collect();
        assert_eq!(texts, ["Hé", "llo", ""]);
        assert_eq!(items[2].usage, Some(json!({"total_tokens": 3})));
        assert_eq!(session.get_history_length(), 2);
        assert_eq!(
            session.get_last_chat().unwrap().get_text_no_think(""),
            "Héllo"
        );
    }
}

#[tokio::test]
async fn last_event_without_blank_line_or_done() {
    let body = format!(
        "data: {}",
        json!({"choices": [{"index": 0, "delta": {"content": "Hi"}}]})
    );
    let (items, session) = read(response_stream(vec![body.into_bytes()])).await;
    assert_eq!(items.len(), 1);
    assert_eq!(session.get_last_chat().unwrap().get_text_no_think(""), "Hi");
}

fn calls(session: &Session) -> Vec<(String, String, Value)> {
    session
        .get_last_chat()
        .unwrap()
        .get_function_calls()
        .map(|call| (call.id().clone(), call.name().clone(), call.args().unwrap()))
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
    let (items, mut session) = read(response_stream(vec![body.into_bytes()])).await;
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
    // Complete calls can be answered
    session.add_function_response("call_b", "Rainy").unwrap();
    session.add_function_response("call_a", "Sunny").unwrap();
    assert!(session.get_pending_function_calls().is_empty());
    assert_eq!(*session.get_last_chat().unwrap().role(), Role::Tool);
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
    let (_, session) = read(response_stream(vec![body.into_bytes()])).await;
    let calls = calls(&session);
    assert_eq!(calls.len(), 4);
    assert_eq!(calls[0], ("a".into(), "f".into(), json!({"x": 1})));
    assert_eq!(calls[1], ("b".into(), "f".into(), json!({"x": 2})));
    // Ids are generated for calls without one, as responses refer to calls by id
    assert!(!calls[2].0.is_empty());
    assert_ne!(calls[2].0, calls[3].0);
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
    let (items, session) = read(response_stream(vec![body.into_bytes()])).await;
    assert_eq!(items.len(), 3);
    assert!(items[0].is_ok());
    match &items[1] {
        Err(OpenAIResponseStreamError::ApiError(e)) => {
            assert_eq!(e.message, "Provider overloaded");
            assert_eq!(e.code, Some(json!(502)));
        }
        other => panic!("Expected ApiError, got {other:?}"),
    }
    match &items[2] {
        Err(OpenAIResponseStreamError::InvalidResponseFormat(e)) => {
            assert!(e.contains("{not json"))
        }
        other => panic!("Expected InvalidResponseFormat, got {other:?}"),
    }
    // What was received before the error is kept
    assert_eq!(session.get_last_chat().unwrap().get_text_no_think(""), "Hi");
}
