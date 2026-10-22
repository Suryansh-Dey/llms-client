use crate::chat::{ChatCompletion, ChatCompletionChunk, FinishReason};
use serde_json::{Value, json};

fn completion(message: Value, finish_reason: &str) -> ChatCompletion {
    serde_json::from_value(json!({
        "id": "1",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "created": 1
    }))
    .unwrap()
}

#[test]
fn parses_reasoning_refusal_and_tool_calls() {
    let completion = completion(
        json!({
            "role": "assistant",
            "content": null,
            "reasoning_content": "Let me think",
            "refusal": "I can't",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{\"x\":1}"}},
                // Some servers send arguments as an object, and no id
                {"type": "function", "function": {"name": "g", "arguments": {"y": 2}}}
            ]
        }),
        "tool_calls",
    );
    let message = completion.message();
    assert_eq!(message.reasoning.as_deref(), Some("Let me think"));
    assert_eq!(message.content, None);
    assert_eq!(completion.text(), "");
    assert_eq!(message.refusal.as_deref(), Some("I can't"));
    assert_eq!(completion.finish_reason(), Some(&FinishReason::ToolCalls));
    let calls = &message.tool_calls;
    assert_eq!(calls[0].id, "call_1");
    assert_eq!(calls[0].args::<Value>().unwrap(), json!({"x": 1}));
    assert_eq!(calls[1].function.name, "g");
    assert_eq!(calls[1].args::<Value>().unwrap(), json!({"y": 2}));
    assert!(!calls[1].id.is_empty(), "id is generated if missing");
    assert_eq!(completion.extra["created"], 1);
}

#[test]
fn parses_other_reasoning_and_content_formats() {
    let completion = completion(
        json!({
            "role": "assistant",
            "reasoning": "Ollama style",
            "content": [
                {"type": "thinking", "thinking": " and parts"},
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " world"}
            ]
        }),
        "eos",
    );
    let message = completion.message();
    assert_eq!(message.reasoning.as_deref(), Some("Ollama style and parts"));
    assert_eq!(completion.text(), "Hello world");
    assert_eq!(completion.finish_reason(), Some(&FinishReason::Other));
}

#[test]
fn round_trip() {
    let original = completion(
        json!({
            "role": "assistant",
            "content": "Hi",
            "reasoning_content": "Thought",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        }),
        "stop",
    );
    let json = serde_json::to_value(&original).unwrap();
    let restored: ChatCompletion = serde_json::from_value(json).unwrap();
    assert_eq!(restored.message(), original.message());
    assert_eq!(restored.extra["created"], 1);
}

#[test]
fn empty_choices() {
    let completion: ChatCompletion =
        serde_json::from_value(json!({"choices": [], "usage": {"total_tokens": 3}})).unwrap();
    assert_eq!(completion.message(), &Default::default());
    assert_eq!(completion.finish_reason(), None);
    assert_eq!(completion.usage.unwrap().total_tokens, 3);
}

#[test]
fn json_ignores_reasoning_and_code_fence() {
    let completion = completion(
        json!({"content": "```json\n{\"a\": [1, 2]}\n```", "reasoning_content": "{not json"}),
        "stop",
    );
    assert_eq!(completion.json::<Value>().unwrap(), json!({"a": [1, 2]}));
    let plain = self::completion(json!({"content": " {\"a\": 1} "}), "stop");
    assert_eq!(plain.json::<Value>().unwrap(), json!({"a": 1}));
}

#[test]
fn parses_chunks() {
    let chunk: ChatCompletionChunk = serde_json::from_value(json!({
        "id": "1",
        "choices": [{"index": 0, "delta": {
            "role": "assistant",
            "content": "Hi",
            "reasoning": "Hmm",
            "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "f", "arguments": {"x": 1}}}]
        }, "finish_reason": null}],
        "usage": null,
        "created": 1
    }))
    .unwrap();
    assert_eq!(chunk.text(), "Hi");
    let delta = chunk.delta();
    assert_eq!(delta.reasoning.as_deref(), Some("Hmm"));
    assert_eq!(delta.tool_calls[0].index, Some(0));
    assert_eq!(delta.tool_calls[0].id.as_deref(), Some("call_1"));
    assert_eq!(delta.tool_calls[0].function.name.as_deref(), Some("f"));
    assert_eq!(
        delta.tool_calls[0].function.arguments.as_deref(),
        Some("{\"x\":1}")
    );
    assert_eq!(chunk.finish_reason(), None);

    // Usage only, as sent by OpenAI at the end of a stream
    let chunk: ChatCompletionChunk =
        serde_json::from_value(json!({"choices": [], "usage": {"prompt_tokens": 1}})).unwrap();
    assert_eq!(chunk.text(), "");
    assert_eq!(chunk.usage.unwrap().prompt_tokens, 1);
}
