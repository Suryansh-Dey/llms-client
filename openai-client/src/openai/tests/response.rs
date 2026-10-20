use crate::openai::types::request::{FunctionCall, Role};
use crate::openai::types::response::{FinishReason, OpenAIResponse};
use serde_json::{Value, json};

fn response(message: Value, finish_reason: &str) -> OpenAIResponse {
    serde_json::from_value(json!({
        "id": "1",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "created": 1
    }))
    .unwrap()
}

#[test]
fn parses_reasoning_refusal_and_tool_calls() {
    let response = response(
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
    let chat = response.get_chat();
    assert_eq!(*chat.role(), Role::Assistant);
    assert_eq!(chat.get_thoughts(""), "Let me think");
    assert_eq!(chat.get_text_no_think(""), "");
    assert_eq!(chat.get_refusal(), Some("I can't"));
    assert_eq!(response.get_finish_reason(), Some(&FinishReason::ToolCalls));
    let calls: Vec<&FunctionCall> = chat.get_function_calls().collect();
    assert_eq!(calls[0].id(), "call_1");
    assert_eq!(calls[0].args().unwrap(), json!({"x": 1}));
    assert_eq!(calls[1].name(), "g");
    assert_eq!(calls[1].args().unwrap(), json!({"y": 2}));
    assert!(!calls[1].id().is_empty(), "id is generated if missing");
    assert_eq!(response.extra["created"], 1);
}

#[test]
fn parses_other_reasoning_and_content_formats() {
    let response = response(
        json!({
            "role": "assistant",
            "reasoning": "Ollama style",
            "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]
        }),
        "eos",
    );
    assert_eq!(response.get_chat().get_thoughts(""), "Ollama style");
    assert_eq!(response.get_chat().get_text_no_think(""), "Hello world");
    assert_eq!(response.get_finish_reason(), Some(&FinishReason::Other));
}

#[test]
fn round_trip() {
    let original = response(
        json!({
            "role": "assistant",
            "content": "Hi",
            "reasoning_content": "Thought",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        }),
        "stop",
    );
    let json = serde_json::to_value(&original).unwrap();
    let restored: OpenAIResponse = serde_json::from_value(json).unwrap();
    let chat = restored.get_chat();
    assert_eq!(chat.get_text_no_think(""), "Hi");
    assert_eq!(chat.get_thoughts(""), "Thought");
    assert_eq!(chat.get_function_calls().next().unwrap().id(), "call_1");
    assert_eq!(restored.extra["created"], 1);
}

#[test]
fn empty_choices() {
    let response: OpenAIResponse =
        serde_json::from_value(json!({"choices": [], "usage": {"total_tokens": 3}})).unwrap();
    assert!(response.get_chat().parts().is_empty());
    assert_eq!(response.get_finish_reason(), None);
}

#[test]
fn get_json_ignores_thoughts_and_code_fence() {
    let response = response(
        json!({"content": "```json\n{\"a\": [1, 2]}\n```", "reasoning_content": "{not json"}),
        "stop",
    );
    let json: Value = response.get_json().unwrap();
    assert_eq!(json, json!({"a": [1, 2]}));
    let plain = self::response(json!({"content": " {\"a\": 1} "}), "stop");
    assert_eq!(plain.get_json::<Value>().unwrap(), json!({"a": 1}));
}
