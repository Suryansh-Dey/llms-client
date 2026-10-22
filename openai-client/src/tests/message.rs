use crate::chat::message::{
    AssistantMessage, AudioFormat, Content, ContentPart, File, FunctionCall, ImageDetail, ImageUrl,
    InputAudio, Message, ToolCall, ToolMessage, UserMessage,
};
use serde_json::{Value, json};

fn to_json(message: impl Into<Message>) -> Value {
    serde_json::to_value(message.into()).unwrap()
}

#[test]
fn user_content_is_sent_as_built() {
    assert_eq!(
        to_json(UserMessage::from("Hi")),
        json!({"role": "user", "content": "Hi"})
    );
    // Parts keep their own type, whatever the mime type
    let message = UserMessage::from(vec![
        "Compare".into(),
        ImageUrl::from_bytes(mime::IMAGE_PNG, b"img").into(),
        ImageUrl::new("https://example.com/a.jpg")
            .with_detail(ImageDetail::Low)
            .into(),
        ImageUrl::from_bytes(mime::APPLICATION_PDF, b"pdf").into(),
        InputAudio::from_bytes(AudioFormat::Mp3, b"aud").into(),
        File::from_bytes("report.pdf", mime::APPLICATION_PDF, b"pdf").into(),
        File::from_id("file-1").into(),
    ]);
    assert_eq!(
        to_json(message.clone()),
        json!({"role": "user", "content": [
            {"type": "text", "text": "Compare"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg", "detail": "low"}},
            {"type": "image_url", "image_url": {"url": "data:application/pdf;base64,cGRm"}},
            {"type": "input_audio", "input_audio": {"data": "YXVk", "format": "mp3"}},
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,cGRm", "filename": "report.pdf"}},
            {"type": "file", "file": {"file_id": "file-1"}}
        ]})
    );
    // Round trip
    let restored: Message = serde_json::from_value(to_json(message.clone())).unwrap();
    assert_eq!(restored, Message::User(message));
    assert_eq!(
        UserMessage::from(vec!["a".into(), "b".into()]).text(),
        "a\nb"
    );
}

#[test]
fn conversions() {
    assert_eq!(Content::from("Hi"), Content::Text("Hi".into()));
    assert_eq!(
        Content::from(ImageUrl::new("u")),
        Content::Parts(vec![ContentPart::ImageUrl {
            image_url: ImageUrl::new("u")
        }])
    );
    assert_eq!(UserMessage::from(String::from("Hi")).text(), "Hi");
    assert_eq!(Message::from(UserMessage::from("Hi")).text(), "Hi");
    assert_eq!(Message::from(AssistantMessage::from("Hey")).text(), "Hey");
    assert_eq!(Message::from(ToolMessage::new("call_1", "42")).text(), "42");
    let call = ToolCall::new("call_1", "f", json!({}));
    assert_eq!(AssistantMessage::from(call.clone()).tool_calls, [call]);
}

#[test]
fn assistant_and_tool_messages() {
    let reply = AssistantMessage {
        content: Some("Let me check.".into()),
        reasoning: Some("Thought".into()),
        tool_calls: vec![
            ToolCall::new("call_1", "weather", json!({"city": "London"})),
            // Empty arguments are sent as `{}`, as some servers reject them
            ToolCall {
                id: "call_2".into(),
                function: FunctionCall {
                    name: "time".into(),
                    arguments: String::new(),
                },
            },
        ],
        ..Default::default()
    };
    // Reasoning is kept, so a saved session is restored with it
    let json = to_json(reply.clone());
    assert_eq!(
        json,
        json!({"role": "assistant", "content": "Let me check.", "reasoning_content": "Thought", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"London\"}"}},
            {"id": "call_2", "type": "function", "function": {"name": "time", "arguments": "{}"}}
        ]})
    );
    let restored: Message = serde_json::from_value(json).unwrap();
    let Message::Assistant(restored) = restored else {
        panic!("Expected an assistant message, got {restored:?}");
    };
    assert_eq!(restored.reasoning, reply.reasoning);
    assert_eq!(restored.tool_calls[1].args::<Value>().unwrap(), json!({}));

    assert_eq!(
        to_json(ToolMessage::new("call_1", "Rainy")),
        json!({"role": "tool", "tool_call_id": "call_1", "content": "Rainy"})
    );
}

#[test]
fn system_message_is_not_in_history() {
    let system = json!({"role": "system", "content": "Be brief"});
    assert!(serde_json::from_value::<Message>(system).is_err());
}
