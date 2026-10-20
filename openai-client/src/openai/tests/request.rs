use crate::openai::types::request::*;
use crate::openai::types::sessions::Session;
use mime::Mime;
use serde_json::{Value, json};
use std::str::FromStr;

fn body(session: &Session, config: Option<&Value>, send_reasoning: bool) -> Value {
    let history = session.get_history();
    let body = OpenAIRequestBody::new(
        "model",
        None,
        &history,
        None,
        None,
        config,
        false,
        send_reasoning,
    );
    serde_json::to_value(&body).unwrap()
}
fn messages(session: &Session, send_reasoning: bool) -> Value {
    body(session, None, send_reasoning)["messages"].take()
}
fn assistant_with_call(id: &str, name: &str, args: Value) -> Chat {
    Chat::new(
        Role::Assistant,
        vec![FunctionCall::new(id, name, args).into()],
    )
}

#[test]
fn text_only_content_is_string() {
    let mut session = Session::new(10);
    session.ask("Hello").ask(" there");
    session.reply("Hi");
    assert_eq!(
        messages(&session, false),
        json!([
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi"}
        ])
    );
}

#[test]
fn multimodal_content_is_parts() {
    let mut session = Session::new(10);
    session.ask_parts(vec![
        "Compare".into(),
        InlineData::new(mime::IMAGE_PNG, "aW1n".into()).into(),
        ImageUrl::new("https://example.com/a.jpg".into(), Some(ImageDetail::Low)).into(),
        InlineData::new(Mime::from_str("audio/mpeg").unwrap(), "YXVk".into()).into(),
        InlineData::new(mime::APPLICATION_PDF, "cGRm".into()).into(),
    ]);
    assert_eq!(
        messages(&session, false),
        json!([{"role": "user", "content": [
            {"type": "text", "text": "Compare"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg", "detail": "low"}},
            {"type": "input_audio", "input_audio": {"data": "YXVk", "format": "mp3"}},
            {"type": "file", "file": {"filename": "file.pdf", "file_data": "data:application/pdf;base64,cGRm"}}
        ]}])
    );
}

#[test]
fn function_calls_and_responses() {
    let mut session = Session::new(10);
    session.ask("Weather in London and Paris?");
    session
        .add_chat(Chat::new(
            Role::Assistant,
            vec![
                "Let me check.".into(),
                FunctionCall::new("call_1", "weather", json!({"city": "London"})).into(),
                FunctionCall::new("call_2", "weather", json!({"city": "Paris"})).into(),
            ],
        ))
        .unwrap();
    // Answered out of order
    session
        .add_function_response("call_2", json!({"temp": 20}))
        .unwrap();
    session.add_function_response("call_1", "Rainy").unwrap();
    session.reply("Rainy in London, 20 in Paris");

    assert_eq!(
        messages(&session, false),
        json!([
            {"role": "user", "content": "Weather in London and Paris?"},
            {"role": "assistant", "content": "Let me check.", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"London\"}"}},
                {"id": "call_2", "type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}}
            ]},
            {"role": "tool", "content": "{\"temp\":20}", "tool_call_id": "call_2"},
            {"role": "tool", "content": "Rainy", "tool_call_id": "call_1"},
            {"role": "assistant", "content": "Rainy in London, 20 in Paris"}
        ])
    );
}

#[test]
fn assistant_with_only_tool_calls_has_null_content() {
    let mut session = Session::new(10);
    session.ask("Hi");
    session
        .add_chat(assistant_with_call("call_1", "f", json!({})))
        .unwrap();
    let messages = messages(&session, false);
    assert_eq!(messages[1]["content"], Value::Null);
    assert!(messages[1].as_object().unwrap().contains_key("content"));
}

#[test]
fn multimodal_function_response_is_sent_in_user_message() {
    let image = || InlineData::new(mime::IMAGE_PNG, "aW1n".into());
    let mut session = Session::new(10);
    session.ask("Take screenshots");
    session
        .add_chat(Chat::new(
            Role::Assistant,
            vec![
                FunctionCall::new("call_1", "screenshot", json!({})).into(),
                FunctionCall::new("call_2", "screenshot", json!({})).into(),
            ],
        ))
        .unwrap();
    session
        .add_function_response(
            "call_1",
            MultimodalResponse::new(vec!["Home page".into(), image().into()]),
        )
        .unwrap();
    session
        .add_function_response("call_2", MultimodalResponse::from(vec![image().into()]))
        .unwrap();

    let attachments = json!([
        {"type": "text", "text": "Attachments of `screenshot` function output (call id call_1):"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}},
        {"type": "text", "text": "Attachments of `screenshot` function output (call id call_2):"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}}
    ]);
    let messages = messages(&session, false);
    assert_eq!(
        messages[2],
        json!({"role": "tool", "content": "Home page", "tool_call_id": "call_1"})
    );
    assert_eq!(
        messages[3],
        json!({"role": "tool", "content": "Output is attached in the next message.", "tool_call_id": "call_2"})
    );
    assert_eq!(messages[4], json!({"role": "user", "content": attachments}));
    assert_eq!(messages.as_array().unwrap().len(), 5);

    // Attachments are put before the next user prompt in the same message
    session.ask("What do you see?");
    let messages = self::messages(&session, false);
    let mut content = attachments.as_array().unwrap().clone();
    content.push(json!({"type": "text", "text": "What do you see?"}));
    assert_eq!(messages[4], json!({"role": "user", "content": content}));
    assert_eq!(messages.as_array().unwrap().len(), 5);

    // and before the next model reply, if there is no user prompt
    session.remove_last_chat();
    session.reply("I see two images");
    let messages = self::messages(&session, false);
    assert_eq!(messages[4], json!({"role": "user", "content": attachments}));
    assert_eq!(messages[5]["role"], "assistant");
}

#[test]
fn reasoning_is_sent_only_in_ongoing_tool_loop() {
    let mut session = Session::new(10);
    session.ask("1");
    session.reply_parts(vec![Part::new_thought("old thought"), "one".into()]);
    session.ask("2");
    session
        .add_chat(Chat::new(
            Role::Assistant,
            vec![
                Part::new_thought("new thought"),
                FunctionCall::new("call_1", "f", json!({})).into(),
            ],
        ))
        .unwrap();
    session.add_function_response("call_1", 1).unwrap();

    let without = messages(&session, false);
    assert!(without[1].get("reasoning_content").is_none());
    assert!(without[3].get("reasoning_content").is_none());
    assert_eq!(without[1]["content"], "one");

    let with = messages(&session, true);
    assert!(with[1].get("reasoning_content").is_none());
    assert_eq!(with[3]["reasoning_content"], "new thought");
}

#[test]
fn generation_config_is_merged_without_reserved_keys() {
    let mut session = Session::new(10);
    session.ask("Hi");
    let config = json!({
        "temperature": 0.2,
        "model": "ignored",
        "stream": true,
        "stream_options": {"include_usage": true},
        "reasoning_effort": null,
        "top_k": 5
    });
    let body = body(&session, Some(&config), false);
    assert_eq!(body["model"], "model");
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(body["top_k"], 5);
    let object = body.as_object().unwrap();
    // `stream_options` is only allowed with `stream`
    for key in ["stream", "stream_options", "reasoning_effort"] {
        assert!(!object.contains_key(key), "{key} must not be sent");
    }

    let history = session.get_history();
    let streamed = OpenAIRequestBody::new(
        "model",
        None,
        &history,
        None,
        None,
        Some(&config),
        true,
        false,
    );
    let streamed = serde_json::to_value(&streamed).unwrap();
    assert_eq!(streamed["stream"], true);
    assert_eq!(streamed["stream_options"], json!({"include_usage": true}));
}

#[test]
fn tools_and_tool_choice() {
    let tools = [
        Tool::Function(json!({"name": "f", "parameters": {"type": "object"}})),
        Tool::Custom(json!({"type": "web_search"})),
    ];
    let session = Session::new(1);
    let history = session.get_history();
    let choice = ToolChoice::Function("f".into());
    let sys_prompt = SystemInstruction::from("System");
    let body = OpenAIRequestBody::new(
        "model",
        Some(&sys_prompt),
        &history,
        Some(&tools),
        Some(&choice),
        None,
        false,
        false,
    );
    let body = serde_json::to_value(&body).unwrap();
    assert_eq!(
        body["tools"],
        json!([
            {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}},
            {"type": "web_search"}
        ])
    );
    assert_eq!(
        body["tool_choice"],
        json!({"type": "function", "function": {"name": "f"}})
    );
    assert_eq!(
        body["messages"],
        json!([{"role": "system", "content": "System"}])
    );
    assert_eq!(
        serde_json::to_value(ToolChoice::Required).unwrap(),
        "required"
    );

    // Round trip
    let tools: Vec<Tool> = serde_json::from_value(body["tools"].clone()).unwrap();
    assert_eq!(tools[0].name(), Some("f"));
    assert!(matches!(tools[1], Tool::Custom(_)));
    let choice: ToolChoice = serde_json::from_value(body["tool_choice"].clone()).unwrap();
    assert_eq!(choice, ToolChoice::Function("f".into()));
}

#[cfg(feature = "reqwest")]
#[test]
fn client_configuration() {
    use crate::openai::ask::{DEFAULT_BASE_URL, OpenAI};
    let schema = json!({"type": "object"});
    let ai = OpenAI::new("key", "model", None);
    assert_eq!(ai.base_url(), DEFAULT_BASE_URL);
    let ai = ai
        .set_base_url("http://localhost:11434/v1/")
        .set_json_mode(schema.clone())
        .set_reasoning_effort(ReasoningEffort::XHigh)
        .set_tool(Tool::Function(json!({"name": "a", "description": "old"})))
        .set_tool(Tool::Function(json!({"name": "b"})))
        .set_tool(Tool::Function(json!({"name": "a", "description": "new"})))
        .set_tool(Tool::Custom(json!({"type": "web_search"})));
    assert_eq!(ai.base_url(), "http://localhost:11434/v1");
    let config = ai.generation_config().as_ref().unwrap();
    assert_eq!(config["reasoning_effort"], "xhigh");
    assert_eq!(
        config["response_format"],
        json!({"type": "json_schema", "json_schema": {"name": "response", "strict": true, "schema": schema}})
    );
    let tools = ai.tools().as_ref().unwrap();
    assert_eq!(tools.len(), 3);
    assert_eq!(
        tools.iter().map(Tool::name).collect::<Vec<_>>(),
        [Some("a"), Some("b"), None]
    );
    if let Tool::Function(a) = &tools[0] {
        assert_eq!(a["description"], "new");
    }

    let ai = ai.remove_json_mode().remove_tools();
    assert!(
        ai.generation_config()
            .as_ref()
            .unwrap()
            .get("response_format")
            .is_none()
    );
    assert!(ai.tools().is_none());
}
