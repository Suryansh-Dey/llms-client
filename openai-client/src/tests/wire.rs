use crate::chat::message::{AssistantMessage, ToolCall};
use crate::chat::session::{Session, SlidingWindow};
use crate::chat::wire::ChatRequest;
use crate::chat::{Tool, ToolChoice};
use serde_json::{Map, Value, json};

fn request<'a>(session: &'a Session, params: &'a Map<String, Value>) -> ChatRequest<'a> {
    ChatRequest {
        model: "model",
        system_prompt: None,
        messages: session.history(),
        tools: &[],
        tool_choice: None,
        params,
        stream: false,
        send_reasoning: false,
    }
}
fn messages(session: &Session, send_reasoning: bool) -> Value {
    let params = Map::new();
    let request = ChatRequest {
        send_reasoning,
        ..request(session, &params)
    };
    serde_json::to_value(&request).unwrap()["messages"].take()
}
fn calls(calls: &[(&str, &str, Value)]) -> AssistantMessage {
    calls
        .iter()
        .map(|(id, name, args)| ToolCall::new(*id, *name, args.clone()))
        .collect::<Vec<ToolCall>>()
        .into()
}

#[test]
fn messages_are_sent_one_by_one() {
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("Hello").ask("there");
    session.reply("Hi");
    assert_eq!(
        messages(&session, false),
        json!([
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "there"},
            {"role": "assistant", "content": "Hi"}
        ])
    );
}

#[test]
fn tool_calls_and_results() {
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("Weather in London and Paris?");
    session.reply(AssistantMessage {
        content: Some("Let me check.".into()),
        ..calls(&[
            ("call_1", "weather", json!({"city": "London"})),
            ("call_2", "weather", json!({"city": "Paris"})),
        ])
    });
    // Answered out of order
    session
        .add_tool_result("call_2", json!({"temp": 20}))
        .unwrap();
    session.add_tool_result("call_1", "Rainy").unwrap();
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
fn assistant_content_is_null_only_with_tool_calls() {
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("Hi");
    session.reply(calls(&[("call_1", "f", json!({}))]));
    session.add_tool_result("call_1", 1).unwrap();
    session.reply(AssistantMessage::default());
    let messages = messages(&session, false);
    assert_eq!(messages[1]["content"], Value::Null);
    assert!(messages[1].as_object().unwrap().contains_key("content"));
    assert_eq!(messages[3]["content"], "");
}

#[test]
fn reasoning_is_sent_only_in_ongoing_tool_loop() {
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("1");
    session.reply(AssistantMessage {
        reasoning: Some("old thought".into()),
        .."one".into()
    });
    session.ask("2");
    session.reply(AssistantMessage {
        reasoning: Some("new thought".into()),
        refusal: Some("I can't".into()),
        ..calls(&[("call_1", "f", json!({}))])
    });
    session.add_tool_result("call_1", 1).unwrap();

    let without = messages(&session, false);
    assert!(without[1].get("reasoning_content").is_none());
    assert!(without[3].get("reasoning_content").is_none());
    assert_eq!(without[1]["content"], "one");
    assert_eq!(without[3]["refusal"], "I can't");

    let with = messages(&session, true);
    assert!(with[1].get("reasoning_content").is_none());
    assert_eq!(with[3]["reasoning_content"], "new thought");
}

#[test]
fn params_are_merged_without_reserved_keys() {
    let mut session = Session::new(SlidingWindow::new(10));
    session.ask("Hi");
    let params = json!({
        "temperature": 0.2,
        "model": "ignored",
        "stream": true,
        "stream_options": {"include_usage": true},
        "reasoning_effort": null,
        "top_k": 5
    });
    let Value::Object(params) = params else {
        unreachable!()
    };
    let body = serde_json::to_value(request(&session, &params)).unwrap();
    assert_eq!(body["model"], "model");
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(body["top_k"], 5);
    let object = body.as_object().unwrap();
    // `stream_options` is only allowed with `stream`
    for key in ["stream", "stream_options", "reasoning_effort"] {
        assert!(!object.contains_key(key), "{key} must not be sent");
    }

    let streamed = ChatRequest {
        stream: true,
        ..request(&session, &params)
    };
    let streamed = serde_json::to_value(&streamed).unwrap();
    assert_eq!(streamed["stream"], true);
    assert_eq!(streamed["stream_options"], json!({"include_usage": true}));
}

#[test]
fn system_prompt_tools_and_tool_choice() {
    let tools = [
        Tool::Function(json!({"name": "f", "parameters": {"type": "object"}})),
        Tool::Custom(json!({"type": "web_search"})),
    ];
    let session = Session::new(SlidingWindow::new(1));
    let params = Map::new();
    let choice = ToolChoice::Function("f".into());
    let body = ChatRequest {
        system_prompt: Some("System"),
        tools: &tools,
        tool_choice: Some(&choice),
        ..request(&session, &params)
    };
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

#[test]
fn client_configuration() {
    use crate::DEFAULT_BASE_URL;
    use crate::chat::{AI, ReasoningEffort, ResponseFormat};
    let schema = json!({"type": "object"});
    let mut ai = AI::new("key", "model", None);
    assert_eq!(ai.get_base_url(), DEFAULT_BASE_URL);
    ai.base_url("http://localhost:11434/v1/")
        .response_format(ResponseFormat::json_schema(schema.clone()))
        .reasoning_effort(ReasoningEffort::XHigh)
        .temperature(0.5)
        .stop(["END"])
        .include_usage(true)
        .param("extra_body", json!({"nested": {"a": 1}}))
        .tool(Tool::Function(json!({"name": "a", "description": "old"})))
        .tool(Tool::Function(json!({"name": "b"})))
        .tool(Tool::Function(json!({"name": "a", "description": "new"})))
        .tool(Tool::Custom(json!({"type": "web_search"})))
        .tool_choice(ToolChoice::Required);
    assert_eq!(ai.get_base_url(), "http://localhost:11434/v1");
    assert_eq!(
        Value::Object(ai.get_params().clone()),
        json!({
            "reasoning_effort": "xhigh",
            "response_format": {"type": "json_schema", "json_schema": {"name": "response", "strict": true, "schema": schema}},
            "temperature": 0.5,
            "stop": ["END"],
            "stream_options": {"include_usage": true},
            "extra_body": {"nested": {"a": 1}}
        })
    );
    let tools = ai.get_tools();
    assert_eq!(
        tools.iter().map(Tool::name).collect::<Vec<_>>(),
        [Some("a"), Some("b"), None]
    );
    assert_eq!(
        tools[0],
        Tool::Function(json!({"name": "a", "description": "new"}))
    );
    assert_eq!(ai.get_tool_choice(), Some(&ToolChoice::Required));

    // `null` removes a parameter
    ai.param("temperature", None::<f64>)
        .tools([])
        .tool_choice(None);
    assert!(!ai.get_params().contains_key("temperature"));
    assert!(ai.get_tools().is_empty());
    assert_eq!(ai.get_tool_choice(), None);
    ai.params_mut()["extra_body"]["nested"]["a"] = 2.into();
    assert_eq!(ai.get_params()["extra_body"], json!({"nested": {"a": 2}}));
}
