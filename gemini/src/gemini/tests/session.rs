use crate::gemini::types::sessions::Session;

#[test]
fn test_absolute_chat_no() {
    let mut session = Session::new(2);
    session.ask("hum 1");
    session.reply("Tum 1");
    session.ask("hum 2");
    session.ask(" aur sirf tum");
    assert!(session.get_chat_by_no(1).is_none());
    assert_eq!(
        session.get_chat_by_no(2).unwrap().get_text_no_think("\n"),
        "Tum 1"
    );
    assert_eq!(
        session
            .get_chat_by_no(session.get_chat_no())
            .unwrap()
            .get_text_no_think("\n"),
        "hum 2 aur sirf tum"
    );
}

#[test]
fn test_function_role_sent_as_user() {
    use crate::gemini::types::request::{GeminiRequestBody, Role};

    let mut session = Session::new(4);
    session.ask("What's the weather?");
    session.reply("calling weather()");
    session
        .add_function_response("weather", serde_json::json!({ "temp": 20 }))
        .unwrap();

    let history = session.get_history();
    let body = GeminiRequestBody::new(None, None, &history, None, None, None, None);
    let roles: Vec<String> = serde_json::to_value(&body).unwrap()["contents"]
        .as_array()
        .unwrap()
        .iter()
        .map(|chat| chat["role"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(roles, ["user", "model", "user"]);

    // Session's own serde still round-trips Role::Function.
    let restored: Session =
        serde_json::from_str(&serde_json::to_string(&session).unwrap()).unwrap();
    assert_eq!(*restored.get_last_chat().unwrap().role(), Role::Function);
}
