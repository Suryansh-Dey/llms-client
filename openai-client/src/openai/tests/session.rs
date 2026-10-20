use crate::openai::types::sessions::Session;

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
fn test_tool_role_serde() {
    use crate::openai::types::request::{Chat, FunctionCall, Role};

    let mut session = Session::new(4);
    session.ask("What's the weather?");
    session
        .add_chat(Chat::new(
            Role::Assistant,
            vec![FunctionCall::new("call_1", "weather", serde_json::json!({})).into()],
        ))
        .unwrap();
    session
        .add_function_response("call_1", serde_json::json!({ "temp": 20 }))
        .unwrap();

    // Session's own serde round-trips Role::Tool and the function response.
    let restored: Session =
        serde_json::from_str(&serde_json::to_string(&session).unwrap()).unwrap();
    let last = restored.get_last_chat().unwrap();
    assert_eq!(*last.role(), Role::Tool);
    assert_eq!(restored.get_history_length(), 3);
}

#[test]
fn test_function_responses_by_id() {
    use crate::openai::types::request::{Chat, FunctionCall, PartType, Role};
    use crate::openai::types::sessions::AddFunctionResponseError;
    use serde_json::json;

    let mut session = Session::new(10);
    assert!(matches!(
        session.add_function_response("call_1", 1),
        Err(AddFunctionResponseError::FunctionResponseAfterUser)
    ));
    session.ask("Weather in London and Paris?");
    assert!(matches!(
        session.add_function_response("call_1", 1),
        Err(AddFunctionResponseError::FunctionResponseAfterUser)
    ));
    session
        .add_chat(Chat::new(
            Role::Assistant,
            vec![
                FunctionCall::new("call_1", "weather", json!({"city": "London"})).into(),
                FunctionCall::new("call_2", "weather", json!({"city": "Paris"})).into(),
            ],
        ))
        .unwrap();
    let pending = |session: &Session| -> Vec<String> {
        session
            .get_pending_function_calls()
            .iter()
            .map(|call| call.id().clone())
            .collect()
    };
    assert_eq!(pending(&session), ["call_1", "call_2"]);

    assert!(matches!(
        session.add_function_response("call_3", 1),
        Err(AddFunctionResponseError::NoPendingCall(id)) if id == "call_3"
    ));
    session
        .add_function_response("call_2", "20 degrees")
        .unwrap();
    assert_eq!(pending(&session), ["call_1"]);
    // Already responded
    assert!(matches!(
        session.add_function_response("call_2", 1),
        Err(AddFunctionResponseError::NoPendingCall(_))
    ));
    session.add_function_response("call_1", 18).unwrap();
    assert!(pending(&session).is_empty());

    // Both responses are in one chat, with names of their calls
    assert_eq!(session.get_history_length(), 3);
    let responses: Vec<(String, String, String)> = session
        .get_last_chat()
        .unwrap()
        .parts()
        .iter()
        .filter_map(|part| match part.data() {
            PartType::FunctionResponse(r) => Some((
                r.id().clone(),
                r.name().clone(),
                Chat::extract_text_all(r.parts(), ""),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        responses,
        [
            ("call_2".into(), "weather".into(), "20 degrees".into()),
            ("call_1".into(), "weather".into(), "18".into())
        ]
    );

    // No pending calls once the model replies
    session.reply("It is 18 in London and 20 in Paris");
    assert!(pending(&session).is_empty());
}

#[test]
fn test_forgetting_keeps_latest_chat() {
    use crate::openai::types::request::{Chat, FunctionCall, Role};

    let mut session = Session::new(10);
    session.ask("only");
    assert!(session.forget_oldest_chat().is_none());

    // Limit 0 used to empty the history on every insertion
    let mut session = Session::new(0);
    session.ask("hi");
    assert_eq!(session.get_history_length(), 1);

    // Forgetting the prompt would also remove the function call, leaving nothing to respond to
    let mut session = Session::new(1);
    session.ask("weather?");
    session
        .add_chat(Chat::new(
            Role::Assistant,
            vec![FunctionCall::new("call_1", "weather", serde_json::json!({})).into()],
        ))
        .unwrap();
    assert_eq!(session.get_history_length(), 2);
    session.add_function_response("call_1", 20).unwrap();
    assert_eq!(session.get_history_length(), 3);

    // Once the model replies, the whole exchange can be forgotten
    session.reply("It is 20 degrees");
    assert_eq!(session.get_history_length(), 1);
    assert_eq!(
        session.get_chat_by_no(4).unwrap().get_text_no_think(""),
        "It is 20 degrees"
    );
}

mod overflow_handler {
    use crate::openai::types::request::{Chat, FunctionCall, Role};
    use crate::openai::types::sessions::{HistoryTrimmer, OverflowHandler, Session};
    use serde::{Deserialize, Serialize};

    /// Counts calls and returns `run_default`.
    #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
    struct Counter {
        calls: usize,
        run_default: bool,
    }
    impl OverflowHandler for Counter {
        fn handle(&mut self, _: &mut HistoryTrimmer<'_>) -> bool {
            self.calls += 1;
            self.run_default
        }
    }
    fn counter(run_default: bool) -> Counter {
        Counter {
            calls: 0,
            run_default,
        }
    }

    /// Forgets oldest chats while total text length exceeds `max_chars`.
    #[derive(Clone, Debug)]
    struct CharBudget {
        max_chars: usize,
    }
    impl OverflowHandler for CharBudget {
        fn handle(&mut self, history: &mut HistoryTrimmer<'_>) -> bool {
            let chars = |history: &HistoryTrimmer| -> usize {
                history
                    .get_history_as_vecdeque()
                    .iter()
                    .map(|chat| chat.get_text_no_think("").len())
                    .sum()
            };
            while chars(history) > self.max_chars && history.forget_oldest_chat().is_some() {}
            false
        }
    }

    fn texts<H: OverflowHandler>(session: &Session<H>) -> Vec<String> {
        session
            .get_history()
            .iter()
            .map(|chat| chat.get_text_no_think(""))
            .collect()
    }

    #[test]
    fn returning_false_skips_default_trimming() {
        let mut session = Session::new(2).set_overflow_handler(counter(false));
        session.ask("1").reply("2").ask("3").reply("4");
        assert_eq!(texts(&session), ["1", "2", "3", "4"]);
        assert_eq!(session.overflow_handler().calls, 4);
    }

    #[test]
    fn returning_true_runs_default_trimming() {
        let mut session = Session::new(2).set_overflow_handler(counter(true));
        session.ask("1").reply("2").ask("3").reply("4");
        assert_eq!(texts(&session), ["3", "4"]);
    }

    #[test]
    fn called_on_concatenation() {
        let mut session = Session::new(10).set_overflow_handler(counter(true));
        session.ask("a").ask("b").reply("c").reply("d");
        assert_eq!(session.get_history_length(), 2);
        assert_eq!(session.overflow_handler().calls, 4);
    }

    #[test]
    fn handler_state_can_be_changed() {
        let mut session = Session::new(2).set_overflow_handler(counter(true));
        session.overflow_handler_mut().run_default = false;
        session.ask("1").reply("2").ask("3");
        assert_eq!(session.get_history_length(), 3);
    }

    #[test]
    fn budget_handler_trims_oldest_and_ignores_history_limit() {
        let mut session = Session::new(1).set_overflow_handler(CharBudget { max_chars: 10 });
        session.ask("aaaa").reply("bbbb").ask("cc");
        assert_eq!(texts(&session), ["aaaa", "bbbb", "cc"]);

        session.reply("dddd");
        assert_eq!(texts(&session), ["bbbb", "cc", "dddd"]);

        // Growing the last chat through concatenation also triggers trimming
        session.reply("eeee");
        assert_eq!(texts(&session), ["cc", "ddddeeee"]);
    }

    #[test]
    fn forgetting_keeps_function_call_and_response_together() {
        let mut session = Session::new(10).set_overflow_handler(CharBudget { max_chars: 6 });
        session.ask("aaaa");
        session
            .add_chat(Chat::new(
                Role::Assistant,
                vec![FunctionCall::new("call_f", "f", serde_json::json!({})).into()],
            ))
            .unwrap();
        session.add_function_response("call_f", 1).unwrap();
        assert_eq!(session.get_history_length(), 3);

        // Forgetting "aaaa" leaves a dangling function call and response at the start,
        // so both are removed too.
        session.reply("bbbbbbb");
        assert_eq!(texts(&session), ["bbbbbbb"]);
    }

    #[test]
    fn never_forgets_latest_chat() {
        let mut session = Session::new(10).set_overflow_handler(CharBudget { max_chars: 2 });
        session.ask("aaaa");
        assert_eq!(texts(&session), ["aaaa"]);

        // Forgetting the prompt would also remove the function call that follows it
        session
            .add_chat(Chat::new(
                Role::Assistant,
                vec![FunctionCall::new("call_f", "f", serde_json::json!({})).into()],
            ))
            .unwrap();
        assert_eq!(session.get_history_length(), 2);
    }

    #[test]
    fn remove_overflow_handler_restores_default() {
        let mut session = Session::new(2).set_overflow_handler(counter(false));
        session.ask("1").reply("2").ask("3");
        let (mut session, handler) = session.remove_overflow_handler();
        assert_eq!(handler.calls, 3);

        // Default trimming brings history back within the limit, even if it was over by more than one
        session.reply("4");
        assert_eq!(texts(&session), ["3", "4"]);
    }

    #[test]
    fn clone_copies_handler_state() {
        let mut session = Session::new(10).set_overflow_handler(counter(true));
        session.ask("1");
        let mut cloned = session.clone();
        cloned.reply("2");
        assert_eq!(session.overflow_handler().calls, 1);
        assert_eq!(cloned.overflow_handler().calls, 2);
    }

    #[test]
    fn boxed_handler_can_be_chosen_at_runtime() {
        let handlers: Vec<Box<dyn OverflowHandler>> = vec![
            Box::new(counter(true)),
            Box::new(CharBudget { max_chars: 4 }),
        ];
        let lengths: Vec<usize> = handlers
            .into_iter()
            .map(|handler| {
                let mut session = Session::new(10).set_overflow_handler(handler);
                session.ask("aa").reply("bb").ask("cc");
                session.get_history_length()
            })
            .collect();
        assert_eq!(lengths, [3, 2]);
    }

    #[test]
    fn custom_handler_state_is_serialized() {
        let mut session = Session::new(10).set_overflow_handler(counter(false));
        session.ask("1").reply("2");
        let json = serde_json::to_string(&session).unwrap();
        let restored: Session<Counter> = serde_json::from_str(&json).unwrap();
        assert_eq!(texts(&restored), ["1", "2"]);
        assert_eq!(restored.overflow_handler(), session.overflow_handler());
        assert_eq!(restored.overflow_handler().calls, 2);

        // A session saved without the handler can't be loaded as one with it
        let json = serde_json::to_string(&Session::new(10)).unwrap();
        assert!(serde_json::from_str::<Session<Counter>>(&json).is_err());
    }

    #[test]
    fn session_saved_before_overflow_handlers_loads() {
        // Without the `overflow_handler` field
        let json = r#"{"history":[{"role":"user","parts":[{"text":"hi"}]}],"history_limit":2,"chat_no":1,"remember_reply":true}"#;
        let mut session: Session = serde_json::from_str(json).unwrap();
        assert_eq!(texts(&session), ["hi"]);
        assert_eq!(session.get_chat_no(), 1);
        session.reply("hello").ask("bye");
        assert_eq!(texts(&session), ["hello", "bye"]);

        // Saved again with the handler as `null`, which loads back too
        let json = serde_json::to_string(&session).unwrap();
        assert!(json.ends_with(r#""overflow_handler":null}"#));
        let restored: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(texts(&restored), ["hello", "bye"]);

        // Custom handlers still require their state to be present
        let old = r#"{"history":[],"history_limit":2,"chat_no":0,"remember_reply":true}"#;
        assert!(serde_json::from_str::<Session<Counter>>(old).is_err());
    }
}
