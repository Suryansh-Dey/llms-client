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

#[test]
fn test_forgetting_keeps_latest_chat() {
    use crate::gemini::types::request::{Chat, FunctionCall, Role};

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
            Role::Model,
            vec![FunctionCall::new("weather".into(), None).into()],
        ))
        .unwrap();
    assert_eq!(session.get_history_length(), 2);
    session.add_function_response("weather", 20).unwrap();
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
    use crate::gemini::types::request::{Chat, FunctionCall, Role};
    use crate::gemini::types::sessions::{HistoryTrimmer, OverflowHandler, Session};
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
                Role::Model,
                vec![FunctionCall::new("f".into(), None).into()],
            ))
            .unwrap();
        session.add_function_response("f", 1).unwrap();
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
                Role::Model,
                vec![FunctionCall::new("f".into(), None).into()],
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
}
