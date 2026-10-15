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

mod overflow_handler {
    use crate::gemini::types::request::{Chat, FunctionCall, Role};
    use crate::gemini::types::sessions::{OverflowHandler, Session};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    /// Counts calls and returns `run_default`.
    #[derive(Clone, Debug)]
    struct Counter {
        calls: Arc<AtomicUsize>,
        run_default: bool,
    }
    impl OverflowHandler for Counter {
        fn handle(&mut self, _: &mut Session) -> bool {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.run_default
        }
    }
    fn counter(run_default: bool) -> (Counter, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let handler = Counter {
            calls: calls.clone(),
            run_default,
        };
        (handler, calls)
    }

    /// Forgets oldest chats while total text length exceeds `max_chars`.
    #[derive(Clone, Debug)]
    struct CharBudget {
        max_chars: usize,
    }
    impl OverflowHandler for CharBudget {
        fn handle(&mut self, session: &mut Session) -> bool {
            let chars = |session: &Session| -> usize {
                session
                    .get_history()
                    .iter()
                    .map(|chat| chat.get_text_no_think("").len())
                    .sum()
            };
            while chars(session) > self.max_chars && session.get_history_length() > 1 {
                session.forget_oldest_chat();
            }
            false
        }
    }

    fn texts(session: &Session) -> Vec<String> {
        session
            .get_history()
            .iter()
            .map(|chat| chat.get_text_no_think(""))
            .collect()
    }

    #[test]
    fn returning_false_skips_default_trimming() {
        let (handler, _) = counter(false);
        let mut session = Session::new(2);
        session.set_overflow_handler(handler);
        session.ask("1").reply("2").ask("3").reply("4");
        assert_eq!(texts(&session), ["1", "2", "3", "4"]);
    }

    #[test]
    fn returning_true_runs_default_trimming() {
        let (handler, calls) = counter(true);
        let mut session = Session::new(2);
        session.set_overflow_handler(handler);
        session.ask("1").reply("2").ask("3").reply("4");
        assert_eq!(texts(&session), ["3", "4"]);
        assert_eq!(calls.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn called_on_concatenation() {
        let (handler, calls) = counter(true);
        let mut session = Session::new(10);
        session.set_overflow_handler(handler);
        session.ask("a").ask("b").reply("c").reply("d");
        assert_eq!(session.get_history_length(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn budget_handler_trims_oldest_and_ignores_history_limit() {
        let mut session = Session::new(1);
        session.set_overflow_handler(CharBudget { max_chars: 10 });
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
        let mut session = Session::new(10);
        session.set_overflow_handler(CharBudget { max_chars: 6 });
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
    fn remove_overflow_handler_restores_default() {
        let (handler, calls) = counter(false);
        let mut session = Session::new(2);
        session.set_overflow_handler(handler);
        session.ask("1").reply("2").ask("3");
        assert_eq!(session.get_history_length(), 3);

        // Default trimming brings history back within the limit, even if it was over by more than one
        session.remove_overflow_handler();
        session.reply("4");
        assert_eq!(texts(&session), ["3", "4"]);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn clone_keeps_handler() {
        let mut session = Session::new(1);
        session.set_overflow_handler(CharBudget { max_chars: 100 });
        let mut cloned = session.clone();
        cloned.ask("1").reply("2");
        assert_eq!(cloned.get_history_length(), 2);
    }

    #[test]
    fn handler_is_not_serialized() {
        let mut session = Session::new(1);
        session.set_overflow_handler(CharBudget { max_chars: 100 });
        session.ask("1").reply("2");
        assert_eq!(session.get_history_length(), 2);

        let mut restored: Session =
            serde_json::from_str(&serde_json::to_string(&session).unwrap()).unwrap();
        restored.ask("3");
        assert_eq!(texts(&restored), ["3"]);

        restored.set_overflow_handler(CharBudget { max_chars: 100 });
        restored.reply("4");
        assert_eq!(texts(&restored), ["3", "4"]);
    }
}
