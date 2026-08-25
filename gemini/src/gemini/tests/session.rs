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
