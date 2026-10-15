use gemini_client_api::gemini::types::sessions::{OverflowHandler, Session};

#[derive(Clone, Debug)]
struct Handler;
impl OverflowHandler for Handler {
    fn handle(&mut self, _: &mut Session) -> bool {
        true
    }
}
#[tokio::main]
async fn main() {
    let mut session = Session::new(2);
    session.set_overflow_handler(Handler);
}
