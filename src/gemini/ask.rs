use super::types::*;
use awc::Client;
use serde_json::Value;

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

pub struct Gemini<'a> {
    client: Client,
    api_key: &'a str,
    model: &'a str,
}
impl<'a> Gemini<'a> {
    pub fn new(api_key: &'a str, model: &'a str) -> Gemini<'a> {
        Self {
            client: Client::new(),
            api_key,
            model,
        }
    }
    pub async fn ask_string(&self, question: &str) -> Result<String, Box<dyn std::error::Error>> {
        let req_url = format!(
            "{BASE_URL}/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let response: Value = self
            .client
            .post(req_url)
            .send_json(&GeminiBody::new(&[Chat::new(
                Role::user,
                &[Part::text(question)],
            )]))
            .await?
            .json()
            .await?;

        Ok(response["candidates"][0]["content"]["parts"][0]["text"].to_string())
    }
}

#[actix_web::test]
async fn it_works() {
    let data = Gemini::new(&std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not found"), "gemini-1.5-flash")
        .ask_string("Hi")
        .await
        .unwrap();
    println!("{data}");
}
