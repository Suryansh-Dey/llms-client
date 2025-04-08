use super::types::*;
use actix_web::http::header;
use awc::Client;
use serde_json::Value;

const BASE_URL: &str = "https://api.openai.com/v1/chat/completions";

pub struct ChatGpt<'a> {
    client: Client,
    api_key: &'a str,
    model: &'a str,
}
impl<'a> ChatGpt<'a> {
    pub fn new(api_key: &'a str, model: &'a str) -> ChatGpt<'a> {
        Self {
            client: Client::new(),
            api_key,
            model,
        }
    }
    pub async fn ask_string(&self, question: &str) -> Result<String, Box<dyn std::error::Error>> {
        let response: Value = self
            .client
            .post(BASE_URL)
            .append_header((header::AUTHORIZATION, format!("Bearer {}", self.api_key)))
            .send_json(&OpenAiBody::new(
                self.model,
                &[Chat::new(Role::user, question)],
            ))
            .await?
            .json()
            .await?;

        Ok(response["choices"][0]["message"]["content"].to_string())
    }
}
