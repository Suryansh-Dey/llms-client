use super::types::*;
use awc::Client;
use serde_json::{Value, json};
use std::time::Duration;

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

pub struct Gemini<'a> {
    client: Client,
    api_key: &'a str,
    model: &'a str,
    sys_prompt: Option<SystemInstruction<'a>>,
    generation_config: Option<Value>,
}
impl<'a> Gemini<'a> {
    pub fn new(
        api_key: &'a str,
        model: &'a str,
        sys_prompt: Option<SystemInstruction<'a>>,
    ) -> Gemini<'a> {
        Self {
            client: Client::builder().timeout(Duration::from_secs(30)).finish(),
            api_key,
            model,
            sys_prompt,
            generation_config: None,
        }
    }
    pub fn set_generation_config(&mut self, generation_config: Value) -> &mut Self {
        self.generation_config = Some(generation_config);
        self
    }
    pub fn set_json_mode(&mut self, schema: Value) -> &Self {
        if let None = self.generation_config {
            self.generation_config = Some(json!({
                "response_mime_type": "application/json",
                "response_schema":schema
            }))
        } else if let Some(config) = self.generation_config.as_mut() {
            config["response_mime_type"] = "application/json".into();
            config["response_schema"] = schema.into();
        }
        self
    }

    pub async fn ask_string(&self, question: &str) -> Result<Value, Box<dyn std::error::Error>> {
        let req_url = format!(
            "{BASE_URL}/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let response: Value = self
            .client
            .post(req_url)
            .send_json(&GeminiBody::new(
                self.sys_prompt.as_ref(),
                &[Chat::new(Role::user, &[Part::text(question)])],
                self.generation_config.as_ref(),
            ))
            .await?
            .json()
            .await?;

        Ok(response)
    }
    pub fn get_response_string(response: &Value) -> String {
        response["candidates"][0]["content"]["parts"][0]["text"].to_string()
    }
    pub fn get_response_json(response: &Value) -> Result<Value, serde_json::Error> {
        let string = response["candidates"][0]["content"]["parts"][0]["text"].to_string();
        let unescaped_str = string.replace("\\\"", "\"").replace("\\n", "\n");
        serde_json::from_str(&unescaped_str[1..unescaped_str.len() - 1])
    }
}
