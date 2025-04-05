use super::types::*;
use awc::Client;
use serde_json::{Value, json};
use std::time::Duration;

const API_TIMEOUT: Duration = Duration::from_secs(30);
const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

pub struct GeminiResponse(Value);
impl GeminiResponse {
    fn new(value: Value) -> Self {
        Self(value)
    }
    pub fn get(&self) -> &Value {
        &self.0
    }
    pub fn get_as_string(&self) -> Result<&str, &Value> {
        self.get()["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .ok_or(self.get())
    }
    pub fn get_as_json(&self) -> Result<Value, &Value> {
        let string = self.get()["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .ok_or(self.get())?;
        let unescaped_str = string.replace("\\\"", "\"").replace("\\n", "\n");
        serde_json::from_str::<Value>(&unescaped_str).map_err(|_| self.get())
    }
}

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
    ) -> Self {
        Self {
            client: Client::builder().timeout(API_TIMEOUT).finish(),
            api_key,
            model,
            sys_prompt,
            generation_config: None,
        }
    }
    pub fn set_model(&mut self, model: &'a str) {
        self.model = model;
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

    pub async fn ask_string(
        &self,
        question: String,
    ) -> Result<GeminiResponse, Box<dyn std::error::Error>> {
        let req_url = format!(
            "{BASE_URL}/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let response: Value = self
            .client
            .post(req_url)
            .send_json(&GeminiBody::new(
                self.sys_prompt.as_ref(),
                &[Chat::new(Role::user, vec![Part::text(question)])],
                self.generation_config.as_ref(),
            ))
            .await?
            .json()
            .await?;

        Ok(GeminiResponse::new(response))
    }
    pub async fn ask(
        &self,
        session: &'a mut Session,
        question: Vec<Part>,
    ) -> Result<GeminiResponse, Box<dyn std::error::Error>> {
        let req_url = format!(
            "{BASE_URL}/{}:generateContent?key={}",
            self.model, self.api_key
        );
        let history = session.get_history_mut();
        history.push(Chat::new(Role::user, question));

        let response: Value = self
            .client
            .post(req_url)
            .send_json(&GeminiBody::new(
                self.sys_prompt.as_ref(),
                history.as_slice(),
                self.generation_config.as_ref(),
            ))
            .await?
            .json()
            .await?;

        Ok(GeminiResponse::new(response))
    }
}
