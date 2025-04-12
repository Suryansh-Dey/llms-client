use super::types::request::*;
use super::types::response::*;
use super::types::sessions::Session;
use awc::Client;
use serde_json::{Value, json};
use std::time::Duration;

const API_TIMEOUT: Duration = Duration::from_secs(30);
const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

pub struct Gemini<'a> {
    client: Client,
    api_key: String,
    model: String,
    sys_prompt: Option<SystemInstruction<'a>>,
    generation_config: Option<Value>,
    tools: Option<Vec<Tool>>,
}
impl<'a> Gemini<'a> {
    /// `sys_prompt` should follow [gemini doc](https://ai.google.dev/gemini-api/docs/text-generation#image-input)
    pub fn new(api_key: String, model: String, sys_prompt: Option<SystemInstruction<'a>>) -> Self {
        Self {
            client: Client::builder().timeout(API_TIMEOUT).finish(),
            api_key,
            model,
            sys_prompt,
            generation_config: None,
            tools: None,
        }
    }
    /// The generation config Schema should follow [Gemini docs](https://ai.google.dev/gemini-api/docs/text-generation#configuration-parameters)
    pub fn set_generation_config(&mut self, generation_config: Value) -> &mut Self {
        self.generation_config = Some(generation_config);
        self
    }
    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }
    pub fn set_api_key(&mut self, api_key: String) {
        self.api_key = api_key;
    }
    /// `schema` should follow [Schema of gemini](https://ai.google.dev/api/caching#Schema)
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
    pub fn unset_json_mode(&mut self) -> &Self {
        if let Some(ref mut generation_config) = self.generation_config {
            generation_config["response_schema"] = None::<Value>.into();
            generation_config["response_mime_type"] = None::<Value>.into();
        }
        self
    }
    ///- `tools` can be None to unset tools from using.  
    ///- Or Vec tools to be allowed
    pub fn set_tools(&mut self, tools: Option<Vec<Tool>>) -> &Self {
        self.tools = tools;
        self
    }
    pub fn unset_code_execution_mode(&mut self) -> &Self {
        self.tools.take();
        self
    }

    pub async fn ask<'b>(
        &self,
        session: &'b mut Session,
    ) -> Result<GeminiResponse, Box<dyn std::error::Error>> {
        let req_url = format!(
            "{BASE_URL}/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let response = self
            .client
            .post(req_url)
            .send_json(&GeminiRequestBody::new(
                self.sys_prompt.as_ref(),
                self.tools.as_deref(),
                &session.get_history().as_slice(),
                self.generation_config.as_ref(),
            ))
            .await?;
        let reply = GeminiResponse::new(response).await?;
        session.update(&reply);
        Ok(reply)
    }
    pub async fn ask_as_stream<'b>(
        &self,
        session: &'b mut Session,
    ) -> Result<GeminiResponseStream<'b>, Box<dyn std::error::Error>> {
        let req_url = format!(
            "{BASE_URL}/{}:streamGenerateContent?key={}",
            self.model, self.api_key
        );

        let response = self
            .client
            .post(req_url)
            .send_json(&GeminiRequestBody::new(
                self.sys_prompt.as_ref(),
                self.tools.as_deref(),
                session.get_history().as_slice(),
                self.generation_config.as_ref(),
            ))
            .await?;
        if !response.status().is_success() {
            return Err(format!(
                "Found status due to {} from Gemini endpoint",
                response.status()
            )
            .into());
        }

        Ok(GeminiResponseStream::new(response, session))
    }
}
