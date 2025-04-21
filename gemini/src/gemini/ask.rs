use super::types::request::*;
use super::types::response::*;
use super::types::sessions::Session;
use awc::Client;
use serde_json::{Value, json};
use std::time::Duration;

const API_TIMEOUT: Duration = Duration::from_secs(60);
const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

#[derive(Clone, Default)]
pub struct Gemini {
    client: Client,
    api_key: String,
    model: String,
    sys_prompt: Option<SystemInstruction>,
    generation_config: Option<Value>,
    tools: Option<Vec<Tool>>,
}
impl Gemini {
    /// `sys_prompt` should follow [gemini doc](https://ai.google.dev/gemini-api/docs/text-generation#image-input)
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        sys_prompt: Option<SystemInstruction>,
    ) -> Self {
        Self {
            client: Client::builder().timeout(API_TIMEOUT).finish(),
            api_key: api_key.into(),
            model: model.into(),
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
    pub fn set_model(&mut self, model: impl Into<String>) -> &mut Self {
        self.model = model.into();
        self
    }
    pub fn set_api_key(&mut self, api_key: impl Into<String>) -> &mut Self {
        self.api_key = api_key.into();
        self
    }
    /// `schema` should follow [Schema of gemini](https://ai.google.dev/api/caching#Schema)
    pub fn set_json_mode(&mut self, schema: Value) -> &mut Self {
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
    pub fn unset_json_mode(&mut self) -> &mut Self {
        if let Some(ref mut generation_config) = self.generation_config {
            generation_config["response_schema"] = None::<Value>.into();
            generation_config["response_mime_type"] = None::<Value>.into();
        }
        self
    }
    ///- `tools` can be None to unset tools from using.  
    ///- Or Vec tools to be allowed
    pub fn set_tools(&mut self, tools: Option<Vec<Tool>>) -> &mut Self {
        self.tools = tools;
        self
    }
    pub fn unset_code_execution_mode(&mut self) -> &mut Self {
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

        let mut response = self
            .client
            .post(req_url)
            .send_json(&GeminiRequestBody::new(
                self.sys_prompt.as_ref(),
                self.tools.as_deref(),
                &session.get_history().as_slice(),
                self.generation_config.as_ref(),
            ))
            .await?;

        if !response.status().is_success() {
            let body = response.body().await?;
            let text = std::str::from_utf8(&body)?;
            return Err(text.into());
        }

        let reply = GeminiResponse::new(response).await?;
        session.update(&reply);
        Ok(reply)
    }
    /// # Warining
    /// You must read the response stream to get reply stored context in sessions.
    /// # Example
    ///```ignore
    ///use futures::StreamExt
    ///let mut response_stream = gemini.ask_as_stream(session).await.unwrap();
    ///while let Some(response) = response_stream.next().await {
    ///    if let Ok(response) = response {
    ///        println!("{}", response.get_text(""));
    ///    }
    ///}
    ///```
    pub async fn ask_as_stream(
        &self,
        session: Session,
    ) -> Result<GeminiResponseStream, Box<dyn std::error::Error>> {
        let req_url = format!(
            "{BASE_URL}/{}:streamGenerateContent?key={}",
            self.model, self.api_key
        );

        let mut response = self
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
            let body = response.body().await?;
            let text = std::str::from_utf8(&body)?;
            return Err(text.into());
        }

        Ok(GeminiResponseStream::new(response, session))
    }
}
