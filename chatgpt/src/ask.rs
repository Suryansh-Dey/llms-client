use crate::error::ChatGptError;
use crate::types::request::{ChatMessage, Role, ToolDefinition};
use crate::types::response::{
    ChatGptResponse, ChatGptResponseStream, ChatGptStreamChunk, ResponseStream,
};
use crate::types::sessions::Session;
use crate::utils::merge_json;
use reqwest::Client;
use serde_json::{Value, json};
use std::time::Duration;

const BASE_URL: &str = "https://api.openai.com/v1/chat/completions";

#[derive(Clone, Debug)]
pub struct ChatGpt {
    client: Client,
    api_key: String,
    model: String,
    sys_prompt: Option<ChatMessage>,
    generation_config: Option<Value>,
    tools: Option<Vec<ToolDefinition>>,
    response_format: Option<Value>,
}

impl ChatGpt {
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        sys_prompt: Option<String>,
    ) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("Unable to build HTTP client"),
            api_key: api_key.into(),
            model: model.into(),
            sys_prompt: sys_prompt
                .map(|prompt| ChatMessage::new(Role::System, prompt, None, None)),
            generation_config: None,
            tools: None,
            response_format: None,
        }
    }

    pub fn new_with_timeout(
        api_key: impl Into<String>,
        model: impl Into<String>,
        sys_prompt: Option<String>,
        api_timeout: Duration,
    ) -> Self {
        Self {
            client: Client::builder()
                .timeout(api_timeout)
                .build()
                .expect("Unable to build HTTP client"),
            api_key: api_key.into(),
            model: model.into(),
            sys_prompt: sys_prompt
                .map(|prompt| ChatMessage::new(Role::System, prompt, None, None)),
            generation_config: None,
            tools: None,
            response_format: None,
        }
    }

    pub fn set_generation_config(&mut self) -> &mut Value {
        if self.generation_config.is_none() {
            self.generation_config = Some(json!({}));
        }
        self.generation_config.as_mut().unwrap()
    }

    pub fn set_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn set_sys_prompt(mut self, sys_prompt: Option<String>) -> Self {
        self.sys_prompt = sys_prompt.map(|prompt| ChatMessage::new(Role::System, prompt, None, None));
        self
    }

    pub fn set_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// `schema` should follow the OpenAI JSON schema format.
    pub fn set_json_mode(mut self, schema: Value) -> Self {
        self.response_format = Some(json!({
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": true
            }
        }));
        self
    }

    pub fn unset_json_mode(mut self) -> Self {
        self.response_format = None;
        self
    }

    pub fn set_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn unset_tools(mut self) -> Self {
        self.tools = None;
        self
    }

    fn build_messages(&self, session: &Session) -> Vec<ChatMessage> {
        let mut messages = Vec::new();
        if let Some(sys_prompt) = &self.sys_prompt {
            messages.push(sys_prompt.clone());
        }
        messages.extend(session.get_history_cloned());
        messages
    }

    fn build_body(&self, session: &Session, stream: bool) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": self.build_messages(session),
        });

        if let Some(gen_config) = &self.generation_config {
            merge_json(&mut body, gen_config);
        }

        if let Some(tools) = &self.tools {
            body["tools"] = Value::Array(tools.clone());
        }

        if let Some(response_format) = &self.response_format {
            body["response_format"] = response_format.clone();
        }

        if stream {
            body["stream"] = Value::Bool(true);
        }

        body
    }

    pub async fn ask(&self, session: &mut Session) -> Result<ChatGptResponse, ChatGptError> {
        let response = self
            .client
            .post(BASE_URL)
            .bearer_auth(&self.api_key)
            .json(&self.build_body(session, false))
            .send()
            .await
            .map_err(ChatGptError::ReqwestError)?;

        if !response.status().is_success() {
            let text = response
                .text()
                .await
                .map_err(ChatGptError::ReqwestError)?;
            return Err(ChatGptError::StatusNotOk(text));
        }

        let reply = ChatGptResponse::new(response)
            .await
            .map_err(ChatGptError::ReqwestError)?;
        session.update_with_response(&reply);
        Ok(reply)
    }

    pub async fn ask_as_stream_with_extractor<F, StreamType>(
        &self,
        session: Session,
        data_extractor: F,
    ) -> Result<ResponseStream<F, StreamType>, (Session, ChatGptError)>
    where
        F: FnMut(&Session, ChatGptStreamChunk) -> StreamType,
    {
        let request = self
            .client
            .post(BASE_URL)
            .bearer_auth(&self.api_key)
            .json(&self.build_body(&session, true))
            .send()
            .await;

        let response = match request {
            Ok(response) => response,
            Err(e) => return Err((session, ChatGptError::ReqwestError(e))),
        };

        if !response.status().is_success() {
            let text = match response.text().await {
                Ok(response) => response,
                Err(e) => return Err((session, ChatGptError::ReqwestError(e))),
            };
            return Err((session, ChatGptError::StatusNotOk(text)));
        }

        Ok(ResponseStream::new(
            Box::new(response.bytes_stream()),
            session,
            data_extractor,
        ))
    }

    pub async fn ask_as_stream(
        &self,
        session: Session,
    ) -> Result<ChatGptResponseStream, (Session, ChatGptError)> {
        self.ask_as_stream_with_extractor(
            session,
            (|_, chunk| chunk) as fn(&Session, ChatGptStreamChunk) -> ChatGptStreamChunk,
        )
        .await
    }
}
