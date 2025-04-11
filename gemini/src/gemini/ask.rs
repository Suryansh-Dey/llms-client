use super::types::*;
use actix_web::dev::{Decompress, Payload};
use awc::{Client, ClientResponse};
use derive_new::new;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

const API_TIMEOUT: Duration = Duration::from_secs(30);
const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

#[derive(Serialize, Deserialize, new)]
#[allow(non_snake_case)]
pub struct Candidate {
    content: Chat,
    pub finishReason: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct GeminiResponse {
    candidates: Vec<Candidate>,
    pub usageMetadata: Value,
    pub modelVersion: String,
}
impl GeminiResponse {
    async fn new(
        mut response: ClientResponse<Decompress<Payload>>,
    ) -> Result<GeminiResponse, awc::error::JsonPayloadError> {
        response.json().await
    }
    fn from(string: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(string)
    }
    pub fn get_parts(&self) -> &Vec<Part> {
        self.candidates[0].content.parts()
    }
    pub fn parse_json(text: &str) -> Result<Value, serde_json::Error> {
        let unescaped_str = text.replace("\\\"", "\"").replace("\\n", "\n");
        serde_json::from_str::<Value>(&unescaped_str)
    }
    pub fn extract_text(parts: &[Part], seperator: &str) -> String {
        let mut concatinated_string = String::new();
        for part in parts {
            if let Part::text(text) = part {
                concatinated_string.push_str(text);
                concatinated_string.push_str(seperator);
            }
        }
        concatinated_string
    }
    ///`seperator` used to concatinate all text parts. TL;DR use "" as seperator.
    pub fn get_text(&self, seperator: &str) -> String {
        Self::extract_text(&self.get_parts(), seperator)
    }
}

pin_project_lite::pin_project! {
#[derive(new)]
    pub struct GeminiResponseStream<'a>{
        #[pin]
        response_stream:ClientResponse<Decompress<Payload>>,
        session: &'a mut Session
    }
}
impl<'a> Stream for GeminiResponseStream<'a> {
    type Item = Result<GeminiResponse, Box<dyn std::error::Error>>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();

        match this.response_stream.poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                let text = String::from_utf8_lossy(&bytes);
                if text == "]" {
                    Poll::Ready(None)
                } else {
                    let response = GeminiResponse::from(text[1..].trim())?;
                    this.session.update(&response)?;
                    Poll::Ready(Some(Ok(response)))
                }
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e.into()))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

pub struct Gemini<'a> {
    client: Client,
    api_key: String,
    model: String,
    sys_prompt: Option<SystemInstruction<'a>>,
    generation_config: Option<Value>,
    tools: Option<Vec<Value>>,
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
    pub fn set_code_execution_mode(&mut self) -> &Self {
        self.tools = Some(vec![json!({ "code_execution":{} })]);
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
            .send_json(&GeminiBody::new(
                self.sys_prompt.as_ref(),
                self.tools.as_deref(),
                &session.get_history().as_slice(),
                self.generation_config.as_ref(),
            ))
            .await?;
        let reply = GeminiResponse::new(response).await?;
        session.update(&reply)?;
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
            .send_json(&GeminiBody::new(
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
