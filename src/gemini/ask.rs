use super::types::*;
use actix_web::dev::{Decompress, Payload};
use awc::{Client, ClientResponse};
use futures::Stream;
use serde_json::{Value, json};
use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

const API_TIMEOUT: Duration = Duration::from_secs(30);
const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

pin_project_lite::pin_project! {
    pub struct GeminiResponseStream<'a>{
        #[pin]
        response_stream:ClientResponse<Decompress<Payload>>,
        reply_storage: &'a mut String
    }
}
impl<'a> GeminiResponseStream<'a> {
    fn new(
        response_stream: ClientResponse<Decompress<Payload>>,
        reply_storage: &'a mut String,
    ) -> Self {
        Self {
            response_stream,
            reply_storage,
        }
    }
    pub fn parse_json(text: &str) -> Result<Value, serde_json::Error> {
        let unescaped_str = text.replace("\\\"", "\"").replace("\\n", "\n");
        serde_json::from_str::<Value>(&unescaped_str)
    }
    fn get_response_text(response: &Value) -> Option<&str> {
        response["candidates"][0]["content"]["parts"][0]["text"].as_str()
    }
}
impl<'a> Stream for GeminiResponseStream<'a> {
    type Item = Result<String, Box<dyn std::error::Error>>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();

        match this.response_stream.poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                let text = String::from_utf8_lossy(&bytes);
                if text == "]" {
                    Poll::Ready(None)
                } else {
                    match serde_json::from_str(text[1..].trim()) {
                        Ok(ref response) => {
                            let reply = GeminiResponseStream::get_response_text(response)
                                .map(|response| {
                                    this.reply_storage.push_str(response);
                                    response.to_string()
                                })
                                .ok_or(
                                    format!("Gemini API sent invalid response:\n{response}").into(),
                                );
                            Poll::Ready(Some(reply))
                        }
                        Err(error) => Poll::Ready(Some(Err(error.into()))),
                    }
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
}
impl<'a> Gemini<'a> {
    pub fn new(api_key: String, model: String, sys_prompt: Option<SystemInstruction<'a>>) -> Self {
        Self {
            client: Client::builder().timeout(API_TIMEOUT).finish(),
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
    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }
    pub fn set_api_key(&mut self, api_key: String) {
        self.api_key = api_key;
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

    pub async fn ask<'b>(&self, session: &'b mut Session) -> Result<&'b str, Box<dyn std::error::Error>> {
        let req_url = format!(
            "{BASE_URL}/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let response: Value = self
            .client
            .post(req_url)
            .send_json(&GeminiBody::new(
                self.sys_prompt.as_ref(),
                &session.get_history().as_slice(),
                self.generation_config.as_ref(),
            ))
            .await?
            .json()
            .await?;
        let reply = GeminiResponseStream::get_response_text(&response)
            .ok_or::<Box<dyn std::error::Error>>(format!("Gemini API sent invalid response:\n{response}").into())?;
        session.update(reply);

        let destination_string = session
            .last_reply_mut()
            .ok_or::<Box<dyn std::error::Error>>(
                "Something went wrong in ask_as_stream, sorry".into(),
            )?;
        Ok(destination_string)
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
        session.update("");
        let destination_string = session
            .last_reply_mut()
            .ok_or::<Box<dyn std::error::Error>>(
                "Something went wrong in ask_as_stream, sorry".into(),
            )?;

        Ok(GeminiResponseStream::new(response, destination_string))
    }
}
