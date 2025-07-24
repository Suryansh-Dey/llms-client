use super::error::GeminiResponseError;
use super::types::request::*;
use super::types::response::*;
use super::types::sessions::Session;
use reqwest::Client;
use serde_json::{Value, json};
use std::time::Duration;

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

#[derive(Clone, Default, Debug)]
pub struct Gemini {
    client: Client,
    api_key: String,
    model: String,
    sys_prompt: Option<SystemInstruction>,
    generation_config: Option<Value>,
    tools: Option<Vec<Tool>>,
}
impl Gemini {
    /// # Arguments
    /// `api_key` get one from [Google AI studio](https://aistudio.google.com/app/apikey)
    /// `model` should be of those mentioned [here](https://ai.google.dev/gemini-api/docs/models#model-variations) in bold black color
    /// `sys_prompt` should follow [gemini doc](https://ai.google.dev/gemini-api/docs/text-generation#image-input)
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        sys_prompt: Option<SystemInstruction>,
    ) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .unwrap(),
            api_key: api_key.into(),
            model: model.into(),
            sys_prompt,
            generation_config: None,
            tools: None,
        }
    }
    /// `sys_prompt` should follow [gemini doc](https://ai.google.dev/gemini-api/docs/text-generation#image-input)
    pub fn new_with_timeout(
        api_key: impl Into<String>,
        model: impl Into<String>,
        sys_prompt: Option<SystemInstruction>,
        api_timeout: Duration,
    ) -> Self {
        Self {
            client: Client::builder().timeout(api_timeout).build().unwrap(),
            api_key: api_key.into(),
            model: model.into(),
            sys_prompt,
            generation_config: None,
            tools: None,
        }
    }
    /// The generation config Schema should follow [Gemini docs](https://ai.google.dev/api/generate-content#generationconfig)
    pub fn set_generation_config(&mut self) -> &mut Value {
        if let None = self.generation_config {
            self.generation_config = Some(json!({}));
        }
        self.generation_config.as_mut().unwrap()
    }
    pub fn set_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
    pub fn set_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }
    /// `schema` should follow [Schema of gemini](https://ai.google.dev/api/caching#Schema)
    /// To verify your schema visit [here](https://aistudio.google.com/prompts/new_chat):
    /// - Under tools, toggle on Structured output
    /// - Click Edit
    /// - Here you can create schema with `Visual Editor` or `Code Editor` with error detection
    pub fn set_json_mode(mut self, schema: Value) -> Self {
        let config = self.set_generation_config();
        config["response_mime_type"] = "application/json".into();
        config["response_schema"] = schema.into();
        self
    }
    pub fn unset_json_mode(mut self) -> Self {
        if let Some(ref mut generation_config) = self.generation_config {
            generation_config["response_schema"] = None::<Value>.into();
            generation_config["response_mime_type"] = None::<Value>.into();
        }
        self
    }
    pub fn set_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }
    pub fn unset_tools(mut self) -> Self {
        self.tools = None;
        self
    }

    pub async fn ask(&self, session: &mut Session) -> Result<GeminiResponse, GeminiResponseError> {
        let req_url = format!(
            "{BASE_URL}/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let response = self
            .client
            .post(req_url)
            .json(&GeminiRequestBody::new(
                self.sys_prompt.as_ref(),
                self.tools.as_deref(),
                &session.get_history().as_slice(),
                self.generation_config.as_ref(),
            ))
            .send()
            .await
            .map_err(|e| GeminiResponseError::ReqwestError(e))?;

        if !response.status().is_success() {
            let text = response
                .text()
                .await
                .map_err(|e| GeminiResponseError::ReqwestError(e))?;
            return Err(GeminiResponseError::StatusNotOk(text));
        }

        let reply = GeminiResponse::new(response)
            .await
            .map_err(|e| GeminiResponseError::ReqwestError(e))?;
        session.update(&reply);
        Ok(reply)
    }
    /// # Warning
    /// You must read the response stream to get reply stored context in `session`.
    /// `data_extractor` is used to extract data that you get as a stream of futures.
    /// # Example
    ///```ignore
    ///use futures::StreamExt
    ///let mut response_stream = gemini.ask_as_stream_with_extractor(session,
    ///|session, _gemini_response| session.get_last_message_text("").unwrap())
    ///.await.unwrap(); // Use _gemini_response.get_text("") to just get the text received in every chunk
    ///
    ///while let Some(response) = response_stream.next().await {
    ///    if let Ok(response) = response {
    ///        println!("{}", response);
    ///    }
    ///}
    ///```
    pub async fn ask_as_stream_with_extractor<F, StreamType>(
        &self,
        session: Session,
        data_extractor: F,
    ) -> Result<ResponseStream<F, StreamType>, (Session, GeminiResponseError)>
    where
        F: FnMut(&Session, GeminiResponse) -> StreamType,
    {
        let req_url = format!(
            "{BASE_URL}/{}:streamGenerateContent?key={}",
            self.model, self.api_key
        );

        let request = self
            .client
            .post(req_url)
            .json(&GeminiRequestBody::new(
                self.sys_prompt.as_ref(),
                self.tools.as_deref(),
                session.get_history().as_slice(),
                self.generation_config.as_ref(),
            ))
            .send()
            .await;
        let response = match request {
            Ok(response) => response,
            Err(e) => return Err((session, GeminiResponseError::ReqwestError(e))),
        };

        if !response.status().is_success() {
            let text = match response.text().await {
                Ok(response) => response,
                Err(e) => return Err((session, GeminiResponseError::ReqwestError(e))),
            };
            return Err((session, GeminiResponseError::StatusNotOk(text.into())));
        }

        Ok(ResponseStream::new(
            Box::new(response.bytes_stream()),
            session,
            data_extractor,
        ))
    }
    /// # Warning
    /// You must read the response stream to get reply stored context in `session`.
    /// # Example
    ///```ignore
    ///use futures::StreamExt
    ///let mut response_stream = gemini.ask_as_stream(session).await.unwrap();
    ///
    ///while let Some(response) = response_stream.next().await {
    ///    if let Ok(response) = response {
    ///        println!("{}", response.get_text(""));
    ///    }
    ///}
    ///```
    pub async fn ask_as_stream(
        &self,
        session: Session,
    ) -> Result<GeminiResponseStream, (Session, GeminiResponseError)> {
        self.ask_as_stream_with_extractor(
            session,
            (|_, gemini_response| gemini_response)
                as fn(&Session, GeminiResponse) -> GeminiResponse,
        )
        .await
    }
}
