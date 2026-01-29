use super::error::GeminiResponseError;
use super::types::request::*;
use super::types::response::*;
use super::types::sessions::Session;
#[cfg(feature = "reqwest")]
use reqwest::Client;
use serde_json::{Value, json};
use std::time::Duration;

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// The main client for interacting with the Gemini API.
///
/// Use `Gemini::new` or `Gemini::new_with_timeout` to create an instance.
/// You can configure various aspects of the request like model, system instructions,
/// generation config, safety settings, and tools using the provided builder-like methods.
#[derive(Clone, Default, Debug)]
pub struct Gemini {
    #[cfg(feature = "reqwest")]
    client: Client,
    api_key: String,
    model: String,
    sys_prompt: Option<SystemInstruction>,
    generation_config: Option<Value>,
    safety_settings: Option<Vec<SafetySetting>>,
    tools: Option<Vec<Tool>>,
    tool_config: Option<ToolConfig>,
}

impl Gemini {
    /// Creates a new `Gemini` client.
    ///
    /// # Arguments
    /// * `api_key` - Your Gemini API key. Get one from [Google AI studio](https://aistudio.google.com/app/apikey).
    /// * `model` - The model variation to use (e.g., "gemini-2.5-flash"). See [model variations](https://ai.google.dev/gemini-api/docs/models#model-variations).
    /// * `sys_prompt` - Optional system instructions. See [system instructions](https://ai.google.dev/gemini-api/docs/text-generation#image-input).
    #[cfg(feature = "reqwest")]
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
            safety_settings: None,
            tools: None,
            tool_config: None,
        }
    }
    /// Creates a new `Gemini` client with a custom API timeout.
    ///
    /// # Arguments
    /// * `api_key` - Your Gemini API key.
    /// * `model` - The model variation to use.
    /// * `sys_prompt` - Optional system instructions.
    /// * `api_timeout` - Custom duration for request timeouts.
    #[cfg(feature = "reqwest")]
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
            safety_settings: None,
            tools: None,
            tool_config: None,
        }
    }
    /// Returns a mutable reference to the generation configuration.
    /// If not already set, initializes it to an empty object.
    ///
    /// See [Gemini docs](https://ai.google.dev/api/generate-content#generationconfig) for schema details.
    pub fn set_generation_config(&mut self) -> &mut Value {
        if let None = self.generation_config {
            self.generation_config = Some(json!({}));
        }
        self.generation_config.as_mut().unwrap()
    }
    pub fn set_tool_config(mut self, config: ToolConfig) -> Self {
        self.tool_config = Some(config);
        self
    }
    pub fn set_thinking_config(mut self, config: ThinkingConfig) -> Self {
        if let Value::Object(map) = self.set_generation_config() {
            if let Ok(thinking_value) = serde_json::to_value(config) {
                map.insert("thinking_config".to_string(), thinking_value);
            }
        }
        self
    }
    pub fn set_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
    pub fn set_sys_prompt(mut self, sys_prompt: Option<SystemInstruction>) -> Self {
        self.sys_prompt = sys_prompt;
        self
    }
    pub fn set_safety_settings(mut self, settings: Option<Vec<SafetySetting>>) -> Self {
        self.safety_settings = settings;
        self
    }
    pub fn set_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }
    /// Sets the response format to JSON mode with a specific schema.
    ///
    /// To use a Rust struct as a schema, decorate it with `#[gemini_schema]` and pass
    /// `StructName::gemini_schema()`.
    ///
    /// # Arguments
    /// * `schema` - The JSON schema for the response. See [Gemini Schema docs](https://ai.google.dev/api/caching#Schema).
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
    /// Sets the tools (functions) available to the model.
    pub fn set_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }
    /// Removes all tools.
    pub fn unset_tools(mut self) -> Self {
        self.tools = None;
        self
    }

    /// Sends a prompt to the model and waits for the full response.
    ///
    /// Updates the `session` history with the model's reply.
    ///
    /// # Errors
    /// Returns `GeminiResponseError::NothingToRespond` if the last message in history is from the model.
    #[cfg(feature = "reqwest")]
    pub async fn ask(&self, session: &mut Session) -> Result<GeminiResponse, GeminiResponseError> {
        if session
            .get_last_chat()
            .is_some_and(|chat| *chat.role() == Role::Model)
        {
            return Err(GeminiResponseError::NothingToRespond);
        }
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
                self.safety_settings.as_deref(),
                self.tool_config.as_ref(),
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
    #[cfg(feature = "reqwest")]
    pub async fn ask_as_stream_with_extractor<F, StreamType>(
        &self,
        session: Session,
        data_extractor: F,
    ) -> Result<ResponseStream<F, StreamType>, (Session, GeminiResponseError)>
    where
        F: FnMut(&Session, GeminiResponse) -> StreamType,
    {
        if session
            .get_last_chat()
            .is_some_and(|chat| *chat.role() == Role::Model)
        {
            return Err((session, GeminiResponseError::NothingToRespond));
        }
        let req_url = format!(
            "{BASE_URL}/{}:streamGenerateContent?alt=sse&key={}",
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
                self.safety_settings.as_deref(),
                self.tool_config.as_ref(),
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
    /// Sends a prompt to the model and returns a stream of responses.
    ///
    /// # Warning
    /// You must exhaust the response stream to ensure the `session` history is correctly updated.
    ///
    /// # Example
    /// ```no_run
    /// use futures::StreamExt;
    /// # async fn run(gemini: gemini_client_api::gemini::ask::Gemini, session: gemini_client_api::gemini::types::sessions::Session) {
    /// let mut response_stream = gemini.ask_as_stream(session).await.unwrap();
    ///
    /// while let Some(response) = response_stream.next().await {
    ///     if let Ok(response) = response {
    ///         println!("{}", response.get_text(""));
    ///     }
    /// }
    /// # }
    /// ```
    #[cfg(feature = "reqwest")]
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
