use super::error::{ApiError, EnvVarError, OpenAIError, OpenAIResponseError};
use super::types::request::*;
use super::types::response::*;
use super::types::sessions::{OverflowHandler, Session};
use derive_getters::{Dissolve, Getters};
use reqwest::{Client, RequestBuilder, Response};
use serde::Deserialize;
use serde_json::{Value, json};

/// Base URL of OpenAI API. Use `set_base_url()` for any other OpenAI compatible API.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Adds `Authorization: Bearer <api_key>` header, unless `api_key` is empty (e.g. for local servers).
pub(crate) fn authorize(request: RequestBuilder, api_key: &str) -> RequestBuilder {
    if api_key.is_empty() {
        request
    } else {
        request.bearer_auth(api_key)
    }
}
/// Reads `OPENAI_API_KEY` (required) and `OPENAI_BASE_URL` (optional, unset or empty means
/// `DEFAULT_BASE_URL`).
pub(crate) fn read_env() -> Result<(String, String), EnvVarError> {
    let var = |name| std::env::var(name).map_err(|source| EnvVarError { name, source });
    let api_key = var("OPENAI_API_KEY")?;
    let base_url = match var("OPENAI_BASE_URL") {
        Ok(base_url) if !base_url.is_empty() => base_url,
        Ok(_)
        | Err(EnvVarError {
            source: std::env::VarError::NotPresent,
            ..
        }) => DEFAULT_BASE_URL.to_string(),
        Err(e) => return Err(e),
    };
    Ok((api_key, base_url))
}
pub(crate) fn join_url(base_url: &str, path: &str) -> String {
    format!("{}/{path}", base_url.trim_end_matches('/'))
}
pub(crate) async fn error_from_response(response: Response) -> OpenAIResponseError {
    let status = response.status();
    match response.text().await {
        Ok(body) => OpenAIResponseError::StatusNotOk(OpenAIError {
            status,
            error: ApiError::from_body(&body),
        }),
        Err(e) => e.into(),
    }
}
/// Reads the JSON body of a response, treating error status or an error object as an error.
pub(crate) async fn read_json(response: Response) -> Result<Value, OpenAIResponseError> {
    if !response.status().is_success() {
        return Err(error_from_response(response).await);
    }
    let status = response.status();
    let body = response.text().await?;
    let value: Value = serde_json::from_str(&body).map_err(|e| {
        OpenAIResponseError::InvalidResponseFormat(format!("JSON parsing error [{e}]: {body}"))
    })?;
    // Some servers report errors with status 200
    if value.get("error").is_some_and(|e| !e.is_null()) && value.get("choices").is_none() {
        return Err(OpenAIResponseError::StatusNotOk(OpenAIError {
            status,
            error: ApiError::from_value(&value),
        }));
    }
    Ok(value)
}
pub(crate) fn deserialize_value<'a, T: Deserialize<'a>>(
    value: &'a Value,
) -> Result<T, OpenAIResponseError> {
    T::deserialize(value).map_err(|e| {
        OpenAIResponseError::InvalidResponseFormat(format!(
            "Unexpected response format [{e}]: {value}"
        ))
    })
}

/// The main client for interacting with any OpenAI compatible chat completions API.
///
/// # Example
/// ```no_run
/// use open_client::openai::ask::OpenAI;
/// let deepseek = OpenAI::new("API_KEY", "deepseek-chat", None)
///     .set_base_url("https://api.deepseek.com");
/// let ollama = OpenAI::with_no_api_key("llama3.2", None)
///     .set_base_url("http://localhost:11434/v1");
/// ```
/// Use `new_with_client()` with a client having default headers, if the server needs other headers.
///
/// `Debug` masks `api_key`, so it can't leak through logs or a panic message.
#[derive(Clone, Dissolve, Getters)]
pub struct OpenAI {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    sys_prompt: Option<SystemInstruction>,
    generation_config: Option<Value>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    send_reasoning: bool,
}
impl std::fmt::Debug for OpenAI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAI")
            .field("client", &self.client)
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("sys_prompt", &self.sys_prompt)
            .field("generation_config", &self.generation_config)
            .field("tools", &self.tools)
            .field("tool_choice", &self.tool_choice)
            .field("send_reasoning", &self.send_reasoning)
            .finish()
    }
}
impl Default for OpenAI {
    fn default() -> Self {
        Self::new_with_client("", "", None, Client::default())
    }
}

impl OpenAI {
    /// Creates a new `OpenAI` client for OpenAI API. Use `set_base_url()` for any other API.
    ///
    /// # Arguments
    /// * `api_key` - Your API key. Use `with_no_api_key()` for servers without one.
    /// * `model` - The model to use (e.g., "gpt-5-mini", "deepseek-chat").
    /// * `sys_prompt` - Optional system instructions, sent as the first message with `system` role.
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        sys_prompt: Option<SystemInstruction>,
    ) -> Self {
        Self::new_with_client(api_key, model, sys_prompt, Client::default())
    }
    /// Creates a new `OpenAI` client configured by environment variables, like the official
    /// OpenAI SDKs:
    /// * `OPENAI_API_KEY` - Your API key. Required.
    /// * `OPENAI_BASE_URL` - Base URL of any OpenAI compatible API. Default is `DEFAULT_BASE_URL`.
    ///
    /// # Errors
    /// If `OPENAI_API_KEY` is not set, or any of them isn't valid unicode.
    pub fn from_env(
        model: impl Into<String>,
        sys_prompt: Option<SystemInstruction>,
    ) -> Result<Self, EnvVarError> {
        let (api_key, base_url) = read_env()?;
        Ok(Self::new(api_key, model, sys_prompt).set_base_url(base_url))
    }
    /// Creates a new `OpenAI` client sending no `Authorization` header, for servers without API
    /// keys, e.g. local servers like Ollama, vLLM or LM Studio. Set their URL with `set_base_url()`.
    pub fn with_no_api_key(
        model: impl Into<String>,
        sys_prompt: Option<SystemInstruction>,
    ) -> Self {
        Self::new("", model, sys_prompt)
    }
    /// Creates a new `OpenAI` client with a custom API reqwest::Client.
    ///
    /// # Arguments
    /// * `api_key` - Your API key.
    /// * `model` - The model to use.
    /// * `sys_prompt` - Optional system instructions.
    /// * `client` - reqwest::Client to request the API, e.g. with a timeout or default headers.
    pub fn new_with_client(
        api_key: impl Into<String>,
        model: impl Into<String>,
        sys_prompt: Option<SystemInstruction>,
        client: Client,
    ) -> Self {
        Self {
            client,
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: model.into(),
            sys_prompt,
            generation_config: None,
            tools: None,
            tool_choice: None,
            send_reasoning: false,
        }
    }
    /// Sets the base URL of any OpenAI compatible API. Requests are sent to `{base_url}/chat/completions`.
    ///
    /// # Example
    /// - OpenAI: `https://api.openai.com/v1` (default)
    /// - DeepSeek: `https://api.deepseek.com`
    /// - Ollama: `http://localhost:11434/v1`
    /// - OpenRouter: `https://openrouter.ai/api/v1`
    /// - Gemini: `https://generativelanguage.googleapis.com/v1beta/openai`
    pub fn set_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }
    /// Returns a mutable reference to the generation configuration.
    /// If not already set, initializes it to an empty object.
    ///
    /// Its entries are added to the top level of the request body, e.g. `temperature`,
    /// `max_completion_tokens`, `seed` or any provider specific parameter. `model`, `messages`,
    /// `tools`, `tool_choice` and `stream` are ignored as they are managed by `OpenAI`.
    ///
    /// See [OpenAI docs](https://platform.openai.com/docs/api-reference/chat/create) for parameters.
    pub fn set_generation_config(&mut self) -> &mut Value {
        if self.generation_config.is_none() {
            self.generation_config = Some(json!({}));
        }
        self.generation_config.as_mut().unwrap()
    }
    pub fn set_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }
    /// Sets `reasoning_effort` in generation config. Supported values depend on the model.
    pub fn set_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.set_generation_config()["reasoning_effort"] = json!(effort);
        self
    }
    /// If true, thoughts of replies in the ongoing tool calling loop (after the last user prompt)
    /// are sent back as `reasoning_content`. Required by DeepSeek in thinking mode with tool
    /// calls. Default is false, as servers like OpenAI reject unknown fields.
    pub fn set_send_reasoning(mut self, send_reasoning: bool) -> Self {
        self.send_reasoning = send_reasoning;
        self
    }
    pub fn set_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
    /// # Warning
    /// Changing sys_prompt in middle of a conversation can confuse the model.
    pub fn set_sys_prompt(mut self, sys_prompt: Option<SystemInstruction>) -> Self {
        self.sys_prompt = sys_prompt;
        self
    }
    pub fn set_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }
    /// Sets the response format to JSON mode with a specific schema, as
    /// `{"type": "json_schema", "json_schema": {"name": "response", "strict": true, "schema": schema}}`
    ///
    /// To use a Rust struct as a schema, decorate it with `#[openai_schema]` and pass
    /// `StructName::openai_schema()`.
    /// Set `response_format` in `set_generation_config()` for other formats, e.g. `json_object`.
    ///
    /// # Arguments
    /// * `schema` - The JSON schema for the response. In strict mode, every object needs
    ///   `"additionalProperties": false` and all its properties in `required`, which is ensured by
    ///   `#[openai_schema]`. See [Structured outputs](https://platform.openai.com/docs/guides/structured-outputs#supported-schemas).
    pub fn set_json_mode(mut self, schema: Value) -> Self {
        self.set_generation_config()["response_format"] = json!({
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": true,
                "schema": schema
            }
        });
        self
    }
    pub fn remove_json_mode(mut self) -> Self {
        if let Some(Value::Object(generation_config)) = &mut self.generation_config {
            generation_config.remove("response_format");
        }
        self
    }
    /// Adds a tool available to the model. A `Tool::Function` replaces the one with the same name.
    pub fn set_tool(mut self, tool: Tool) -> Self {
        let tools = self.tools.get_or_insert_with(Vec::new);
        match tool
            .name()
            .and_then(|name| tools.iter().position(|t| t.name() == Some(name)))
        {
            Some(position) => tools[position] = tool,
            None => tools.push(tool),
        }
        self
    }
    /// Sets the tools (functions) available to the model, replacing all the previous ones.
    pub fn set_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }
    /// Removes all tools.
    pub fn remove_tools(mut self) -> Self {
        self.tools = None;
        self
    }

    fn request(&self, history: &[&Chat], stream: bool) -> RequestBuilder {
        let body = OpenAIRequestBody {
            model: &self.model,
            sys_prompt: self.sys_prompt.as_ref(),
            messages: history,
            tools: self.tools.as_deref(),
            tool_choice: self.tool_choice.as_ref(),
            generation_config: self.generation_config.as_ref(),
            stream,
            send_reasoning: self.send_reasoning,
        };
        let request = self
            .client
            .post(join_url(&self.base_url, "chat/completions"))
            .json(&body);
        authorize(request, &self.api_key)
    }

    /// Sends a prompt to the model and waits for the full response.
    ///
    /// Updates the `session` history with the model's reply.
    ///
    /// # Errors
    /// Returns `OpenAIResponseError::NothingToRespond` if the last message in history is from the model.
    pub async fn ask<H: OverflowHandler>(
        &self,
        session: &mut Session<H>,
    ) -> Result<OpenAIResponse, OpenAIResponseError> {
        if session
            .get_last_chat()
            .is_some_and(|chat| *chat.role() == Role::Assistant)
        {
            return Err(OpenAIResponseError::NothingToRespond);
        }
        let response = self.request(&session.get_history(), false).send().await?;
        let reply: OpenAIResponse = deserialize_value(&read_json(response).await?)?;
        session.update(&reply);
        Ok(reply)
    }
    /// Like `ask_as_stream()`, but each item is what `data_extractor` returns. It is called with
    /// the session, already updated with the chunk, and the chunk.
    ///
    /// # Warning
    /// You must exhaust the response stream to ensure the `session` history is correctly updated.
    ///
    /// # Errors
    /// The session is returned along with the error, as it was moved into this function.
    ///
    /// # Example
    /// ```no_run
    /// use open_client::futures::StreamExt;
    /// # async fn run(ai: open_client::openai::ask::OpenAI, session: open_client::openai::types::sessions::Session) {
    /// // Each item is the whole reply received so far.
    /// let mut response_stream = ai
    ///     .ask_as_stream_with_extractor(session, |session, _chunk| {
    ///         session.get_last_chat().unwrap().get_text_no_think("")
    ///     })
    ///     .await
    ///     .unwrap();
    /// while let Some(reply) = response_stream.next().await {
    ///     println!("{}", reply.unwrap());
    /// }
    /// # }
    /// ```
    #[allow(
        clippy::result_large_err,
        reason = "the session is handed back on error, as it is moved into the stream"
    )]
    pub async fn ask_as_stream_with_extractor<F, StreamType, H: OverflowHandler>(
        &self,
        session: Session<H>,
        data_extractor: F,
    ) -> Result<ResponseStream<F, StreamType, H>, (Session<H>, OpenAIResponseError)>
    where
        F: FnMut(&Session<H>, OpenAIResponse) -> StreamType,
    {
        if session
            .get_last_chat()
            .is_some_and(|chat| *chat.role() == Role::Assistant)
        {
            return Err((session, OpenAIResponseError::NothingToRespond));
        }
        let request = self.request(&session.get_history(), true).send().await;
        let response = match request {
            Ok(response) => response,
            Err(e) => return Err((session, e.into())),
        };

        if !response.status().is_success() {
            return Err((session, error_from_response(response).await));
        }

        Ok(ResponseStream::new(
            Box::new(response.bytes_stream()),
            session,
            data_extractor,
        ))
    }
    /// Sends a prompt to the model and returns a stream of responses.
    ///
    /// Each item has only the newly received part of the reply. Use `session.get_last_chat()` (or
    /// `response_stream.get_session()`) for the reply received so far, e.g. for complete function
    /// calls, as their arguments arrive in pieces.
    ///
    /// # Warning
    /// You must exhaust the response stream to ensure the `session` history is correctly updated.
    ///
    /// # Example
    /// ```no_run
    /// use open_client::futures::StreamExt;
    /// # async fn run(ai: open_client::openai::ask::OpenAI, session: open_client::openai::types::sessions::Session) {
    /// let mut response_stream = ai.ask_as_stream(session).await.unwrap();
    ///
    /// while let Some(response) = response_stream.next().await {
    ///     if let Ok(response) = response {
    ///         println!("{}", response.get_chat().get_text_no_think("\n"));
    ///     }
    /// }
    /// # }
    /// ```
    #[allow(
        clippy::result_large_err,
        reason = "the session is handed back on error, as it is moved into the stream"
    )]
    pub async fn ask_as_stream<H: OverflowHandler>(
        &self,
        session: Session<H>,
    ) -> Result<OpenAIResponseStream<H>, (Session<H>, OpenAIResponseError)> {
        self.ask_as_stream_with_extractor(
            session,
            (|_, response| response) as fn(&Session<H>, OpenAIResponse) -> OpenAIResponse,
        )
        .await
    }
}
