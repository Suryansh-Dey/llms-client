use super::completion::ChatCompletion;
use super::message::Message;
use super::params::{ReasoningEffort, ResponseFormat, Tool, ToolChoice};
use super::session::{OverflowHandler, Session};
use super::stream::ChatStream;
use super::wire::ChatRequest;
use crate::http::{
    authorize, deserialize_value, error_from_response, join_url, read_env, read_json,
};
use crate::{DEFAULT_BASE_URL, EnvVarError, Error};
use reqwest::{Client, RequestBuilder};
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::VecDeque;

/// Client of any OpenAI compatible chat completions API.
///
/// # Example
/// ```no_run
/// use open_client::chat::AI;
/// let mut deepseek = AI::new("API_KEY", "deepseek-chat", None);
/// deepseek.base_url("https://api.deepseek.com").temperature(0.7);
/// let mut ollama = AI::with_no_api_key("llama3.2", None);
/// ollama.base_url("http://localhost:11434/v1");
/// ```
/// Use `with_client()` with a client having default headers, if the server needs other headers.
///
/// Like `std::process::Command`, options are set by methods with their plain name, which take and
/// return `&mut Self`, so getters have a `get_` prefix. `Debug` masks the API key, so it can't
/// leak through logs or a panic message.
#[derive(Clone)]
pub struct AI {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    system_prompt: Option<String>,
    tools: Vec<Tool>,
    tool_choice: Option<ToolChoice>,
    params: Map<String, Value>,
    send_reasoning: bool,
}
impl std::fmt::Debug for AI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AI")
            .field("client", &self.client)
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("system_prompt", &self.system_prompt)
            .field("tools", &self.tools)
            .field("tool_choice", &self.tool_choice)
            .field("params", &self.params)
            .field("send_reasoning", &self.send_reasoning)
            .finish()
    }
}

impl AI {
    /// A client of the OpenAI API. Set `base_url()` for any other API.
    ///
    /// # Arguments
    /// * `api_key` - Your API key. Use `with_no_api_key()` for servers without one.
    /// * `model` - The model to use (e.g., "gpt-5-mini", "deepseek-chat").
    /// * `system_prompt` - Sent as the first message, with `system` role.
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        system_prompt: Option<String>,
    ) -> Self {
        Self::with_client(api_key, model, system_prompt, Client::default())
    }
    /// A client configured by environment variables, like the official OpenAI SDKs:
    /// * `OPENAI_API_KEY` - Your API key. Required.
    /// * `OPENAI_BASE_URL` - Base URL of any OpenAI compatible API. Default is `DEFAULT_BASE_URL`.
    ///
    /// # Errors
    /// If `OPENAI_API_KEY` is not set, or any of them isn't valid unicode.
    pub fn from_env(
        model: impl Into<String>,
        system_prompt: Option<String>,
    ) -> Result<Self, EnvVarError> {
        let (api_key, base_url) = read_env()?;
        let mut ai = Self::new(api_key, model, system_prompt);
        ai.base_url(base_url);
        Ok(ai)
    }
    /// A client sending no `Authorization` header, for servers without API keys, e.g. local
    /// servers like Ollama, vLLM or LM Studio. Set their URL with `base_url()`.
    pub fn with_no_api_key(model: impl Into<String>, system_prompt: Option<String>) -> Self {
        Self::new("", model, system_prompt)
    }
    /// Like `new()`, with a `reqwest::Client` configured e.g. with a timeout or default headers.
    pub fn with_client(
        api_key: impl Into<String>,
        model: impl Into<String>,
        system_prompt: Option<String>,
        client: Client,
    ) -> Self {
        Self {
            client,
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: model.into(),
            system_prompt,
            tools: Vec::new(),
            tool_choice: None,
            params: Map::new(),
            send_reasoning: false,
        }
    }

    /// Base URL of any OpenAI compatible API. Requests are sent to `{base_url}/chat/completions`.
    ///
    /// # Example
    /// - OpenAI: `https://api.openai.com/v1` (default)
    /// - DeepSeek: `https://api.deepseek.com`
    /// - Ollama: `http://localhost:11434/v1`
    /// - OpenRouter: `https://openrouter.ai/api/v1`
    /// - Gemini: `https://generativelanguage.googleapis.com/v1beta/openai`
    pub fn base_url(&mut self, base_url: impl Into<String>) -> &mut Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }
    pub fn api_key(&mut self, api_key: impl Into<String>) -> &mut Self {
        self.api_key = api_key.into();
        self
    }
    pub fn model(&mut self, model: impl Into<String>) -> &mut Self {
        self.model = model.into();
        self
    }
    /// # Warning
    /// Changing the system prompt in the middle of a conversation can confuse the model.
    pub fn system_prompt(&mut self, system_prompt: Option<String>) -> &mut Self {
        self.system_prompt = system_prompt;
        self
    }
    /// Adds a tool available to the model. A `Tool::Function` replaces the one with the same name.
    pub fn tool(&mut self, tool: Tool) -> &mut Self {
        let position = tool
            .name()
            .and_then(|name| self.tools.iter().position(|t| t.name() == Some(name)));
        match position {
            Some(position) => self.tools[position] = tool,
            None => self.tools.push(tool),
        }
        self
    }
    /// Sets the tools available to the model, replacing all the previous ones.
    pub fn tools(&mut self, tools: impl IntoIterator<Item = Tool>) -> &mut Self {
        self.tools = tools.into_iter().collect();
        self
    }
    /// `None` leaves it to the server, which is `ToolChoice::Auto` if there are tools.
    pub fn tool_choice(&mut self, tool_choice: impl Into<Option<ToolChoice>>) -> &mut Self {
        self.tool_choice = tool_choice.into();
        self
    }
    /// If true, reasoning of every reply in the history is sent back as `reasoning_content`.
    /// Required by DeepSeek after tool calls in thinking mode, and by models that keep their
    /// reasoning across turns, like GLM with preserved thinking and Kimi. Default is false, as the
    /// field isn't in the OpenAI specification.
    pub fn send_reasoning(&mut self, send_reasoning: bool) -> &mut Self {
        self.send_reasoning = send_reasoning;
        self
    }

    /// Sets any parameter of the request body, e.g. a provider specific one. `value` can be
    /// anything `Serialize`, including nested objects like `json!({"type": "enabled"})`. A `null`
    /// value (e.g. `None`) removes the parameter.
    ///
    /// `model`, `messages`, `tools`, `tool_choice` and `stream` are ignored as `AI` sets them.
    /// See [OpenAI docs](https://platform.openai.com/docs/api-reference/chat/create) for
    /// parameters.
    ///
    /// # Panics
    /// If `value` can't be serialized to JSON, e.g. a map with non-string keys.
    pub fn param(&mut self, key: impl Into<String>, value: impl Serialize) -> &mut Self {
        let key = key.into();
        let value = serde_json::to_value(value)
            .unwrap_or_else(|e| panic!("parameter `{key}` can't be serialized to JSON: {e}"));
        if value.is_null() {
            self.params.remove(&key);
        } else {
            self.params.insert(key, value);
        }
        self
    }
    /// Parameters added to the top level of the request body, for editing nested values in place.
    pub fn params_mut(&mut self) -> &mut Map<String, Value> {
        &mut self.params
    }
    /// Between 0 and 2. Higher values make the reply more random.
    pub fn temperature(&mut self, temperature: f64) -> &mut Self {
        self.param("temperature", temperature)
    }
    /// Nucleus sampling: only tokens in the top `top_p` probability mass are considered.
    pub fn top_p(&mut self, top_p: f64) -> &mut Self {
        self.param("top_p", top_p)
    }
    /// Upper bound of tokens generated, including reasoning tokens.
    pub fn max_completion_tokens(&mut self, max_completion_tokens: u64) -> &mut Self {
        self.param("max_completion_tokens", max_completion_tokens)
    }
    /// For deterministic sampling, where supported.
    pub fn seed(&mut self, seed: i64) -> &mut Self {
        self.param("seed", seed)
    }
    /// Sequences where the model stops generating.
    pub fn stop<S: Into<String>>(&mut self, stop: impl IntoIterator<Item = S>) -> &mut Self {
        self.param(
            "stop",
            stop.into_iter().map(Into::into).collect::<Vec<String>>(),
        )
    }
    /// Between -2 and 2. Positive values make new topics more likely.
    pub fn presence_penalty(&mut self, presence_penalty: f64) -> &mut Self {
        self.param("presence_penalty", presence_penalty)
    }
    /// Between -2 and 2. Positive values make repetition less likely.
    pub fn frequency_penalty(&mut self, frequency_penalty: f64) -> &mut Self {
        self.param("frequency_penalty", frequency_penalty)
    }
    /// Whether the model can call many tools in one reply.
    pub fn parallel_tool_calls(&mut self, parallel_tool_calls: bool) -> &mut Self {
        self.param("parallel_tool_calls", parallel_tool_calls)
    }
    /// How much a reasoning model thinks. Supported values depend on the model.
    pub fn reasoning_effort(&mut self, reasoning_effort: ReasoningEffort) -> &mut Self {
        self.param("reasoning_effort", reasoning_effort)
    }
    /// Format of the reply, e.g. `ResponseFormat::json_schema(Type::openai_schema())` for
    /// structured output.
    pub fn response_format(&mut self, response_format: ResponseFormat) -> &mut Self {
        self.param("response_format", response_format)
    }
    /// If true, the token usage of a streamed reply is sent as `ChatCompletionChunk::usage`. Sets
    /// `stream_options`, which is only sent when streaming.
    pub fn include_usage(&mut self, include_usage: bool) -> &mut Self {
        let options = self
            .params
            .entry("stream_options")
            .or_insert_with(|| json!({}));
        if !options.is_object() {
            *options = json!({});
        }
        options["include_usage"] = include_usage.into();
        self
    }

    pub fn get_base_url(&self) -> &str {
        &self.base_url
    }
    pub fn get_model(&self) -> &str {
        &self.model
    }
    pub fn get_system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }
    pub fn get_tools(&self) -> &[Tool] {
        &self.tools
    }
    pub fn get_tool_choice(&self) -> Option<&ToolChoice> {
        self.tool_choice.as_ref()
    }
    pub fn get_send_reasoning(&self) -> bool {
        self.send_reasoning
    }
    pub fn get_params(&self) -> &Map<String, Value> {
        &self.params
    }

    fn request(&self, messages: &VecDeque<Message>, stream: bool) -> RequestBuilder {
        let body = ChatRequest {
            model: &self.model,
            system_prompt: self.system_prompt.as_deref(),
            messages,
            tools: &self.tools,
            tool_choice: self.tool_choice.as_ref(),
            params: &self.params,
            stream,
            send_reasoning: self.send_reasoning,
        };
        let request = self
            .client
            .post(join_url(&self.base_url, "chat/completions"))
            .json(&body);
        authorize(request, &self.api_key)
    }
    fn check_turn<H>(session: &Session<H>) -> Result<(), Error> {
        match session.last() {
            Some(Message::Assistant(_)) => Err(Error::NothingToRespond),
            _ => Ok(()),
        }
    }

    /// Sends the session and waits for the whole reply. It is added to the session, then the
    /// session's `OverflowHandler` runs.
    ///
    /// # Errors
    /// `Error::NothingToRespond` if the last message of the session is from the assistant.
    pub async fn ask<H: OverflowHandler>(
        &self,
        session: &mut Session<H>,
    ) -> Result<ChatCompletion, Error> {
        Self::check_turn(session)?;
        let response = self.request(session.history(), false).send().await?;
        let completion: ChatCompletion = deserialize_value(&read_json(response).await?)?;
        session.add_completion(&completion).await;
        Ok(completion)
    }
    /// Sends the session and streams the reply. Each item has the newly received part of it, and
    /// `ChatStream::message()` has the reply so far. When the stream ends, the whole reply is added
    /// to the session and the session's `OverflowHandler` runs. If the stream is dropped before
    /// that, the reply so far is added without running the handler.
    ///
    /// # Errors
    /// `Error::NothingToRespond` if the last message of the session is from the assistant.
    ///
    /// # Example
    /// ```no_run
    /// use open_client::futures::StreamExt;
    /// # async fn run(ai: open_client::chat::AI) -> Result<(), open_client::Error> {
    /// let mut session = open_client::chat::session::Session::default();
    /// let mut stream = ai.stream(session.ask("Hi")).await?;
    /// while let Some(chunk) = stream.next().await {
    ///     print!("{}", chunk?.text());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stream<'a, H: OverflowHandler>(
        &self,
        session: &'a mut Session<H>,
    ) -> Result<ChatStream<'a, H>, Error> {
        Self::check_turn(session)?;
        let response = self.request(session.history(), true).send().await?;
        if !response.status().is_success() {
            return Err(error_from_response(response).await);
        }
        Ok(ChatStream::new(Box::new(response.bytes_stream()), session))
    }
}
