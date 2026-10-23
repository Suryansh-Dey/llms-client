# Open Client

A fast, flexible, and feature-rich Rust library for **any OpenAI compatible API**: OpenAI, DeepSeek, Ollama, vLLM, OpenRouter, Groq, LM Studio, Gemini and more. Featuring macros for automatic JSON schema generation and function calling.

Types follow the [Chat Completions](https://platform.openai.com/docs/api-reference/chat) specification and are sent as you build them, so what you see in the OpenAI docs is what you write.

## Features

- **Any OpenAI compatible endpoint**: `base_url()` and you're done. Responses and errors are parsed leniently, so servers that only partially follow the spec still work.
- **Automatic Context Management**: Simple `Session` struct to handle conversation history, kept in check after each reply by a `SlidingWindow` of messages or your own async `OverflowHandler`, e.g. a token budget using the `usage` of the response, or a summary of what is forgotten.
- **Procedural Macros**:
    - `#[openai_schema]`: Generate strict mode compatible JSON schemas directly from your Rust structs and enums.
    - `#[openai_function]`: Turn Rust functions into tools with minimal boilerplate.
    - `execute_function_calls!`: Seamlessly dispatch and handle multiple (parallel) tool calls.
- **Multimodal Support**: Images, audio and files (e.g. PDF), from local files, URLs or bytes.
- **Streaming**: Tool call arguments streamed in pieces are assembled for you.
- **Reasoning**: `reasoning_effort()`, and reasoning from `reasoning_content` / `reasoning` (DeepSeek, Qwen, vLLM, Ollama, OpenRouter..).
- **Embeddings**: Dense vector embeddings with `Embedder`.
- **Framework Agnostic**: Modular design that works anywhere, including Actix, Axum, and WASM environments.

### Basic Chat Example

```rust
use open_client::chat::AI;
use open_client::chat::session::{Session, SlidingWindow};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut session = Session::new(SlidingWindow::new(10)); // Keep last 10 messages
    let mut ai = AI::new(
        std::env::var("DEEPSEEK_API_KEY")?,
        "deepseek-chat",
        None, // Optional system prompt
    );
    ai.base_url("https://api.deepseek.com") // Skip for OpenAI
        .temperature(0.7);

    let completion = ai.ask(session.ask("Hello!")).await?;
    println!("AI: {}", completion.text());
    Ok(())
}
```

`AI::from_env(model, system_prompt)` reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` instead, like the official OpenAI SDKs. Local servers without API keys use `AI::with_no_api_key(model, system_prompt)`.

| Provider | Base URL |
|---|---|
| OpenAI (default) | `https://api.openai.com/v1` |
| DeepSeek | `https://api.deepseek.com` |
| Ollama | `http://localhost:11434/v1` (`with_no_api_key()`) |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` |

Common parameters have their own methods, e.g. `temperature()`, `max_completion_tokens()` or `response_format()`. Anything else, including provider specific and nested parameters, is set with `param(key, value)`, e.g. `.param("provider", json!({"sort": "price"}))`. Servers needing extra headers can be used with `AI::with_client()` and a `reqwest::Client` having default headers.

## Modules

- `open_client::chat`: the `AI` client, its parameters (`Tool`, `ToolChoice`, `ResponseFormat`, `ReasoningEffort`) and responses (`ChatCompletion`, and `ChatCompletionChunk` of a `ChatStream`).
    - `chat::message`: `Message` and its parts: `UserMessage` with `ImageUrl`, `InputAudio` and `File` parts, `AssistantMessage` with `ToolCall`s, and `ToolMessage`.
    - `chat::session`: `Session`, and `OverflowHandler`s that keep it in check.
- `open_client::embeddings`: the `Embedder` client.
- `open_client`: `Error`, the macros, and the `OpenAISchema` trait.

`reqwest`, `mime`, `serde`, `serde_json` and `futures` are re-exported (`open_client::reqwest`...), so you don't need them as dependencies. Add `serde` to derive `Deserialize` for structured output.

## Multimodal

```rust
use open_client::chat::message::{ContentPart, File, ImageUrl};
use open_client::mime;

session.ask(vec![
    ContentPart::from("Summarise the chart and the report."),
    ImageUrl::from_path("chart.png", mime::IMAGE_PNG).await?.into(),
    File::from_path("report.pdf", mime::APPLICATION_PDF).await?.into(),
]);
```

Each part is sent as its own type, so what works depends on the server:

| | Images | PDFs | Audio |
|---|---|---|---|
| OpenAI | `ImageUrl` | `File` | `InputAudio` (audio models) |
| Gemini | `ImageUrl` | `ImageUrl` (rejects `File`) | `InputAudio` |

Tool results are text, as per the specification. To show the model an image a tool produced, e.g. a screenshot, send it in a user message after the results.

## Examples

Examples use `AI::from_env()`. Set `OPENAI_API_KEY`, and `OPENAI_BASE_URL` (and the model in the example) for APIs other than OpenAI.

- [**Basic Chat**](examples/basic_chat.rs): A conversation, with follow-up questions relying on the history.
- [**Providers**](examples/providers.rs): DeepSeek, Gemini, OpenRouter (extra headers and parameters) and local Ollama.
- [**Streaming**](examples/streaming.rs): Printing the reply while it is generated.
- [**Function Calling**](examples/function_calling.rs): Rust functions as tools with `#[openai_function]` and `execute_function_calls!`.
- [**Manual Function Calling**](examples/manual_function_calling.rs): Checking calls before acting, and showing the model a screenshot.
- [**Structured Output**](examples/structured_output.rs): A reply in your own type using `#[openai_schema]`.
- [**Multimodal**](examples/multimodal.rs): Local images, image URLs, PDFs and audio.
- [**Thinking**](examples/thinking.rs): Reasoning of thinking models in a tool calling loop (DeepSeek).
- [**Context Management**](examples/context_management.rs): A token budget `OverflowHandler` calibrated with `usage`.
- [**Embeddings**](examples/embedding.rs): Semantic search.

For WASM environments, disable default features. Messages, `Session` and the macros work without the clients. Pass responses received with your own HTTP client to `Session::add_completion()`, which adds the reply and runs the `OverflowHandler`:

```toml
[dependencies]
open-client = { version = "...", default-features = false }
```

## Testing

```bash
cargo test # Offline tests
OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=... OPENAI_EMBEDDING_MODEL=... cargo test -- --ignored # Live tests
```
