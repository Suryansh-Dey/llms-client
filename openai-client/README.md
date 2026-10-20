# Open Client

A fast, flexible, and feature-rich Rust library for **any OpenAI compatible API**: OpenAI, DeepSeek, Ollama, vLLM, OpenRouter, Groq, LM Studio, Gemini and more. Featuring macros for automatic JSON schema generation and function calling.

Sibling of [gemini-client-api](https://crates.io/crates/gemini-client-api) with the same design. Moving from it? See [migration.md](migration.md).

## Features

- **Any OpenAI compatible endpoint**: `set_base_url()` and you're done. Text-only messages are sent as plain strings, and responses and errors are parsed leniently, so servers that only partially follow the spec still work.
- **Automatic Context Management**: Simple `Session` struct to handle conversation history, with custom `OverflowHandler`s (e.g. token budget).
- **Procedural Macros**:
    - `#[openai_schema]`: Generate strict mode compatible JSON schemas directly from your Rust structs and enums.
    - `#[openai_function]`: Turn Rust functions into tools with minimal boilerplate.
    - `execute_function_calls!`: Seamlessly dispatch and handle multiple (parallel) function calls.
- **Multimodal Support**: Images, audio and files (e.g. PDF), plus a built-in markdown parser for images and local files.
- **Multimodal function responses**: Functions can return images too, e.g. a screenshot.
- **Streaming**: Function call arguments streamed in pieces are assembled for you.
- **Reasoning**: `reasoning_effort` and thoughts from `reasoning_content` / `reasoning` (DeepSeek, Qwen, vLLM, Ollama, OpenRouter..).
- **Embeddings**: Generate dense vector embeddings using the `OpenAIEmbedding` client.
- **Framework Agnostic**: Modular design that works anywhere, including Actix, Axum, and WASM environments.

### Basic Chat Example

```rust
use open_client::openai::ask::OpenAI;
use open_client::openai::types::sessions::Session;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut session = Session::new(10); // Keep last 10 messages
    let ai = OpenAI::new(
        std::env::var("DEEPSEEK_API_KEY")?,
        "deepseek-chat",
        None, // Optional system instruction
    )
    .set_base_url("https://api.deepseek.com"); // Skip for OpenAI

    let response = ai.ask(session.ask("Hello!")).await?;
    println!("AI: {}", response.get_chat().get_text_no_think("\n"));
    Ok(())
}
```

`OpenAI::from_env(model, sys_prompt)` reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` instead, like the official OpenAI SDKs. Local servers without API keys use `OpenAI::with_no_api_key(model, sys_prompt)`.

| Provider | Base URL |
|---|---|
| OpenAI (default) | `https://api.openai.com/v1` |
| DeepSeek | `https://api.deepseek.com` |
| Ollama | `http://localhost:11434/v1` (`with_no_api_key()`) |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` |

Provider specific parameters can be set with `ai.set_generation_config()["param"] = value`, which goes to the top level of the request body. Servers needing extra headers can be used with `OpenAI::new_with_client()` and a `reqwest::Client` having default headers.

`reqwest`, `mime`, `serde_json` and `futures` are re-exported (`open_client::reqwest`...), so you don't need them as dependencies. Add `serde` to derive `Deserialize` for structured output.

## Examples

Examples use `OpenAI::from_env()`. Set `OPENAI_API_KEY`, and `OPENAI_BASE_URL` (and the model in the example) for APIs other than OpenAI.

- [**Basic Chat**](examples/basic_chat.rs): A conversation, with follow-up questions relying on the history.
- [**Providers**](examples/providers.rs): DeepSeek, Gemini, OpenRouter (extra headers and parameters) and local Ollama.
- [**Streaming**](examples/streaming.rs): Printing the reply while it is generated.
- [**Function Calling**](examples/function_calling.rs): Rust functions as tools with `#[openai_function]` and `execute_function_calls!`.
- [**Manual Function Calling**](examples/manual_function_calling.rs): Checking calls before acting, and responding with an image.
- [**Structured Output**](examples/structured_output.rs): A reply in your own type using `#[openai_schema]`.
- [**Multimodal**](examples/multimodal.rs): Local files, image URLs and markdown with images.
- [**Thinking**](examples/thinking.rs): Thoughts of reasoning models in a tool calling loop (DeepSeek).
- [**Context Management**](examples/context_management.rs): A token budget `OverflowHandler` calibrated with `usage`.
- [**Embeddings**](examples/embedding.rs): Semantic search.

For WASM environments, disable default features:

```toml
[dependencies]
open-client = { version = "...", default-features = false }
```

## Testing

```bash
cargo test # Offline tests
OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=... OPENAI_EMBEDDING_MODEL=... cargo test -- --ignored # Live tests
```
