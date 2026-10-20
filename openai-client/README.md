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
async fn main() {
    let mut session = Session::new(10); // Keep last 10 messages
    let ai = OpenAI::new(
        std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set"),
        "deepseek-chat",
        None, // Optional system instruction
    )
    .set_base_url("https://api.deepseek.com"); // Skip for OpenAI

    let response = ai.ask(session.ask("Hello!")).await.unwrap();
    println!("AI: {}", response.get_chat().get_text_no_think("\n"));
}
```

| Provider | Base URL |
|---|---|
| OpenAI (default) | `https://api.openai.com/v1` |
| DeepSeek | `https://api.deepseek.com` |
| Ollama | `http://localhost:11434/v1` (empty API key) |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` |

Provider specific parameters can be set with `ai.set_generation_config()["param"] = value`, which goes to the top level of the request body. Servers needing extra headers can be used with `OpenAI::new_with_client()` and a `reqwest::Client` having default headers.

## Examples

- [**Basic Chat**](examples/basic_chat.rs): Simple request-response interaction.
- [**Custom Endpoint**](examples/custom_endpoint.rs): DeepSeek, Ollama, OpenRouter, Gemini..
- [**Multimodal**](examples/multimodal.rs): Sending images and files.
- [**Streaming**](examples/streaming.rs): Handling real-time chunks for a snappier UI experience.
- [**Structured Output**](examples/structured_output.rs): Forcing a reply in a specific JSON format using `#[openai_schema]`.
- [**Function Calling**](examples/function_calling.rs): Giving the model tools to interact with the real world using `#[openai_function]`.
- [**Thinking**](examples/thinking.rs): Reasoning effort and thoughts.
- [**Advanced Context Management**](examples/advanced_context_management.rs): Custom history trimming by token budget.
- [**Embeddings**](examples/embedding.rs): Generating vector embeddings for search and retrieval.

For WASM environments, disable default features:

```toml
[dependencies]
open-client = { version = "...", default-features = false }
```

## Testing

```bash
cargo test -- --skip live # Offline tests, using a local mock server
OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=... OPENAI_EMBEDDING_MODEL=... cargo test live
```
