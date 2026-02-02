# Gemini Client Library

A fast, flexible, and feature-rich Rust library for Google's Gemini API, featuring macros for automatic JSON schema generation and function calling. Supports all tools like Google search and maps.

## Features

- **Automatic Context Management**: Simple `Session` struct to handle conversation history.
- **Procedural Macros**:
    - `#[gemini_schema]`: Generate JSON schemas directly from your Rust structs and enums.
    - `#[gemini_function]`: Turn Rust functions into Gemini-callable tools with minimal boilerplate.
    - `execute_function_calls!`: Seamlessly dispatch and handle multiple function calls requested by Gemini.
- **Multimodal Support**: Built-in markdown parser for images and local files.
- **Advanced Capabilities**: Code execution, PDF/document/audio reading, and "Thinking" mode support.
- **Context Caching**: Efficiently manage and reuse large context windows.
- **Framework Agnostic**: Modular design that works anywhere, including Actix, Axum, and WASM environments.

### Basic Chat Example

```rust
use gemini_client_api::gemini::ask::Gemini;
use gemini_client_api::gemini::types::sessions::Session;

#[tokio::main]
async fn main() {
    let mut session = Session::new(10); // Keep last 10 messages
    let ai = Gemini::new(
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set"),
        "gemini-2.5-flash",
        None, // Optional system instruction
    );

    let response = ai.ask(session.ask("Hello, Gemini!")).await.unwrap();
    println!("Gemini: {}", response.get_chat().get_text_no_think("\n"));
}
```

## Deep Dive

Explore our comprehensive examples to learn how to use advanced features:

- [**Basic Chat**](examples/basic_chat.rs): Simple request-response interaction.
- [**Streaming**](examples/streaming.rs): Handling real-time chunks for a snappier UI experience.
- [**Structured Output**](examples/structured_output.rs): Forcing Gemini to reply in a specific JSON format using `#[gemini_schema]`.
- [**Function Calling**](examples/function_calling.rs): Giving Gemini tools to interact with the real world using `#[gemini_function]`.
- [**Thinking Mode**](examples/thinking.rs): Enabling Gemini's reasoning capabilities.
- [**Multimodal**](examples/multimodal.rs): Sending images and files to Gemini.
- [**Context Caching**](examples/context_caching.rs): Creating and using cached content for large contexts.

For WASM environments, disable default features:

```toml
[dependencies]
gemini-client-api = { version = "...", default-features = false }
```

**[Change log](https://github.com/Suryansh-Dey/llms-client/blob/main/gemini/CHANGELOG.md)**
