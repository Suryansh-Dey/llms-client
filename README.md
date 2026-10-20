> [!Warning]
> **Status**: gemini-client-api project is currently in maintenance mode / archived. Fewer updates are expected.
> If you are starting, [open-client](https://crates.io/crates/open-client) is preferred because Gemini supports OpenAI complaint endpoints hence you get access to Gemini with flexibility to switch to any OpenAI complaint API.

# Gemini
## Installation
```bash
cargo add gemini-client-api
```
view at [crate.io](https://crates.io/crates/gemini-client-api)

## Overview
A Rust library to use Google's Gemini API with macro super powers! It is extremely flexible and modular to integrate with any framework.  
For example, since Actix supports stream of `Result<Bytes, Error>` for response streaming, you can get it directly instead of making a wrapper stream around a response stream of futures, which is a pain.

### Features
- Automatic context management
- Automatic function calling. Trust me!
- Automatic JSON schema generation
- Inbuilt markdown to parts parser enables AI to see markdown images or files, even if they are from your device storage!
- Vision to see images
- Code execution by Gemini
- File reading like PDF or any document, even audio files like MP3
- Function call support
- Thinking and Safety setting
- Context Caching
- Supports Session management in WASM environment with `default-features = false`


# Open Client
## Installation
```bash
cargo add open-client
```
Same as above, for **any OpenAI compatible API**: OpenAI, DeepSeek, Ollama, vLLM, OpenRouter, Groq, Gemini... Just `set_base_url(url)`.
See [openai-client](openai-client/README.md), and [migration.md](openai-client/migration.md) for differences from the Gemini client.
