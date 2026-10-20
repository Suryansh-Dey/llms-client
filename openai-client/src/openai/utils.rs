mod macros;
pub use macros::OpenAISchema;
pub use open_client_proc_macros::{
    execute_function_calls, execute_function_calls_with_callback, openai_function, openai_schema,
};
