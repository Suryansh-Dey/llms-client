// Lets the proc-macros' `::open_client::..` paths resolve inside this crate too.
extern crate self as open_client;

pub mod openai;
#[cfg(feature = "reqwest")]
mod utils;
pub use futures;
pub use mime;
#[cfg(feature = "reqwest")]
pub use reqwest;
pub use serde;
pub use serde_json;
