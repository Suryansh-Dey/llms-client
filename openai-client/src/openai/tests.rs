//! Offline tests of crate internals. Tests of the public API, including live ones, are in `tests/`.
mod response;
// Need the `reqwest` feature
#[cfg(feature = "reqwest")]
mod error;
#[cfg(feature = "reqwest")]
mod http;
#[cfg(feature = "reqwest")]
mod request;
#[cfg(feature = "reqwest")]
mod stream;
#[cfg(feature = "reqwest")]
mod utils;
