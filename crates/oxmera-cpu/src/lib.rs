//! The oxmera CPU backend: multi-threaded (rayon) elementwise ops,
//! reductions, and cache-tiled matrix multiplication.
//!
//! The implementation lives in `oxmera_tensor::cpu` so the reference
//! backend can be registered lazily and is always available; this crate is
//! its public face and test home.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub use oxmera_tensor::cpu::{CpuBackend, register};
