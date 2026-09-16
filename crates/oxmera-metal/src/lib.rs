//! The Apple-Silicon Metal backend: MSL compute pipelines for elementwise
//! ops, reductions (threadgroup memory for full reduces), and tiled matrix
//! multiplication, over unified-memory (`StorageModeShared`) buffers.
//!
//! On non-macOS targets this crate compiles to an empty stub so the
//! workspace builds everywhere. Registration is explicit: call
//! [`register_default`] — or `oxmera_runtime::init()` — on macOS.
//!
//! Unsafe policy: every `unsafe` block in this crate is FFI-adjacent —
//! reading a Metal buffer's contents pointer — and carries a `// SAFETY:`
//! justification.

#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(target_os = "macos")]
mod backend;

#[cfg(target_os = "macos")]
pub use backend::{MetalBackend, device_summary, register_default};

/// No Metal off macOS: registration is a no-op stub so callers need no
/// cfg of their own.
#[cfg(not(target_os = "macos"))]
pub fn register_default() {}
