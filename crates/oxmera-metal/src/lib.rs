//! The Apple-Silicon Metal backend: MSL compute pipelines for elementwise
//! ops, reductions (threadgroup memory for full reduces), and tiled matrix
//! multiplication, over unified-memory (`StorageModeShared`) buffers.
//!
//! On non-macOS targets this crate compiles to an empty stub so the
//! workspace builds everywhere; the backend registers itself at load time
//! on macOS only.
//!
//! Unsafe policy: every `unsafe` block in this crate is FFI-adjacent —
//! reading a Metal buffer's contents pointer or life-before-main
//! registration — and carries a `// SAFETY:` justification.

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

// SAFETY: runs before main via the platform's initializer section. The
// body only queries the Metal device list and inserts into the
// std-synchronized backend registry; no thread-locals, no other crate's
// statics.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
#[ctor::ctor(crate_path = ::ctor)]
unsafe fn auto_register() {
    register_default();
}
