//! oxmera-cuda — the CUDA execution path, compiled with `cuda-oxide`.
//!
//! Nothing lives here yet. This crate exists so the nightly research
//! workspace is real from commit one; the `cuda-oxide` dependency and the
//! first `#[kernel]` seam arrive with the exercise ladder (tier B).
//!
//! Constraints this crate inherits (see ARCHITECTURE.md):
//! - pinned nightly, separate workspace — never a member of the stable root;
//! - kernels cannot be *built* on macOS, but `cargo check` of this workspace
//!   must stay green on every development machine with no CUDA toolkit;
//! - the correctness gate is `cargo reconverge check --strict`, which needs
//!   no GPU. Timings come only from real hardware, and are per-part.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
