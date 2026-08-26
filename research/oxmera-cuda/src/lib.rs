//! oxmera-cuda — the CUDA kernels for oxmera's deferred GPU backend,
//! written as ordinary Rust with `cuda-oxide`'s `#[kernel]`.
//!
//! Everything here is verified on a laptop, with no GPU:
//! `cargo reconverge check --strict` proves barrier and warp-collective
//! convergence statically, and `launchbound prune` disqualifies unsafe
//! launch configurations (per compute capability) before anything would
//! compile to PTX. Real execution and timings arrive only with metered
//! tier-2 sessions and are recorded when they happen — never predicted.
//!
//! Kernel families:
//! - [`elementwise`] — grid-stride loops, one disjoint slice walk per
//!   thread, no barriers by construction;
//! - [`reduce`] — two-stage reductions: a shared-memory stage tile, warp
//!   butterfly reductions (`warp::reduce_*_f32`), and uniform
//!   `sync_threads()` placements only;
//! - [`matmul`] — persistent-block tiled GEMM with double-buffered
//!   shared-memory tiles.

#![warn(missing_docs)]

pub mod elementwise;
pub mod matmul;
pub mod params;
pub mod reduce;
