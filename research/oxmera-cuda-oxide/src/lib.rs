//! oxmera-cuda-oxide — research CUDA kernels written as ordinary Rust with
//! `cuda-oxide`'s `#[kernel]`, verified without a GPU by `reconverge` and
//! `launchbound`. This is the *research* CUDA path; the shipped
//! `Device::Cuda` backend is the `oxmera-cuda` crate in the stable
//! workspace, which drives the CUDA driver API through `cudarc`.
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
