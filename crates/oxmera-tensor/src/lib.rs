//! The oxmera tensor: shared storage viewed through a layout, with
//! multi-device backends and reverse-mode autograd.
//!
//! This crate is the hub of the framework:
//!
//! - [`Tensor`] — the value type: constructors, zero-copy strided views,
//!   element access, and the full differentiable op surface (`add`,
//!   `matmul`, `softmax`, …) plus `std::ops` operator sugar.
//! - [`einsum`] — Einstein-summation contractions lowered onto `matmul`,
//!   `permute` and `sum`.
//! - Small batched linear algebra on `Tensor`: `eye`, `diag`, `diag_embed`,
//!   `trace`, `cholesky`, `logdet`, `det`, `eigh`.
//! - [`backend`] — the op vocabulary ([`backend::UnaryOp`],
//!   [`backend::BinaryOp`], [`backend::ReduceOp`]), the [`backend::Backend`]
//!   trait every device implements, and the registry that resolves a
//!   [`oxmera_core::Device`] handle. The traits live here so tensor
//!   methods and operator overloads can dispatch without violating the
//!   orphan rule.
//! - [`autograd`] — the tape: recording switch, [`autograd::no_grad`],
//!   and gradient propagation; `Tensor::backward` drives it.
//!
//! Backends register themselves at load time (linking `oxmera-cpu` or
//! `oxmera-metal` is what makes their device usable); the `oxmera`
//! umbrella crate links every backend for the current platform.

#![deny(unsafe_code)] // the two Metal Send/Sync impls in `storage` opt in locally
#![warn(missing_docs)]

pub mod autograd;
pub mod backend;
pub mod cpu;
mod cpu_f64;
mod cpu_iter;
mod cpu_linalg;
mod cpu_matmul;
mod einsum;
mod linalg;
pub use linalg::EIGH_SYMMETRY_TOL;
pub mod ops;
pub mod overload;
pub mod storage;
pub mod tensor;

pub use autograd::{NoGradGuard, no_grad};
pub use backend::{Backend, BinaryOp, ReduceOp, UnaryOp, backend_for, register_backend};
pub use einsum::einsum;
pub use storage::{CpuStorage, OpaqueBuffer, Storage, StorageData};
pub use tensor::Tensor;

/// What this crate can do, for `oxmera doctor`.
///
/// It lives here, beside the code, because the alternative was a list of
/// string literals in `oxmera-cli` that nobody editing an op ever opened:
/// that list named neither CUDA nor `f64` nor the linear algebra two
/// releases after they shipped, and three golden files pinned it, so
/// adding a feature to it cost more than leaving it stale.
///
/// Adding an op family here is one line in the crate that implements it.
pub const CAPABILITIES: &[(&str, &str)] = &[
    // Each row must fit an 80-column terminal after the 12-character
    // "  area     " prefix, so keep the text under 68 characters: the
    // doctor goldens are 100 wide and a wrapped row costs two of them.
    // `capability_rows_fit_a_narrow_terminal` in oxmera-cli enforces it.
    (
        "tensor",
        "strided views, broadcasting, batched matmul, device transfer",
    ),
    (
        "dtypes",
        "f32 everywhere; f64 on the CPU; i64 indices; no promotion",
    ),
    (
        "linalg",
        "eye diag trace, batched cholesky (differentiable) logdet det eigh",
    ),
    (
        "einsum",
        "one- and two-operand contractions with an explicit output",
    ),
    (
        "indexing",
        "index_select / index_add, native on Metal and CUDA",
    ),
];
