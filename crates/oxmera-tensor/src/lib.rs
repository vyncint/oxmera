//! The oxmera tensor: shared storage viewed through a layout, with
//! multi-device backends and reverse-mode autograd.
//!
//! This crate is the hub of the framework:
//!
//! - [`Tensor`] — the value type: constructors, zero-copy strided views,
//!   element access, and the full differentiable op surface (`add`,
//!   `matmul`, `softmax`, …) plus `std::ops` operator sugar.
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
mod cpu_iter;
mod cpu_matmul;
pub mod ops;
pub mod overload;
pub mod storage;
pub mod tensor;

pub use autograd::{NoGradGuard, no_grad};
pub use backend::{Backend, BinaryOp, ReduceOp, UnaryOp, backend_for, register_backend};
pub use storage::{CpuStorage, OpaqueBuffer, Storage, StorageData};
pub use tensor::Tensor;
