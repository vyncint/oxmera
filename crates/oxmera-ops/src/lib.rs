//! The operation vocabulary oxmera backends implement.
//!
//! The trait and enums are defined in `oxmera-tensor` (they must live with
//! `Tensor` so its methods and `std::ops` overloads can dispatch under the
//! orphan rule); this crate re-exports them as the stable, documented
//! vocabulary for backend authors.
//!
//! Semantics every backend must honor:
//!
//! - Binary elementwise operations broadcast per
//!   [`oxmera_core::shape::broadcast_shapes`].
//! - Operands must share a device and dtype; violations are typed errors,
//!   never coercions.
//! - The output of every operation is a fresh contiguous tensor on the
//!   operands' device.
//! - The CPU backend defines correct answers; every other backend is
//!   validated against it within floating-point tolerance.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub use oxmera_tensor::backend::{
    Backend, BinaryOp, ReduceOp, UnaryOp, backend_for, register_backend, registered_devices,
};
