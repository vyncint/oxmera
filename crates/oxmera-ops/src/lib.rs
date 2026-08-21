//! Operation traits — the contract every backend implements.
//!
//! This layer owns the *signatures* of the operations and the contracts in
//! their documentation; it contains no implementations of any kind, and
//! that is permanent — implementations live in backend crates. It must
//! never know which backends exist. It may depend only on `oxmera-core`
//! and `oxmera-tensor`.
//!
//! Semantics every backend must honor:
//!
//! - Binary elementwise operations broadcast per
//!   [`oxmera_core::shape::broadcast_shapes`].
//! - Operands must share a device and (for now) a dtype; violations are
//!   typed errors, never coercions. Implicit promotion is a future ADR,
//!   not a backend's improvisation.
//! - The output of every operation is a fresh contiguous tensor on the
//!   operands' device.
//! - The CPU reference backend defines correct answers; every other
//!   backend is validated against it.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use oxmera_core::Result;
use oxmera_tensor::Tensor;

/// Unary elementwise operations.
pub trait UnaryOps {
    /// Elementwise negation.
    fn neg(&self, a: &Tensor) -> Result<Tensor>;
    /// Elementwise natural exponential. Float dtypes only.
    fn exp(&self, a: &Tensor) -> Result<Tensor>;
    /// Elementwise natural logarithm. Float dtypes only.
    fn ln(&self, a: &Tensor) -> Result<Tensor>;
    /// Elementwise absolute value.
    fn abs(&self, a: &Tensor) -> Result<Tensor>;
}

/// Binary elementwise operations, broadcasting.
pub trait BinaryOps {
    /// Elementwise addition.
    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>;
    /// Elementwise subtraction.
    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>;
    /// Elementwise multiplication.
    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>;
    /// Elementwise division. Integer division by zero is a typed error;
    /// float division follows IEEE-754.
    fn div(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>;
}

/// Matrix multiplication.
pub trait MatmulOps {
    /// Matrix product of two rank-2 tensors: `[m, k] x [k, n] -> [m, n]`.
    ///
    /// Rank-2 only until batched matmul earns an ADR. Shape or dtype
    /// violations are typed errors.
    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>;
}

/// Reductions along axes.
pub trait ReduceOps {
    /// Sum over `axes`, removing them from the shape. Empty `axes` means
    /// all axes (a scalar result).
    fn sum(&self, a: &Tensor, axes: &[usize]) -> Result<Tensor>;
    /// Maximum over `axes`, removing them. Errors on empty reduction
    /// extents — there is no identity to invent.
    fn max(&self, a: &Tensor, axes: &[usize]) -> Result<Tensor>;
}
