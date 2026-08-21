//! Dispatch and scheduling: the seam between a device handle and the
//! backend that serves it.
//!
//! This layer owns the [`Backend`] trait (the union of the op traits plus
//! identity), the resolution of a [`Device`] to a backend, and the
//! user-facing operation surface ([`TensorOps`]) that routes through it.
//! It must never know how any operation computes its answer, and it must
//! never know about `cuda-oxide`, `reconverge`, or `launchbound` — GPU
//! backends register themselves; the runtime does not reach for them.
//! `oxmera-cpu` and future backends depend on this crate; this crate
//! depends on no backend. Dispatch is dynamic (`Arc<dyn Backend>`) per
//! ADR-0004, which also records what it would take to switch.
//!
//! Status: seams only; bodies are `todo!()`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::sync::Arc;

use oxmera_core::{Device, Result};
use oxmera_ops::{BinaryOps, MatmulOps, ReduceOps, UnaryOps};
use oxmera_tensor::Tensor;

/// A complete backend: every op family, plus identity.
///
/// Object-safe by construction — the runtime holds backends as
/// `Arc<dyn Backend>`.
pub trait Backend: UnaryOps + BinaryOps + MatmulOps + ReduceOps + Send + Sync {
    /// The device this backend serves.
    fn device(&self) -> Device;
    /// A short stable name for reports and `oxmera doctor`.
    fn name(&self) -> &'static str;
}

/// The backend serving `device`.
///
/// The CPU reference backend is always available; GPU backends resolve
/// only when their crate is present and registered. An unregistered device
/// is [`oxmera_core::error::Error::BackendUnavailable`] — never a panic.
pub fn backend_for(device: Device) -> Result<Arc<dyn Backend>> {
    let _ = device;
    todo!("runtime dispatch lands with the first solved backend rung (A4)")
}

/// Register a backend for its device, replacing any previous registration.
///
/// Backend crates call this from their initialization; the umbrella crate
/// wires the defaults.
pub fn register_backend(backend: Arc<dyn Backend>) {
    let _ = backend;
    todo!("runtime dispatch lands with the first solved backend rung (A4)")
}

/// The user-facing operation surface: tensor methods that resolve the
/// backend from the tensor's device and dispatch.
pub trait TensorOps {
    /// Elementwise addition, broadcasting.
    fn add(&self, rhs: &Tensor) -> Result<Tensor>;
    /// Elementwise subtraction, broadcasting.
    fn sub(&self, rhs: &Tensor) -> Result<Tensor>;
    /// Elementwise multiplication, broadcasting.
    fn mul(&self, rhs: &Tensor) -> Result<Tensor>;
    /// Elementwise division, broadcasting.
    fn div(&self, rhs: &Tensor) -> Result<Tensor>;
    /// Elementwise negation.
    fn neg(&self) -> Result<Tensor>;
    /// Rank-2 matrix product.
    fn matmul(&self, rhs: &Tensor) -> Result<Tensor>;
    /// Sum over `axes` (empty means all).
    fn sum(&self, axes: &[usize]) -> Result<Tensor>;
}

impl TensorOps for Tensor {
    fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        let _ = rhs;
        todo!("dispatch plumbing lands with the first solved backend rung (A4)")
    }

    fn sub(&self, rhs: &Tensor) -> Result<Tensor> {
        let _ = rhs;
        todo!("dispatch plumbing lands with the first solved backend rung (A4)")
    }

    fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        let _ = rhs;
        todo!("dispatch plumbing lands with the first solved backend rung (A4)")
    }

    fn div(&self, rhs: &Tensor) -> Result<Tensor> {
        let _ = rhs;
        todo!("dispatch plumbing lands with the first solved backend rung (A4)")
    }

    fn neg(&self) -> Result<Tensor> {
        todo!("dispatch plumbing lands with the first solved backend rung (A4)")
    }

    fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        let _ = rhs;
        todo!("dispatch plumbing lands with the first solved backend rung (A4)")
    }

    fn sum(&self, axes: &[usize]) -> Result<Tensor> {
        let _ = axes;
        todo!("dispatch plumbing lands with the first solved backend rung (A4)")
    }
}
