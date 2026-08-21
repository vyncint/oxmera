//! The CPU reference backend: correct, deliberately unoptimized, always
//! available.
//!
//! This layer owns the ground truth. Its implementations define what every
//! operation *means*; every other backend is validated against it, and no
//! optimization is ever accepted here at the cost of readability. It must
//! never know about other backends, dispatch policy, or anything
//! GPU-shaped. It depends on `core`, `tensor`, `ops`, and `runtime`;
//! nothing depends on it except the umbrella.
//!
//! Status: every operation body is `todo!()`. The reference
//! implementations are the point of the exercise ladder (A4 and onward)
//! and are written by the maintainer, not scaffolded.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use oxmera_core::{Device, Result};
use oxmera_ops::{BinaryOps, MatmulOps, ReduceOps, UnaryOps};
use oxmera_runtime::Backend;
use oxmera_tensor::Tensor;

/// The reference backend. Stateless; one instance serves all CPU tensors.
#[derive(Debug, Default)]
pub struct CpuBackend;

impl UnaryOps for CpuBackend {
    fn neg(&self, a: &Tensor) -> Result<Tensor> {
        let _ = a;
        todo!("exercise rung: reference elementwise ops")
    }

    fn exp(&self, a: &Tensor) -> Result<Tensor> {
        let _ = a;
        todo!("exercise rung: reference elementwise ops")
    }

    fn ln(&self, a: &Tensor) -> Result<Tensor> {
        let _ = a;
        todo!("exercise rung: reference elementwise ops")
    }

    fn abs(&self, a: &Tensor) -> Result<Tensor> {
        let _ = a;
        todo!("exercise rung: reference elementwise ops")
    }
}

impl BinaryOps for CpuBackend {
    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        let _ = (lhs, rhs);
        todo!("exercise rung: reference elementwise ops")
    }

    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        let _ = (lhs, rhs);
        todo!("exercise rung: reference elementwise ops")
    }

    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        let _ = (lhs, rhs);
        todo!("exercise rung: reference elementwise ops")
    }

    fn div(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        let _ = (lhs, rhs);
        todo!("exercise rung: reference elementwise ops")
    }
}

impl MatmulOps for CpuBackend {
    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        let _ = (lhs, rhs);
        todo!("exercise A4: the reference matmul")
    }
}

impl ReduceOps for CpuBackend {
    fn sum(&self, a: &Tensor, axes: &[usize]) -> Result<Tensor> {
        let _ = (a, axes);
        todo!("exercise rung: reference reductions")
    }

    fn max(&self, a: &Tensor, axes: &[usize]) -> Result<Tensor> {
        let _ = (a, axes);
        todo!("exercise rung: reference reductions")
    }
}

impl Backend for CpuBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn name(&self) -> &'static str {
        "cpu-reference"
    }
}
