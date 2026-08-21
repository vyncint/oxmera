//! oxmera — a Rust-native machine-learning and GPU-computing framework
//! built from first principles, in the open, as a learning project that
//! does not lie about what it is.
//!
//! This is the umbrella crate: it re-exports the public surface of the
//! oxmera workspace and contains no logic of its own, permanently. The
//! layer crates are re-exported as modules; the most common types are also
//! re-exported at the root.
//!
//! Status: skeleton under construction. Every operation body is `todo!()`
//! — the implementations are the maintainer's exercise ladder, and no
//! computation works yet. See the repository README for what this project
//! deliberately is and is not.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub use oxmera_core as core;
pub use oxmera_cpu as cpu;
pub use oxmera_ops as ops;
pub use oxmera_runtime as runtime;
pub use oxmera_tensor as tensor;

pub use oxmera_core::{DType, Device, Error, Layout, Result, Shape, Strides};
pub use oxmera_runtime::{Backend, TensorOps};
pub use oxmera_tensor::{Storage, Tensor};
