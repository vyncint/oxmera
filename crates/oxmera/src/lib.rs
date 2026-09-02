//! oxmera — a Rust-native tensor and deep-learning framework with
//! multi-threaded CPU, Apple-Silicon Metal and NVIDIA CUDA backends, reverse-mode
//! autograd, neural-network layers, optimizers, and a terminal UI.
//!
//! This umbrella crate re-exports the public surface of the workspace and
//! links every backend for the current platform, so `use oxmera::*`-style
//! consumers get working devices with no setup:
//!
//! ```
//! use oxmera::{Device, Tensor};
//!
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
//! let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], [2, 2]).unwrap();
//! let c = a.matmul(&b).unwrap();
//! assert_eq!(c.get_f32(&[0, 0]).unwrap(), 19.0);
//! assert_eq!(c.device(), Device::Cpu);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub use oxmera_autograd as autograd;
pub use oxmera_core as core;
pub use oxmera_cpu as cpu;
pub use oxmera_cuda as cuda;
#[cfg(target_os = "macos")]
pub use oxmera_metal as metal;
pub use oxmera_nn as nn;
pub use oxmera_ops as ops;
pub use oxmera_optim as optim;
pub use oxmera_runtime as runtime;
pub use oxmera_tensor as tensor;

pub use oxmera_core::{DType, Device, Error, Layout, Result, Shape, Strides};
pub use oxmera_runtime::{default_device, init, no_grad};
pub use oxmera_tensor::{Backend, NoGradGuard, Storage, Tensor};
