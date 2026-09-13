//! Core types for oxmera: dtype, shape, strides, layout, device handle, and
//! the error taxonomy.
//!
//! This layer owns the vocabulary every other layer speaks. It must never
//! know about tensors, storage, backends, or dispatch. Everything may depend
//! on it; it depends on nothing but `std` and `thiserror`.
//!
//! It defines tensor data types, validated shapes and strides, memory layouts,
//! device handles, and the shared error taxonomy used throughout the workspace.
//! These backend-independent primitives keep representation and validation
//! rules consistent across CPU, Metal, and CUDA implementations.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod device;
pub mod dtype;
pub mod error;
pub mod layout;
pub mod shape;

pub use device::Device;
pub use dtype::DType;
pub use error::{Error, Result};
pub use layout::{Layout, Strides};
pub use shape::Shape;
