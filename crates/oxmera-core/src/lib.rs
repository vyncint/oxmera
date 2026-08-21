//! Core types for oxmera: dtype, shape, strides, layout, device handle, and
//! the error taxonomy.
//!
//! This layer owns the vocabulary every other layer speaks. It must never
//! know about tensors, storage, backends, or dispatch. Everything may depend
//! on it; it depends on nothing but `std` and `thiserror`.
//!
//! Status: seams only. Every function that computes something is `todo!()` —
//! the shape/stride arithmetic (A1), broadcasting rules (A2), and error
//! taxonomy refinement (A5) are exercise rungs, implemented by the
//! maintainer, specified by the tests in `exercises/`.

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
