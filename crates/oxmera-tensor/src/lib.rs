//! The oxmera tensor type: storage ownership and views.
//!
//! This layer owns the `Tensor` value itself — which buffer it references,
//! with what layout, on which device — and the view operations that change
//! layout without touching data. It must never know about backends,
//! dispatch, or how any operation is computed. It may depend only on
//! `oxmera-core`.
//!
//! Status: seams only. Constructors and view arithmetic are exercise rungs
//! (A1 shape and strides, A3 strided views); bodies are `todo!()`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod storage;
pub mod tensor;

pub use storage::Storage;
pub use tensor::Tensor;
