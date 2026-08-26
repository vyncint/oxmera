//! Neural-network building blocks for oxmera: the [`Module`] trait,
//! layers, losses, initializers, and safetensors serialization.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod conv;
mod dropout;
mod embedding;
pub mod functional;
pub mod init;
mod linear;
mod loss;
mod norm;
mod param;
mod sequential;
pub mod serialize;

pub use conv::Conv2d;
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use linear::Linear;
pub use loss::{BCEWithLogitsLoss, CrossEntropyLoss, MSELoss};
pub use norm::{BatchNorm2d, LayerNorm};
pub use param::Param;
pub use sequential::Sequential;

use oxmera_core::Result;
use oxmera_tensor::tensor::Tensor;

/// A neural-network component: a differentiable function of its input and
/// a set of learnable parameters.
pub trait Module: Send + Sync {
    /// Apply the module.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Every learnable parameter, as shared handles the optimizer updates
    /// in place. (Handles rather than plain tensors, so updates written by
    /// the optimizer are seen by the module — Rust's answer to PyTorch's
    /// shared parameter objects.)
    fn parameters(&self) -> Vec<Param>;

    /// Parameters with their serialization names, prefixed by `prefix`
    /// (`""` at the root; `Sequential` prepends child indices).
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Param)>;

    /// Clear every parameter's accumulated gradient.
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    /// Switch training-mode behaviour (dropout, batch-norm statistics).
    /// Modules without mode-dependent behaviour ignore this.
    fn set_training(&self, _training: bool) {}
}
