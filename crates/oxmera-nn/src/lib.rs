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

use oxmera_core::{Error, Result};
use oxmera_tensor::tensor::Tensor;

/// Refuse an input whose dtype the module's parameters cannot meet.
///
/// Every layer here holds `f32` parameters and there is no way to build
/// one that does not — `Linear::new` takes no dtype, and `Module` has no
/// `to_dtype`. That is a deliberate boundary (`f64` exists for research
/// metrics and set-likelihood normalisers, not for training), but until
/// 0.4.0 nothing stated it and the error a caller got named an internal
/// operation:
///
/// ```text
/// dtype mismatch: expected F64, got F32 in matmul
/// ```
///
/// A user who has just read that `f64` covers "every CPU primitive and
/// autograd", called `model.forward(&x_f64)`, and been told about a
/// matmul they did not write has to go source-diving to learn a fact the
/// documentation could have given them. This says it instead.
///
/// # Errors
///
/// [`Error::DTypeMismatch`] naming the layer and the cast to make.
pub fn check_param_dtype(layer: &'static str, input: &Tensor, params: &[Param]) -> Result<()> {
    let Some(first) = params.first() else {
        return Ok(()); // a module with no parameters imposes nothing
    };
    let want = first.value().dtype();
    let got = input.dtype();
    if got == want {
        return Ok(());
    }
    Err(Error::InvalidArgument {
        op: "forward",
        detail: format!(
            "{layer} holds {want:?} parameters but the input is {got:?}; \
             nn layers and optimizer state are f32 regardless of input dtype \
             (docs/LIMITATIONS.md) — cast with input.to_dtype(DType::{want:?})"
        ),
    })
}

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

/// What this crate can do, for `oxmera doctor`. See
/// [`oxmera_tensor::CAPABILITIES`] for why the list lives beside the code.
pub const CAPABILITIES: &[(&str, &str)] = &[
    (
        "nn",
        "Linear Conv2d Embedding LayerNorm BatchNorm2d Dropout Sequential",
    ),
    ("losses", "MSE CrossEntropy BCEWithLogits"),
    ("weights", "safetensors save/load by parameter name"),
];
