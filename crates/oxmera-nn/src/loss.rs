//! Loss functions. Each is a plain struct with a two-argument `forward`;
//! all are composites of differentiable primitives.

use oxmera_core::{Error, Result, Shape};
use oxmera_tensor::tensor::Tensor;

use crate::functional::one_hot;

/// Mean squared error: `mean((input - target)^2)`.
#[derive(Debug, Default, Clone, Copy)]
pub struct MSELoss;

impl MSELoss {
    /// The scalar loss.
    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Result<Tensor> {
        if input.shape() != target.shape() {
            return Err(Error::ShapeMismatch {
                expected: input.shape().clone(),
                got: target.shape().clone(),
                op: "MSELoss",
            });
        }
        let diff = input.sub(target)?;
        diff.mul(&diff)?.mean(&[])
    }
}

/// Cross-entropy over logits: numerically stable log-softmax plus
/// negative log-likelihood. Logits `[n, classes]` (`F32`), targets `[n]`
/// (`I64` class indices).
#[derive(Debug, Default, Clone, Copy)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// The scalar loss, averaged over the batch.
    pub fn forward(&self, logits: &Tensor, target: &Tensor) -> Result<Tensor> {
        if logits.ndim() != 2 {
            return Err(Error::InvalidArgument {
                op: "CrossEntropyLoss",
                detail: format!("logits must be [n, classes], got {:?}", logits.shape()),
            });
        }
        let n = logits.dims()[0];
        if n == 0 {
            return Err(Error::InvalidArgument {
                op: "CrossEntropyLoss",
                detail: "empty batch: the mean over 0 samples is undefined".into(),
            });
        }
        if target.ndim() != 1 || target.dims()[0] != n {
            return Err(Error::ShapeMismatch {
                expected: Shape::from([n]),
                got: target.shape().clone(),
                op: "CrossEntropyLoss",
            });
        }
        let classes = logits.dims()[1];
        let log_p = logits.log_softmax(1)?;
        let onehot = one_hot(&target.to_device(oxmera_core::Device::Cpu)?, classes)?
            .to_device(logits.device())?;
        log_p
            .mul(&onehot)?
            .sum(&[])?
            .neg()?
            .mul_scalar(1.0 / n as f32)
    }
}

/// Binary cross-entropy over logits, computed with the stable
/// `max(x, 0) - x·z + ln(1 + e^{-|x|})` form.
#[derive(Debug, Default, Clone, Copy)]
pub struct BCEWithLogitsLoss;

impl BCEWithLogitsLoss {
    /// The scalar loss, averaged over every element.
    pub fn forward(&self, logits: &Tensor, target: &Tensor) -> Result<Tensor> {
        if logits.shape() != target.shape() {
            return Err(Error::ShapeMismatch {
                expected: logits.shape().clone(),
                got: target.shape().clone(),
                op: "BCEWithLogitsLoss",
            });
        }
        let zero = Tensor::scalar_on(logits, 0.0)?;
        let relu_x = logits.maximum(&zero)?;
        let xz = logits.mul(target)?;
        let softplus = logits.abs()?.neg()?.exp()?.add_scalar(1.0)?.ln()?;
        relu_x.sub(&xz)?.add(&softplus)?.mean(&[])
    }
}
