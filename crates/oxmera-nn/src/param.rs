//! Shared parameter handles.

use std::sync::{Arc, RwLock};

use oxmera_tensor::tensor::Tensor;

/// A learnable parameter: a shared, replaceable handle to a
/// gradient-accumulating leaf tensor.
///
/// Modules read the current value each forward pass; optimizers write
/// updated values back through the same handle.
#[derive(Debug, Clone)]
pub struct Param {
    inner: Arc<RwLock<Tensor>>,
}

impl Param {
    /// Wrap `value` as a learnable leaf.
    pub fn new(value: Tensor) -> Self {
        Self {
            inner: Arc::new(RwLock::new(value.detach().requires_grad_(true))),
        }
    }

    /// The current value (a cheap clone sharing storage and tape state).
    pub fn value(&self) -> Tensor {
        self.inner.read().expect("param lock poisoned").clone()
    }

    /// Replace the value with a fresh gradient-accumulating leaf.
    pub fn set(&self, value: Tensor) {
        *self.inner.write().expect("param lock poisoned") = value.detach().requires_grad_(true);
    }

    /// The gradient accumulated on the current value, if any.
    pub fn grad(&self) -> Option<Tensor> {
        self.inner.read().expect("param lock poisoned").grad()
    }

    /// Clear the accumulated gradient.
    pub fn zero_grad(&self) {
        self.inner.read().expect("param lock poisoned").zero_grad();
    }
}
