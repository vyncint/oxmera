//! Shared parameter handles.

use std::sync::{Arc, RwLock};

use oxmera_core::{Device, Result};
use oxmera_tensor::tensor::Tensor;

/// A learnable parameter: a shared, replaceable handle to a
/// gradient-accumulating leaf tensor.
///
/// Modules read the current value each forward pass; optimizers write
/// updated values back through the same handle. Cloning a `Param` shares
/// that handle intentionally, while [`Param::detached_copy`] creates an
/// independent parameter for explicitly copying a module.
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

    /// Create an independent parameter with the same current value.
    ///
    /// Unlike [`Clone::clone`], this copies the tensor storage and starts a
    /// fresh gradient-accumulating leaf. The copy remains on the same device
    /// and keeps the source tensor's dtype and shape.
    pub fn detached_copy(&self) -> Result<Self> {
        Ok(Self::new(self.value().contiguous_untracked()?))
    }

    /// The current value (a cheap clone sharing storage and tape state).
    pub fn value(&self) -> Tensor {
        self.inner.read().expect("param lock poisoned").clone()
    }

    /// Replace the value with a fresh gradient-accumulating leaf.
    pub fn set(&self, value: Tensor) {
        *self.inner.write().expect("param lock poisoned") = value.detach().requires_grad_(true);
    }

    /// Move the parameter's value to `device` in place.
    ///
    /// The shared handle means every holder — the owning module and any
    /// optimizer state keyed on this `Param` — sees the moved tensor, so a
    /// model's weights upload once instead of the forward pass re-uploading
    /// them on every call. A no-op when the value is already on `device`.
    pub fn to_device(&self, device: Device) -> Result<()> {
        let current = self.value();
        if current.device() == device {
            return Ok(());
        }
        self.set(current.to_device(device)?);
        Ok(())
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
