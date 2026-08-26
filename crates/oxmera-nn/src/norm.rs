//! Normalization layers.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use oxmera_core::{Error, Result, Shape};
use oxmera_tensor::no_grad;
use oxmera_tensor::tensor::Tensor;

use crate::param::Param;
use crate::{Module, init};

/// Layer normalization over the last dimension:
/// `y = (x - mean) / sqrt(var + eps) * gamma + beta`.
#[derive(Debug)]
pub struct LayerNorm {
    gamma: Param,
    beta: Param,
    features: usize,
    eps: f32,
}

impl LayerNorm {
    /// Normalize the trailing `features` dimension.
    pub fn new(features: usize) -> Self {
        Self {
            gamma: Param::new(init::ones([features])),
            beta: Param::new(init::zeros([features])),
            features,
            eps: 1e-5,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let last = input.ndim().checked_sub(1).ok_or(Error::InvalidArgument {
            op: "LayerNorm",
            detail: "input must have at least one dimension".into(),
        })?;
        if input.dims()[last] != self.features {
            return Err(Error::ShapeMismatch {
                expected: Shape::from([self.features]),
                got: input.shape().clone(),
                op: "LayerNorm",
            });
        }
        let mu = input.mean_keepdim(&[last], true)?;
        let centered = input.sub(&mu)?;
        let var = centered.mul(&centered)?.mean_keepdim(&[last], true)?;
        let xhat = centered.div(&var.add_scalar(self.eps)?.sqrt()?)?;
        let device = input.device();
        let g = self.gamma.value().to_device(device)?;
        let b = self.beta.value().to_device(device)?;
        xhat.mul(&g)?.add(&b)
    }

    fn parameters(&self) -> Vec<Param> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Param)> {
        vec![
            (format!("{prefix}weight"), self.gamma.clone()),
            (format!("{prefix}bias"), self.beta.clone()),
        ]
    }
}

/// Batch normalization over `[n, c, h, w]`: statistics per channel across
/// batch and space; running estimates used in eval mode.
#[derive(Debug)]
pub struct BatchNorm2d {
    gamma: Param,
    beta: Param,
    running_mean: Mutex<Tensor>,
    running_var: Mutex<Tensor>,
    channels: usize,
    momentum: f32,
    eps: f32,
    training: AtomicBool,
}

impl BatchNorm2d {
    /// Batch norm over `channels`.
    pub fn new(channels: usize) -> Self {
        Self {
            gamma: Param::new(init::ones([channels])),
            beta: Param::new(init::zeros([channels])),
            running_mean: Mutex::new(Tensor::zeros([channels])),
            running_var: Mutex::new(Tensor::ones([channels])),
            channels,
            momentum: 0.1,
            eps: 1e-5,
            training: AtomicBool::new(true),
        }
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.ndim() != 4 || input.dims()[1] != self.channels {
            return Err(Error::ShapeMismatch {
                expected: Shape::from([0, self.channels, 0, 0]),
                got: input.shape().clone(),
                op: "BatchNorm2d",
            });
        }
        let stat_shape = Shape::from([1, self.channels, 1, 1]);
        let (mu, var) = if self.training.load(Ordering::Relaxed) {
            let mu = input.mean_keepdim(&[0, 2, 3], true)?;
            let centered = input.sub(&mu)?;
            let var = centered.mul(&centered)?.mean_keepdim(&[0, 2, 3], true)?;
            no_grad(|| -> Result<()> {
                let flat_mu = mu
                    .detach()
                    .to_device(oxmera_core::Device::Cpu)?
                    .reshape(Shape::from([self.channels]))?;
                let flat_var = var
                    .detach()
                    .to_device(oxmera_core::Device::Cpu)?
                    .reshape(Shape::from([self.channels]))?;
                let mut rm = self.running_mean.lock().expect("bn lock poisoned");
                let mut rv = self.running_var.lock().expect("bn lock poisoned");
                *rm = rm
                    .mul_scalar(1.0 - self.momentum)?
                    .add(&flat_mu.mul_scalar(self.momentum)?)?;
                *rv = rv
                    .mul_scalar(1.0 - self.momentum)?
                    .add(&flat_var.mul_scalar(self.momentum)?)?;
                Ok(())
            })?;
            (mu, var)
        } else {
            let rm = self.running_mean.lock().expect("bn lock poisoned").clone();
            let rv = self.running_var.lock().expect("bn lock poisoned").clone();
            (
                rm.reshape(stat_shape.clone())?.to_device(input.device())?,
                rv.reshape(stat_shape.clone())?.to_device(input.device())?,
            )
        };
        let xhat = input.sub(&mu)?.div(&var.add_scalar(self.eps)?.sqrt()?)?;
        let device = input.device();
        let g = self
            .gamma
            .value()
            .to_device(device)?
            .reshape(stat_shape.clone())?;
        let b = self.beta.value().to_device(device)?.reshape(stat_shape)?;
        xhat.mul(&g)?.add(&b)
    }

    fn parameters(&self) -> Vec<Param> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Param)> {
        vec![
            (format!("{prefix}weight"), self.gamma.clone()),
            (format!("{prefix}bias"), self.beta.clone()),
        ]
    }

    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }
}
