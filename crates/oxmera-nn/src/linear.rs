//! The fully connected layer.

use oxmera_core::{Result, Shape};
use oxmera_tensor::tensor::Tensor;

use crate::param::Param;
use crate::{Module, init};

/// `y = x Wᵀ + b`: input `[batch, in_features]`, output
/// `[batch, out_features]`.
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Param,
    bias: Option<Param>,
}

impl Linear {
    /// A linear layer with Kaiming-uniform weights and zero bias.
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        Self {
            weight: Param::new(init::kaiming_uniform(
                [out_features, in_features],
                in_features,
                seed,
            )),
            bias: Some(Param::new(init::zeros([out_features]))),
        }
    }

    /// A linear layer without a bias term.
    pub fn without_bias(in_features: usize, out_features: usize, seed: u64) -> Self {
        Self {
            weight: Param::new(init::kaiming_uniform(
                [out_features, in_features],
                in_features,
                seed,
            )),
            bias: None,
        }
    }

    /// The `[out, in]` weight handle.
    pub fn weight(&self) -> &Param {
        &self.weight
    }

    /// The bias handle, when the layer has one.
    pub fn bias(&self) -> Option<&Param> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        crate::check_param_dtype("Linear", input, &self.parameters())?;
        let w = self.weight.value().to_device(input.device())?;
        let mut y = input.matmul(&w.t()?)?;
        if let Some(bias) = &self.bias {
            let b = bias.value().to_device(input.device())?;
            let features = b.dims()[0];
            y = y.add(&b.reshape(Shape::from([1, features]))?)?;
        }
        Ok(y)
    }

    fn parameters(&self) -> Vec<Param> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Param)> {
        let mut named = vec![(format!("{prefix}weight"), self.weight.clone())];
        if let Some(b) = &self.bias {
            named.push((format!("{prefix}bias"), b.clone()));
        }
        named
    }
}
