//! Module composition.

use oxmera_core::Result;
use oxmera_tensor::tensor::Tensor;

use crate::{Module, Param};

/// Modules applied in order. Parameter names are prefixed by child index
/// (`0.weight`, `1.bias`, …), matching the common convention.
#[derive(Default)]
pub struct Sequential {
    children: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// An empty pipeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a module, builder-style.
    pub fn push(mut self, module: impl Module + 'static) -> Self {
        self.children.push(Box::new(module));
        self
    }

    /// The number of children.
    pub fn len(&self) -> usize {
        self.children.len()
    }

    /// Whether the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for child in &self.children {
            x = child.forward(&x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<Param> {
        self.children.iter().flat_map(|c| c.parameters()).collect()
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Param)> {
        self.children
            .iter()
            .enumerate()
            .flat_map(|(i, c)| c.named_parameters(&format!("{prefix}{i}.")))
            .collect()
    }

    fn set_training(&self, training: bool) {
        for child in &self.children {
            child.set_training(training);
        }
    }
}
