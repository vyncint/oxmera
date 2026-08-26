//! The lookup-table layer.

use oxmera_core::{Result, Shape};
use oxmera_tensor::tensor::Tensor;

use crate::param::Param;
use crate::{Module, init};

/// A learnable lookup table: `I64` indices of any shape in, embeddings of
/// shape `indices.shape() + [dim]` out.
#[derive(Debug, Clone)]
pub struct Embedding {
    weight: Param,
    dim: usize,
}

impl Embedding {
    /// A table of `vocab` embeddings of size `dim`.
    pub fn new(vocab: usize, dim: usize, seed: u64) -> Self {
        Self {
            weight: Param::new(init::xavier_uniform([vocab, dim], vocab, dim, seed)),
            dim,
        }
    }

    /// The `[vocab, dim]` weight handle.
    pub fn weight(&self) -> &Param {
        &self.weight
    }

    /// Look up `indices` (`I64`).
    pub fn lookup(&self, indices: &Tensor) -> Result<Tensor> {
        let flat_n = indices.numel();
        let flat = indices.reshape(Shape::from([flat_n]))?;
        let table = self.weight.value();
        let rows = table.index_select(0, &flat)?;
        let mut out_dims = indices.dims().to_vec();
        out_dims.push(self.dim);
        rows.reshape(Shape::new(out_dims))
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.lookup(input)
    }

    fn parameters(&self) -> Vec<Param> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Param)> {
        vec![(format!("{prefix}weight"), self.weight.clone())]
    }
}
