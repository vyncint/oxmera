//! Inverted dropout.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use oxmera_core::{Error, Result};
use oxmera_tensor::tensor::Tensor;
use rand::{Rng, SeedableRng};

use crate::{Module, Param};

/// Inverted dropout: in training mode, zero each element with probability
/// `p` and scale survivors by `1/(1-p)`; identity in eval mode.
#[derive(Debug)]
pub struct Dropout {
    p: f32,
    training: AtomicBool,
    /// Per-forward seed counter so masks vary between calls but the layer
    /// stays deterministic for a fixed construction seed.
    seed: AtomicU64,
}

impl Dropout {
    /// Dropout with drop probability `p` in `[0, 1)`.
    ///
    /// Returns [`Error::InvalidArgument`] when `p` is outside `[0, 1)`: at
    /// `p >= 1` every element is dropped and the survivors would be scaled
    /// by `1/(1 - p) <= 0`, and a negative `p` is silently ignored by the
    /// forward pass.
    pub fn new(p: f32, seed: u64) -> Result<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(Error::InvalidArgument {
                op: "Dropout",
                detail: format!("drop probability must be in [0, 1), got {p}"),
            });
        }
        Ok(Self {
            p,
            training: AtomicBool::new(true),
            seed: AtomicU64::new(seed),
        })
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !self.training.load(Ordering::Relaxed) || self.p <= 0.0 {
            return Ok(input.clone());
        }
        let seed = self.seed.fetch_add(1, Ordering::Relaxed);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let keep = 1.0 - self.p;
        let mask: Vec<f32> = (0..input.numel())
            .map(|_| {
                if rng.random::<f32>() < keep {
                    1.0 / keep
                } else {
                    0.0
                }
            })
            .collect();
        let mask = Tensor::from_vec_f32(mask, input.shape().clone())?.to_device(input.device())?;
        input.mul(&mask)
    }

    fn parameters(&self) -> Vec<Param> {
        Vec::new()
    }

    fn named_parameters(&self, _prefix: &str) -> Vec<(String, Param)> {
        Vec::new()
    }

    fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }
}
