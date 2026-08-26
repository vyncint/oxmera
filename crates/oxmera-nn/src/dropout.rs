//! Inverted dropout.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use oxmera_core::Result;
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
    pub fn new(p: f32, seed: u64) -> Self {
        Self {
            p,
            training: AtomicBool::new(true),
            seed: AtomicU64::new(seed),
        }
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
