//! Weight initializers.

use oxmera_core::Shape;
use oxmera_tensor::tensor::Tensor;
use rand::{Rng, SeedableRng};

fn uniform(shape: Shape, bound: f32, seed: u64) -> Tensor {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let numel = shape.numel();
    let data: Vec<f32> = (0..numel)
        .map(|_| rng.random_range(-bound..=bound))
        .collect();
    Tensor::from_vec_f32(data, shape).expect("lengths match by construction")
}

/// Kaiming/He uniform: `U(-√(6/fan_in), √(6/fan_in))` — the ReLU-family
/// default.
pub fn kaiming_uniform(shape: impl Into<Shape>, fan_in: usize, seed: u64) -> Tensor {
    uniform(shape.into(), (6.0 / fan_in as f32).sqrt(), seed)
}

/// Xavier/Glorot uniform: `U(-√(6/(fan_in+fan_out)), …)` — the
/// tanh/sigmoid-family default.
pub fn xavier_uniform(shape: impl Into<Shape>, fan_in: usize, fan_out: usize, seed: u64) -> Tensor {
    uniform(shape.into(), (6.0 / (fan_in + fan_out) as f32).sqrt(), seed)
}

/// All zeros (biases, norms' beta).
pub fn zeros(shape: impl Into<Shape>) -> Tensor {
    Tensor::zeros(shape)
}

/// All ones (norms' gamma).
pub fn ones(shape: impl Into<Shape>) -> Tensor {
    Tensor::ones(shape)
}
