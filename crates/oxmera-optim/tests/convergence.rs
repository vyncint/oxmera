//! Optimizers must optimize: each one drives a quadratic to its minimum,
//! and Adam trains a small MLP to solve XOR.

use oxmera_core::Shape;
use oxmera_nn::{Linear, MSELoss, Module, Param, Sequential};
use oxmera_optim::{Adam, AdamW, Optimizer, RmsProp, Sgd};
use oxmera_tensor::tensor::Tensor;

/// Minimize `sum((x - c)^2)` from a fixed start; return final distance.
fn descend(make: impl Fn(Vec<Param>) -> Box<dyn Optimizer>, steps: usize) -> f32 {
    let target = Tensor::from_slice(&[3.0, -2.0, 0.5], [3]).unwrap();
    let x = Param::new(Tensor::zeros([3]));
    let mut opt = make(vec![x.clone()]);
    for _ in 0..steps {
        opt.zero_grad();
        let diff = x.value().sub(&target).unwrap();
        let loss = diff.mul(&diff).unwrap().sum(&[]).unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();
    }
    let end = x
        .value()
        .sub(&target)
        .unwrap()
        .abs()
        .unwrap()
        .max(&[])
        .unwrap();
    end.get_f32(&[]).unwrap()
}

#[test]
fn every_optimizer_minimizes_a_quadratic() {
    assert!(descend(|p| Box::new(Sgd::new(p, 0.1)), 200) < 1e-3, "sgd");
    assert!(
        descend(|p| Box::new(Sgd::with_config(p, 0.05, 0.9, 0.0)), 200) < 1e-3,
        "sgd+momentum"
    );
    assert!(descend(|p| Box::new(Adam::new(p, 0.1)), 300) < 1e-2, "adam");
    assert!(
        descend(|p| Box::new(AdamW::new(p, 0.1, 0.0)), 300) < 1e-2,
        "adamw"
    );
    assert!(
        descend(|p| Box::new(RmsProp::new(p, 0.05)), 400) < 1e-2,
        "rmsprop"
    );
}

#[test]
fn weight_decay_shrinks_toward_zero() {
    // With a zero-gradient loss, decoupled decay must shrink the weights.
    let x = Param::new(Tensor::ones([4]));
    let mut opt = AdamW::new(vec![x.clone()], 0.01, 0.5);
    for _ in 0..50 {
        opt.zero_grad();
        // Constant loss: gradient of x * 0 is zero everywhere.
        let loss = x.value().mul_scalar(0.0).unwrap().sum(&[]).unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();
    }
    let norm = x
        .value()
        .abs()
        .unwrap()
        .max(&[])
        .unwrap()
        .get_f32(&[])
        .unwrap();
    assert!(norm < 0.85, "decay had no effect: {norm}");
}

#[test]
fn adam_trains_an_mlp_to_solve_xor() {
    let model = Sequential::new()
        .push(Linear::new(2, 16, 1))
        .push(Tanh)
        .push(Linear::new(16, 1, 2));
    let x = Tensor::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2]).unwrap();
    let y = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], [4, 1]).unwrap();

    let mut opt = Adam::new(model.parameters(), 0.05);
    let loss_fn = MSELoss;
    let mut last = f32::INFINITY;
    for step in 0..400 {
        opt.zero_grad();
        let out = model.forward(&x).unwrap();
        let loss = loss_fn.forward(&out, &y).unwrap();
        last = loss.get_f32(&[]).unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();
        if last < 1e-4 {
            eprintln!("converged at step {step}: loss {last}");
            break;
        }
    }
    assert!(last < 1e-2, "XOR did not converge: final loss {last}");

    let pred = model.forward(&x).unwrap().to_vec_f32().unwrap();
    for (p, want) in pred.iter().zip([0.0, 1.0, 1.0, 0.0]) {
        assert!((p - want).abs() < 0.2, "prediction {p} vs {want}");
    }
}

/// A parameter-free activation module for test pipelines.
struct Tanh;

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> oxmera_core::Result<Tensor> {
        input.tanh()
    }
    fn parameters(&self) -> Vec<Param> {
        Vec::new()
    }
    fn named_parameters(&self, _prefix: &str) -> Vec<(String, Param)> {
        Vec::new()
    }
}

#[test]
fn optimizer_reports_missing_gradients() {
    let x = Param::new(Tensor::zeros(Shape::from([2])));
    let mut opt = Sgd::new(vec![x], 0.1);
    assert!(
        opt.step().is_err(),
        "step without backward must be a typed error"
    );
}
