//! Optimizers for oxmera: SGD (momentum, weight decay), Adam, AdamW, and
//! RMSprop, updating shared [`Param`] handles in place.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use oxmera_core::{Error, Result};
use oxmera_nn::Param;
use oxmera_tensor::no_grad;
use oxmera_tensor::tensor::Tensor;

/// A gradient-based parameter updater.
pub trait Optimizer {
    /// Apply one update from the gradients currently accumulated on the
    /// parameters.
    fn step(&mut self) -> Result<()>;

    /// Clear every parameter's accumulated gradient.
    fn zero_grad(&self);
}

fn grad_of(param: &Param) -> Result<Tensor> {
    param.grad().ok_or(Error::InvalidArgument {
        op: "Optimizer::step",
        detail: "parameter has no gradient; run backward() first".into(),
    })
}

/// Stochastic gradient descent with optional momentum and decoupled
/// weight decay.
pub struct Sgd {
    params: Vec<Param>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    velocity: Vec<Option<Tensor>>,
}

impl Sgd {
    /// Plain SGD.
    pub fn new(params: Vec<Param>, lr: f32) -> Self {
        Self::with_config(params, lr, 0.0, 0.0)
    }

    /// SGD with momentum and L2 weight decay.
    pub fn with_config(params: Vec<Param>, lr: f32, momentum: f32, weight_decay: f32) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            momentum,
            weight_decay,
            velocity: vec![None; n],
        }
    }
}

impl Optimizer for Sgd {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for (i, param) in self.params.iter().enumerate() {
                let value = param.value().detach();
                let mut grad = grad_of(param)?;
                if self.weight_decay != 0.0 {
                    grad = grad.add(&value.mul_scalar(self.weight_decay)?)?;
                }
                let update = if self.momentum != 0.0 {
                    let v = match &self.velocity[i] {
                        Some(v) => v.mul_scalar(self.momentum)?.add(&grad)?,
                        None => grad.clone(),
                    };
                    self.velocity[i] = Some(v.clone());
                    v
                } else {
                    grad
                };
                param.set(value.sub(&update.mul_scalar(self.lr)?)?);
            }
            Ok(())
        })
    }

    fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

/// Shared Adam machinery; `decoupled` selects AdamW's weight-decay
/// placement.
struct AdamCore {
    params: Vec<Param>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    decoupled: bool,
    step: i32,
    m: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
}

impl AdamCore {
    fn new(params: Vec<Param>, lr: f32, weight_decay: f32, decoupled: bool) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            decoupled,
            step: 0,
            m: vec![None; n],
            v: vec![None; n],
        }
    }

    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            self.step += 1;
            let bc1 = 1.0 - self.beta1.powi(self.step);
            let bc2 = 1.0 - self.beta2.powi(self.step);
            for (i, param) in self.params.iter().enumerate() {
                let mut value = param.value().detach();
                let mut grad = grad_of(param)?;
                if self.weight_decay != 0.0 {
                    if self.decoupled {
                        // AdamW: decay applied to the weights directly.
                        value = value.mul_scalar(1.0 - self.lr * self.weight_decay)?;
                    } else {
                        grad = grad.add(&value.mul_scalar(self.weight_decay)?)?;
                    }
                }
                let m = match &self.m[i] {
                    Some(m) => m
                        .mul_scalar(self.beta1)?
                        .add(&grad.mul_scalar(1.0 - self.beta1)?)?,
                    None => grad.mul_scalar(1.0 - self.beta1)?,
                };
                let g2 = grad.mul(&grad)?;
                let v = match &self.v[i] {
                    Some(v) => v
                        .mul_scalar(self.beta2)?
                        .add(&g2.mul_scalar(1.0 - self.beta2)?)?,
                    None => g2.mul_scalar(1.0 - self.beta2)?,
                };
                self.m[i] = Some(m.clone());
                self.v[i] = Some(v.clone());
                let m_hat = m.mul_scalar(1.0 / bc1)?;
                let v_hat = v.mul_scalar(1.0 / bc2)?;
                let update = m_hat.div(&v_hat.sqrt()?.add_scalar(self.eps)?)?;
                param.set(value.sub(&update.mul_scalar(self.lr)?)?);
            }
            Ok(())
        })
    }

    fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

/// Adam with the standard defaults (β₁ 0.9, β₂ 0.999, ε 1e-8) and coupled
/// L2 weight decay.
pub struct Adam(AdamCore);

impl Adam {
    /// Adam without weight decay.
    pub fn new(params: Vec<Param>, lr: f32) -> Self {
        Self(AdamCore::new(params, lr, 0.0, false))
    }

    /// Adam with coupled L2 weight decay.
    pub fn with_weight_decay(params: Vec<Param>, lr: f32, weight_decay: f32) -> Self {
        Self(AdamCore::new(params, lr, weight_decay, false))
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> Result<()> {
        self.0.step()
    }
    fn zero_grad(&self) {
        self.0.zero_grad()
    }
}

/// AdamW: Adam with decoupled weight decay.
pub struct AdamW(AdamCore);

impl AdamW {
    /// AdamW with the standard defaults.
    pub fn new(params: Vec<Param>, lr: f32, weight_decay: f32) -> Self {
        Self(AdamCore::new(params, lr, weight_decay, true))
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) -> Result<()> {
        self.0.step()
    }
    fn zero_grad(&self) {
        self.0.zero_grad()
    }
}

/// RMSprop with the standard defaults (α 0.99, ε 1e-8).
pub struct RmsProp {
    params: Vec<Param>,
    lr: f32,
    alpha: f32,
    eps: f32,
    sq: Vec<Option<Tensor>>,
}

impl RmsProp {
    /// RMSprop with smoothing constant α = 0.99.
    pub fn new(params: Vec<Param>, lr: f32) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            alpha: 0.99,
            eps: 1e-8,
            sq: vec![None; n],
        }
    }
}

impl Optimizer for RmsProp {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for (i, param) in self.params.iter().enumerate() {
                let value = param.value().detach();
                let grad = grad_of(param)?;
                let g2 = grad.mul(&grad)?;
                let sq = match &self.sq[i] {
                    Some(s) => s
                        .mul_scalar(self.alpha)?
                        .add(&g2.mul_scalar(1.0 - self.alpha)?)?,
                    None => g2.mul_scalar(1.0 - self.alpha)?,
                };
                self.sq[i] = Some(sq.clone());
                let update = grad.div(&sq.sqrt()?.add_scalar(self.eps)?)?;
                param.set(value.sub(&update.mul_scalar(self.lr)?)?);
            }
            Ok(())
        })
    }

    fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}
