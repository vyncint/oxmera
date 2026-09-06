//! Optimizers for oxmera: SGD (momentum, weight decay), Adam, AdamW, and
//! RMSprop, updating shared [`Param`] handles in place.
//!
//! Every optimizer takes its parameters as one or more [`ParamGroup`]s. A
//! group carries its own learning rate and weight decay, so a model can
//! decay its interaction weights harder than its biases, or freeze a
//! block by giving it a learning rate of zero, without a second optimizer.
//! The plain constructors (`AdamW::new(params, lr, wd)` and friends) are
//! the single-group case and behave exactly as before.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use oxmera_core::{Device, Error, Result};
use oxmera_nn::Param;
use oxmera_tensor::backend::{AdamStep, backend_for};
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

/// A set of parameters that share one learning rate and one weight decay.
///
/// Groups are the unit of hyper-parameter control: two groups with
/// different `weight_decay` values on the same optimizer are updated in one
/// `step()`, each with its own settings, and share the optimizer's global
/// state (Adam's bias-correction step count, for instance).
#[derive(Debug, Clone)]
pub struct ParamGroup {
    /// The parameters in this group.
    pub params: Vec<Param>,
    /// The learning rate applied to this group.
    pub lr: f32,
    /// The weight decay applied to this group (coupled L2 for SGD/Adam,
    /// decoupled for AdamW; ignored by RMSprop).
    pub weight_decay: f32,
}

impl ParamGroup {
    /// A group with an explicit learning rate and weight decay.
    pub fn new(params: Vec<Param>, lr: f32, weight_decay: f32) -> Self {
        Self {
            params,
            lr,
            weight_decay,
        }
    }

    /// A group with the given learning rate and no weight decay.
    pub fn with_lr(params: Vec<Param>, lr: f32) -> Self {
        Self::new(params, lr, 0.0)
    }
}

fn grad_of(param: &Param) -> Result<Tensor> {
    param.grad().ok_or(Error::InvalidArgument {
        op: "Optimizer::step",
        detail: "parameter has no gradient; run backward() first".into(),
    })
}

fn zero_all(groups: &[ParamGroup]) {
    for g in groups {
        for p in &g.params {
            p.zero_grad();
        }
    }
}

fn state_slots(groups: &[ParamGroup]) -> Vec<Vec<Option<Tensor>>> {
    groups.iter().map(|g| vec![None; g.params.len()]).collect()
}

/// Stochastic gradient descent with optional momentum and (coupled L2)
/// weight decay, per group.
pub struct Sgd {
    groups: Vec<ParamGroup>,
    momentum: f32,
    velocity: Vec<Vec<Option<Tensor>>>,
}

impl Sgd {
    /// Plain SGD: one group, no momentum, no weight decay.
    pub fn new(params: Vec<Param>, lr: f32) -> Self {
        Self::with_config(params, lr, 0.0, 0.0)
    }

    /// SGD with momentum and L2 weight decay, as one group.
    pub fn with_config(params: Vec<Param>, lr: f32, momentum: f32, weight_decay: f32) -> Self {
        Self::with_groups(vec![ParamGroup::new(params, lr, weight_decay)], momentum)
    }

    /// SGD over several parameter groups, each with its own learning rate
    /// and weight decay; `momentum` is shared.
    pub fn with_groups(groups: Vec<ParamGroup>, momentum: f32) -> Self {
        let velocity = state_slots(&groups);
        Self {
            groups,
            momentum,
            velocity,
        }
    }

    /// The parameter groups, for schedules that adjust `lr` between steps.
    pub fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }
}

impl Optimizer for Sgd {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for (gi, group) in self.groups.iter().enumerate() {
                for (i, param) in group.params.iter().enumerate() {
                    let value = param.value().detach();
                    let mut grad = grad_of(param)?;
                    if group.weight_decay != 0.0 {
                        grad = grad.add(&value.mul_scalar(group.weight_decay)?)?;
                    }
                    let update = if self.momentum != 0.0 {
                        let v = match &self.velocity[gi][i] {
                            Some(v) => v.mul_scalar(self.momentum)?.add(&grad)?,
                            None => grad.clone(),
                        };
                        self.velocity[gi][i] = Some(v.clone());
                        v
                    } else {
                        grad
                    };
                    param.set(value.sub(&update.mul_scalar(group.lr)?)?);
                }
            }
            Ok(())
        })
    }

    fn zero_grad(&self) {
        zero_all(&self.groups);
    }
}

/// Shared Adam machinery; `decoupled` selects AdamW's weight-decay
/// placement. The bias-correction step count is global — every group
/// advances together — while the learning rate and decay are per group.
struct AdamCore {
    groups: Vec<ParamGroup>,
    beta1: f32,
    beta2: f32,
    eps: f32,
    decoupled: bool,
    step: i32,
    m: Vec<Vec<Option<Tensor>>>,
    v: Vec<Vec<Option<Tensor>>>,
}

impl AdamCore {
    fn new(groups: Vec<ParamGroup>, decoupled: bool) -> Self {
        let m = state_slots(&groups);
        let v = state_slots(&groups);
        Self {
            groups,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            decoupled,
            step: 0,
            m,
            v,
        }
    }

    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            self.step += 1;
            let bc1 = 1.0 - self.beta1.powi(self.step);
            let bc2 = 1.0 - self.beta2.powi(self.step);
            for (gi, group) in self.groups.iter().enumerate() {
                for (i, param) in group.params.iter().enumerate() {
                    let mut value = param.value().detach();
                    let mut grad = grad_of(param)?;
                    // A GPU backend fuses the whole update into one launch;
                    // the composite path below is the reference it must match.
                    if value.device() != Device::Cpu {
                        let step = AdamStep {
                            param: &value,
                            grad: &grad,
                            m: self.m[gi][i].as_ref(),
                            v: self.v[gi][i].as_ref(),
                            lr: group.lr,
                            beta1: self.beta1,
                            beta2: self.beta2,
                            eps: self.eps,
                            weight_decay: group.weight_decay,
                            decoupled: self.decoupled,
                            bias_correction1: bc1,
                            bias_correction2: bc2,
                        };
                        match backend_for(value.device())?.adam_step(&step) {
                            Ok((p, m, v)) => {
                                self.m[gi][i] = Some(m);
                                self.v[gi][i] = Some(v);
                                param.set(p);
                                continue;
                            }
                            Err(Error::NotImplemented { .. }) => {}
                            Err(e) => return Err(e),
                        }
                    }
                    if group.weight_decay != 0.0 {
                        if self.decoupled {
                            // AdamW: decay applied to the weights directly.
                            value = value.mul_scalar(1.0 - group.lr * group.weight_decay)?;
                        } else {
                            grad = grad.add(&value.mul_scalar(group.weight_decay)?)?;
                        }
                    }
                    let m = match &self.m[gi][i] {
                        Some(m) => m
                            .mul_scalar(self.beta1)?
                            .add(&grad.mul_scalar(1.0 - self.beta1)?)?,
                        None => grad.mul_scalar(1.0 - self.beta1)?,
                    };
                    let g2 = grad.mul(&grad)?;
                    let v = match &self.v[gi][i] {
                        Some(v) => v
                            .mul_scalar(self.beta2)?
                            .add(&g2.mul_scalar(1.0 - self.beta2)?)?,
                        None => g2.mul_scalar(1.0 - self.beta2)?,
                    };
                    self.m[gi][i] = Some(m.clone());
                    self.v[gi][i] = Some(v.clone());
                    let m_hat = m.mul_scalar(1.0 / bc1)?;
                    let v_hat = v.mul_scalar(1.0 / bc2)?;
                    let update = m_hat.div(&v_hat.sqrt()?.add_scalar(self.eps)?)?;
                    param.set(value.sub(&update.mul_scalar(group.lr)?)?);
                }
            }
            Ok(())
        })
    }

    fn zero_grad(&self) {
        zero_all(&self.groups);
    }
}

/// Adam with the standard defaults (β₁ 0.9, β₂ 0.999, ε 1e-8) and coupled
/// L2 weight decay.
pub struct Adam(AdamCore);

impl Adam {
    /// Adam without weight decay, as one group.
    pub fn new(params: Vec<Param>, lr: f32) -> Self {
        Self::with_groups(vec![ParamGroup::with_lr(params, lr)])
    }

    /// Adam with coupled L2 weight decay, as one group.
    pub fn with_weight_decay(params: Vec<Param>, lr: f32, weight_decay: f32) -> Self {
        Self::with_groups(vec![ParamGroup::new(params, lr, weight_decay)])
    }

    /// Adam over several parameter groups, each with its own learning rate
    /// and (coupled) weight decay.
    pub fn with_groups(groups: Vec<ParamGroup>) -> Self {
        Self(AdamCore::new(groups, false))
    }

    /// The parameter groups, for schedules that adjust `lr` between steps.
    pub fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.0.groups
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
    /// AdamW with the standard defaults, as one group.
    pub fn new(params: Vec<Param>, lr: f32, weight_decay: f32) -> Self {
        Self::with_groups(vec![ParamGroup::new(params, lr, weight_decay)])
    }

    /// AdamW over several parameter groups, each with its own learning
    /// rate and decoupled weight decay.
    pub fn with_groups(groups: Vec<ParamGroup>) -> Self {
        Self(AdamCore::new(groups, true))
    }

    /// The parameter groups, for schedules that adjust `lr` between steps.
    pub fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.0.groups
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

/// RMSprop with the standard defaults (α 0.99, ε 1e-8). Weight decay on a
/// group is ignored: RMSprop here is the classic, decay-free rule.
pub struct RmsProp {
    groups: Vec<ParamGroup>,
    alpha: f32,
    eps: f32,
    sq: Vec<Vec<Option<Tensor>>>,
}

impl RmsProp {
    /// RMSprop with smoothing constant α = 0.99, as one group.
    pub fn new(params: Vec<Param>, lr: f32) -> Self {
        Self::with_groups(vec![ParamGroup::with_lr(params, lr)])
    }

    /// RMSprop over several parameter groups, each with its own learning
    /// rate.
    pub fn with_groups(groups: Vec<ParamGroup>) -> Self {
        let sq = state_slots(&groups);
        Self {
            groups,
            alpha: 0.99,
            eps: 1e-8,
            sq,
        }
    }

    /// The parameter groups, for schedules that adjust `lr` between steps.
    pub fn groups_mut(&mut self) -> &mut [ParamGroup] {
        &mut self.groups
    }
}

impl Optimizer for RmsProp {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for (gi, group) in self.groups.iter().enumerate() {
                for (i, param) in group.params.iter().enumerate() {
                    let value = param.value().detach();
                    let grad = grad_of(param)?;
                    let g2 = grad.mul(&grad)?;
                    let sq = match &self.sq[gi][i] {
                        Some(s) => s
                            .mul_scalar(self.alpha)?
                            .add(&g2.mul_scalar(1.0 - self.alpha)?)?,
                        None => g2.mul_scalar(1.0 - self.alpha)?,
                    };
                    self.sq[gi][i] = Some(sq.clone());
                    let update = grad.div(&sq.sqrt()?.add_scalar(self.eps)?)?;
                    param.set(value.sub(&update.mul_scalar(group.lr)?)?);
                }
            }
            Ok(())
        })
    }

    fn zero_grad(&self) {
        zero_all(&self.groups);
    }
}

/// What this crate can do, for `oxmera doctor`. See
/// [`oxmera_tensor::CAPABILITIES`] for why the list lives beside the code.
pub const CAPABILITIES: &[(&str, &str)] = &[(
    "optim",
    "SGD Adam AdamW RMSprop, per-group lr/decay, fused GPU step",
)];
