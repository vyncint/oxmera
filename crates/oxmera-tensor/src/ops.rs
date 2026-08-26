//! The differentiable operation layer: every method dispatches to the
//! device backend for the forward pass and, when recording is on and an
//! input is tracked, attaches the exact vector-Jacobian product to the
//! output's tape node.

use oxmera_core::{Device, Error, Result, Shape};

use crate::autograd::{GradFn, is_recording};
use crate::backend::{Backend, BinaryOp, ReduceOp, UnaryOp, backend_for};
use crate::tensor::{Tensor, ViewKind};

use std::sync::Arc;

fn same_device(a: &Tensor, b: &Tensor, op: &'static str) -> Result<Device> {
    if a.device() != b.device() {
        return Err(Error::DeviceMismatch {
            lhs: a.device(),
            rhs: b.device(),
            op,
        });
    }
    Ok(a.device())
}

fn record(
    out: Tensor,
    inputs: Vec<Tensor>,
    vjp: impl Fn(&Tensor) -> Result<Vec<Option<Tensor>>> + Send + Sync + 'static,
) -> Tensor {
    if is_recording() && inputs.iter().any(Tensor::is_tracked) {
        out.with_grad_fn(GradFn {
            inputs,
            vjp: Box::new(vjp),
        })
    } else {
        out
    }
}

/// Sum `grad` down to `shape` (undo broadcasting): reduce the leading
/// extra axes and every axis the target holds as 1, then reshape.
pub(crate) fn reduce_to_shape(grad: &Tensor, shape: &Shape) -> Result<Tensor> {
    if grad.shape() == shape {
        return Ok(grad.clone());
    }
    let gdims = grad.dims().to_vec();
    let tdims = shape.dims();
    let lead = gdims.len() - tdims.len();
    let mut axes: Vec<usize> = (0..lead).collect();
    for (i, &td) in tdims.iter().enumerate() {
        if td == 1 && gdims[lead + i] != 1 {
            axes.push(lead + i);
        }
    }
    let reduced = if axes.is_empty() {
        grad.clone()
    } else {
        grad.sum_keepdim(&axes, true)?
    };
    reduced.reshape(shape.clone())
}

/// Attach the view VJP to a freshly built view (called from `tensor.rs`).
pub(crate) fn record_view(input: &Tensor, out: Tensor, kind: ViewKind) -> Tensor {
    let in_shape = input.shape().clone();
    record(out, vec![input.clone()], move |g| {
        let gi = match &kind {
            ViewKind::Reshape | ViewKind::Contiguous => g.reshape(in_shape.clone())?,
            ViewKind::Permute(perm) => {
                let mut inverse = vec![0usize; perm.len()];
                for (i, &p) in perm.iter().enumerate() {
                    inverse[p] = i;
                }
                g.permute(&inverse)?
            }
            ViewKind::Narrow { dim, start, len } => {
                let indices: Vec<i64> = (*start..start + len).map(|i| i as i64).collect();
                let indices = Tensor::from_vec_i64(indices, Shape::from([*len]))?;
                Tensor::zeros(in_shape.clone())
                    .to_device(g.device())?
                    .index_add(*dim, &indices, g)?
            }
            ViewKind::Broadcast => reduce_to_shape(g, &in_shape)?,
        };
        Ok(vec![Some(gi)])
    })
}

impl Tensor {
    fn backend(&self) -> Result<Arc<dyn Backend>> {
        backend_for(self.device())
    }

    // ---- unary -----------------------------------------------------------

    fn unary_op(&self, op: UnaryOp) -> Result<Tensor> {
        let out = self.backend()?.unary(op, self)?;
        let a = self.clone();
        let o = out.clone();
        Ok(record(out, vec![self.clone()], move |g| {
            let gi = match op {
                UnaryOp::Neg => g.neg()?,
                UnaryOp::Exp => g.mul(&o)?,
                UnaryOp::Ln => g.div(&a)?,
                UnaryOp::Abs => {
                    let sign = a
                        .gt_mask(&Tensor::scalar_on(&a, 0.0)?)?
                        .sub(&Tensor::scalar_on(&a, 0.0)?.gt_mask(&a)?)?;
                    g.mul(&sign)?
                }
                UnaryOp::Sqrt => g.mul(&Tensor::scalar_on(&a, 0.5)?)?.div(&o)?,
                UnaryOp::Sin => g.mul(&a.cos()?)?,
                UnaryOp::Cos => g.mul(&a.sin()?.neg()?)?,
                UnaryOp::Tanh => {
                    let one = Tensor::scalar_on(&a, 1.0)?;
                    g.mul(&one.sub(&o.mul(&o)?)?)?
                }
                UnaryOp::Relu => g.mul(&a.gt_mask(&Tensor::scalar_on(&a, 0.0)?)?)?,
                UnaryOp::Gelu => {
                    // d/dx [0.5x(1+tanh(u))], u = c(x + 0.044715 x^3),
                    // c = sqrt(2/pi).
                    let c = Tensor::scalar_on(&a, 0.797_884_6)?;
                    let k = Tensor::scalar_on(&a, 0.044_715)?;
                    let one = Tensor::scalar_on(&a, 1.0)?;
                    let half = Tensor::scalar_on(&a, 0.5)?;
                    let three_k = Tensor::scalar_on(&a, 3.0 * 0.044_715)?;
                    let x2 = a.mul(&a)?;
                    let u = c.mul(&a.add(&k.mul(&x2.mul(&a)?)?)?)?;
                    let t = u.tanh()?;
                    let sech2 = one.sub(&t.mul(&t)?)?;
                    let du = c.mul(&one.add(&three_k.mul(&x2)?)?)?;
                    let d = half
                        .mul(&one.add(&t)?)?
                        .add(&half.mul(&a)?.mul(&sech2)?.mul(&du)?)?;
                    g.mul(&d)?
                }
                UnaryOp::Sigmoid => {
                    let one = Tensor::scalar_on(&a, 1.0)?;
                    g.mul(&o)?.mul(&one.sub(&o)?)?
                }
            };
            Ok(vec![Some(gi)])
        }))
    }

    /// Elementwise negation.
    pub fn neg(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Neg)
    }
    /// Elementwise `e^x`.
    pub fn exp(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Exp)
    }
    /// Elementwise natural logarithm.
    pub fn ln(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Ln)
    }
    /// Elementwise absolute value.
    pub fn abs(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Abs)
    }
    /// Elementwise square root.
    pub fn sqrt(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Sqrt)
    }
    /// Elementwise sine.
    pub fn sin(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Sin)
    }
    /// Elementwise cosine.
    pub fn cos(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Cos)
    }
    /// Elementwise hyperbolic tangent.
    pub fn tanh(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Tanh)
    }
    /// Elementwise rectified linear unit.
    pub fn relu(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Relu)
    }
    /// Elementwise GELU (tanh approximation).
    pub fn gelu(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Gelu)
    }
    /// Elementwise logistic sigmoid.
    pub fn sigmoid(&self) -> Result<Tensor> {
        self.unary_op(UnaryOp::Sigmoid)
    }

    /// A scalar constant on the same device as `like` (plumbing for VJPs
    /// and scalar operator overloads).
    pub fn scalar_on(like: &Tensor, value: f32) -> Result<Tensor> {
        Tensor::scalar(value).to_device(like.device())
    }

    // ---- binary ----------------------------------------------------------

    fn binary_op(&self, op: BinaryOp, rhs: &Tensor) -> Result<Tensor> {
        let device = same_device(self, rhs, "binary")?;
        let out = backend_for(device)?.binary(op, self, rhs)?;
        let (a, b) = (self.clone(), rhs.clone());
        let o = out.clone();
        Ok(record(out, vec![self.clone(), rhs.clone()], move |g| {
            let (ga, gb): (Option<Tensor>, Option<Tensor>) = match op {
                BinaryOp::Add => (Some(g.clone()), Some(g.clone())),
                BinaryOp::Sub => (Some(g.clone()), Some(g.neg()?)),
                BinaryOp::Mul => (Some(g.mul_raw(&b)?), Some(g.mul_raw(&a)?)),
                BinaryOp::Div => {
                    let ga = g.div_raw(&b)?;
                    let gb = g.mul_raw(&o)?.div_raw(&b)?.neg()?;
                    (Some(ga), Some(gb))
                }
                BinaryOp::Pow => {
                    let one = Tensor::scalar_on(&a, 1.0)?;
                    let ga = g.mul_raw(&b)?.mul_raw(&a.pow(&b.sub(&one)?)?)?;
                    let gb = g.mul_raw(&o)?.mul_raw(&a.ln()?)?;
                    (Some(ga), Some(gb))
                }
                BinaryOp::Maximum => {
                    let mask = a.gt_mask(&b)?;
                    let one = Tensor::scalar_on(&a, 1.0)?;
                    let ga = g.mul_raw(&mask)?;
                    let gb = g.mul_raw(&one.sub(&mask)?)?;
                    (Some(ga), Some(gb))
                }
                BinaryOp::Minimum => {
                    let mask = b.gt_mask(&a)?;
                    let one = Tensor::scalar_on(&a, 1.0)?;
                    let ga = g.mul_raw(&mask)?;
                    let gb = g.mul_raw(&one.sub(&mask)?)?;
                    (Some(ga), Some(gb))
                }
                BinaryOp::Gt | BinaryOp::Eq => (None, None),
            };
            let ga = match ga {
                Some(t) => Some(reduce_to_shape(&t, a.shape())?),
                None => None,
            };
            let gb = match gb {
                Some(t) => Some(reduce_to_shape(&t, b.shape())?),
                None => None,
            };
            Ok(vec![ga, gb])
        }))
    }

    /// Untracked multiply, for use inside VJP closures (recording is
    /// already off during backward; this is belt and braces).
    fn mul_raw(&self, rhs: &Tensor) -> Result<Tensor> {
        let device = same_device(self, rhs, "mul")?;
        backend_for(device)?.binary(BinaryOp::Mul, self, rhs)
    }

    fn div_raw(&self, rhs: &Tensor) -> Result<Tensor> {
        let device = same_device(self, rhs, "div")?;
        backend_for(device)?.binary(BinaryOp::Div, self, rhs)
    }

    /// Elementwise addition, broadcasting.
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(BinaryOp::Add, rhs)
    }
    /// Elementwise subtraction, broadcasting.
    pub fn sub(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(BinaryOp::Sub, rhs)
    }
    /// Elementwise multiplication, broadcasting.
    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(BinaryOp::Mul, rhs)
    }
    /// Elementwise division, broadcasting.
    pub fn div(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(BinaryOp::Div, rhs)
    }
    /// Elementwise power, broadcasting.
    pub fn pow(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(BinaryOp::Pow, rhs)
    }
    /// Elementwise maximum, broadcasting.
    pub fn maximum(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(BinaryOp::Maximum, rhs)
    }
    /// Elementwise minimum, broadcasting.
    pub fn minimum(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(BinaryOp::Minimum, rhs)
    }
    /// Elementwise `a > b` as a 0.0/1.0 mask. Not differentiable.
    pub fn gt_mask(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(BinaryOp::Gt, rhs)
    }
    /// Elementwise `a == b` as a 0.0/1.0 mask. Not differentiable.
    pub fn eq_mask(&self, rhs: &Tensor) -> Result<Tensor> {
        self.binary_op(BinaryOp::Eq, rhs)
    }

    /// Add a scalar, broadcasting.
    pub fn add_scalar(&self, s: f32) -> Result<Tensor> {
        self.add(&Tensor::scalar_on(self, s)?)
    }
    /// Multiply by a scalar, broadcasting.
    pub fn mul_scalar(&self, s: f32) -> Result<Tensor> {
        self.mul(&Tensor::scalar_on(self, s)?)
    }

    // ---- matmul ----------------------------------------------------------

    /// Matrix product: rank-2 `[m, k] x [k, n]`, or batched rank-3
    /// `[b, m, k] x [b, k, n]`.
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        let device = same_device(self, rhs, "matmul")?;
        let out = backend_for(device)?.matmul(self, rhs)?;
        let (a, b) = (self.clone(), rhs.clone());
        Ok(record(out, vec![self.clone(), rhs.clone()], move |g| {
            let ga = backend_for(g.device())?.matmul(g, &b.t()?)?;
            let gb = backend_for(g.device())?.matmul(&a.t()?, g)?;
            Ok(vec![Some(ga), Some(gb)])
        }))
    }

    // ---- reductions --------------------------------------------------------

    fn reduce_op(&self, op: ReduceOp, axes: &[usize], keepdim: bool) -> Result<Tensor> {
        let axes = normalize_axes(axes, self.ndim(), "reduce")?;
        let out = self.backend()?.reduce(op, self, &axes, keepdim)?;
        let a = self.clone();
        let o = out.clone();
        let axes_c = axes.clone();
        Ok(record(out, vec![self.clone()], move |g| {
            // Re-insert reduced axes as size 1 so broadcasting lines up.
            let g_keep = if keepdim {
                g.clone()
            } else {
                unsqueeze_axes(g, &axes_c)?
            };
            let gi = match op {
                ReduceOp::Sum => g_keep.broadcast_to(a.shape().clone())?.contiguous()?,
                ReduceOp::Max | ReduceOp::Min => {
                    let o_keep = if keepdim {
                        o.clone()
                    } else {
                        unsqueeze_axes(&o, &axes_c)?
                    };
                    let mask = a.eq_mask(&o_keep.broadcast_to(a.shape().clone())?)?;
                    let count = mask.sum_keepdim(&axes_c, true)?;
                    g_keep
                        .broadcast_to(a.shape().clone())?
                        .mul_raw(&mask)?
                        .div_raw(&count.broadcast_to(a.shape().clone())?.contiguous()?)?
                }
            };
            Ok(vec![Some(gi)])
        }))
    }

    /// Sum over `axes` (empty means all), removing them from the shape.
    pub fn sum(&self, axes: &[usize]) -> Result<Tensor> {
        self.reduce_op(ReduceOp::Sum, axes, false)
    }

    /// Sum over `axes` with explicit `keepdim`.
    pub fn sum_keepdim(&self, axes: &[usize], keepdim: bool) -> Result<Tensor> {
        self.reduce_op(ReduceOp::Sum, axes, keepdim)
    }

    /// Maximum over `axes` (empty means all).
    pub fn max(&self, axes: &[usize]) -> Result<Tensor> {
        self.reduce_op(ReduceOp::Max, axes, false)
    }

    /// Maximum over `axes` with explicit `keepdim`.
    pub fn max_keepdim(&self, axes: &[usize], keepdim: bool) -> Result<Tensor> {
        self.reduce_op(ReduceOp::Max, axes, keepdim)
    }

    /// Minimum over `axes` (empty means all).
    pub fn min(&self, axes: &[usize]) -> Result<Tensor> {
        self.reduce_op(ReduceOp::Min, axes, false)
    }

    /// Mean over `axes` (empty means all) — composite, so its gradient
    /// flows through `sum` and scalar multiply.
    pub fn mean(&self, axes: &[usize]) -> Result<Tensor> {
        self.mean_keepdim(axes, false)
    }

    /// Mean over `axes` with explicit `keepdim`.
    pub fn mean_keepdim(&self, axes: &[usize], keepdim: bool) -> Result<Tensor> {
        let axes_n = normalize_axes(axes, self.ndim(), "mean")?;
        let n: usize = axes_n.iter().map(|&ax| self.dims()[ax]).product();
        self.sum_keepdim(&axes_n, keepdim)?
            .mul_scalar(1.0 / n as f32)
    }

    /// Index of the maximum along `dim`, as an `I64` tensor. Not
    /// differentiable.
    pub fn argmax(&self, dim: usize, keepdim: bool) -> Result<Tensor> {
        if dim >= self.ndim() {
            return Err(Error::InvalidArgument {
                op: "argmax",
                detail: format!("dim {dim} out of range for rank {}", self.ndim()),
            });
        }
        self.backend()?.argmax(self, dim, keepdim)
    }

    /// Numerically stable softmax along `dim` — composite.
    pub fn softmax(&self, dim: usize) -> Result<Tensor> {
        let shifted = self.sub(
            &self
                .max_keepdim(&[dim], true)?
                .detach()
                .broadcast_to(self.shape().clone())?
                .contiguous()?,
        )?;
        let e = shifted.exp()?;
        let denom = e.sum_keepdim(&[dim], true)?;
        e.div(&denom.broadcast_to(self.shape().clone())?.contiguous()?)
    }

    /// Numerically stable log-softmax along `dim` — composite.
    pub fn log_softmax(&self, dim: usize) -> Result<Tensor> {
        let shifted = self.sub(
            &self
                .max_keepdim(&[dim], true)?
                .detach()
                .broadcast_to(self.shape().clone())?
                .contiguous()?,
        )?;
        let lse = shifted.exp()?.sum_keepdim(&[dim], true)?.ln()?;
        shifted.sub(&lse.broadcast_to(self.shape().clone())?.contiguous()?)
    }

    // ---- indexing -----------------------------------------------------------

    /// Rows of `self` along `dim` selected by `indices` (`I64`).
    pub fn index_select(&self, dim: usize, indices: &Tensor) -> Result<Tensor> {
        let out = dispatch_index(self, |be, t| be.index_select(t, dim, indices))?;
        let in_shape = self.shape().clone();
        let idx = indices.clone();
        Ok(record(out, vec![self.clone()], move |g| {
            let zeros = Tensor::zeros(in_shape.clone()).to_device(g.device())?;
            Ok(vec![Some(zeros.index_add(dim, &idx, g)?)])
        }))
    }

    /// `out[indices[i]] += src[i]` along `dim`, on a fresh copy of `self`.
    pub fn index_add(&self, dim: usize, indices: &Tensor, src: &Tensor) -> Result<Tensor> {
        let out = dispatch_index(self, |be, t| be.index_add(t, dim, indices, src))?;
        let idx = indices.clone();
        Ok(record(out, vec![self.clone(), src.clone()], move |g| {
            Ok(vec![Some(g.clone()), Some(g.index_select(dim, &idx)?)])
        }))
    }

    // ---- device movement ------------------------------------------------------

    /// This tensor's data on `device` (a cheap clone when already there).
    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        if self.device() == device {
            return Ok(self.clone());
        }
        let out = match (self.device(), device) {
            (Device::Cpu, target) => backend_for(target)?.upload(&self.contiguous_data()?)?,
            (_, Device::Cpu) => self.backend()?.download(self)?,
            (_, target) => {
                let host = self.backend()?.download(self)?;
                backend_for(target)?.upload(&host)?
            }
        };
        let source = self.device();
        Ok(record(out, vec![self.clone()], move |g| {
            Ok(vec![Some(g.to_device(source)?)])
        }))
    }
}

/// Insert size-1 axes at `axes` (sorted ascending) — plumbing for reduce
/// VJPs.
fn unsqueeze_axes(t: &Tensor, axes: &[usize]) -> Result<Tensor> {
    let mut out = t.clone();
    let mut sorted = axes.to_vec();
    sorted.sort_unstable();
    for &ax in &sorted {
        out = out.unsqueeze(ax)?;
    }
    Ok(out)
}

/// Validate and canonicalize reduce axes; empty means all axes.
fn normalize_axes(axes: &[usize], ndim: usize, op: &'static str) -> Result<Vec<usize>> {
    let mut axes: Vec<usize> = if axes.is_empty() {
        (0..ndim).collect()
    } else {
        axes.to_vec()
    };
    axes.sort_unstable();
    axes.dedup();
    if let Some(&bad) = axes.iter().find(|&&a| a >= ndim) {
        return Err(Error::InvalidArgument {
            op,
            detail: format!("axis {bad} out of range for rank {ndim}"),
        });
    }
    Ok(axes)
}

/// Run an index op on the tensor's backend, falling back to a CPU
/// round-trip when the backend declines.
fn dispatch_index(
    t: &Tensor,
    f: impl Fn(&dyn Backend, &Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    let backend = backend_for(t.device())?;
    match f(backend.as_ref(), t) {
        Err(Error::NotImplemented { .. }) if t.device() != Device::Cpu => {
            let cpu = backend.download(t)?;
            let cpu_backend = backend_for(Device::Cpu)?;
            let out = f(cpu_backend.as_ref(), &cpu)?;
            backend_for(t.device())?.upload(&out)
        }
        other => other,
    }
}
