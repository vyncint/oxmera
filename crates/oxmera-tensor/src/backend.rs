//! The backend seam: the op vocabulary every device implements, and the
//! registry that resolves a [`Device`] handle to an implementation.
//!
//! The traits live here (rather than a separate crate) so that `Tensor`'s
//! methods and `std::ops` overloads can dispatch through them without
//! violating the orphan rule; `oxmera-ops` re-exports this vocabulary.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use oxmera_core::{Device, Error, Result};

use crate::tensor::Tensor;

/// Elementwise unary operations. Float (`f32`) tensors only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnaryOp {
    /// `-x`
    Neg,
    /// `e^x`
    Exp,
    /// `ln(x)`
    Ln,
    /// `|x|`
    Abs,
    /// `√x`
    Sqrt,
    /// `sin(x)`
    Sin,
    /// `cos(x)`
    Cos,
    /// `tanh(x)`
    Tanh,
    /// `max(x, 0)`
    Relu,
    /// GELU with the tanh approximation.
    Gelu,
    /// `1 / (1 + e^-x)`
    Sigmoid,
}

impl UnaryOp {
    /// Stable lowercase name, used for kernel lookup and error messages.
    pub fn name(self) -> &'static str {
        match self {
            UnaryOp::Neg => "neg",
            UnaryOp::Exp => "exp",
            UnaryOp::Ln => "ln",
            UnaryOp::Abs => "abs",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Relu => "relu",
            UnaryOp::Gelu => "gelu",
            UnaryOp::Sigmoid => "sigmoid",
        }
    }

    /// Every unary op, for exhaustive backend tests.
    pub fn all() -> &'static [UnaryOp] {
        &[
            UnaryOp::Neg,
            UnaryOp::Exp,
            UnaryOp::Ln,
            UnaryOp::Abs,
            UnaryOp::Sqrt,
            UnaryOp::Sin,
            UnaryOp::Cos,
            UnaryOp::Tanh,
            UnaryOp::Relu,
            UnaryOp::Gelu,
            UnaryOp::Sigmoid,
        ]
    }

    /// Apply the op to one scalar — the CPU reference semantics every
    /// backend must reproduce.
    pub fn eval(self, x: f32) -> f32 {
        match self {
            UnaryOp::Neg => -x,
            UnaryOp::Exp => x.exp(),
            UnaryOp::Ln => x.ln(),
            UnaryOp::Abs => x.abs(),
            UnaryOp::Sqrt => x.sqrt(),
            UnaryOp::Sin => x.sin(),
            UnaryOp::Cos => x.cos(),
            UnaryOp::Tanh => x.tanh(),
            UnaryOp::Relu => x.max(0.0),
            UnaryOp::Gelu => {
                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
            }
            UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }
}

/// Elementwise binary operations with NumPy broadcasting. `f32` only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BinaryOp {
    /// `a + b`
    Add,
    /// `a - b`
    Sub,
    /// `a * b`
    Mul,
    /// `a / b`
    Div,
    /// `a ^ b`
    Pow,
    /// `max(a, b)`
    Maximum,
    /// `min(a, b)`
    Minimum,
    /// `a > b` as a 0.0/1.0 mask. Not differentiable.
    Gt,
    /// `a == b` as a 0.0/1.0 mask. Not differentiable.
    Eq,
}

impl BinaryOp {
    /// Stable lowercase name, used for kernel lookup and error messages.
    pub fn name(self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Pow => "pow",
            BinaryOp::Maximum => "maximum",
            BinaryOp::Minimum => "minimum",
            BinaryOp::Gt => "gt",
            BinaryOp::Eq => "eq",
        }
    }

    /// Every binary op, for exhaustive backend tests.
    pub fn all() -> &'static [BinaryOp] {
        &[
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Pow,
            BinaryOp::Maximum,
            BinaryOp::Minimum,
            BinaryOp::Gt,
            BinaryOp::Eq,
        ]
    }

    /// Apply the op to one scalar pair — the reference semantics.
    pub fn eval(self, a: f32, b: f32) -> f32 {
        match self {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Pow => a.powf(b),
            BinaryOp::Maximum => a.max(b),
            BinaryOp::Minimum => a.min(b),
            BinaryOp::Gt => f32::from(a > b),
            BinaryOp::Eq => f32::from(a == b),
        }
    }
}

/// Reductions along axes. `f32` only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReduceOp {
    /// Sum of the reduced elements.
    Sum,
    /// Maximum of the reduced elements.
    Max,
    /// Minimum of the reduced elements.
    Min,
}

impl ReduceOp {
    /// Stable lowercase name, used for kernel lookup and error messages.
    pub fn name(self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
        }
    }

    /// The identity element the reduction starts from.
    pub fn identity(self) -> f32 {
        match self {
            ReduceOp::Sum => 0.0,
            ReduceOp::Max => f32::NEG_INFINITY,
            ReduceOp::Min => f32::INFINITY,
        }
    }

    /// Combine an accumulator with one element.
    pub fn combine(self, acc: f32, x: f32) -> f32 {
        match self {
            ReduceOp::Sum => acc + x,
            ReduceOp::Max => acc.max(x),
            ReduceOp::Min => acc.min(x),
        }
    }
}

/// A complete backend: every primitive the tensor method layer dispatches.
///
/// Composite operations (mean, softmax, losses, convolution, …) are built
/// from these primitives device-generically; only what is listed here is
/// implemented per device.
pub trait Backend: Send + Sync {
    /// The device this backend serves.
    fn device(&self) -> Device;

    /// A short stable name for reports and `oxmera doctor`.
    fn name(&self) -> &'static str;

    /// Elementwise unary op over a (possibly strided) `f32` tensor,
    /// producing a fresh contiguous tensor of the same shape.
    fn unary(&self, op: UnaryOp, a: &Tensor) -> Result<Tensor>;

    /// Elementwise binary op with broadcasting, producing a fresh
    /// contiguous tensor of the broadcast shape.
    fn binary(&self, op: BinaryOp, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Matrix product: rank-2 `[m, k] x [k, n] -> [m, n]`, or batched
    /// rank-3 `[b, m, k] x [b, k, n] -> [b, m, n]`.
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Reduce over `axes` (sorted, deduplicated by the caller; empty means
    /// all axes). `keepdim` keeps reduced axes as size 1.
    fn reduce(&self, op: ReduceOp, a: &Tensor, axes: &[usize], keepdim: bool) -> Result<Tensor>;

    /// Index of the maximum along `dim`, as an `I64` tensor.
    fn argmax(&self, a: &Tensor, dim: usize, keepdim: bool) -> Result<Tensor>;

    /// A fresh contiguous tensor with the same logical elements.
    fn contiguous(&self, a: &Tensor) -> Result<Tensor>;

    /// Download to a contiguous CPU tensor.
    fn download(&self, a: &Tensor) -> Result<Tensor>;

    /// Upload a contiguous CPU tensor to this backend's device.
    fn upload(&self, a: &Tensor) -> Result<Tensor>;

    /// Rows of `a` along `dim` selected by `indices` (`I64`).
    ///
    /// Backends may return [`Error::NotImplemented`]; the method layer
    /// then falls back to the CPU backend with a device round-trip.
    fn index_select(&self, a: &Tensor, dim: usize, indices: &Tensor) -> Result<Tensor> {
        let _ = (dim, indices);
        Err(Error::NotImplemented {
            op: "index_select",
            detail: format!("backend {}", a.device().kind_name()),
        })
    }

    /// `out[indices[i]] += src[i]` along `dim`, on a fresh copy of `a`.
    ///
    /// Same fallback contract as [`Backend::index_select`].
    fn index_add(&self, a: &Tensor, dim: usize, indices: &Tensor, src: &Tensor) -> Result<Tensor> {
        let _ = (dim, indices, src);
        Err(Error::NotImplemented {
            op: "index_add",
            detail: format!("backend {}", a.device().kind_name()),
        })
    }
}

type Registry = RwLock<HashMap<Device, Arc<dyn Backend>>>;

fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register a backend for its device, replacing any previous registration.
///
/// Backend crates call this from their load-time constructors; linking a
/// backend crate is what makes its device usable.
pub fn register_backend(backend: Arc<dyn Backend>) {
    registry()
        .write()
        .expect("backend registry poisoned")
        .insert(backend.device(), backend);
}

/// The backend serving `device`, or a typed error when none is registered.
///
/// The CPU backend is registered lazily on first use, so it can never be
/// unavailable; GPU backends register at load time or via
/// `oxmera_runtime::init()`.
pub fn backend_for(device: Device) -> Result<Arc<dyn Backend>> {
    let found = registry()
        .read()
        .expect("backend registry poisoned")
        .get(&device)
        .cloned();
    match found {
        Some(b) => Ok(b),
        None if device == Device::Cpu => {
            crate::cpu::register();
            registry()
                .read()
                .expect("backend registry poisoned")
                .get(&device)
                .cloned()
                .ok_or(Error::BackendUnavailable { device })
        }
        None => Err(Error::BackendUnavailable { device }),
    }
}

/// Every registered device, sorted for stable reporting.
pub fn registered_devices() -> Vec<Device> {
    let mut devices: Vec<Device> = registry()
        .read()
        .expect("backend registry poisoned")
        .keys()
        .copied()
        .collect();
    devices.sort_by_key(|d| (d.kind_name(), device_index(*d)));
    devices
}

fn device_index(d: Device) -> usize {
    match d {
        Device::Cpu => 0,
        Device::Metal { index } | Device::Cuda { index } => index,
        _ => 0,
    }
}
