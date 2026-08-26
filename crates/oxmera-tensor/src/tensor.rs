//! The tensor value: a layout over shared storage, with an optional
//! autograd tape node.

use std::sync::{Arc, Mutex};

use oxmera_core::layout::contiguous_strides;
use oxmera_core::{DType, Device, Error, Layout, Result, Shape, Strides};
use rand::SeedableRng;
use rand_distr::Distribution;

use crate::autograd::{AutogradMeta, GradFn};
use crate::backend::backend_for;
use crate::storage::{CpuStorage, Storage};

/// A tensor: shared storage viewed through a layout.
///
/// Cloning a tensor is cheap — it clones the layout and bumps the storage
/// refcount, never the data. View operations (`reshape`, `permute`,
/// `narrow`, …) produce new tensors over the same storage whenever the
/// layout arithmetic allows it.
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Arc<Storage>,
    layout: Layout,
    autograd: Option<Arc<AutogradMeta>>,
}

impl Tensor {
    // ---- construction ---------------------------------------------------

    /// A tensor over existing storage with an explicit layout.
    ///
    /// Errors when the layout addresses elements outside the storage.
    pub fn from_storage(storage: Arc<Storage>, layout: Layout) -> Result<Self> {
        let needed = max_addressed(&layout);
        let available = storage_len(&storage);
        if needed > available {
            return Err(Error::InvalidArgument {
                op: "Tensor::from_storage",
                detail: format!("layout addresses {needed} elements, storage holds {available}"),
            });
        }
        Ok(Self {
            storage,
            layout,
            autograd: None,
        })
    }

    /// A contiguous CPU tensor holding `data` with shape `shape`.
    ///
    /// Errors when `data.len()` does not equal `shape.numel()`.
    pub fn from_vec_f32(data: Vec<f32>, shape: Shape) -> Result<Self> {
        if data.len() != shape.numel() {
            return Err(Error::ShapeMismatch {
                expected: Shape::from([data.len()]),
                got: shape,
                op: "Tensor::from_vec_f32",
            });
        }
        Ok(Self {
            storage: Arc::new(Storage::from_f32_vec(data)),
            layout: Layout::contiguous(shape),
            autograd: None,
        })
    }

    /// A contiguous CPU tensor copying `data` with shape `shape`.
    pub fn from_slice(data: &[f32], shape: impl Into<Shape>) -> Result<Self> {
        Self::from_vec_f32(data.to_vec(), shape.into())
    }

    /// A contiguous CPU `I64` tensor holding `data` (indices, targets).
    pub fn from_vec_i64(data: Vec<i64>, shape: Shape) -> Result<Self> {
        if data.len() != shape.numel() {
            return Err(Error::ShapeMismatch {
                expected: Shape::from([data.len()]),
                got: shape,
                op: "Tensor::from_vec_i64",
            });
        }
        Ok(Self {
            storage: Arc::new(Storage::from_i64_vec(data)),
            layout: Layout::contiguous(shape),
            autograd: None,
        })
    }

    /// A CPU tensor of zeros.
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let numel = shape.numel();
        Self::from_vec_f32(vec![0.0; numel], shape).expect("lengths match by construction")
    }

    /// A CPU tensor of ones.
    pub fn ones(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let numel = shape.numel();
        Self::from_vec_f32(vec![1.0; numel], shape).expect("lengths match by construction")
    }

    /// A CPU tensor filled with `value`.
    pub fn full(shape: impl Into<Shape>, value: f32) -> Self {
        let shape = shape.into();
        let numel = shape.numel();
        Self::from_vec_f32(vec![value; numel], shape).expect("lengths match by construction")
    }

    /// A rank-0 scalar tensor.
    pub fn scalar(value: f32) -> Self {
        Self::from_vec_f32(vec![value], Shape::from([])).expect("scalar always fits")
    }

    /// Standard-normal random CPU tensor, seeded from the OS.
    pub fn randn(shape: impl Into<Shape>) -> Self {
        Self::randn_with_seed(shape, rand::random())
    }

    /// Standard-normal random CPU tensor with a fixed seed, for
    /// reproducible tests and examples.
    pub fn randn_with_seed(shape: impl Into<Shape>, seed: u64) -> Self {
        let shape = shape.into();
        let numel = shape.numel();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = rand_distr::StandardNormal;
        let data: Vec<f32> = (0..numel).map(|_| normal.sample(&mut rng)).collect();
        Self::from_vec_f32(data, shape).expect("lengths match by construction")
    }

    // ---- accessors -------------------------------------------------------

    /// The shape of this view.
    pub fn shape(&self) -> &Shape {
        &self.layout.shape
    }

    /// The dimension extents, outermost first.
    pub fn dims(&self) -> &[usize] {
        self.layout.shape.dims()
    }

    /// The rank (number of dimensions).
    pub fn ndim(&self) -> usize {
        self.layout.shape.ndim()
    }

    /// The total number of elements.
    pub fn numel(&self) -> usize {
        self.layout.shape.numel()
    }

    /// The full layout of this view.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The element type.
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// The device the storage lives on.
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// The shared storage behind this view.
    pub fn storage(&self) -> &Arc<Storage> {
        &self.storage
    }

    // ---- element access (CPU) --------------------------------------------

    /// The element at a logical index, as `f32`.
    ///
    /// Errors on rank mismatch, out-of-bounds, non-float dtype, or non-CPU
    /// storage.
    pub fn get_f32(&self, index: &[usize]) -> Result<f32> {
        let offset = self.layout.offset_of(index)?;
        Ok(self.storage.cpu()?.f32s()?[offset])
    }

    /// The element at a logical index, as `i64`.
    pub fn get_i64(&self, index: &[usize]) -> Result<i64> {
        let offset = self.layout.offset_of(index)?;
        Ok(self.storage.cpu()?.i64s()?[offset])
    }

    /// Every element in logical (row-major) order, as `f32`, from CPU
    /// storage.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let src = self.storage.cpu()?.f32s()?;
        Ok(gather_logical(src, &self.layout))
    }

    /// Every element in logical (row-major) order, as `i64`.
    pub fn to_vec_i64(&self) -> Result<Vec<i64>> {
        let src = self.storage.cpu()?.i64s()?;
        Ok(gather_logical(src, &self.layout))
    }

    // ---- views -----------------------------------------------------------

    fn view(&self, layout: Layout) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            layout,
            autograd: None,
        }
    }

    /// A view (or copy, when this view is not contiguous) with the same
    /// elements in a new shape.
    pub fn reshape(&self, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        if shape.numel() != self.numel() {
            return Err(Error::ShapeMismatch {
                expected: self.shape().clone(),
                got: shape,
                op: "reshape",
            });
        }
        let base = if self.layout.is_contiguous() {
            self.clone()
        } else {
            self.contiguous_data()?
        };
        let layout = Layout {
            strides: contiguous_strides(&shape),
            shape,
            offset: base.layout.offset,
        };
        let out = base.view(layout);
        Ok(crate::ops::record_view(self, out, ViewKind::Reshape))
    }

    /// A view with dimensions reordered by `perm` (a permutation of
    /// `0..ndim`).
    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        let n = self.ndim();
        if perm.len() != n || {
            let mut seen = vec![false; n];
            perm.iter()
                .any(|&p| p >= n || std::mem::replace(&mut seen[p], true))
        } {
            return Err(Error::InvalidArgument {
                op: "permute",
                detail: format!("{perm:?} is not a permutation of 0..{n}"),
            });
        }
        let dims = self.dims();
        let strides = self.layout.strides.values();
        let new_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
        let new_strides: Vec<isize> = perm.iter().map(|&p| strides[p]).collect();
        let layout = Layout {
            shape: Shape::new(new_dims),
            strides: Strides::new(new_strides),
            offset: self.layout.offset,
        };
        let out = self.view(layout);
        Ok(crate::ops::record_view(
            self,
            out,
            ViewKind::Permute(perm.to_vec()),
        ))
    }

    /// A view with dimensions `d0` and `d1` swapped.
    pub fn transpose(&self, d0: usize, d1: usize) -> Result<Self> {
        let mut perm: Vec<usize> = (0..self.ndim()).collect();
        if d0 >= perm.len() || d1 >= perm.len() {
            return Err(Error::InvalidArgument {
                op: "transpose",
                detail: format!("dims ({d0}, {d1}) out of range for rank {}", perm.len()),
            });
        }
        perm.swap(d0, d1);
        self.permute(&perm)
    }

    /// The matrix transpose: the last two dimensions swapped.
    pub fn t(&self) -> Result<Self> {
        let n = self.ndim();
        if n < 2 {
            return Err(Error::InvalidArgument {
                op: "t",
                detail: format!("needs rank >= 2, got {n}"),
            });
        }
        self.transpose(n - 2, n - 1)
    }

    /// A view of `len` elements of dimension `dim` starting at `start`.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() || start + len > dims[dim] {
            return Err(Error::InvalidArgument {
                op: "narrow",
                detail: format!(
                    "dim {dim}, range {start}..{} against shape {:?}",
                    start + len,
                    self.shape()
                ),
            });
        }
        let mut new_dims = dims.to_vec();
        new_dims[dim] = len;
        let strides = self.layout.strides.values().to_vec();
        let offset = (self.layout.offset as isize + start as isize * strides[dim]) as usize;
        let layout = Layout {
            shape: Shape::new(new_dims),
            strides: Strides::new(strides),
            offset,
        };
        let out = self.view(layout);
        Ok(crate::ops::record_view(
            self,
            out,
            ViewKind::Narrow { dim, start, len },
        ))
    }

    /// A view of `range` along `dim` — sugar over [`Tensor::narrow`].
    pub fn slice(&self, dim: usize, range: std::ops::Range<usize>) -> Result<Self> {
        let len = range.end.saturating_sub(range.start);
        self.narrow(dim, range.start, len)
    }

    /// A zero-copy broadcast view to `shape` (stride 0 on expanded axes).
    pub fn broadcast_to(&self, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let layout = broadcast_layout(&self.layout, &shape)?;
        let out = self.view(layout);
        Ok(crate::ops::record_view(self, out, ViewKind::Broadcast))
    }

    /// A broadcast view that records nothing on the tape — backend
    /// plumbing; prefer [`Tensor::broadcast_to`] in user code.
    pub fn broadcast_view(&self, shape: &Shape) -> Result<Self> {
        let layout = broadcast_layout(&self.layout, shape)?;
        Ok(self.view(layout))
    }

    /// A view with a new size-1 dimension inserted at `dim`.
    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let mut dims = self.dims().to_vec();
        if dim > dims.len() {
            return Err(Error::InvalidArgument {
                op: "unsqueeze",
                detail: format!("dim {dim} out of range for rank {}", dims.len()),
            });
        }
        dims.insert(dim, 1);
        let mut strides = self.layout.strides.values().to_vec();
        strides.insert(dim, 0);
        let layout = Layout {
            shape: Shape::new(dims),
            strides: Strides::new(strides),
            offset: self.layout.offset,
        };
        let out = self.view(layout);
        Ok(crate::ops::record_view(self, out, ViewKind::Reshape))
    }

    /// This tensor's elements, in logical order, in fresh contiguous
    /// storage on the same device. A no-op clone when already contiguous.
    pub fn contiguous(&self) -> Result<Self> {
        if self.layout.is_contiguous() && self.layout.offset == 0 {
            return Ok(self.clone());
        }
        let out = self.contiguous_data()?;
        Ok(crate::ops::record_view(self, out, ViewKind::Contiguous))
    }

    /// The contiguous copy without autograd recording — backend plumbing;
    /// prefer [`Tensor::contiguous`] in user code.
    pub fn contiguous_untracked(&self) -> Result<Self> {
        self.contiguous_data()
    }

    /// The contiguous copy without autograd recording (plumbing).
    pub(crate) fn contiguous_data_crate(&self) -> Result<Self> {
        self.contiguous_data()
    }

    /// The contiguous copy without autograd recording (plumbing).
    pub(crate) fn contiguous_data(&self) -> Result<Self> {
        match self.device() {
            Device::Cpu => {
                let shape = self.shape().clone();
                match self.storage.cpu()? {
                    CpuStorage::F32(_) => Tensor::from_vec_f32(self.to_vec_f32()?, shape),
                    CpuStorage::I64(_) => Tensor::from_vec_i64(self.to_vec_i64()?, shape),
                    CpuStorage::U8(_) => Err(Error::UnsupportedDType {
                        dtype: DType::U8,
                        op: "contiguous",
                    }),
                }
            }
            device => backend_for(device)?.contiguous(self),
        }
    }

    // ---- autograd surface -------------------------------------------------

    /// Mark (or unmark) this tensor as a gradient-accumulating leaf, in
    /// place, returning it for chaining.
    pub fn requires_grad_(mut self, requires: bool) -> Self {
        match (&self.autograd, requires) {
            (Some(meta), _) if meta.grad_fn.is_none() => {
                // Leaf: rebuild the meta with the new flag.
                self.autograd = requires.then(|| {
                    Arc::new(AutogradMeta {
                        requires_grad: true,
                        grad: Mutex::new(None),
                        grad_fn: None,
                    })
                });
            }
            (Some(_), true) => { /* non-leaf already tracked; nothing to do */ }
            (Some(_), false) => self.autograd = None,
            (None, true) => {
                self.autograd = Some(Arc::new(AutogradMeta {
                    requires_grad: true,
                    grad: Mutex::new(None),
                    grad_fn: None,
                }));
            }
            (None, false) => {}
        }
        self
    }

    /// Whether gradients accumulate on this tensor during `backward`.
    pub fn requires_grad(&self) -> bool {
        self.autograd.as_ref().is_some_and(|m| m.requires_grad)
    }

    /// Whether this tensor participates in the tape at all.
    pub(crate) fn is_tracked(&self) -> bool {
        self.autograd.is_some()
    }

    /// The accumulated gradient, if a backward pass has produced one.
    pub fn grad(&self) -> Option<Tensor> {
        self.autograd
            .as_ref()
            .and_then(|m| m.grad.lock().expect("grad mutex poisoned").clone())
    }

    /// Clear this tensor's accumulated gradient.
    pub fn zero_grad(&self) {
        if let Some(meta) = &self.autograd {
            *meta.grad.lock().expect("grad mutex poisoned") = None;
        }
    }

    /// The same view without any tape connection.
    pub fn detach(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            layout: self.layout.clone(),
            autograd: None,
        }
    }

    /// Propagate gradients from this scalar through the recorded tape.
    ///
    /// Errors when the tensor is not a scalar; use
    /// [`Tensor::backward_with`] to seed a non-scalar output.
    pub fn backward(&self) -> Result<()> {
        if self.numel() != 1 {
            return Err(Error::InvalidArgument {
                op: "backward",
                detail: format!(
                    "output has {} elements; seed a non-scalar with backward_with",
                    self.numel()
                ),
            });
        }
        self.backward_with(Tensor::ones(self.shape().clone()))
    }

    /// Propagate gradients seeding this tensor's gradient with `seed`.
    pub fn backward_with(&self, seed: Tensor) -> Result<()> {
        crate::autograd::run_backward(self, seed)
    }

    pub(crate) fn autograd_meta(&self) -> Option<Arc<AutogradMeta>> {
        self.autograd.clone()
    }

    /// Attach a tape node to this tensor (used by the op layer).
    pub(crate) fn with_grad_fn(mut self, grad_fn: GradFn) -> Self {
        self.autograd = Some(Arc::new(AutogradMeta {
            requires_grad: false,
            grad: Mutex::new(None),
            grad_fn: Some(grad_fn),
        }));
        self
    }
}

/// The internal view taxonomy the op layer uses to build view VJPs.
#[derive(Debug, Clone)]
pub(crate) enum ViewKind {
    /// Reshape/unsqueeze: gradient reshapes back to the input shape.
    Reshape,
    /// Permute by this permutation: gradient permutes by the inverse.
    Permute(Vec<usize>),
    /// Narrow: gradient scatters back into zeros of the input shape.
    Narrow {
        /// Narrowed dimension.
        dim: usize,
        /// Range start.
        start: usize,
        /// Range length.
        len: usize,
    },
    /// Broadcast view: gradient sum-reduces back to the input shape.
    Broadcast,
    /// Contiguous copy: gradient passes through (reshaped if needed).
    Contiguous,
}

fn storage_len(storage: &Storage) -> usize {
    match storage.data() {
        crate::storage::StorageData::Cpu(c) => c.len(),
        #[cfg(target_os = "macos")]
        crate::storage::StorageData::Metal(b) => {
            b.buffer().length() as usize / storage.dtype().size_in_bytes()
        }
    }
}

fn max_addressed(layout: &Layout) -> usize {
    if layout.shape.numel() == 0 {
        return 0;
    }
    let mut max = layout.offset as isize;
    for (&d, &s) in layout.shape.dims().iter().zip(layout.strides.values()) {
        if d > 1 && s > 0 {
            max += (d as isize - 1) * s;
        }
    }
    (max + 1) as usize
}

/// Broadcast `layout` to `target`, stride 0 on expanded axes.
pub(crate) fn broadcast_layout(layout: &Layout, target: &Shape) -> Result<Layout> {
    let src = layout.shape.dims();
    let dst = target.dims();
    if dst.len() < src.len() {
        return Err(Error::BroadcastIncompatible {
            lhs: layout.shape.clone(),
            rhs: target.clone(),
        });
    }
    let lead = dst.len() - src.len();
    let mut strides = vec![0isize; dst.len()];
    for i in 0..src.len() {
        let (s, d) = (src[i], dst[lead + i]);
        if s == d {
            strides[lead + i] = layout.strides.values()[i];
        } else if s == 1 {
            strides[lead + i] = 0;
        } else {
            return Err(Error::BroadcastIncompatible {
                lhs: layout.shape.clone(),
                rhs: target.clone(),
            });
        }
    }
    Ok(Layout {
        shape: target.clone(),
        strides: Strides::new(strides),
        offset: layout.offset,
    })
}

/// Gather a strided view's elements into logical row-major order.
pub(crate) fn gather_logical<T: Copy>(src: &[T], layout: &Layout) -> Vec<T> {
    let numel = layout.shape.numel();
    let mut out = Vec::with_capacity(numel);
    if numel == 0 {
        return out;
    }
    let dims = layout.shape.dims();
    if layout.is_contiguous() {
        let start = layout.offset;
        out.extend_from_slice(&src[start..start + numel]);
        return out;
    }
    let strides = layout.strides.values();
    let mut index = vec![0usize; dims.len()];
    let mut offset = layout.offset as isize;
    loop {
        out.push(src[offset as usize]);
        // Odometer increment, last dimension fastest, offset updated
        // incrementally.
        let mut d = dims.len();
        loop {
            if d == 0 {
                return out;
            }
            d -= 1;
            index[d] += 1;
            offset += strides[d];
            if index[d] < dims[d] {
                break;
            }
            offset -= dims[d] as isize * strides[d];
            index[d] = 0;
        }
        if out.len() == numel {
            return out;
        }
    }
}
