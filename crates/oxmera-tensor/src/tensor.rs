//! The tensor value: a layout over shared storage.

use std::sync::Arc;

use oxmera_core::{DType, Device, Layout, Result, Shape};

use crate::storage::Storage;

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
}

impl Tensor {
    /// A tensor over existing storage with an explicit layout.
    ///
    /// Errors when the layout addresses elements outside the storage.
    pub fn from_storage(storage: Arc<Storage>, layout: Layout) -> Result<Self> {
        let _ = (storage, layout);
        todo!("exercise A1: shape and strides")
    }

    /// A contiguous CPU tensor holding `data` with shape `shape`.
    ///
    /// Errors when `data.len()` does not equal `shape.numel()`.
    pub fn from_vec_f32(data: Vec<f32>, shape: Shape) -> Result<Self> {
        let _ = (data, shape);
        todo!("exercise A1: shape and strides")
    }

    /// The shape of this view.
    pub fn shape(&self) -> &Shape {
        &self.layout.shape
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

    /// A view with the same elements in a new shape.
    ///
    /// Succeeds without copying only when the current layout permits it;
    /// errors on element-count mismatch. Never copies — `contiguous` is the
    /// explicit spelling for that.
    pub fn reshape(&self, shape: Shape) -> Result<Self> {
        let _ = shape;
        todo!("exercise A3: strided views")
    }

    /// A view with dimensions reordered by `perm` (a permutation of
    /// `0..ndim`).
    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        let _ = perm;
        todo!("exercise A3: strided views")
    }

    /// A view of `len` elements of dimension `dim` starting at `start`.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let _ = (dim, start, len);
        todo!("exercise A3: strided views")
    }

    /// This tensor's elements, in logical order, in fresh contiguous
    /// storage on the same device. A no-op clone when already contiguous.
    pub fn contiguous(&self) -> Result<Self> {
        todo!("exercise A3: strided views")
    }

    /// The element at a logical index, as `f32`, for tests and debugging.
    ///
    /// Errors on rank mismatch, out-of-bounds, non-float dtype, or non-CPU
    /// storage.
    pub fn get_f32(&self, index: &[usize]) -> Result<f32> {
        let _ = index;
        todo!("exercise A1: shape and strides")
    }
}
