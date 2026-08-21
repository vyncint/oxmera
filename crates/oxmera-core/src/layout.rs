//! Strides and memory layout.

use crate::error::Result;
use crate::shape::Shape;

/// Per-dimension element strides (not byte strides), outermost first.
///
/// Strides are signed so that flipped views are representable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides(Vec<isize>);

impl Strides {
    /// Wrap per-dimension element strides.
    pub fn new(strides: Vec<isize>) -> Self {
        Self(strides)
    }

    /// The stride values, outermost first.
    pub fn values(&self) -> &[isize] {
        &self.0
    }
}

/// How a tensor's logical index space maps onto its storage: a shape, the
/// strides, and a start offset (in elements) into the underlying buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// Extents per dimension.
    pub shape: Shape,
    /// Element strides per dimension.
    pub strides: Strides,
    /// Start offset into the buffer, in elements.
    pub offset: usize,
}

impl Layout {
    /// The contiguous row-major layout for `shape`, offset 0.
    pub fn contiguous(shape: Shape) -> Self {
        let _ = shape;
        todo!("exercise A1: shape and strides")
    }

    /// The storage offset (in elements) of a logical index.
    ///
    /// Errors when `index` has the wrong rank or is out of bounds.
    pub fn offset_of(&self, index: &[usize]) -> Result<usize> {
        let _ = index;
        todo!("exercise A1: shape and strides")
    }

    /// Whether this layout is contiguous row-major with offset semantics
    /// preserved (a scalar and any empty tensor count as contiguous).
    pub fn is_contiguous(&self) -> bool {
        todo!("exercise A1: shape and strides")
    }
}

/// The contiguous row-major strides for `shape`.
pub fn contiguous_strides(shape: &Shape) -> Strides {
    let _ = shape;
    todo!("exercise A1: shape and strides")
}
