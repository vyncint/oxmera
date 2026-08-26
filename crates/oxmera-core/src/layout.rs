//! Strides and memory layout.

use crate::error::{Error, Result};
use crate::shape::Shape;

/// Per-dimension element strides (not byte strides), outermost first.
///
/// Strides are signed so that flipped views are representable, and a
/// stride of 0 represents a broadcast dimension.
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
        let strides = contiguous_strides(&shape);
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    /// The storage offset (in elements) of a logical index.
    ///
    /// Errors when `index` has the wrong rank or is out of bounds.
    pub fn offset_of(&self, index: &[usize]) -> Result<usize> {
        let dims = self.shape.dims();
        if index.len() != dims.len() {
            return Err(Error::RankMismatch {
                index_rank: index.len(),
                shape_rank: dims.len(),
            });
        }
        for (i, (&ix, &d)) in index.iter().zip(dims).enumerate() {
            let _ = i;
            if ix >= d {
                return Err(Error::IndexOutOfBounds {
                    index: index.to_vec(),
                    shape: self.shape.clone(),
                });
            }
        }
        let mut off = self.offset as isize;
        for (&ix, &s) in index.iter().zip(self.strides.values()) {
            off += ix as isize * s;
        }
        debug_assert!(off >= 0, "negative absolute offset from a valid layout");
        Ok(off as usize)
    }

    /// Whether this layout is contiguous row-major (a scalar and any empty
    /// tensor count as contiguous).
    pub fn is_contiguous(&self) -> bool {
        if self.shape.numel() == 0 {
            return true;
        }
        let expected = contiguous_strides(&self.shape);
        // Dimensions of extent 1 have no observable stride; ignore them.
        self.shape
            .dims()
            .iter()
            .zip(self.strides.values())
            .zip(expected.values())
            .all(|((&d, &got), &want)| d == 1 || got == want)
    }
}

/// The contiguous row-major strides for `shape`.
pub fn contiguous_strides(shape: &Shape) -> Strides {
    let dims = shape.dims();
    let mut strides = vec![0isize; dims.len()];
    let mut acc = 1isize;
    for (i, &d) in dims.iter().enumerate().rev() {
        strides[i] = acc;
        acc *= d as isize;
    }
    Strides(strides)
}
