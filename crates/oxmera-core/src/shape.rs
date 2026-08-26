//! Tensor shapes and the broadcasting rules.

use crate::error::{Error, Result};

/// The extents of a tensor, one entry per dimension, outermost first
/// (row-major convention throughout the project).
///
/// A rank-0 shape (`[]`) is a scalar and is valid. A dimension of size 0 is
/// valid and makes the element count 0.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Wrap dimension extents as a shape.
    pub fn new(dims: Vec<usize>) -> Self {
        Self(dims)
    }

    /// The dimension extents, outermost first.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// The rank (number of dimensions).
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// The total number of elements.
    ///
    /// A scalar has 1 element; any zero-sized dimension makes this 0.
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

/// The shape two operands broadcast to, or a typed error when they are
/// incompatible.
///
/// The rules are NumPy's: align trailing dimensions; each pair must be
/// equal or one of them 1. This is a total function over pairs of shapes —
/// every input has a defined answer, success or a specific error.
pub fn broadcast_shapes(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    let (a, b) = (lhs.dims(), rhs.dims());
    let ndim = a.len().max(b.len());
    let mut out = vec![0usize; ndim];
    for i in 0..ndim {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        out[ndim - 1 - i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(Error::BroadcastIncompatible {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            });
        };
    }
    Ok(Shape(out))
}
