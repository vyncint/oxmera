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
        self.checked_numel().expect(
            "shape element count overflows usize — reject the shape before it becomes a tensor",
        )
    }

    /// Total number of elements, or `None` when the product of the
    /// dimensions does not fit in `usize`.
    ///
    /// Every constructor that accepts a caller-supplied shape validates
    /// with this before allocating: a wrapped product would let a shape
    /// claim 2^64 elements over an empty buffer and pass the
    /// `data.len() == numel` check (a reported class of bug); with the
    /// check, such a shape is a typed error instead.
    pub fn checked_numel(&self) -> Option<usize> {
        self.0.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
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

#[cfg(test)]
mod tests {
    use super::Shape;

    #[test]
    fn checked_numel_matches_numel_when_it_fits() {
        assert_eq!(Shape::from([2, 3, 4]).checked_numel(), Some(24));
        assert_eq!(Shape::from([0, 3]).checked_numel(), Some(0));
        assert_eq!(Shape::new(vec![]).checked_numel(), Some(1));
    }

    #[test]
    fn checked_numel_is_none_on_overflow() {
        assert_eq!(
            Shape::from([1usize << 32, 1usize << 32]).checked_numel(),
            None
        );
        assert_eq!(Shape::from([usize::MAX, 2]).checked_numel(), None);
    }

    #[test]
    #[should_panic(expected = "overflows usize")]
    fn numel_refuses_to_wrap_silently() {
        let _ = Shape::from([1usize << 32, 1usize << 32]).numel();
    }
}
