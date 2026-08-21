//! The error taxonomy.
//!
//! The design rule (exercise A5 refines it): invalid states are
//! unrepresentable where the type system can afford it, and every
//! representable failure is typed — no stringly errors on any seam.

use crate::device::Device;
use crate::dtype::DType;
use crate::shape::Shape;

/// Any failure an oxmera operation can report.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Two shapes were required to match and did not.
    #[error("shape mismatch: expected {expected:?}, got {got:?} in {op}")]
    ShapeMismatch {
        /// The shape the operation required.
        expected: Shape,
        /// The shape it received.
        got: Shape,
        /// The operation reporting the mismatch.
        op: &'static str,
    },

    /// Two shapes cannot broadcast together.
    #[error("cannot broadcast {lhs:?} with {rhs:?}")]
    BroadcastIncompatible {
        /// Left operand shape.
        lhs: Shape,
        /// Right operand shape.
        rhs: Shape,
    },

    /// Two dtypes were required to match and did not.
    #[error("dtype mismatch: expected {expected:?}, got {got:?} in {op}")]
    DTypeMismatch {
        /// The dtype the operation required.
        expected: DType,
        /// The dtype it received.
        got: DType,
        /// The operation reporting the mismatch.
        op: &'static str,
    },

    /// An operation does not support a dtype at all.
    #[error("dtype {dtype:?} is not supported by {op}")]
    UnsupportedDType {
        /// The offending dtype.
        dtype: DType,
        /// The operation that cannot carry it.
        op: &'static str,
    },

    /// Operands live on different devices.
    #[error("device mismatch: {lhs:?} vs {rhs:?} in {op}")]
    DeviceMismatch {
        /// Left operand device.
        lhs: Device,
        /// Right operand device.
        rhs: Device,
        /// The operation reporting the mismatch.
        op: &'static str,
    },

    /// A logical index was outside a tensor's bounds.
    #[error("index {index:?} out of bounds for shape {shape:?}")]
    IndexOutOfBounds {
        /// The offending index.
        index: Vec<usize>,
        /// The shape it was applied to.
        shape: Shape,
    },

    /// An index had the wrong number of dimensions for the shape.
    #[error("rank mismatch: index of rank {index_rank} against shape of rank {shape_rank}")]
    RankMismatch {
        /// Rank of the supplied index.
        index_rank: usize,
        /// Rank of the shape it was applied to.
        shape_rank: usize,
    },

    /// No backend is registered for a device.
    #[error("no backend available for device {device:?}")]
    BackendUnavailable {
        /// The device with no backend.
        device: Device,
    },

    /// The operation is not implemented yet. The skeleton phase of this
    /// project returns this nowhere — `todo!()` is used instead so that
    /// unimplemented paths are loud — but backends need it for genuinely
    /// unsupported combinations.
    #[error("{op} is not implemented for {detail}")]
    NotImplemented {
        /// The operation.
        op: &'static str,
        /// What combination is unsupported.
        detail: String,
    },
}

/// The result type every fallible oxmera API returns.
pub type Result<T> = std::result::Result<T, Error>;
