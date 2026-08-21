//! Element data types.

/// The element type of a tensor.
///
/// The set is deliberately small; it grows only when a backend can actually
/// carry the new type end to end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// 32-bit IEEE-754 float.
    F32,
    /// 64-bit IEEE-754 float.
    F64,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 8-bit unsigned integer.
    U8,
    /// Boolean, stored one byte per element.
    Bool,
}

impl DType {
    /// The size of one element of this type, in bytes.
    pub fn size_in_bytes(self) -> usize {
        todo!("exercise A1: shape and strides")
    }

    /// Whether this type is a floating-point type.
    pub fn is_float(self) -> bool {
        todo!("exercise A1: shape and strides")
    }
}
