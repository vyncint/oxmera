//! Storage: the owned buffer behind one or more tensors.

use oxmera_core::{DType, Device};

/// An owned, reference-counted buffer of elements on one device.
///
/// Storage is untyped at the buffer level — a flat byte buffer plus the
/// `DType` that says how to read it. Typed access is the business of the
/// backend that allocated it; nothing else may reinterpret the bytes.
/// Multiple tensors (views) may share one storage; storage never knows how
/// many.
#[derive(Debug)]
pub struct Storage {
    data: StorageData,
    dtype: DType,
    device: Device,
}

/// Where the bytes actually live.
///
/// One variant per device family. GPU variants hold opaque backend handles
/// once those backends exist; they are deliberately absent until then so
/// this enum never carries a stub it cannot honor.
#[derive(Debug)]
enum StorageData {
    /// Host memory for the CPU reference backend.
    // The expect below is a tripwire: solving exercise A1 makes the
    // constructors build this variant, the expectation stops holding, and
    // clippy -D warnings forces its removal.
    #[expect(dead_code, reason = "constructed once exercise A1 is solved")]
    Cpu(Vec<u8>),
}

impl Storage {
    /// Allocate zero-initialized CPU storage for `numel` elements of
    /// `dtype`.
    pub fn cpu_zeros(numel: usize, dtype: DType) -> Self {
        let _ = (numel, dtype);
        todo!("exercise A1: shape and strides")
    }

    /// Wrap raw host bytes as CPU storage. The byte length must be a
    /// multiple of the dtype size; the caller asserts the encoding.
    pub fn cpu_from_bytes(bytes: Vec<u8>, dtype: DType) -> Self {
        let _ = (bytes, dtype);
        todo!("exercise A1: shape and strides")
    }

    /// The element type of this buffer.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The device this buffer lives on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The raw bytes, when the storage is on the CPU.
    ///
    /// Backends other than the CPU reference return `None`; they expose
    /// their own typed access instead.
    pub fn cpu_bytes(&self) -> Option<&[u8]> {
        match &self.data {
            StorageData::Cpu(bytes) => Some(bytes),
        }
    }
}
