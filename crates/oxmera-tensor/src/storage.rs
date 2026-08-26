//! Storage: the owned buffer behind one or more tensors.

use oxmera_core::{DType, Device, Error, Result};

/// Typed host memory for the CPU backend.
///
/// One variant per supported dtype, so element access never reinterprets
/// bytes and the crate stays free of `unsafe` on the CPU path.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    /// 32-bit floats.
    F32(Vec<f32>),
    /// 64-bit signed integers (indices, argmax results, class targets).
    I64(Vec<i64>),
    /// Raw bytes (`U8`/`Bool`).
    U8(Vec<u8>),
}

impl CpuStorage {
    /// The dtype this buffer holds.
    pub fn dtype(&self) -> DType {
        match self {
            CpuStorage::F32(_) => DType::F32,
            CpuStorage::I64(_) => DType::I64,
            CpuStorage::U8(_) => DType::U8,
        }
    }

    /// Number of elements in the buffer.
    pub fn len(&self) -> usize {
        match self {
            CpuStorage::F32(v) => v.len(),
            CpuStorage::I64(v) => v.len(),
            CpuStorage::U8(v) => v.len(),
        }
    }

    /// Whether the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `f32` elements, or a typed error for other dtypes.
    pub fn f32s(&self) -> Result<&[f32]> {
        match self {
            CpuStorage::F32(v) => Ok(v),
            other => Err(Error::DTypeMismatch {
                expected: DType::F32,
                got: other.dtype(),
                op: "CpuStorage::f32s",
            }),
        }
    }

    /// The `i64` elements, or a typed error for other dtypes.
    pub fn i64s(&self) -> Result<&[i64]> {
        match self {
            CpuStorage::I64(v) => Ok(v),
            other => Err(Error::DTypeMismatch {
                expected: DType::I64,
                got: other.dtype(),
                op: "CpuStorage::i64s",
            }),
        }
    }
}

/// A Metal buffer that is safe to share across threads.
///
/// `metal::Buffer` is a smart pointer to an `MTLBuffer`.
#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct MetalBuffer {
    buffer: metal::Buffer,
    /// The Metal device index the buffer was allocated on.
    pub device_index: usize,
}

#[cfg(target_os = "macos")]
impl MetalBuffer {
    /// Wrap a Metal buffer allocated on device `device_index`.
    pub fn new(buffer: metal::Buffer, device_index: usize) -> Self {
        Self {
            buffer,
            device_index,
        }
    }

    /// The underlying Metal buffer.
    pub fn buffer(&self) -> &metal::Buffer {
        &self.buffer
    }
}

// SAFETY: MTLBuffer is documented by Apple as thread-safe ("Most Metal
// objects can be used from multiple threads"; command *encoders* are the
// exception and are never stored here). The wrapper only hands out shared
// references; all mutation happens through Metal command buffers, which
// serialize on the command queue.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
unsafe impl Send for MetalBuffer {}
// SAFETY: see the `Send` justification above; `&MetalBuffer` exposes no
// interior mutability outside Metal's own synchronized command path.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
unsafe impl Sync for MetalBuffer {}

/// Where the bytes actually live.
#[derive(Debug)]
pub enum StorageData {
    /// Host memory for the CPU backend.
    Cpu(CpuStorage),
    /// A GPU buffer for the Apple Metal backend.
    #[cfg(target_os = "macos")]
    Metal(MetalBuffer),
}

/// An owned, reference-counted buffer of elements on one device.
///
/// Multiple tensors (views) may share one storage; storage never knows how
/// many.
#[derive(Debug)]
pub struct Storage {
    data: StorageData,
    dtype: DType,
    device: Device,
}

impl Storage {
    /// Zero-initialized CPU storage for `numel` `f32` elements.
    pub fn cpu_f32_zeros(numel: usize) -> Self {
        Self::from_f32_vec(vec![0.0; numel])
    }

    /// CPU storage owning `data` as `f32` elements.
    pub fn from_f32_vec(data: Vec<f32>) -> Self {
        Self {
            data: StorageData::Cpu(CpuStorage::F32(data)),
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    /// CPU storage owning `data` as `i64` elements.
    pub fn from_i64_vec(data: Vec<i64>) -> Self {
        Self {
            data: StorageData::Cpu(CpuStorage::I64(data)),
            dtype: DType::I64,
            device: Device::Cpu,
        }
    }

    /// Storage backed by a Metal buffer holding `dtype` elements.
    #[cfg(target_os = "macos")]
    pub fn from_metal(buffer: MetalBuffer, dtype: DType) -> Self {
        let device = Device::Metal {
            index: buffer.device_index,
        };
        Self {
            data: StorageData::Metal(buffer),
            dtype,
            device,
        }
    }

    /// The element type of this buffer.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The device this buffer lives on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The raw storage data.
    pub fn data(&self) -> &StorageData {
        &self.data
    }

    /// The typed CPU buffer, or a typed error when the storage is on a
    /// different device.
    pub fn cpu(&self) -> Result<&CpuStorage> {
        match &self.data {
            StorageData::Cpu(c) => Ok(c),
            #[cfg(target_os = "macos")]
            StorageData::Metal(_) => Err(Error::DeviceMismatch {
                lhs: self.device,
                rhs: Device::Cpu,
                op: "Storage::cpu",
            }),
        }
    }
}
