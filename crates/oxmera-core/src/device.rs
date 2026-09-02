//! Device handles.

/// Where a tensor's storage lives and where its work runs.
///
/// This is a *handle*, not a backend: it names a place. The backend
/// registry resolves a handle to an implementation; this crate must never
/// know how.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// The multi-threaded CPU backend — always available.
    Cpu,
    /// An Apple-Silicon GPU via Metal, by device index.
    Metal {
        /// Zero-based device index.
        index: usize,
    },
    /// An NVIDIA GPU, by device index — served by `oxmera-cuda` when a
    /// driver and a device are present at load time.
    Cuda {
        /// Zero-based device index.
        index: usize,
    },
}

impl Device {
    /// A short stable name for the device kind (`"cpu"`, `"metal"`,
    /// `"cuda"`), used in error messages and `oxmera doctor` output.
    pub fn kind_name(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Metal { .. } => "metal",
            Device::Cuda { .. } => "cuda",
        }
    }
}
