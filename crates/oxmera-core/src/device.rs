//! Device handles.

/// Where a tensor's storage lives and where its work runs.
///
/// This is a *handle*, not a backend: it names a place. The runtime layer
/// resolves a handle to a backend implementation; this crate must never
/// know how.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// The CPU reference backend — always available, defines correctness.
    Cpu,
    /// An Apple-Silicon GPU, by device index. No convergence gate exists on
    /// this path; see ARCHITECTURE.md.
    Metal {
        /// Zero-based device index.
        index: usize,
    },
    /// An NVIDIA GPU, by device index. Feature-gated, Linux, non-default.
    Cuda {
        /// Zero-based device index.
        index: usize,
    },
}

impl Device {
    /// A short stable name for the device kind (`"cpu"`, `"metal"`,
    /// `"cuda"`), used in error messages and `oxmera doctor` output.
    pub fn kind_name(self) -> &'static str {
        todo!("exercise A5: the error taxonomy")
    }
}
