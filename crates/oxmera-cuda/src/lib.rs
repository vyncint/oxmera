//! The NVIDIA CUDA backend: the same eight kernels as the Metal backend
//! (strided elementwise, axis and full reductions, tiled matmul), written
//! in CUDA C, shipped as PTX for `compute_75`, and driven through the CUDA
//! driver API via `cudarc`.
//!
//! Nothing here needs a CUDA toolkit to *build*: `libcuda` is located at
//! runtime with `dlopen`, and when it is absent the crate registers no
//! device — `Device::Cuda { .. }` is simply unavailable, exactly like Metal
//! off macOS. When the driver rejects the checked-in PTX (an older driver
//! than the toolkit that produced it), the embedded CUDA C source is
//! recompiled through NVRTC if `libnvrtc` is present.
//!
//! This is the *shipped* CUDA path. The `research/oxmera-cuda-oxide` crate
//! is the convergence-verified research line built on `cuda-oxide`; the two
//! share no code and the dependency firewall keeps that toolchain out of
//! this workspace.
//!
//! Unsafe policy: every `unsafe` block is FFI-adjacent — a kernel launch
//! whose argument list is checked against the kernel signature, a `dlopen`
//! probe, a `#[repr(C)]` argument marker, or life-before-main registration —
//! and carries a `// SAFETY:` justification.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod backend;

pub use backend::{CudaBackend, device_summary, is_driver_present, register_default};

// SAFETY: runs before main via the platform's initializer section. The body
// probes for libcuda (a dlopen that fails cleanly when the library is
// absent), optionally creates a context, and inserts into the
// std-synchronized backend registry; no thread-locals, no other crate's
// statics.
#[allow(unsafe_code)]
#[ctor::ctor(crate_path = ::ctor)]
unsafe fn auto_register() {
    register_default();
}
