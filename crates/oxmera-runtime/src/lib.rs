//! Runtime concerns: backend availability, explicit initialization, and
//! the inference (`no_grad`) context.
//!
//! Backends self-register at load time when linked; this crate links the
//! CPU backend unconditionally, the Metal backend on macOS, and the CUDA
//! backend when the default `cuda` feature is on — so depending on
//! `oxmera-runtime` (or the `oxmera` umbrella) guarantees a working
//! default device set. [`init`] exists for contexts that want the
//! registration to be explicit and checkable.
//!
//! Turning `cuda` off removes `cudarc`, `libloading` and the `ctor`/`dtor`
//! pair from the graph, the shipped PTX from the binary, and a pre-`main`
//! constructor that `dlopen`s `libcuda` from the process. What it does not
//! change is the *type*: [`oxmera_core::Device::Cuda`] still exists and
//! still resolves to a typed error at run time, so a caller that names the
//! device compiles either way and finds out the same way it would on a
//! machine with no NVIDIA card.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use oxmera_core::{Device, Result};

pub use oxmera_tensor::autograd::{NoGradGuard, no_grad};
pub use oxmera_tensor::backend::{Backend, backend_for, register_backend, registered_devices};

/// Ensure the default backends for this platform are registered, and
/// report the devices available.
///
/// Load-time constructors normally make this unnecessary; calling it is
/// harmless and returns the registered device list either way.
pub fn init() -> Vec<Device> {
    oxmera_cpu::register();
    #[cfg(target_os = "macos")]
    oxmera_metal::register_default();
    #[cfg(feature = "cuda")]
    oxmera_cuda::register_default();
    registered_devices()
}

/// The preferred compute device on this machine: a Metal device when one
/// is registered, else a CUDA device, else the CPU.
pub fn default_device() -> Device {
    let devices = registered_devices();
    devices
        .iter()
        .copied()
        .find(|d| matches!(d, Device::Metal { .. }))
        .or_else(|| {
            devices
                .iter()
                .copied()
                .find(|d| matches!(d, Device::Cuda { .. }))
        })
        .unwrap_or(Device::Cpu)
}

/// A backend for `device`, initializing defaults first if needed.
pub fn backend(device: Device) -> Result<std::sync::Arc<dyn Backend>> {
    if backend_for(device).is_err() {
        init();
    }
    backend_for(device)
}
