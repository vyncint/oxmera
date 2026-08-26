//! Runtime concerns: backend availability, explicit initialization, and
//! the inference (`no_grad`) context.
//!
//! Backends self-register at load time when linked; this crate links the
//! CPU backend unconditionally and the Metal backend on macOS, so
//! depending on `oxmera-runtime` (or the `oxmera` umbrella) guarantees a
//! working default device set. [`init`] exists for contexts that want the
//! registration to be explicit and checkable.

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
    registered_devices()
}

/// The preferred compute device on this machine: the first Metal device
/// when one is registered, the CPU otherwise.
pub fn default_device() -> Device {
    registered_devices()
        .into_iter()
        .find(|d| matches!(d, Device::Metal { .. }))
        .unwrap_or(Device::Cpu)
}

/// A backend for `device`, initializing defaults first if needed.
pub fn backend(device: Device) -> Result<std::sync::Arc<dyn Backend>> {
    if backend_for(device).is_err() {
        init();
    }
    backend_for(device)
}
