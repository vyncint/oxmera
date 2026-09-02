//! Device registry semantics.

use oxmera_core::{Device, Error};
use oxmera_runtime::{backend_for, default_device, init};

#[test]
fn cpu_is_always_available() {
    let backend = backend_for(Device::Cpu).expect("cpu is lazily registered");
    assert_eq!(backend.name(), "cpu");
}

#[test]
fn unregistered_device_is_a_typed_error_not_a_panic() {
    let result = backend_for(Device::Cuda { index: 7 });
    assert!(matches!(
        result.map(|_| ()),
        Err(Error::BackendUnavailable { .. })
    ));
}

#[test]
fn init_reports_the_platform_devices() {
    let devices = init();
    assert!(devices.contains(&Device::Cpu));
    #[cfg(target_os = "macos")]
    assert!(
        devices.iter().any(|d| matches!(d, Device::Metal { .. })),
        "Apple Silicon must register Metal: {devices:?}"
    );
    let preferred = default_device();
    #[cfg(target_os = "macos")]
    assert!(matches!(preferred, Device::Metal { .. }));
    // Off macOS the preference is CUDA when a device registered at load
    // time, else the CPU — a Linux box with an NVIDIA card is a real host.
    #[cfg(not(target_os = "macos"))]
    {
        let has_cuda = devices.iter().any(|d| matches!(d, Device::Cuda { .. }));
        if has_cuda {
            assert!(matches!(preferred, Device::Cuda { .. }), "{devices:?}");
        } else {
            assert_eq!(preferred, Device::Cpu);
        }
    }
}
