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

/// With the `cuda` feature off, naming the device must still **compile**
/// and must still fail at run time the way it does on a machine with no
/// NVIDIA card.
///
/// This is the contract that makes the feature safe to turn off: a
/// consumer who writes `Device::Cuda { index: 0 }` behind a config flag
/// keeps building. If turning the backend off ever made the *type*
/// disappear, every such consumer would break at compile time on a
/// dependency-graph change they did not make.
///
/// Runs in both configurations on purpose — with the feature on and no
/// hardware the outcome is the same, and a test that only runs in one
/// configuration would not be pinning a contract about both.
#[test]
fn naming_cuda_compiles_and_fails_at_run_time_without_the_backend() {
    oxmera_runtime::init();
    let t = oxmera_tensor::Tensor::from_vec_f32(vec![1.0, 2.0], [2]).unwrap();
    match t.to_device(oxmera_core::Device::Cuda { index: 0 }) {
        // A machine with a driver and the feature on: the round trip is exact.
        Ok(g) => assert_eq!(
            g.to_device(oxmera_core::Device::Cpu)
                .unwrap()
                .to_vec_f32()
                .unwrap(),
            vec![1.0, 2.0]
        ),
        Err(e) => {
            let msg = e.to_string().to_lowercase();
            assert!(
                msg.contains("cuda") || msg.contains("backend") || msg.contains("device"),
                "the error has to name what was unavailable, got: {e}"
            );
        }
    }
}

/// The CPU is always there, feature or no feature — the property that
/// makes `--no-default-features` a reduction rather than a mutilation.
#[test]
fn the_cpu_backend_survives_every_feature_combination() {
    let devices = oxmera_runtime::init();
    assert!(
        devices.contains(&oxmera_core::Device::Cpu),
        "registered devices: {devices:?}"
    );
    let t = oxmera_tensor::Tensor::from_vec_f32(vec![1.0, -2.0], [2]).unwrap();
    assert_eq!(t.relu().unwrap().to_vec_f32().unwrap(), vec![1.0, 0.0]);
}
