//! What every machine can check: the crate loads, probes for a driver
//! without panicking, and reports CUDA as simply unavailable when there is
//! none — the same contract Metal keeps off macOS. On a machine with a GPU
//! these still hold; the parity suite covers the rest.
use oxmera_core::Device;
use oxmera_tensor::backend::backend_for;
use oxmera_tensor::tensor::Tensor;

#[test]
fn registering_without_a_driver_is_a_quiet_no_op() {
    oxmera_cuda::register_default();
    let present = oxmera_cuda::is_driver_present();
    let registered = backend_for(Device::Cuda { index: 0 }).is_ok();
    // A device can only be registered when a driver is present.
    assert!(
        present || !registered,
        "registered a CUDA backend with no driver"
    );
}

#[test]
fn asking_for_cuda_without_a_device_is_a_typed_error() {
    oxmera_cuda::register_default();
    let t = Tensor::from_slice(&[1.0, 2.0], [2]).unwrap();
    match t.to_device(Device::Cuda { index: 0 }) {
        Ok(g) => {
            // A real device: the round trip must be exact.
            let back = g.to_device(Device::Cpu).unwrap();
            assert_eq!(back.to_vec_f32().unwrap(), vec![1.0, 2.0]);
        }
        Err(e) => {
            let msg = e.to_string().to_lowercase();
            assert!(
                msg.contains("cuda") || msg.contains("backend") || msg.contains("unavailable"),
                "unhelpful error: {e}"
            );
        }
    }
}

#[test]
fn device_summary_agrees_with_registration() {
    oxmera_cuda::register_default();
    let summary = oxmera_cuda::device_summary();
    let registered = backend_for(Device::Cuda { index: 0 }).is_ok();
    assert_eq!(summary.is_some(), registered);
    if let Some((name, mem, (major, minor))) = summary {
        assert!(!name.is_empty());
        assert!(mem > 0);
        assert!(major >= 5, "compute capability {major}.{minor}");
    }
}
