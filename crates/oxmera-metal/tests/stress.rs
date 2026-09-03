//! Concurrency stress for the asynchronous dispatch path: many threads
//! upload, compute, and read back small tensors at once. Every read must
//! see the kernel's output, never a fresh buffer's zeros — the shape of
//! the race the parity suite hit under `--test-threads` when dispatch
//! first went asynchronous.

#![cfg(target_os = "macos")]

use oxmera_core::Device;
use oxmera_tensor::backend::backend_for;
use oxmera_tensor::tensor::Tensor;

fn metal() -> Device {
    oxmera_metal::register_default();
    let device = Device::Metal { index: 0 };
    assert!(backend_for(device).is_ok(), "no Metal device");
    device
}

#[test]
fn concurrent_threads_never_read_an_unwritten_output() {
    let device = metal();
    let threads: Vec<_> = (0..8)
        .map(|t| {
            std::thread::spawn(move || {
                for round in 0..200u64 {
                    let n = 5 + (round % 7) as usize;
                    let a = Tensor::randn_with_seed([n, 6], t * 1000 + round)
                        .abs()
                        .unwrap()
                        .add_scalar(1.0)
                        .unwrap();
                    let cpu = a.neg().unwrap().add_scalar(0.5).unwrap();
                    let gpu = a
                        .to_device(device)
                        .unwrap()
                        .neg()
                        .unwrap()
                        .add_scalar(0.5)
                        .unwrap()
                        .to_device(Device::Cpu)
                        .unwrap();
                    let (c, g) = (cpu.to_vec_f32().unwrap(), gpu.to_vec_f32().unwrap());
                    for (i, (x, y)) in c.iter().zip(&g).enumerate() {
                        assert!(
                            (x - y).abs() < 1e-6,
                            "thread {t} round {round} element {i}: cpu {x} vs metal {y}"
                        );
                    }
                }
            })
        })
        .collect();
    for h in threads {
        h.join().expect("a stress thread panicked");
    }
}
