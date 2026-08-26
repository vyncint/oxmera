//! CPU ↔ Metal numerical parity: every op family computed on both
//! backends must agree within `1e-5` (relative for large magnitudes).
//! macOS only; on other targets this file compiles to nothing.

#![cfg(target_os = "macos")]

use oxmera_core::{Device, Shape};
use oxmera_tensor::backend::{BinaryOp, UnaryOp, backend_for};
use oxmera_tensor::tensor::Tensor;

const TOL: f32 = 1e-5;

fn metal() -> Device {
    oxmera_metal::register_default();
    let device = Device::Metal { index: 0 };
    assert!(
        backend_for(device).is_ok(),
        "no Metal device on this machine"
    );
    device
}

fn assert_close(cpu: &[f32], gpu: &[f32], context: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{context}: length");
    for (i, (&x, &y)) in cpu.iter().zip(gpu).enumerate() {
        let tol = TOL * 1.0f32.max(x.abs());
        assert!(
            (x - y).abs() <= tol || (x.is_nan() && y.is_nan()),
            "{context}: element {i}: cpu {x} vs metal {y}"
        );
    }
}

#[test]
fn upload_download_roundtrip() {
    let device = metal();
    let t = Tensor::randn_with_seed([7, 9], 1);
    let back = t.to_device(device).unwrap().to_device(Device::Cpu).unwrap();
    assert_close(
        &t.to_vec_f32().unwrap(),
        &back.to_vec_f32().unwrap(),
        "roundtrip",
    );
}

#[test]
fn every_unary_op_matches_cpu() {
    let device = metal();
    // Positive so ln/sqrt agree everywhere.
    let t = Tensor::randn_with_seed([5, 33], 2)
        .abs()
        .unwrap()
        .add_scalar(0.25)
        .unwrap();
    let g = t.to_device(device).unwrap();
    for &op in UnaryOp::all() {
        let cpu = backend_for(Device::Cpu).unwrap().unary(op, &t).unwrap();
        let gpu = backend_for(device).unwrap().unary(op, &g).unwrap();
        assert_close(
            &cpu.to_vec_f32().unwrap(),
            &gpu.to_device(Device::Cpu).unwrap().to_vec_f32().unwrap(),
            op.name(),
        );
    }
}

#[test]
fn every_binary_op_matches_cpu_with_broadcast() {
    let device = metal();
    let a = Tensor::randn_with_seed([4, 1, 6], 3)
        .abs()
        .unwrap()
        .add_scalar(0.5)
        .unwrap();
    let b = Tensor::randn_with_seed([3, 6], 4)
        .abs()
        .unwrap()
        .add_scalar(0.5)
        .unwrap();
    let (ga, gb) = (a.to_device(device).unwrap(), b.to_device(device).unwrap());
    for &op in BinaryOp::all() {
        let cpu = backend_for(Device::Cpu)
            .unwrap()
            .binary(op, &a, &b)
            .unwrap();
        let gpu = backend_for(device).unwrap().binary(op, &ga, &gb).unwrap();
        assert_eq!(cpu.dims(), gpu.dims(), "{}", op.name());
        assert_close(
            &cpu.to_vec_f32().unwrap(),
            &gpu.to_device(Device::Cpu).unwrap().to_vec_f32().unwrap(),
            op.name(),
        );
    }
}

#[test]
fn matmul_matches_cpu() {
    let device = metal();
    for (m, k, n) in [(2, 2, 2), (17, 33, 9), (64, 64, 64), (37, 41, 29)] {
        let a = Tensor::randn_with_seed([m, k], (m + k) as u64);
        let b = Tensor::randn_with_seed([k, n], (k + n + 1) as u64);
        let cpu = a.matmul(&b).unwrap();
        let gpu = a
            .to_device(device)
            .unwrap()
            .matmul(&b.to_device(device).unwrap())
            .unwrap()
            .to_device(Device::Cpu)
            .unwrap();
        // Different summation orders: tolerance scales with k.
        let scale = (k as f32).sqrt();
        for (i, (&x, &y)) in cpu
            .to_vec_f32()
            .unwrap()
            .iter()
            .zip(&gpu.to_vec_f32().unwrap())
            .enumerate()
        {
            assert!(
                (x - y).abs() <= 1e-5 * scale * 1.0f32.max(x.abs()),
                "matmul {m}x{k}x{n} element {i}: {x} vs {y}"
            );
        }
    }
}

#[test]
fn batched_matmul_matches_cpu() {
    let device = metal();
    let a = Tensor::randn_with_seed([3, 8, 5], 40);
    let b = Tensor::randn_with_seed([3, 5, 7], 41);
    let cpu = a.matmul(&b).unwrap();
    let gpu = a
        .to_device(device)
        .unwrap()
        .matmul(&b.to_device(device).unwrap())
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap();
    assert_close(
        &cpu.to_vec_f32().unwrap(),
        &gpu.to_vec_f32().unwrap(),
        "bmm",
    );
}

#[test]
fn reductions_match_cpu_including_threadgroup_path() {
    let device = metal();
    // Small: per-axis kernel. Large: the two-stage threadgroup path.
    for (shape, axes) in [
        (vec![6usize, 7], vec![0usize]),
        (vec![6, 7], vec![1]),
        (vec![6, 7], vec![]),
        (vec![300, 400], vec![]), // 120k elements: threadgroup path
        (vec![300, 400], vec![0]),
    ] {
        let t = Tensor::randn_with_seed(Shape::new(shape.clone()), 50);
        let g = t.to_device(device).unwrap();
        for (name, cpu_r, gpu_r) in [
            ("sum", t.sum(&axes).unwrap(), g.sum(&axes).unwrap()),
            ("max", t.max(&axes).unwrap(), g.max(&axes).unwrap()),
            ("min", t.min(&axes).unwrap(), g.min(&axes).unwrap()),
            ("mean", t.mean(&axes).unwrap(), g.mean(&axes).unwrap()),
        ] {
            let gpu_host = gpu_r.to_device(Device::Cpu).unwrap();
            let cv = cpu_r.to_vec_f32().unwrap();
            let gv = gpu_host.to_vec_f32().unwrap();
            for (i, (&x, &y)) in cv.iter().zip(&gv).enumerate() {
                let tol = 1e-4 * 1.0f32.max(x.abs()); // summation order differs
                assert!(
                    (x - y).abs() <= tol,
                    "{name} {shape:?}/{axes:?} elem {i}: {x} vs {y}"
                );
            }
        }
    }
}

#[test]
fn strided_views_compute_correctly_on_metal() {
    let device = metal();
    let t = Tensor::randn_with_seed([6, 8], 60);
    let g = t.to_device(device).unwrap();
    let cpu = t.permute(&[1, 0]).unwrap().exp().unwrap();
    let gpu = g
        .permute(&[1, 0])
        .unwrap()
        .exp()
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap();
    assert_close(
        &cpu.to_vec_f32().unwrap(),
        &gpu.to_vec_f32().unwrap(),
        "exp over transpose",
    );

    let cpu_n = t.narrow(1, 2, 3).unwrap().contiguous().unwrap();
    let gpu_n = g
        .narrow(1, 2, 3)
        .unwrap()
        .contiguous()
        .unwrap()
        .to_device(Device::Cpu)
        .unwrap();
    assert_close(
        &cpu_n.to_vec_f32().unwrap(),
        &gpu_n.to_vec_f32().unwrap(),
        "narrow+contiguous",
    );
}

#[test]
fn autograd_flows_through_metal() {
    let device = metal();
    let x_host = Tensor::randn_with_seed([3, 4], 70);
    let x = x_host.to_device(device).unwrap().requires_grad_(true);
    let w = Tensor::randn_with_seed([4, 2], 71)
        .to_device(device)
        .unwrap()
        .requires_grad_(true);
    let y = x.matmul(&w).unwrap().tanh().unwrap().sum(&[]).unwrap();
    y.to_device(Device::Cpu).unwrap().backward().unwrap();
    let gx = x.grad().expect("x gradient");
    let gw = w.grad().expect("w gradient");
    assert_eq!(gx.dims(), &[3, 4]);
    assert_eq!(gw.dims(), &[4, 2]);
    // Non-degenerate gradients.
    let s: f32 = gx
        .to_device(Device::Cpu)
        .unwrap()
        .to_vec_f32()
        .unwrap()
        .iter()
        .map(|v| v.abs())
        .sum();
    assert!(s > 1e-3, "gradient collapsed to zero: {s}");
}
