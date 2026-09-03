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

/// Issue #17: a zero-element tensor must round-trip and run through every
/// op family without touching memory it does not own. Every shape here
/// has numel 0.
#[test]
fn zero_element_tensors_are_safe_on_every_path() {
    let device = metal();
    for dims in [vec![0usize, 3], vec![2, 0], vec![0], vec![2, 0, 5]] {
        let t = Tensor::zeros(Shape::new(dims.clone()));
        let g = t.to_device(device).unwrap();
        assert_eq!(g.numel(), 0, "{dims:?}");
        let back = g.to_device(Device::Cpu).unwrap();
        assert_eq!(back.dims(), &dims[..]);
        assert!(back.to_vec_f32().unwrap().is_empty());
        assert_eq!(g.relu().unwrap().numel(), 0, "{dims:?} unary");
        assert_eq!(g.add(&g).unwrap().numel(), 0, "{dims:?} binary");
        assert_eq!(g.contiguous().unwrap().numel(), 0, "{dims:?} contiguous");
        let axes: Vec<usize> = (0..dims.len()).collect();
        let s = g.sum(&axes).unwrap().to_device(Device::Cpu).unwrap();
        assert_eq!(
            s.to_vec_f32().unwrap(),
            vec![0.0],
            "{dims:?} sum of nothing is 0"
        );
    }
    // Empty matmul: [2,0] x [0,3] is a 2x3 block of zeros.
    let a = Tensor::zeros([2usize, 0]).to_device(device).unwrap();
    let b = Tensor::zeros([0usize, 3]).to_device(device).unwrap();
    let c = a.matmul(&b).unwrap().to_device(Device::Cpu).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.to_vec_f32().unwrap(), vec![0.0; 6]);
}

/// Issue #22: batch broadcasting and mixed-rank matmul agree with the CPU.
#[test]
fn matmul_batch_broadcast_matches_cpu() {
    let device = metal();
    let cases: &[(&[usize], &[usize])] = &[
        (&[2, 2, 3], &[1, 3, 2]),
        (&[1, 2, 3], &[4, 3, 2]),
        (&[2, 3], &[4, 3, 5]),
        (&[4, 2, 3], &[3, 5]),
        (&[3, 5, 7], &[3, 7, 2]),
    ];
    for (i, (sa, sb)) in cases.iter().enumerate() {
        let a = Tensor::randn_with_seed(Shape::new(sa.to_vec()), 100 + i as u64);
        let b = Tensor::randn_with_seed(Shape::new(sb.to_vec()), 200 + i as u64);
        let cpu = a.matmul(&b).unwrap();
        let gpu = a
            .to_device(device)
            .unwrap()
            .matmul(&b.to_device(device).unwrap())
            .unwrap();
        assert_eq!(cpu.dims(), gpu.dims(), "{sa:?} x {sb:?}");
        assert_close(
            &cpu.to_vec_f32().unwrap(),
            &gpu.to_device(Device::Cpu).unwrap().to_vec_f32().unwrap(),
            &format!("{sa:?} x {sb:?}"),
        );
    }
}

/// backward() on a device-resident loss: the seed and every gradient stay
/// on the device, and the leaf gradients match the CPU's.
#[test]
fn autograd_runs_end_to_end_on_the_device() {
    let device = metal();
    let x = Tensor::randn_with_seed([8, 4], 9)
        .to_device(device)
        .unwrap()
        .requires_grad_(true);
    let w = Tensor::randn_with_seed([4, 3], 10)
        .to_device(device)
        .unwrap()
        .requires_grad_(true);
    let loss = x.matmul(&w).unwrap().relu().unwrap().sum(&[0, 1]).unwrap();
    loss.backward().unwrap();
    let gw = w.grad().expect("grad on device");
    assert_eq!(gw.device(), device);
    let xc = x
        .detach()
        .to_device(Device::Cpu)
        .unwrap()
        .requires_grad_(true);
    let wc = w
        .detach()
        .to_device(Device::Cpu)
        .unwrap()
        .requires_grad_(true);
    xc.matmul(&wc)
        .unwrap()
        .relu()
        .unwrap()
        .sum(&[0, 1])
        .unwrap()
        .backward()
        .unwrap();
    let scale = 1.0 + 8.0f32.sqrt();
    let cpu = wc.grad().unwrap().to_vec_f32().unwrap();
    let gpu = gw.to_device(Device::Cpu).unwrap().to_vec_f32().unwrap();
    for (i, (&c, &g)) in cpu.iter().zip(&gpu).enumerate() {
        assert!(
            (c - g).abs() <= TOL * scale * c.abs().max(1.0),
            "grad[{i}]: {c} vs {g}"
        );
    }
}

/// Native gather/scatter (issue #25): index_select and index_add run on
/// the device — including duplicate indices, a strided (transposed)
/// source, dim 0 and dim 1 — and match the CPU exactly (the scatter adds
/// in index order, like the CPU reference).
#[test]
fn index_select_and_index_add_match_cpu() {
    let device = metal();
    let a = Tensor::randn_with_seed([5, 7], 61);
    let g = a.to_device(device).unwrap();
    for dim in 0..2 {
        let extent = a.dims()[dim] as i64;
        let idx = Tensor::from_vec_i64(vec![0, extent - 1, 2, 2, 1], Shape::from([5])).unwrap();
        let cpu = a.index_select(dim, &idx).unwrap();
        let gpu = g.index_select(dim, &idx).unwrap();
        assert_eq!(gpu.device(), device, "result stays on the device");
        assert_eq!(cpu.dims(), gpu.dims());
        assert_close(
            &cpu.to_vec_f32().unwrap(),
            &gpu.to_device(Device::Cpu).unwrap().to_vec_f32().unwrap(),
            &format!("index_select dim {dim}"),
        );
        let src = Tensor::randn_with_seed(cpu.shape().clone(), 62 + dim as u64);
        let cpu_add = a.index_add(dim, &idx, &src).unwrap();
        let gpu_add = g
            .index_add(dim, &idx, &src.to_device(device).unwrap())
            .unwrap();
        assert_eq!(gpu_add.device(), device);
        assert_close(
            &cpu_add.to_vec_f32().unwrap(),
            &gpu_add
                .to_device(Device::Cpu)
                .unwrap()
                .to_vec_f32()
                .unwrap(),
            &format!("index_add dim {dim} with duplicate indices"),
        );
    }
    // A transposed (strided) source and a strided src for index_add.
    let at = a.t().unwrap();
    let gt = g.t().unwrap();
    let idx = Tensor::from_vec_i64(vec![6, 0, 3], Shape::from([3])).unwrap();
    assert_close(
        &at.index_select(0, &idx).unwrap().to_vec_f32().unwrap(),
        &gt.index_select(0, &idx)
            .unwrap()
            .to_device(Device::Cpu)
            .unwrap()
            .to_vec_f32()
            .unwrap(),
        "index_select on a transposed view",
    );
    let src = Tensor::randn_with_seed([5, 3], 63).t().unwrap(); // [3, 5], strided
    assert_close(
        &at.index_add(0, &idx, &src).unwrap().to_vec_f32().unwrap(),
        &gt.index_add(0, &idx, &src.to_device(device).unwrap())
            .unwrap()
            .to_device(Device::Cpu)
            .unwrap()
            .to_vec_f32()
            .unwrap(),
        "index_add on transposed views",
    );
    // Out-of-range indices are typed errors, not kernel faults.
    let bad = Tensor::from_vec_i64(vec![0, 9], Shape::from([2])).unwrap();
    assert!(g.index_select(0, &bad).is_err());
    // Empty selection.
    let none = Tensor::from_vec_i64(vec![], Shape::from([0])).unwrap();
    let e = g.index_select(1, &none).unwrap();
    assert_eq!(e.dims(), &[5, 0]);
}

/// The narrow VJP is index_add into zeros on the loss's device: this is
/// exactly the shape that failed with a DeviceMismatch on CUDA (oxmega's
/// k-DPP log-det) before the fallback moved every operand.
#[test]
fn narrow_backward_runs_on_the_device() {
    let device = metal();
    let x = Tensor::randn_with_seed([4, 6], 64)
        .to_device(device)
        .unwrap()
        .requires_grad_(true);
    let loss = x
        .narrow(1, 2, 3)
        .unwrap()
        .mul_scalar(2.0)
        .unwrap()
        .sum(&[0, 1])
        .unwrap();
    loss.backward().unwrap();
    let g = x.grad().unwrap();
    assert_eq!(g.device(), device);
    let gv = g.to_device(Device::Cpu).unwrap().to_vec_f32().unwrap();
    for (i, v) in gv.iter().enumerate() {
        let col = i % 6;
        let want = if (2..5).contains(&col) { 2.0 } else { 0.0 };
        assert_eq!(*v, want, "element {i}");
    }
}

/// The fused Adam/AdamW step (issue #28) must reproduce the composite
/// CPU update — with and without decoupled decay, on the first step (no
/// state) and after several — within f32 rounding of the same formula.
#[test]
fn fused_adam_step_matches_the_composite_cpu_optimizer() {
    use oxmera_nn::Param;
    use oxmera_optim::{Adam, AdamW, Optimizer, ParamGroup};
    let device = metal();
    for decoupled in [false, true] {
        let init = Tensor::randn_with_seed([6, 7], 71);
        let w_cpu = Param::new(init.clone());
        let w_gpu = Param::new(init.to_device(device).unwrap());
        let mk = |p: Param| -> Box<dyn Optimizer> {
            let groups = vec![ParamGroup::new(vec![p], 0.05, 0.1)];
            if decoupled {
                Box::new(AdamW::with_groups(groups))
            } else {
                Box::new(Adam::with_groups(groups))
            }
        };
        let mut o_cpu = mk(w_cpu.clone());
        let mut o_gpu = mk(w_gpu.clone());
        for step in 0..4 {
            for w in [&w_cpu, &w_gpu] {
                // loss = Σ (w * seed)², a different gradient per element and step.
                let scale = Tensor::randn_with_seed([6, 7], 80 + step)
                    .to_device(w.value().device())
                    .unwrap();
                let v = w.value();
                let l = v.mul(&scale).unwrap();
                l.mul(&l).unwrap().sum(&[0, 1]).unwrap().backward().unwrap();
            }
            o_cpu.step().unwrap();
            o_gpu.step().unwrap();
            o_cpu.zero_grad();
            o_gpu.zero_grad();
            assert_eq!(
                w_gpu.value().device(),
                device,
                "parameter stays on the device"
            );
            assert_close(
                &w_cpu.value().to_vec_f32().unwrap(),
                &w_gpu
                    .value()
                    .to_device(Device::Cpu)
                    .unwrap()
                    .to_vec_f32()
                    .unwrap(),
                &format!("adam decoupled={decoupled} step {step}"),
            );
        }
    }
}
