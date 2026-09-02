//! CPU ↔ CUDA numerical parity, the same contract as the Metal suite:
//! every op family computed on both backends agrees within 1e-5 (relative
//! for large magnitudes). Needs a GPU: run with
//! `cargo test -p oxmera-cuda --features hardware`; without the feature
//! the tests compile and are ignored, so CI stays green on plain runners.
use oxmera_core::{Device, Shape};
use oxmera_tensor::backend::{BinaryOp, UnaryOp, backend_for};
use oxmera_tensor::tensor::Tensor;

const TOL: f32 = 1e-5;

fn cuda() -> Device {
    oxmera_cuda::register_default();
    let device = Device::Cuda { index: 0 };
    assert!(
        backend_for(device).is_ok(),
        "no CUDA device on this machine (driver present: {})",
        oxmera_cuda::is_driver_present()
    );
    device
}

fn assert_close(cpu: &[f32], gpu: &[f32], context: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{context}: length");
    for (i, (&x, &y)) in cpu.iter().zip(gpu).enumerate() {
        let tol = TOL * 1.0f32.max(x.abs());
        assert!(
            (x - y).abs() <= tol || (x.is_nan() && y.is_nan()),
            "{context}: element {i}: cpu {x} vs cuda {y}"
        );
    }
}

fn back(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu).unwrap().to_vec_f32().unwrap()
}

#[test]
#[cfg_attr(not(feature = "hardware"), ignore)]
fn upload_download_roundtrip() {
    let device = cuda();
    let t = Tensor::randn_with_seed([7, 9], 1);
    let g = t.to_device(device).unwrap();
    assert_eq!(g.device(), device);
    assert_close(&t.to_vec_f32().unwrap(), &back(&g), "roundtrip");
}

#[test]
#[cfg_attr(not(feature = "hardware"), ignore)]
fn every_unary_op_matches_cpu() {
    let device = cuda();
    let t = Tensor::randn_with_seed([5, 33], 2)
        .abs()
        .unwrap()
        .add_scalar(0.25)
        .unwrap();
    let g = t.to_device(device).unwrap();
    for &op in UnaryOp::all() {
        let cpu = backend_for(Device::Cpu).unwrap().unary(op, &t).unwrap();
        let gpu = backend_for(device).unwrap().unary(op, &g).unwrap();
        assert_close(&cpu.to_vec_f32().unwrap(), &back(&gpu), &format!("{op:?}"));
    }
}

#[test]
#[cfg_attr(not(feature = "hardware"), ignore)]
fn every_binary_op_matches_cpu_with_broadcasting() {
    let device = cuda();
    let a = Tensor::randn_with_seed([4, 6], 3)
        .abs()
        .unwrap()
        .add_scalar(0.5)
        .unwrap();
    let b = Tensor::randn_with_seed([1, 6], 4)
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
        assert_close(&cpu.to_vec_f32().unwrap(), &back(&gpu), &format!("{op:?}"));
    }
}

#[test]
#[cfg_attr(not(feature = "hardware"), ignore)]
fn reductions_match_cpu_on_every_axis_and_in_full() {
    let device = cuda();
    let t = Tensor::randn_with_seed([6, 7, 8], 5);
    let g = t.to_device(device).unwrap();
    for axes in [&[0usize][..], &[1], &[2], &[0, 2], &[0, 1, 2]] {
        for keep in [false, true] {
            let cpu = t.sum_keepdim(axes, keep).unwrap();
            let gpu = g.sum_keepdim(axes, keep).unwrap();
            assert_eq!(cpu.dims(), gpu.dims());
            assert_close(
                &cpu.to_vec_f32().unwrap(),
                &back(&gpu),
                &format!("sum {axes:?} keep={keep}"),
            );
            let cpu = t.max_keepdim(axes, keep).unwrap();
            let gpu = g.max_keepdim(axes, keep).unwrap();
            assert_close(
                &cpu.to_vec_f32().unwrap(),
                &back(&gpu),
                &format!("max {axes:?} keep={keep}"),
            );
        }
    }
    // Large enough for the two-stage block kernel.
    let big = Tensor::randn_with_seed([300, 400], 6);
    let gbig = big.to_device(device).unwrap();
    let cpu = big.sum(&[0, 1]).unwrap().to_vec_f32().unwrap()[0];
    let gpu = back(&gbig.sum(&[0, 1]).unwrap())[0];
    let scale = 1.0 + (300.0f32 * 400.0).sqrt();
    assert!(
        (cpu - gpu).abs() <= TOL * scale * cpu.abs().max(1.0),
        "full sum: {cpu} vs {gpu}"
    );
    assert_close(
        &back(&gbig.max(&[0, 1]).unwrap()),
        &big.max(&[0, 1]).unwrap().to_vec_f32().unwrap(),
        "full max",
    );
}

#[test]
#[cfg_attr(not(feature = "hardware"), ignore)]
fn matmul_matches_cpu_including_batch_broadcast() {
    let device = cuda();
    let cases: &[(&[usize], &[usize])] = &[
        (&[5, 7], &[7, 3]),
        (&[33, 65], &[65, 17]),
        (&[2, 2, 3], &[1, 3, 2]),
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
        let k = sa[sa.len() - 1] as f32;
        let scale = 1.0 + k.sqrt();
        for (j, (&x, y)) in cpu.to_vec_f32().unwrap().iter().zip(back(&gpu)).enumerate() {
            assert!(
                (x - y).abs() <= TOL * scale * x.abs().max(1.0),
                "{sa:?} x {sb:?} [{j}]: {x} vs {y}"
            );
        }
    }
}

#[test]
#[cfg_attr(not(feature = "hardware"), ignore)]
fn views_softmax_and_argmax_match_cpu() {
    let device = cuda();
    let t = Tensor::randn_with_seed([9, 11], 7);
    let g = t.to_device(device).unwrap();
    assert_close(
        &t.t().unwrap().contiguous().unwrap().to_vec_f32().unwrap(),
        &back(&g.t().unwrap().contiguous().unwrap()),
        "transpose contiguous",
    );
    assert_close(
        &t.narrow(1, 2, 5)
            .unwrap()
            .contiguous()
            .unwrap()
            .to_vec_f32()
            .unwrap(),
        &back(&g.narrow(1, 2, 5).unwrap().contiguous().unwrap()),
        "narrow",
    );
    assert_close(
        &t.softmax(1).unwrap().to_vec_f32().unwrap(),
        &back(&g.softmax(1).unwrap()),
        "softmax",
    );
    assert_close(
        &t.log_softmax(1).unwrap().to_vec_f32().unwrap(),
        &back(&g.log_softmax(1).unwrap()),
        "log_softmax",
    );
    assert_eq!(
        t.argmax(1, false).unwrap().to_vec_i64().unwrap(),
        g.argmax(1, false).unwrap().to_vec_i64().unwrap()
    );
}

#[test]
#[cfg_attr(not(feature = "hardware"), ignore)]
fn zero_element_tensors_are_safe_on_every_path() {
    let device = cuda();
    for dims in [vec![0usize, 3], vec![2, 0], vec![0], vec![2, 0, 5]] {
        let t = Tensor::zeros(Shape::new(dims.clone()));
        let g = t.to_device(device).unwrap();
        assert_eq!(g.numel(), 0);
        assert!(back(&g).is_empty());
        assert_eq!(g.relu().unwrap().numel(), 0);
        assert_eq!(g.add(&g).unwrap().numel(), 0);
        let axes: Vec<usize> = (0..dims.len()).collect();
        assert_eq!(back(&g.sum(&axes).unwrap()), vec![0.0], "{dims:?}");
    }
    let a = Tensor::zeros([2usize, 0]).to_device(device).unwrap();
    let b = Tensor::zeros([0usize, 3]).to_device(device).unwrap();
    assert_eq!(back(&a.matmul(&b).unwrap()), vec![0.0; 6]);
}

#[test]
#[cfg_attr(not(feature = "hardware"), ignore)]
fn autograd_runs_end_to_end_on_the_device() {
    let device = cuda();
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
    // Same computation on the CPU.
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
    for (i, (&c, g)) in wc
        .grad()
        .unwrap()
        .to_vec_f32()
        .unwrap()
        .iter()
        .zip(back(&gw))
        .enumerate()
    {
        assert!(
            (c - g).abs() <= TOL * scale * c.abs().max(1.0),
            "grad[{i}]: {c} vs {g}"
        );
    }
}
