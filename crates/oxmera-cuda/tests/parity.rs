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

/// Native gather/scatter (issue #25): index_select and index_add run on
/// the device — duplicate indices, a strided source, dim 0 and dim 1 —
/// and match the CPU exactly.
#[test]
#[cfg_attr(not(feature = "hardware"), ignore)]
fn index_select_and_index_add_match_cpu() {
    let device = cuda();
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
            &back(&gpu),
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
            &back(&gpu_add),
            &format!("index_add dim {dim} with duplicate indices"),
        );
    }
    let at = a.t().unwrap();
    let gt = g.t().unwrap();
    let idx = Tensor::from_vec_i64(vec![6, 0, 3], Shape::from([3])).unwrap();
    assert_close(
        &at.index_select(0, &idx).unwrap().to_vec_f32().unwrap(),
        &back(&gt.index_select(0, &idx).unwrap()),
        "index_select on a transposed view",
    );
    let src = Tensor::randn_with_seed([5, 3], 63).t().unwrap();
    assert_close(
        &at.index_add(0, &idx, &src).unwrap().to_vec_f32().unwrap(),
        &back(
            &gt.index_add(0, &idx, &src.to_device(device).unwrap())
                .unwrap(),
        ),
        "index_add on transposed views",
    );
    let bad = Tensor::from_vec_i64(vec![0, 9], Shape::from([2])).unwrap();
    assert!(g.index_select(0, &bad).is_err());
    let none = Tensor::from_vec_i64(vec![], Shape::from([0])).unwrap();
    assert_eq!(g.index_select(1, &none).unwrap().dims(), &[5, 0]);
}

/// The narrow VJP (index_add into zeros on the loss's device) — the shape
/// that failed with a DeviceMismatch in oxmega's k-DPP loss on this
/// backend before 0.3.0.
#[test]
#[cfg_attr(not(feature = "hardware"), ignore)]
fn narrow_backward_runs_on_the_device() {
    let device = cuda();
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
    for (i, v) in back(&g).iter().enumerate() {
        let want = if (2..5).contains(&(i % 6)) { 2.0 } else { 0.0 };
        assert_eq!(*v, want, "element {i}");
    }
}

/// The fused Adam/AdamW step (issue #28) reproduces the composite CPU
/// update on the device, first step and after several, both decays.
#[test]
#[cfg_attr(not(feature = "hardware"), ignore)]
fn fused_adam_step_matches_the_composite_cpu_optimizer() {
    use oxmera_nn::Param;
    use oxmera_optim::{Adam, AdamW, Optimizer, ParamGroup};
    let device = cuda();
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
            assert_eq!(w_gpu.value().device(), device);
            assert_close(
                &w_cpu.value().to_vec_f32().unwrap(),
                &back(&w_gpu.value()),
                &format!("adam decoupled={decoupled} step {step}"),
            );
        }
    }
}
