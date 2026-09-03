//! `DType::F64` on the CPU backend (issue #27): storage, conversion, every
//! op family the CPU implements, autograd through f64, and the typed
//! errors for what f64 does not do (GPUs, mixed-dtype arithmetic).
use oxmera_core::{DType, Device};
use oxmera_tensor::tensor::Tensor;

fn f64s(v: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_vec_f64(v.to_vec(), shape.to_vec()).unwrap()
}

#[test]
fn storage_conversion_and_element_access() {
    let a = f64s(&[1.5, -2.25, 3.0, 4.125], &[2, 2]);
    assert_eq!(a.dtype(), DType::F64);
    assert_eq!(a.device(), Device::Cpu);
    assert_eq!(a.get_f64(&[1, 0]).unwrap(), 3.0);
    assert!(
        a.get_f32(&[0, 0]).is_err(),
        "f32 access on an f64 tensor is typed"
    );
    let f32v = a.to_dtype(DType::F32).unwrap();
    assert_eq!(f32v.dtype(), DType::F32);
    assert_eq!(f32v.to_vec_f32().unwrap(), vec![1.5, -2.25, 3.0, 4.125]);
    let back = f32v.to_dtype(DType::F64).unwrap();
    assert_eq!(back.to_vec_f64().unwrap(), vec![1.5, -2.25, 3.0, 4.125]);
    // Same dtype is a clone, I64 converts, transposed views convert in
    // logical order.
    assert_eq!(
        a.to_dtype(DType::F64).unwrap().to_vec_f64().unwrap(),
        a.to_vec_f64().unwrap()
    );
    let i = Tensor::from_vec_i64(vec![3, -1], [2]).unwrap();
    assert_eq!(
        i.to_dtype(DType::F64).unwrap().to_vec_f64().unwrap(),
        vec![3.0, -1.0]
    );
    assert_eq!(
        a.t().unwrap().to_vec_f64().unwrap(),
        vec![1.5, 3.0, -2.25, 4.125]
    );
    assert_eq!(
        a.t()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec_f32()
            .unwrap(),
        vec![1.5, 3.0, -2.25, 4.125]
    );
}

#[test]
fn f64_carries_what_f32_cannot() {
    // 100000001 is not an f32: storage in f64 keeps it, f32 rounds it away.
    let big = f64s(&[1e8 + 1.0, -1e8], &[2]);
    assert_eq!(big.sum(&[]).unwrap().to_vec_f64().unwrap(), vec![1.0]);
    let small = big.to_dtype(DType::F32).unwrap();
    assert_eq!(small.to_vec_f32().unwrap()[0], 1e8);
    assert_eq!(small.sum(&[]).unwrap().to_vec_f32().unwrap(), vec![0.0]);
    // (The f32 reduction itself is Neumaier-compensated: [1e8, 1, -1e8]
    // sums to exactly 1 in both dtypes — the loss is in the storage.)
    let three = f64s(&[1e8, 1.0, -1e8], &[3]);
    assert_eq!(three.sum(&[]).unwrap().to_vec_f64().unwrap(), vec![1.0]);
    assert_eq!(
        three
            .to_dtype(DType::F32)
            .unwrap()
            .sum(&[])
            .unwrap()
            .to_vec_f32()
            .unwrap(),
        vec![1.0]
    );
    // A product spanning 8e6 keeps 15 digits.
    let w = f64s(&[8145060.0, 1.0 / 8145060.0], &[2]);
    let p = w
        .narrow(0, 0, 1)
        .unwrap()
        .mul(&w.narrow(0, 1, 1).unwrap())
        .unwrap();
    assert!((p.to_vec_f64().unwrap()[0] - 1.0).abs() < 1e-15);
}

#[test]
fn every_op_family_runs_in_f64_and_matches_the_f32_formula() {
    let a32 = Tensor::randn_with_seed([3, 4], 91)
        .abs()
        .unwrap()
        .add_scalar(0.5)
        .unwrap();
    let b32 = Tensor::randn_with_seed([1, 4], 92)
        .abs()
        .unwrap()
        .add_scalar(0.5)
        .unwrap();
    let (a, b) = (
        a32.to_dtype(DType::F64).unwrap(),
        b32.to_dtype(DType::F64).unwrap(),
    );
    let close = |x: &Tensor, y: &Tensor, ctx: &str| {
        let (xv, yv) = (x.to_vec_f64().unwrap(), y.to_vec_f32().unwrap());
        assert_eq!(xv.len(), yv.len(), "{ctx}: length");
        for (i, (p, q)) in xv.iter().zip(yv).enumerate() {
            assert!(
                (p - q as f64).abs() <= 1e-5 * 1.0f64.max(q.abs() as f64),
                "{ctx}[{i}]: {p} vs {q}"
            );
        }
    };
    close(&a.exp().unwrap(), &a32.exp().unwrap(), "exp");
    close(&a.ln().unwrap(), &a32.ln().unwrap(), "ln");
    close(&a.gelu().unwrap(), &a32.gelu().unwrap(), "gelu");
    close(
        &a.add(&b).unwrap(),
        &a32.add(&b32).unwrap(),
        "add broadcast",
    );
    close(
        &a.div(&b).unwrap(),
        &a32.div(&b32).unwrap(),
        "div broadcast",
    );
    close(&a.pow(&b).unwrap(), &a32.pow(&b32).unwrap(), "pow");
    close(&a.gt_mask(&b).unwrap(), &a32.gt_mask(&b32).unwrap(), "gt");
    close(&a.sum(&[0]).unwrap(), &a32.sum(&[0]).unwrap(), "sum axis 0");
    close(&a.max(&[1]).unwrap(), &a32.max(&[1]).unwrap(), "max axis 1");
    close(&a.mean(&[]).unwrap(), &a32.mean(&[]).unwrap(), "mean");
    close(&a.softmax(1).unwrap(), &a32.softmax(1).unwrap(), "softmax");
    close(
        &a.matmul(&a.t().unwrap()).unwrap(),
        &a32.matmul(&a32.t().unwrap()).unwrap(),
        "matmul",
    );
    close(
        &a.t().unwrap().contiguous().unwrap(),
        &a32.t().unwrap().contiguous().unwrap(),
        "contiguous",
    );
    assert_eq!(
        a.argmax(1, false).unwrap().to_vec_i64().unwrap(),
        a32.argmax(1, false).unwrap().to_vec_i64().unwrap()
    );
    let idx = Tensor::from_vec_i64(vec![2, 0, 2], [3]).unwrap();
    close(
        &a.index_select(0, &idx).unwrap(),
        &a32.index_select(0, &idx).unwrap(),
        "index_select",
    );
    let src = a.index_select(0, &idx).unwrap();
    close(
        &a.index_add(0, &idx, &src).unwrap(),
        &a32.index_add(0, &idx, &src.to_dtype(DType::F32).unwrap())
            .unwrap(),
        "index_add",
    );
    // Linear algebra: an SPD matrix built in f64.
    let spd = a
        .matmul(&a.t().unwrap())
        .unwrap()
        .add(&Tensor::eye(3).to_dtype(DType::F64).unwrap())
        .unwrap();
    let spd32 = spd.to_dtype(DType::F32).unwrap();
    close(
        &spd.cholesky().unwrap(),
        &spd32.cholesky().unwrap(),
        "cholesky",
    );
    close(&spd.logdet().unwrap(), &spd32.logdet().unwrap(), "logdet");
    close(&spd.trace().unwrap(), &spd32.trace().unwrap(), "trace");
    let (w, _) = spd.eigh().unwrap();
    assert_eq!(w.dtype(), DType::F64);
    close(&w, &spd32.eigh().unwrap().0, "eigh values");
}

#[test]
fn autograd_flows_through_f64_and_through_to_dtype() {
    let x = f64s(&[0.5, 1.5, -2.0], &[3]).requires_grad_(true);
    // loss = Σ x² — gradient 2x, in f64.
    let loss = x.mul(&x).unwrap().sum(&[]).unwrap();
    assert_eq!(loss.dtype(), DType::F64);
    loss.backward().unwrap();
    let g = x.grad().unwrap();
    assert_eq!(g.dtype(), DType::F64);
    assert_eq!(g.to_vec_f64().unwrap(), vec![1.0, 3.0, -4.0]);
    // Through a narrow (index_add VJP in f64) and a mean (scalar_on in f64).
    let y = f64s(&[1.0, 2.0, 3.0, 4.0], &[4]).requires_grad_(true);
    y.narrow(0, 1, 2)
        .unwrap()
        .mean(&[])
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(
        y.grad().unwrap().to_vec_f64().unwrap(),
        vec![0.0, 0.5, 0.5, 0.0]
    );
    // An f32 leaf, evaluated in f64: the gradient comes back as f32.
    let z = Tensor::from_slice(&[2.0, 3.0], [2])
        .unwrap()
        .requires_grad_(true);
    z.to_dtype(DType::F64)
        .unwrap()
        .mul_scalar(10.0)
        .unwrap()
        .sum(&[])
        .unwrap()
        .backward()
        .unwrap();
    let gz = z.grad().unwrap();
    assert_eq!(gz.dtype(), DType::F32);
    assert_eq!(gz.to_vec_f32().unwrap(), vec![10.0, 10.0]);
}

#[test]
fn what_f64_does_not_do_is_a_typed_error() {
    let a = f64s(&[1.0, 2.0], &[2]);
    let b = Tensor::from_slice(&[1.0, 2.0], [2]).unwrap();
    let e = a.add(&b).unwrap_err().to_string();
    assert!(e.contains("dtype mismatch"), "{e}");
    let e = a
        .matmul(&Tensor::zeros([2usize, 2]))
        .unwrap_err()
        .to_string();
    assert!(e.contains("dtype mismatch"), "{e}");
    let e = a
        .to_device(Device::Metal { index: 0 })
        .unwrap_err()
        .to_string();
    assert!(e.contains("F64") && e.contains("CPU"), "{e}");
    let e = a.to_dtype(DType::U8).unwrap_err().to_string();
    assert!(e.contains("U8"), "{e}");
}
