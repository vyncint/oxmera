//! CPU backend correctness: every elementwise op against its scalar
//! reference (including strided/broadcast inputs and rayon-sized tensors),
//! reductions and argmax against hand-walked answers.

use oxmera_core::Shape;
use oxmera_tensor::backend::{BinaryOp, UnaryOp};
use oxmera_tensor::tensor::Tensor;

fn assert_close(a: &[f32], b: &[f32], tol: f32, context: &str) {
    assert_eq!(a.len(), b.len(), "{context}: length");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let ok = (x - y).abs() <= tol || (x.is_nan() && y.is_nan());
        assert!(ok, "{context}: element {i}: {x} vs {y}");
    }
}

fn unary_method(op: UnaryOp, t: &Tensor) -> Tensor {
    match op {
        UnaryOp::Neg => t.neg(),
        UnaryOp::Exp => t.exp(),
        UnaryOp::Ln => t.ln(),
        UnaryOp::Abs => t.abs(),
        UnaryOp::Sqrt => t.sqrt(),
        UnaryOp::Sin => t.sin(),
        UnaryOp::Cos => t.cos(),
        UnaryOp::Tanh => t.tanh(),
        UnaryOp::Relu => t.relu(),
        UnaryOp::Gelu => t.gelu(),
        UnaryOp::Sigmoid => t.sigmoid(),
        _ => unreachable!("all() is exhaustive today"),
    }
    .unwrap()
}

fn binary_method(op: BinaryOp, a: &Tensor, b: &Tensor) -> Tensor {
    match op {
        BinaryOp::Add => a.add(b),
        BinaryOp::Sub => a.sub(b),
        BinaryOp::Mul => a.mul(b),
        BinaryOp::Div => a.div(b),
        BinaryOp::Pow => a.pow(b),
        BinaryOp::Maximum => a.maximum(b),
        BinaryOp::Minimum => a.minimum(b),
        BinaryOp::Gt => a.gt_mask(b),
        BinaryOp::Eq => a.eq_mask(b),
        _ => unreachable!("all() is exhaustive today"),
    }
    .unwrap()
}

#[test]
fn every_unary_op_matches_its_scalar_reference() {
    // Positive values so ln/sqrt are defined everywhere.
    let data: Vec<f32> = (1..=24).map(|i| i as f32 * 0.37).collect();
    let t = Tensor::from_vec_f32(data.clone(), Shape::from([4, 6])).unwrap();
    for &op in UnaryOp::all() {
        let got = unary_method(op, &t).to_vec_f32().unwrap();
        let want: Vec<f32> = data.iter().map(|&x| op.eval(x)).collect();
        assert_close(&got, &want, 1e-6, op.name());
    }
}

#[test]
fn unary_ops_respect_strided_views() {
    let t = Tensor::from_vec_f32(
        (0..12).map(|i| i as f32 + 1.0).collect(),
        Shape::from([3, 4]),
    )
    .unwrap();
    let view = t.permute(&[1, 0]).unwrap();
    let got = view.exp().unwrap().to_vec_f32().unwrap();
    let want: Vec<f32> = view
        .to_vec_f32()
        .unwrap()
        .iter()
        .map(|&x| x.exp())
        .collect();
    assert_close(&got, &want, 1e-6, "exp over transpose");
}

#[test]
fn every_binary_op_matches_its_scalar_reference_with_broadcast() {
    let a_data: Vec<f32> = (1..=6).map(|i| i as f32).collect();
    let b_data: Vec<f32> = vec![0.5, 2.0, 3.0];
    let a = Tensor::from_vec_f32(a_data.clone(), Shape::from([2, 3])).unwrap();
    let b = Tensor::from_vec_f32(b_data.clone(), Shape::from([3])).unwrap();
    for &op in BinaryOp::all() {
        let got = binary_method(op, &a, &b).to_vec_f32().unwrap();
        let want: Vec<f32> = (0..6).map(|i| op.eval(a_data[i], b_data[i % 3])).collect();
        assert_close(&got, &want, 1e-6, op.name());
    }
}

#[test]
fn large_tensors_take_the_parallel_path_and_agree() {
    let n = 100_000usize;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-3 - 50.0).collect();
    let t = Tensor::from_vec_f32(data.clone(), Shape::from([n])).unwrap();
    let got = t.tanh().unwrap().to_vec_f32().unwrap();
    let want: Vec<f32> = data.iter().map(|&x| x.tanh()).collect();
    assert_close(&got, &want, 1e-6, "parallel tanh");

    let sum = t.sum(&[]).unwrap().get_f32(&[]).unwrap();
    let want_sum: f32 = data.iter().sum();
    assert!(
        (sum - want_sum).abs() <= want_sum.abs() * 1e-4 + 1e-2,
        "{sum} vs {want_sum}"
    );
}

#[test]
fn reductions_hand_cases() {
    // [[1, 2, 3], [4, 5, 6]]
    let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from([2, 3])).unwrap();
    assert_eq!(t.sum(&[]).unwrap().get_f32(&[]).unwrap(), 21.0);
    assert_eq!(
        t.sum(&[0]).unwrap().to_vec_f32().unwrap(),
        vec![5.0, 7.0, 9.0]
    );
    assert_eq!(t.sum(&[1]).unwrap().to_vec_f32().unwrap(), vec![6.0, 15.0]);
    assert_eq!(t.sum_keepdim(&[1], true).unwrap().dims(), &[2, 1]);
    assert_eq!(t.max(&[1]).unwrap().to_vec_f32().unwrap(), vec![3.0, 6.0]);
    assert_eq!(
        t.min(&[0]).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(t.mean(&[]).unwrap().get_f32(&[]).unwrap(), 3.5);
    assert_eq!(
        t.mean(&[0]).unwrap().to_vec_f32().unwrap(),
        vec![2.5, 3.5, 4.5]
    );
}

#[test]
fn reductions_over_strided_views() {
    let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from([2, 3])).unwrap();
    let tt = t.permute(&[1, 0]).unwrap(); // [[1,4],[2,5],[3,6]]
    assert_eq!(
        tt.sum(&[1]).unwrap().to_vec_f32().unwrap(),
        vec![5.0, 7.0, 9.0]
    );
}

#[test]
fn argmax_hand_cases() {
    let t = Tensor::from_vec_f32(vec![1.0, 9.0, 3.0, 4.0, 5.0, 6.0], Shape::from([2, 3])).unwrap();
    assert_eq!(
        t.argmax(1, false).unwrap().to_vec_i64().unwrap(),
        vec![1, 2]
    );
    assert_eq!(
        t.argmax(0, false).unwrap().to_vec_i64().unwrap(),
        vec![1, 0, 1]
    );
    assert_eq!(t.argmax(1, true).unwrap().dims(), &[2, 1]);
}

#[test]
fn index_select_and_index_add_hand_cases() {
    let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from([3, 2])).unwrap();
    let idx = Tensor::from_vec_i64(vec![2, 0], Shape::from([2])).unwrap();
    let sel = t.index_select(0, &idx).unwrap();
    assert_eq!(sel.to_vec_f32().unwrap(), vec![5.0, 6.0, 1.0, 2.0]);

    let src = Tensor::from_vec_f32(vec![10.0, 20.0, 30.0, 40.0], Shape::from([2, 2])).unwrap();
    let added = t.index_add(0, &idx, &src).unwrap();
    assert_eq!(
        added.to_vec_f32().unwrap(),
        vec![31.0, 42.0, 3.0, 4.0, 15.0, 26.0]
    );
}
