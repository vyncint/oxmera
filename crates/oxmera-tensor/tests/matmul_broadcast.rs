//! matmul batch semantics (issue #22): batch dims broadcast, a rank-2
//! operand is batch 1, and the result is rank 2 only when both inputs are.
use oxmera_tensor::backend::plan_matmul;
use oxmera_tensor::tensor::Tensor;

fn seq(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * scale - 1.0).collect()
}

fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k).map(|x| a[i * k + x] * b[x * n + j]).sum();
        }
    }
    out
}

#[test]
fn a_batch_of_one_broadcasts_against_a_batch_of_two() {
    let a = Tensor::from_slice(&seq(2 * 2 * 3, 0.1), [2, 2, 3]).unwrap();
    let b = Tensor::from_slice(&seq(3 * 2, 0.2), [1, 3, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[2, 2, 2]);
    // Same as multiplying each batch of `a` by the one `b`.
    let av = a.to_vec_f32().unwrap();
    let bv = b.to_vec_f32().unwrap();
    let mut want = reference(&av[..6], &bv, 2, 3, 2);
    want.extend(reference(&av[6..], &bv, 2, 3, 2));
    assert_eq!(c.to_vec_f32().unwrap(), want);
}

#[test]
fn rank_two_times_rank_three_is_batched() {
    let a = Tensor::from_slice(&seq(2 * 3, 0.3), [2, 3]).unwrap();
    let b = Tensor::from_slice(&seq(4 * 3 * 5, 0.05), [4, 3, 5]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[4, 2, 5]);
    let (av, bv) = (a.to_vec_f32().unwrap(), b.to_vec_f32().unwrap());
    let got = c.to_vec_f32().unwrap();
    for i in 0..4 {
        assert_eq!(
            &got[i * 10..(i + 1) * 10],
            &reference(&av, &bv[i * 15..(i + 1) * 15], 2, 3, 5)[..],
            "batch {i}"
        );
    }
}

#[test]
fn rank_three_times_rank_two_is_batched() {
    let a = Tensor::from_slice(&seq(4 * 2 * 3, 0.05), [4, 2, 3]).unwrap();
    let b = Tensor::from_slice(&seq(3 * 5, 0.3), [3, 5]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[4, 2, 5]);
    let (av, bv) = (a.to_vec_f32().unwrap(), b.to_vec_f32().unwrap());
    let got = c.to_vec_f32().unwrap();
    for i in 0..4 {
        assert_eq!(
            &got[i * 10..(i + 1) * 10],
            &reference(&av[i * 6..(i + 1) * 6], &bv, 2, 3, 5)[..],
            "batch {i}"
        );
    }
}

#[test]
fn equal_batches_and_plain_rank_two_are_unchanged() {
    let a = Tensor::from_slice(&seq(2 * 2 * 3, 0.1), [2, 2, 3]).unwrap();
    let b = Tensor::from_slice(&seq(2 * 3 * 2, 0.2), [2, 3, 2]).unwrap();
    assert_eq!(a.matmul(&b).unwrap().dims(), &[2, 2, 2]);
    let a2 = Tensor::from_slice(&seq(6, 0.1), [2, 3]).unwrap();
    let b2 = Tensor::from_slice(&seq(6, 0.2), [3, 2]).unwrap();
    assert_eq!(a2.matmul(&b2).unwrap().dims(), &[2, 2]);
}

#[test]
fn incompatible_batches_and_inner_dims_are_typed_errors() {
    let a = Tensor::zeros([2usize, 2, 3]);
    let b = Tensor::zeros([3usize, 3, 2]);
    let e = a.matmul(&b).unwrap_err().to_string();
    assert!(e.contains("broadcast"), "{e}");
    let b = Tensor::zeros([2usize, 4, 2]);
    let e = a.matmul(&b).unwrap_err().to_string();
    assert!(e.contains("matmul"), "{e}");
    assert!(
        Tensor::zeros([2usize, 3])
            .matmul(&Tensor::zeros([2usize, 3, 4, 5]))
            .is_err()
    );
}

#[test]
fn the_plan_reports_zero_stride_for_the_broadcast_operand() {
    let p = plan_matmul(&[2usize, 2, 3].into(), &[1usize, 3, 2].into()).unwrap();
    assert_eq!((p.batch, p.m, p.k, p.n), (2, 2, 3, 2));
    assert_eq!((p.a_batch_stride, p.b_batch_stride), (6, 0));
    let p = plan_matmul(&[2usize, 3].into(), &[4usize, 3, 5].into()).unwrap();
    assert_eq!((p.a_batch_stride, p.b_batch_stride), (0, 15));
    assert_eq!(p.out_shape.dims(), &[4, 2, 5]);
}
