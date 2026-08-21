//! Spec for exercise A4 — the reference matmul.
//!
//! This is the rung that creates ground truth: every backend oxmera ever
//! grows is validated against what you write here. Correct and deliberately
//! naive; the spec asserts hand-computed products and algebraic laws, and
//! contains no multiplication loop of its own.

use oxmera_core::{Error, Shape};
use oxmera_cpu::CpuBackend;
use oxmera_ops::MatmulOps;
use oxmera_tensor::Tensor;
use proptest::prelude::*;

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec_f32(data.to_vec(), Shape::from(shape)).unwrap()
}

fn assert_tensor_eq(got: &Tensor, expected_data: &[f32], expected_shape: &[usize]) {
    assert_eq!(got.shape(), &Shape::from(expected_shape));
    let dims = got.shape().dims().to_vec();
    let mut idx = 0usize;
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            let v = got.get_f32(&[i, j]).unwrap();
            assert_eq!(v, expected_data[idx], "at [{i}, {j}]");
            idx += 1;
        }
    }
}

#[test]
fn hand_computed_2x2() {
    // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
    let c = CpuBackend
        .matmul(&t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), &t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();
    assert_tensor_eq(&c, &[19.0, 22.0, 43.0, 50.0], &[2, 2]);
}

#[test]
fn hand_computed_rectangular() {
    // [2,3] x [3,1]: [[1,2,3],[4,5,6]] x [[1],[10],[100]] = [[321],[654]]
    let c = CpuBackend
        .matmul(
            &t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
            &t(&[1.0, 10.0, 100.0], &[3, 1]),
        )
        .unwrap();
    assert_tensor_eq(&c, &[321.0, 654.0], &[2, 1]);
}

#[test]
fn identity_leaves_a_matrix_alone() {
    let a = t(&[1.5, -2.0, 0.25, 4.0], &[2, 2]);
    let eye = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = CpuBackend.matmul(&a, &eye).unwrap();
    assert_tensor_eq(&c, &[1.5, -2.0, 0.25, 4.0], &[2, 2]);
}

#[test]
fn zero_annihilates() {
    let a = t(&[3.0, -1.0, 2.0, 7.0, 0.5, -4.0], &[2, 3]);
    let zero = t(&[0.0; 6], &[3, 2]);
    let c = CpuBackend.matmul(&a, &zero).unwrap();
    assert_tensor_eq(&c, &[0.0; 4], &[2, 2]);
}

#[test]
fn inner_dimension_mismatch_is_a_typed_error() {
    let a = t(&[1.0; 6], &[2, 3]);
    let b = t(&[1.0; 8], &[4, 2]);
    assert!(matches!(
        CpuBackend.matmul(&a, &b),
        Err(Error::ShapeMismatch { .. })
    ));
}

#[test]
fn non_rank_2_operands_are_refused() {
    let a = t(&[1.0; 6], &[6]);
    let b = t(&[1.0; 6], &[6]);
    assert!(CpuBackend.matmul(&a, &b).is_err(), "rank-1 operands must refuse");
}

#[test]
fn matmul_works_on_a_transposed_view() {
    // (Bᵀ has layout strides, not fresh bytes — the reference must read
    // through the view correctly.)
    let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = t(&[5.0, 7.0, 6.0, 8.0], &[2, 2]); // Bᵀ in memory
    let bt = b.permute(&[1, 0]).unwrap(); // logical [[5,6],[7,8]]
    let c = CpuBackend.matmul(&a, &bt).unwrap();
    assert_tensor_eq(&c, &[19.0, 22.0, 43.0, 50.0], &[2, 2]);
}

fn arb_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Tensor> {
    proptest::collection::vec(-8.0f32..8.0, rows * cols)
        .prop_map(move |d| t(&d, &[rows, cols]))
}

proptest! {
    /// The transpose law: (A·B)ᵀ = Bᵀ·Aᵀ — an algebraic identity the
    /// implementation must satisfy, checked through views.
    #[test]
    fn transpose_law(a in arb_matrix(2, 3), b in arb_matrix(3, 2)) {
        let ab_t = CpuBackend.matmul(&a, &b).unwrap().permute(&[1, 0]).unwrap();
        let bt_at = CpuBackend
            .matmul(&b.permute(&[1, 0]).unwrap(), &a.permute(&[1, 0]).unwrap())
            .unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let x = ab_t.get_f32(&[i, j]).unwrap();
                let y = bt_at.get_f32(&[i, j]).unwrap();
                prop_assert!((x - y).abs() <= 1e-4 * (1.0 + x.abs().max(y.abs())),
                    "({i},{j}): {x} vs {y}");
            }
        }
    }

    /// Column locality: column j of A·B equals A · (column j of B).
    #[test]
    fn columns_are_independent(a in arb_matrix(3, 3), b in arb_matrix(3, 2), j in 0usize..2) {
        let full = CpuBackend.matmul(&a, &b).unwrap();
        let col = CpuBackend.matmul(&a, &b.narrow(1, j, 1).unwrap()).unwrap();
        for i in 0..3 {
            let x = full.get_f32(&[i, j]).unwrap();
            let y = col.get_f32(&[i, 0]).unwrap();
            prop_assert!((x - y).abs() <= 1e-4 * (1.0 + x.abs().max(y.abs())));
        }
    }
}
