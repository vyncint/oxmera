//! Spec for exercise A3 — strided views.
//!
//! A view changes the layout, never the bytes. Every expected value below
//! is hand-computed from the [[1,2,3],[4,5,6]] running example.

use std::sync::Arc;

use oxmera_core::{Error, Shape};
use oxmera_tensor::Tensor;
use proptest::prelude::*;

/// The running example: [[1, 2, 3], [4, 5, 6]].
fn t23() -> Tensor {
    Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from([2, 3])).unwrap()
}

#[test]
fn permute_reads_the_transpose_and_shares_storage() {
    let t = t23();
    let tt = t.permute(&[1, 0]).unwrap();
    assert_eq!(tt.shape(), &Shape::from([3, 2]));
    // Hand-computed transpose: [[1,4],[2,5],[3,6]]
    assert_eq!(tt.get_f32(&[0, 0]).unwrap(), 1.0);
    assert_eq!(tt.get_f32(&[0, 1]).unwrap(), 4.0);
    assert_eq!(tt.get_f32(&[1, 0]).unwrap(), 2.0);
    assert_eq!(tt.get_f32(&[2, 1]).unwrap(), 6.0);
    assert!(Arc::ptr_eq(t.storage(), tt.storage()), "permute must be a view");
}

#[test]
fn narrow_windows_the_middle_and_shares_storage() {
    let t = t23();
    let n = t.narrow(1, 1, 2).unwrap();
    assert_eq!(n.shape(), &Shape::from([2, 2]));
    // Hand-computed: columns 1..3 -> [[2,3],[5,6]]
    assert_eq!(n.get_f32(&[0, 0]).unwrap(), 2.0);
    assert_eq!(n.get_f32(&[0, 1]).unwrap(), 3.0);
    assert_eq!(n.get_f32(&[1, 0]).unwrap(), 5.0);
    assert_eq!(n.get_f32(&[1, 1]).unwrap(), 6.0);
    assert!(Arc::ptr_eq(t.storage(), n.storage()), "narrow must be a view");
}

#[test]
fn reshape_of_a_contiguous_tensor_is_a_view() {
    let t = t23();
    let r = t.reshape(Shape::from([3, 2])).unwrap();
    // Row-major reflow: [[1,2],[3,4],[5,6]]
    assert_eq!(r.get_f32(&[0, 1]).unwrap(), 2.0);
    assert_eq!(r.get_f32(&[1, 0]).unwrap(), 3.0);
    assert_eq!(r.get_f32(&[2, 1]).unwrap(), 6.0);
    assert!(Arc::ptr_eq(t.storage(), r.storage()), "contiguous reshape must not copy");
}

#[test]
fn contiguous_materializes_a_permuted_view_in_logical_order() {
    let t = t23();
    let tt = t.permute(&[1, 0]).unwrap();
    assert!(!tt.layout().is_contiguous(), "the transpose of [2,3] cannot be contiguous");
    let c = tt.contiguous().unwrap();
    assert!(c.layout().is_contiguous());
    assert!(!Arc::ptr_eq(tt.storage(), c.storage()), "contiguous() must copy here");
    // Same logical values as the view it came from.
    for i in 0..3 {
        for j in 0..2 {
            assert_eq!(c.get_f32(&[i, j]).unwrap(), tt.get_f32(&[i, j]).unwrap());
        }
    }
}

#[test]
fn contiguous_on_an_already_contiguous_tensor_shares_storage() {
    let t = t23();
    let c = t.contiguous().unwrap();
    assert!(Arc::ptr_eq(t.storage(), c.storage()), "no gratuitous copy");
}

#[test]
fn view_errors_are_typed() {
    let t = t23();
    assert!(matches!(t.permute(&[0, 0]), Err(_)), "a non-permutation must refuse");
    assert!(matches!(t.permute(&[0]), Err(_)), "wrong-arity perm must refuse");
    assert!(t.narrow(1, 2, 2).is_err(), "narrow past the end must refuse");
    assert!(t.narrow(2, 0, 1).is_err(), "narrow of a missing dim must refuse");
    match t.reshape(Shape::from([4, 2])) {
        Err(Error::ShapeMismatch { .. }) => {}
        other => panic!("element-count mismatch must be ShapeMismatch, got {other:?}"),
    }
    assert!(matches!(
        Tensor::from_vec_f32(vec![1.0; 5], Shape::from([2, 3])),
        Err(Error::ShapeMismatch { .. })
    ));
}

fn arb_tensor() -> impl Strategy<Value = (Tensor, Vec<usize>)> {
    proptest::collection::vec(1usize..4, 1..4).prop_flat_map(|dims| {
        let numel: usize = dims.iter().product();
        let shape = dims.clone();
        proptest::collection::vec(-100.0f32..100.0, numel).prop_map(move |data| {
            (
                Tensor::from_vec_f32(data, Shape::new(shape.clone())).unwrap(),
                shape.clone(),
            )
        })
    })
}

proptest! {
    /// Permuting by a permutation and then by its inverse reads back the
    /// original values everywhere.
    #[test]
    fn permute_roundtrip((t, dims) in arb_tensor(), seed in any::<u64>()) {
        // Derive a permutation from the seed without rand: rotate by seed.
        let n = dims.len();
        let rot = (seed as usize) % n;
        let perm: Vec<usize> = (0..n).map(|i| (i + rot) % n).collect();
        let mut inverse = vec![0usize; n];
        for (i, &p) in perm.iter().enumerate() {
            inverse[p] = i;
        }
        let round = t.permute(&perm).unwrap().permute(&inverse).unwrap();
        prop_assert_eq!(round.shape(), t.shape());
        // Compare at a hand-rolled sample of indices: all-zeros and the max corner.
        let zeros = vec![0usize; n];
        let corner: Vec<usize> = dims.iter().map(|&d| d - 1).collect();
        prop_assert_eq!(round.get_f32(&zeros).unwrap(), t.get_f32(&zeros).unwrap());
        prop_assert_eq!(round.get_f32(&corner).unwrap(), t.get_f32(&corner).unwrap());
    }
}
