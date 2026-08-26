//! Zero-copy strided views (originally the A3 spec), extended with
//! `broadcast_to`, `slice`, and `unsqueeze`.

use std::sync::Arc;

use oxmera_core::{Error, Shape};
use oxmera_tensor::Tensor;

fn t23() -> Tensor {
    Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from([2, 3])).unwrap()
}

#[test]
fn permute_reads_the_transpose_and_shares_storage() {
    let t = t23();
    let tt = t.permute(&[1, 0]).unwrap();
    assert_eq!(tt.shape(), &Shape::from([3, 2]));
    assert_eq!(tt.get_f32(&[0, 1]).unwrap(), 4.0);
    assert_eq!(tt.get_f32(&[2, 1]).unwrap(), 6.0);
    assert!(
        Arc::ptr_eq(t.storage(), tt.storage()),
        "permute must be a view"
    );
}

#[test]
fn narrow_and_slice_window_and_share_storage() {
    let t = t23();
    let n = t.narrow(1, 1, 2).unwrap();
    assert_eq!(n.to_vec_f32().unwrap(), vec![2.0, 3.0, 5.0, 6.0]);
    assert!(Arc::ptr_eq(t.storage(), n.storage()));
    let s = t.slice(1, 1..3).unwrap();
    assert_eq!(s.to_vec_f32().unwrap(), n.to_vec_f32().unwrap());
}

#[test]
fn reshape_of_a_contiguous_tensor_is_a_view() {
    let t = t23();
    let r = t.reshape(Shape::from([3, 2])).unwrap();
    assert_eq!(r.get_f32(&[2, 1]).unwrap(), 6.0);
    assert!(Arc::ptr_eq(t.storage(), r.storage()));
}

#[test]
fn contiguous_materializes_a_permuted_view() {
    let t = t23();
    let tt = t.permute(&[1, 0]).unwrap();
    assert!(!tt.layout().is_contiguous());
    let c = tt.contiguous().unwrap();
    assert!(c.layout().is_contiguous());
    assert!(!Arc::ptr_eq(tt.storage(), c.storage()));
    assert_eq!(c.to_vec_f32().unwrap(), tt.to_vec_f32().unwrap());

    let already = t.contiguous().unwrap();
    assert!(
        Arc::ptr_eq(t.storage(), already.storage()),
        "no gratuitous copy"
    );
}

#[test]
fn broadcast_to_is_a_zero_copy_view() {
    let row = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], Shape::from([3])).unwrap();
    let b = row.broadcast_to(Shape::from([2, 3])).unwrap();
    assert!(Arc::ptr_eq(row.storage(), b.storage()));
    assert_eq!(b.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn unsqueeze_inserts_a_size_one_axis() {
    let t = t23();
    let u = t.unsqueeze(0).unwrap();
    assert_eq!(u.dims(), &[1, 2, 3]);
    assert_eq!(u.to_vec_f32().unwrap(), t.to_vec_f32().unwrap());
}

#[test]
fn view_errors_are_typed() {
    let t = t23();
    assert!(t.permute(&[0, 0]).is_err());
    assert!(t.narrow(1, 2, 2).is_err());
    assert!(matches!(
        t.reshape(Shape::from([4, 2])),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        Tensor::from_vec_f32(vec![1.0; 5], Shape::from([2, 3])),
        Err(Error::ShapeMismatch { .. })
    ));
}

#[test]
fn negative_strides_are_representable() {
    use oxmera_core::{Layout, Strides};
    // A manually flipped view of [1, 2, 3]: offset 2, stride -1.
    let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], Shape::from([3])).unwrap();
    let flipped = Tensor::from_storage(
        t.storage().clone(),
        Layout {
            shape: Shape::from([3]),
            strides: Strides::new(vec![-1]),
            offset: 2,
        },
    )
    .unwrap();
    assert_eq!(flipped.to_vec_f32().unwrap(), vec![3.0, 2.0, 1.0]);
}
