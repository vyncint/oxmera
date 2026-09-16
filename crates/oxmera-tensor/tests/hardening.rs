//! Regression tests for the 0.5 construction/shape hardening: typed errors
//! where the pre-0.5 code panicked or wrapped, and a `Debug` that prints
//! metadata instead of the whole storage buffer.

use oxmera_core::{Error, Layout, Shape, Strides};
use oxmera_tensor::Tensor;

const HUGE: [usize; 2] = [1usize << 32, 1usize << 32];

#[test]
fn from_storage_rejects_a_negative_stride_that_underflows() {
    // offset 1, stride -1, len 3 addresses indices 1, 0, -1 — the last is
    // before the buffer. Pre-0.5 the bounds check saw only the offset and
    // let it through to an out-of-bounds read. A valid flip (offset 2,
    // covered by views.rs::negative_strides_are_representable) still works.
    let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], Shape::from([3])).unwrap();
    let bad = Layout {
        shape: Shape::from([3]),
        strides: Strides::new(vec![-1]),
        offset: 1,
    };
    let e = Tensor::from_storage(t.storage().clone(), bad).unwrap_err();
    assert!(matches!(e, Error::InvalidArgument { .. }), "{e}");
    // The valid flip (offset 2) is accepted and reads in reverse.
    let ok = Tensor::from_storage(
        t.storage().clone(),
        Layout {
            shape: Shape::from([3]),
            strides: Strides::new(vec![-1]),
            offset: 2,
        },
    )
    .unwrap();
    assert_eq!(ok.to_vec_f32().unwrap(), vec![3.0, 2.0, 1.0]);
}

#[test]
fn narrow_reports_an_overflowing_range_instead_of_panicking() {
    let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], Shape::from([3])).unwrap();
    // start + len overflows usize: pre-0.5 this panicked in the bounds check.
    let e = t.narrow(0, 2, usize::MAX).unwrap_err();
    assert!(matches!(e, Error::InvalidArgument { .. }), "{e}");
}

#[test]
fn try_constructors_return_a_typed_error_on_an_overflowing_shape() {
    assert!(Tensor::try_zeros(HUGE).is_err());
    assert!(Tensor::try_ones(HUGE).is_err());
    assert!(Tensor::try_full(HUGE, 1.0).is_err());
    assert_eq!(Tensor::try_zeros([2, 3]).unwrap().numel(), 6);
    assert_eq!(
        Tensor::try_full([2, 2], 7.0).unwrap().to_vec_f32().unwrap(),
        vec![7.0; 4]
    );
}

#[test]
fn debug_prints_metadata_not_the_buffer() {
    let t = Tensor::zeros([128, 128]);
    let s = format!("{t:?}");
    assert!(s.contains("Tensor"), "{s}");
    assert!(s.contains("shape"), "{s}");
    // 16384 elements; the old derived Debug dumped every one of them.
    assert!(s.len() < 160, "Debug output is {} bytes: {s}", s.len());
    assert!(
        !s.contains("0.0, 0.0, 0.0"),
        "must not dump the storage: {s}"
    );
}

#[test]
fn argmax_reports_nan_instead_of_hiding_it() {
    let t = Tensor::from_slice(&[1.0, f32::NAN, 2.0], [3]).unwrap();
    assert!(
        t.argmax(0, false).is_err(),
        "NaN input must be a typed error"
    );
    let ok = Tensor::from_slice(&[1.0, 3.0, 2.0], [3]).unwrap();
    assert_eq!(ok.argmax(0, false).unwrap().to_vec_i64().unwrap(), vec![1]);
}

#[test]
fn matmul_agrees_for_contiguous_and_strided_operands() {
    // b = [[1,0],[0,1],[1,1]]
    let b = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2]).unwrap();
    // Contiguous a = [[1,2,3],[4,5,6]] (borrow path).
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
    let want = vec![4.0, 5.0, 10.0, 11.0];
    assert_eq!(a.matmul(&b).unwrap().to_vec_f32().unwrap(), want);
    // Same logical a as a non-contiguous view: eᵀ where e = [[1,4],[2,5],[3,6]]
    // (gather path). Both must give the same product.
    let e = Tensor::from_slice(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], [3, 2]).unwrap();
    let a_view = e.t().unwrap();
    assert_eq!(a_view.matmul(&b).unwrap().to_vec_f32().unwrap(), want);
}
