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

// ---- 0.5.1: defects found by deep-testing the published 0.5.0 ----

#[test]
fn from_storage_rejects_a_layout_that_overflows_the_address_space() {
    let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], Shape::from([3])).unwrap();
    // (d - 1) * stride overflows isize: the check must refuse, not wrap
    // (accepting it in release) and not panic inside a Result constructor.
    for (dims, stride) in [
        (3usize, isize::MAX),
        (4, isize::MAX / 2),
        (3, isize::MIN + 1),
    ] {
        let bad = Layout {
            shape: Shape::from([dims]),
            strides: Strides::new(vec![stride]),
            offset: 0,
        };
        let e = Tensor::from_storage(t.storage().clone(), bad).unwrap_err();
        assert!(
            matches!(e, Error::InvalidArgument { .. }),
            "{dims}x{stride}: {e}"
        );
    }
    // A huge offset must not wrap to a negative address either.
    let bad = Layout {
        shape: Shape::from([3]),
        strides: Strides::new(vec![1]),
        offset: usize::MAX - 1,
    };
    assert!(Tensor::from_storage(t.storage().clone(), bad).is_err());
}

#[test]
fn matmul_with_a_zero_size_dimension_is_empty_not_a_panic() {
    let out = Tensor::zeros([2usize, 0, 3])
        .matmul(&Tensor::zeros([2usize, 3, 4]))
        .unwrap();
    assert_eq!(out.dims(), &[2, 0, 4]);
    assert!(out.to_vec_f32().unwrap().is_empty());
    // The rank-2 path already returned the empty tensor; keep it that way.
    let r2 = Tensor::zeros([0usize, 3])
        .matmul(&Tensor::zeros([3usize, 4]))
        .unwrap();
    assert_eq!(r2.dims(), &[0, 4]);
    // A zero *inner* dimension still produces a real, zero-filled output.
    let inner = Tensor::zeros([2usize, 0])
        .matmul(&Tensor::zeros([0usize, 3]))
        .unwrap();
    assert_eq!(inner.dims(), &[2, 3]);
    assert_eq!(inner.to_vec_f32().unwrap(), vec![0.0; 6]);
}

#[test]
fn mean_of_an_f64_tensor_is_computed_in_f64() {
    let x = Tensor::from_vec_f64(vec![1.0, 2.0, 3.0], Shape::from([3])).unwrap();
    let m = x.mean(&[]).unwrap().to_vec_f64().unwrap()[0];
    // Scaling by an f32 reciprocal gave 2.0000000596046448.
    assert_eq!(m, 2.0, "f64 mean must not carry f32 precision: {m:.17}");
}

#[test]
fn maximum_and_minimum_split_a_tie_evenly() {
    let x = Tensor::from_slice(&[0.0, 1.0, -1.0], [3])
        .unwrap()
        .requires_grad_(true);
    let z = Tensor::from_slice(&[0.0, 0.0, 0.0], [3]).unwrap();
    x.maximum(&z).unwrap().sum(&[]).unwrap().backward().unwrap();
    assert_eq!(x.grad().unwrap().to_vec_f32().unwrap(), vec![0.5, 1.0, 0.0]);

    let y = Tensor::from_slice(&[0.0, 1.0, -1.0], [3])
        .unwrap()
        .requires_grad_(true);
    y.minimum(&z).unwrap().sum(&[]).unwrap().backward().unwrap();
    assert_eq!(y.grad().unwrap().to_vec_f32().unwrap(), vec![0.5, 0.0, 1.0]);

    // relu keeps its own documented convention: relu'(0) = 0.
    let r = Tensor::from_slice(&[0.0, 1.0, -1.0], [3])
        .unwrap()
        .requires_grad_(true);
    r.relu().unwrap().sum(&[]).unwrap().backward().unwrap();
    assert_eq!(r.grad().unwrap().to_vec_f32().unwrap(), vec![0.0, 1.0, 0.0]);
}
