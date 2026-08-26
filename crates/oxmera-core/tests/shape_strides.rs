//! Shape, stride, and layout arithmetic — hand-computed cases and
//! properties (originally the A1 exercise spec).

use oxmera_core::layout::contiguous_strides;
use oxmera_core::{DType, Error, Layout, Shape};
use proptest::prelude::*;

#[test]
fn dtype_sizes_are_what_the_hardware_says() {
    assert_eq!(DType::F32.size_in_bytes(), 4);
    assert_eq!(DType::F64.size_in_bytes(), 8);
    assert_eq!(DType::I32.size_in_bytes(), 4);
    assert_eq!(DType::I64.size_in_bytes(), 8);
    assert_eq!(DType::U8.size_in_bytes(), 1);
    assert_eq!(DType::Bool.size_in_bytes(), 1);
    assert!(DType::F32.is_float() && DType::F64.is_float());
    assert!(!DType::I64.is_float() && !DType::Bool.is_float());
}

#[test]
fn numel_hand_cases() {
    assert_eq!(Shape::from([]).numel(), 1, "a scalar has one element");
    assert_eq!(Shape::from([0]).numel(), 0);
    assert_eq!(Shape::from([2, 3]).numel(), 6);
    assert_eq!(Shape::from([2, 0, 3]).numel(), 0);
}

#[test]
fn contiguous_strides_hand_cases() {
    assert_eq!(
        contiguous_strides(&Shape::from([])).values(),
        &[] as &[isize]
    );
    assert_eq!(contiguous_strides(&Shape::from([4])).values(), &[1]);
    assert_eq!(contiguous_strides(&Shape::from([2, 3])).values(), &[3, 1]);
    assert_eq!(
        contiguous_strides(&Shape::from([4, 2, 3])).values(),
        &[6, 3, 1]
    );
}

#[test]
fn offset_of_hand_cases_and_errors() {
    let l = Layout::contiguous(Shape::from([2, 3]));
    assert_eq!(l.offset_of(&[0, 0]).unwrap(), 0);
    assert_eq!(l.offset_of(&[1, 2]).unwrap(), 5);
    assert!(matches!(l.offset_of(&[0]), Err(Error::RankMismatch { .. })));
    assert!(matches!(
        l.offset_of(&[2, 0]),
        Err(Error::IndexOutOfBounds { .. })
    ));
    assert_eq!(
        Layout::contiguous(Shape::from([])).offset_of(&[]).unwrap(),
        0
    );
}

#[test]
fn contiguity_hand_cases() {
    use oxmera_core::Strides;
    assert!(Layout::contiguous(Shape::from([2, 3])).is_contiguous());
    assert!(Layout::contiguous(Shape::from([0, 3])).is_contiguous());
    let transposed = Layout {
        shape: Shape::from([2, 3]),
        strides: Strides::new(vec![1, 2]),
        offset: 0,
    };
    assert!(!transposed.is_contiguous());
}

proptest! {
    /// For a contiguous layout, index -> offset is a bijection onto
    /// [0, numel).
    #[test]
    fn contiguous_offsets_are_a_bijection(dims in proptest::collection::vec(1usize..4, 0..4)) {
        let shape = Shape::new(dims.clone());
        let numel = shape.numel();
        let layout = Layout::contiguous(shape);
        let mut index = vec![0usize; dims.len()];
        let mut seen = std::collections::HashSet::new();
        let mut count = 0usize;
        loop {
            let offset = layout.offset_of(&index).unwrap();
            prop_assert!(offset < numel);
            prop_assert!(seen.insert(offset));
            count += 1;
            let mut d = dims.len();
            loop {
                if d == 0 { break; }
                d -= 1;
                index[d] += 1;
                if index[d] < dims[d] { break; }
                index[d] = 0;
                if d == 0 { d = usize::MAX; break; }
            }
            if dims.is_empty() || d == usize::MAX { break; }
        }
        prop_assert_eq!(count, numel);
    }
}
