//! NumPy broadcasting rules as a total function (originally the A2 spec).

use oxmera_core::shape::broadcast_shapes;
use oxmera_core::{Error, Shape};
use proptest::prelude::*;

fn s(dims: &[usize]) -> Shape {
    Shape::from(dims)
}

#[test]
fn hand_cases_that_broadcast() {
    let cases: &[(&[usize], &[usize], &[usize])] = &[
        (&[2, 3], &[3], &[2, 3]),
        (&[2, 1], &[1, 3], &[2, 3]),
        (&[], &[2, 3], &[2, 3]),
        (&[4, 1, 5], &[3, 1], &[4, 3, 5]),
        (&[0], &[1], &[0]),
    ];
    for (lhs, rhs, expected) in cases {
        assert_eq!(
            broadcast_shapes(&s(lhs), &s(rhs)).unwrap(),
            s(expected),
            "{lhs:?} x {rhs:?}"
        );
    }
}

#[test]
fn hand_cases_that_must_refuse() {
    for (lhs, rhs) in [(&[2usize][..], &[3usize][..]), (&[2, 3][..], &[2, 4][..])] {
        match broadcast_shapes(&s(lhs), &s(rhs)) {
            Err(Error::BroadcastIncompatible { lhs: l, rhs: r }) => {
                assert_eq!((l, r), (s(lhs), s(rhs)));
            }
            other => panic!("{lhs:?} x {rhs:?} must refuse, got {other:?}"),
        }
    }
}

fn arb_shape() -> impl Strategy<Value = Shape> {
    proptest::collection::vec(0usize..4, 0..4).prop_map(Shape::new)
}

proptest! {
    #[test]
    fn commutative(a in arb_shape(), b in arb_shape()) {
        match (broadcast_shapes(&a, &b), broadcast_shapes(&b, &a)) {
            (Ok(x), Ok(y)) => prop_assert_eq!(x, y),
            (Err(_), Err(_)) => {}
            (x, y) => prop_assert!(false, "asymmetric outcome: {:?} vs {:?}", x, y),
        }
    }

    #[test]
    fn idempotent_and_scalar_identity(a in arb_shape()) {
        prop_assert_eq!(broadcast_shapes(&a, &a).unwrap(), a.clone());
        prop_assert_eq!(broadcast_shapes(&a, &Shape::from([])).unwrap(), a);
    }
}
