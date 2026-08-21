//! Spec for exercise A2 — broadcasting.
//!
//! NumPy's rules, as a total function: align trailing dimensions; each
//! aligned pair must be equal or contain a 1. Every expected shape below
//! is hand-computed.

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
        (&[3], &[2, 3], &[2, 3]),
        (&[2, 1], &[1, 3], &[2, 3]),
        (&[], &[2, 3], &[2, 3]),
        (&[2, 3], &[], &[2, 3]),
        (&[], &[], &[]),
        (&[4, 1, 5], &[3, 1], &[4, 3, 5]),
        (&[1], &[1], &[1]),
        (&[5], &[1], &[5]),
        (&[0], &[1], &[0]),
        (&[2, 0], &[1, 1], &[2, 0]),
    ];
    for (lhs, rhs, expected) in cases {
        let got = broadcast_shapes(&s(lhs), &s(rhs)).unwrap_or_else(|e| {
            panic!("{lhs:?} with {rhs:?} must broadcast to {expected:?}, got error: {e}")
        });
        assert_eq!(got, s(expected), "{lhs:?} with {rhs:?}");
    }
}

#[test]
fn hand_cases_that_must_refuse() {
    let cases: &[(&[usize], &[usize])] = &[
        (&[2], &[3]),
        (&[2, 3], &[2, 4]),
        (&[4, 3], &[2, 1, 2]),
        (&[0], &[2]),
    ];
    for (lhs, rhs) in cases {
        match broadcast_shapes(&s(lhs), &s(rhs)) {
            Err(Error::BroadcastIncompatible { lhs: l, rhs: r }) => {
                assert_eq!((l, r), (s(lhs), s(rhs)), "error must carry the operands verbatim");
            }
            other => panic!("{lhs:?} with {rhs:?} must refuse with BroadcastIncompatible, got {other:?}"),
        }
    }
}

fn arb_shape() -> impl Strategy<Value = Shape> {
    proptest::collection::vec(0usize..4, 0..4).prop_map(Shape::new)
}

proptest! {
    /// Broadcasting is commutative — in the result and in the refusal.
    #[test]
    fn commutative(a in arb_shape(), b in arb_shape()) {
        match (broadcast_shapes(&a, &b), broadcast_shapes(&b, &a)) {
            (Ok(x), Ok(y)) => prop_assert_eq!(x, y),
            (Err(_), Err(_)) => {}
            (x, y) => prop_assert!(false, "asymmetric outcome: {:?} vs {:?}", x, y),
        }
    }

    /// A shape broadcast with itself is itself.
    #[test]
    fn idempotent(a in arb_shape()) {
        prop_assert_eq!(broadcast_shapes(&a, &a).unwrap(), a);
    }

    /// A scalar is the identity element.
    #[test]
    fn scalar_is_identity(a in arb_shape()) {
        prop_assert_eq!(broadcast_shapes(&a, &Shape::from([])).unwrap(), a);
    }

    /// The result rank is the larger operand rank, and the result never
    /// shrinks either operand's extent at any aligned position.
    #[test]
    fn result_dominates(a in arb_shape(), b in arb_shape()) {
        if let Ok(c) = broadcast_shapes(&a, &b) {
            prop_assert_eq!(c.ndim(), a.ndim().max(b.ndim()));
            for (operand_dims, out) in [(a.dims(), &c), (b.dims(), &c)] {
                for (i, &d) in operand_dims.iter().rev().enumerate() {
                    let o = out.dims()[out.ndim() - 1 - i];
                    prop_assert!(o == d || d == 1, "extent {d} at trailing pos {i} became {o}");
                }
            }
        }
    }
}
