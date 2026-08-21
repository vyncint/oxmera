//! Spec for exercise A5 — the error taxonomy.
//!
//! The design rule under test: invalid states unrepresentable where the
//! type system can afford it, and every representable failure typed and
//! self-describing. An error a user cannot act on is a bug with better
//! manners.

use oxmera_core::shape::broadcast_shapes;
use oxmera_core::{Device, Error, Layout, Shape};

#[test]
fn device_kind_names_are_short_and_stable() {
    assert_eq!(Device::Cpu.kind_name(), "cpu");
    assert_eq!(Device::Metal { index: 0 }.kind_name(), "metal");
    assert_eq!(Device::Cuda { index: 1 }.kind_name(), "cuda");
}

#[test]
fn errors_are_real_errors() {
    // Compile-time contract: Error is a std Error, sendable across
    // threads, and printable. If this stops compiling, dispatch and CLI
    // reporting break with it.
    fn assert_error<E: std::error::Error + Send + Sync + 'static>() {}
    assert_error::<Error>();
}

#[test]
fn broadcast_refusal_carries_both_operands() {
    let e = broadcast_shapes(&Shape::from([2]), &Shape::from([3])).unwrap_err();
    match &e {
        Error::BroadcastIncompatible { lhs, rhs } => {
            assert_eq!(lhs, &Shape::from([2]));
            assert_eq!(rhs, &Shape::from([3]));
        }
        other => panic!("expected BroadcastIncompatible, got {other:?}"),
    }
    let msg = e.to_string();
    assert!(msg.contains('2') && msg.contains('3'), "message must show the shapes: {msg}");
}

#[test]
fn index_errors_say_where_and_against_what() {
    let l = Layout::contiguous(Shape::from([2, 3]));

    let e = l.offset_of(&[0]).unwrap_err();
    match &e {
        Error::RankMismatch { index_rank, shape_rank } => {
            assert_eq!((*index_rank, *shape_rank), (1, 2));
        }
        other => panic!("expected RankMismatch, got {other:?}"),
    }

    let e = l.offset_of(&[0, 3]).unwrap_err();
    match &e {
        Error::IndexOutOfBounds { index, shape } => {
            assert_eq!(index, &vec![0, 3]);
            assert_eq!(shape, &Shape::from([2, 3]));
        }
        other => panic!("expected IndexOutOfBounds, got {other:?}"),
    }
    let msg = e.to_string();
    assert!(msg.contains("[0, 3]"), "message must show the offending index: {msg}");
}

#[test]
fn messages_name_the_operation() {
    let e = Error::ShapeMismatch {
        expected: Shape::from([2, 2]),
        got: Shape::from([2, 3]),
        op: "matmul",
    };
    assert!(e.to_string().contains("matmul"), "an error that hides its origin is useless");

    let e = Error::DeviceMismatch {
        lhs: Device::Cpu,
        rhs: Device::Metal { index: 0 },
        op: "add",
    };
    let msg = e.to_string();
    assert!(msg.contains("add") && msg.contains("Cpu") && msg.contains("Metal"), "{msg}");
}
