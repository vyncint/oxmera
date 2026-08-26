//! The error taxonomy stays typed and self-describing (originally the A5
//! spec).

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
    fn assert_error<E: std::error::Error + Send + Sync + 'static>() {}
    assert_error::<Error>();
}

#[test]
fn refusals_carry_their_operands() {
    let e = broadcast_shapes(&Shape::from([2]), &Shape::from([3])).unwrap_err();
    let msg = e.to_string();
    assert!(
        msg.contains('2') && msg.contains('3'),
        "message must show the shapes: {msg}"
    );

    let l = Layout::contiguous(Shape::from([2, 3]));
    let e = l.offset_of(&[0, 3]).unwrap_err();
    assert!(
        e.to_string().contains("[0, 3]"),
        "message must show the index: {e}"
    );

    let e = Error::ShapeMismatch {
        expected: Shape::from([2, 2]),
        got: Shape::from([2, 3]),
        op: "matmul",
    };
    assert!(e.to_string().contains("matmul"));
}
