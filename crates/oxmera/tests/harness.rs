//! Harness placeholder for the umbrella crate: the first end-to-end
//! user-facing flow, ignored until the rungs it needs are solved.

use oxmera::{Shape, Tensor, TensorOps};

#[test]
#[ignore = "unignore after rungs A1-A4 are solved; end-to-end flow needs constructors, dispatch, and a backend"]
fn end_to_end_add_through_the_public_surface() {
    // Hand-computed: [1, 2] + [10, 20] = [11, 22].
    let a = Tensor::from_vec_f32(vec![1.0, 2.0], Shape::from([2])).unwrap();
    let b = Tensor::from_vec_f32(vec![10.0, 20.0], Shape::from([2])).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.get_f32(&[0]).unwrap(), 11.0);
    assert_eq!(c.get_f32(&[1]).unwrap(), 22.0);
}
