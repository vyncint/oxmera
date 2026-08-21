//! Harness placeholder for the tensor seam.
//!
//! The real specs live in `exercises/` (rungs A1 and A3). This test exists
//! so the crate has a test target from its first commit; it is ignored
//! until the exercises are solved.

use std::sync::Arc;

use oxmera_core::Shape;
use oxmera_tensor::Tensor;

#[test]
#[ignore = "unignore after exercise A3 is solved; the spec lives in exercises/a3-strided-views"]
fn permute_is_a_view_not_a_copy() {
    // Hand-computed: [[1, 2, 3], [4, 5, 6]] transposed puts 6 at [2, 1].
    let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from([2, 3])).unwrap();
    let tt = t.permute(&[1, 0]).unwrap();
    assert_eq!(tt.get_f32(&[2, 1]).unwrap(), 6.0);
    assert!(Arc::ptr_eq(t.storage(), tt.storage()));
}
