//! Harness placeholder for the core seam.
//!
//! The real specs for shape/stride arithmetic, broadcasting, and the error
//! taxonomy live in `exercises/` (rungs A1, A2, A5) so the ladder owns
//! them. This test exists so the crate has a test target from its first
//! commit; it is ignored until the exercises are solved.

use oxmera_core::{Layout, Shape};

#[test]
#[ignore = "unignore after exercise A1 is solved; the spec lives in exercises/a1-shape-and-strides"]
fn contiguous_layout_roundtrips_a_hand_computed_index() {
    // Hand-computed: in a contiguous [2, 3] row-major layout, index [1, 2]
    // lives at offset 1*3 + 2 = 5.
    let layout = Layout::contiguous(Shape::from([2, 3]));
    assert_eq!(layout.offset_of(&[1, 2]).unwrap(), 5);
}
