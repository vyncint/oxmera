//! Harness placeholder for the CPU reference backend.
//!
//! The behavioural specs live in `exercises/` (rung A4 is the matmul
//! ground truth). This hand-computed case is ignored until that rung is
//! solved; when it is, this is the smallest end-to-end proof that dispatch
//! and the reference implementation agree.

use oxmera_core::Shape;
use oxmera_cpu::CpuBackend;
use oxmera_ops::MatmulOps;
use oxmera_tensor::Tensor;

#[test]
#[ignore = "unignore after exercise A4 is solved; the spec lives in exercises/a4-reference-matmul"]
fn hand_computed_2x2_matmul() {
    // Hand-computed: [[1, 2], [3, 4]] x [[5, 6], [7, 8]] = [[19, 22], [43, 50]].
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from([2, 2])).unwrap();
    let b = Tensor::from_vec_f32(vec![5.0, 6.0, 7.0, 8.0], Shape::from([2, 2])).unwrap();
    let c = CpuBackend.matmul(&a, &b).unwrap();
    assert_eq!(c.get_f32(&[0, 0]).unwrap(), 19.0);
    assert_eq!(c.get_f32(&[0, 1]).unwrap(), 22.0);
    assert_eq!(c.get_f32(&[1, 0]).unwrap(), 43.0);
    assert_eq!(c.get_f32(&[1, 1]).unwrap(), 50.0);
}
