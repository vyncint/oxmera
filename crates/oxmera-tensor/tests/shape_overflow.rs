//! A shape whose element count overflows usize must never become a
//! tensor: with a wrapping product, [2^32, 2^32] has "0" elements and an
//! empty buffer passes the length check, leaving a tensor that claims
//! 2^64 elements over no storage.
use oxmera_tensor::tensor::Tensor;

const HUGE: [usize; 2] = [1usize << 32, 1usize << 32];

#[test]
fn from_vec_rejects_an_overflowing_shape() {
    let e = Tensor::from_vec_f32(vec![], HUGE).unwrap_err().to_string();
    assert!(e.contains("usize"), "{e}");
    assert!(Tensor::from_vec_i64(vec![], HUGE).is_err());
}

#[test]
fn reshape_and_broadcast_reject_an_overflowing_shape() {
    let t = Tensor::zeros([0usize, 3]);
    assert!(t.reshape(HUGE).is_err());
    let one = Tensor::zeros([1usize, 1]);
    assert!(one.broadcast_to(HUGE).is_err());
}

#[test]
#[should_panic(expected = "overflows usize")]
fn infallible_constructors_fail_loudly_rather_than_wrap() {
    let _ = Tensor::zeros(HUGE);
}
