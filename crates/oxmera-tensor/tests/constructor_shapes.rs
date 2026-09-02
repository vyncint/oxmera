//! Every constructor accepts the same shape spellings (issue #19): a bare
//! array, a slice, a Vec, or a Shape — from_vec_f32/from_vec_i64 used to be
//! the two exceptions, and they are the only way to build I64 index and
//! target tensors.
use oxmera_core::Shape;
use oxmera_tensor::tensor::Tensor;

#[test]
fn from_vec_i64_takes_a_bare_array_like_everything_else() {
    let t = Tensor::from_vec_i64(vec![3, 1, 3], [3]).unwrap();
    assert_eq!(t.dims(), &[3]);
    assert_eq!(t.to_vec_i64().unwrap(), vec![3, 1, 3]);
}

#[test]
fn from_vec_f32_takes_every_shape_spelling() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::from_vec_f32(data.clone(), [2, 3]).unwrap();
    let b = Tensor::from_vec_f32(data.clone(), &[2usize, 3][..]).unwrap();
    let c = Tensor::from_vec_f32(data.clone(), vec![2, 3]).unwrap();
    let d = Tensor::from_vec_f32(data.clone(), Shape::from([2, 3])).unwrap();
    for t in [&a, &b, &c, &d] {
        assert_eq!(t.dims(), &[2, 3]);
        assert_eq!(t.to_vec_f32().unwrap(), data);
    }
}

#[test]
fn length_mismatch_is_still_a_typed_error() {
    assert!(Tensor::from_vec_f32(vec![1.0, 2.0], [3]).is_err());
    assert!(Tensor::from_vec_i64(vec![1, 2], [3]).is_err());
}
