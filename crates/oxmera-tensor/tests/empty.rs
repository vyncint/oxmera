//! Zero-extent tensors: every reduction returns its identity instead of
//! reading an element that does not exist (issue #18), and argmax of
//! nothing is a typed error rather than a fabricated index.
use oxmera_core::Shape;
use oxmera_tensor::tensor::Tensor;

fn empty(dims: &[usize]) -> Tensor {
    Tensor::zeros(Shape::new(dims.to_vec()))
}

#[test]
fn sum_of_nothing_is_zero_for_every_empty_shape() {
    for dims in [&[0usize][..], &[0, 3], &[2, 0], &[2, 0, 5], &[0, 0]] {
        let t = empty(dims);
        let all: Vec<usize> = (0..dims.len()).collect();
        assert_eq!(
            t.sum(&all).unwrap().to_vec_f32().unwrap(),
            vec![0.0],
            "{dims:?}"
        );
    }
}

#[test]
fn max_and_min_of_nothing_are_their_identities() {
    let t = empty(&[0, 3]);
    assert_eq!(
        t.max(&[0, 1]).unwrap().to_vec_f32().unwrap(),
        vec![f32::NEG_INFINITY]
    );
    assert_eq!(
        t.min(&[0, 1]).unwrap().to_vec_f32().unwrap(),
        vec![f32::INFINITY]
    );
}

#[test]
fn reducing_only_the_empty_axis_keeps_the_others() {
    // [2, 0, 3] summed over axis 1 → [2, 3] of zeros; maxed → [2, 3] of -inf.
    let t = empty(&[2, 0, 3]);
    let s = t.sum(&[1]).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
    assert_eq!(s.to_vec_f32().unwrap(), vec![0.0; 6]);
    let m = t.max_keepdim(&[1], true).unwrap();
    assert_eq!(m.dims(), &[2, 1, 3]);
    assert!(
        m.to_vec_f32()
            .unwrap()
            .iter()
            .all(|&x| x == f32::NEG_INFINITY)
    );
}

#[test]
fn reducing_a_nonempty_axis_of_a_tensor_with_an_empty_one_gives_an_empty_result() {
    // [2, 0, 3] summed over axis 2 → [2, 0]: no outputs at all.
    let s = empty(&[2, 0, 3]).sum(&[2]).unwrap();
    assert_eq!(s.dims(), &[2, 0]);
    assert!(s.to_vec_f32().unwrap().is_empty());
}

#[test]
fn mean_of_nothing_is_nan_not_a_panic() {
    let m = empty(&[0, 3]).mean(&[0, 1]).unwrap().to_vec_f32().unwrap();
    assert_eq!(m.len(), 1);
    assert!(
        m[0].is_nan(),
        "0/0 → NaN, as in IEEE and NumPy; got {}",
        m[0]
    );
}

#[test]
fn softmax_over_an_empty_axis_is_empty() {
    let s = empty(&[2, 0]).softmax(1).unwrap();
    assert_eq!(s.dims(), &[2, 0]);
}

#[test]
fn argmax_over_an_empty_axis_is_an_error() {
    let e = empty(&[2, 0]).argmax(1, false).unwrap_err().to_string();
    assert!(e.contains("argmax"), "{e}");
    // A non-empty axis of the same tensor is fine and empty.
    assert_eq!(empty(&[2, 0]).argmax(0, false).unwrap().dims(), &[0]);
}

#[test]
fn elementwise_on_empty_is_unchanged() {
    let t = empty(&[0, 3]);
    assert_eq!(t.relu().unwrap().numel(), 0);
    assert_eq!(t.add(&t).unwrap().numel(), 0);
    assert_eq!(t.t().unwrap().contiguous().unwrap().dims(), &[3, 0]);
}
