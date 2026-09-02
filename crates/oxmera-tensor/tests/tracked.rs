//! `is_tracked` is the public predicate for tape membership (issue #20);
//! `requires_grad` is leaf-ness and is the same in and out of `no_grad`.
use oxmera_tensor::autograd::no_grad;
use oxmera_tensor::tensor::Tensor;

fn leaf() -> Tensor {
    Tensor::from_slice(&[1.0, 2.0], [2])
        .unwrap()
        .requires_grad_(true)
}

#[test]
fn computed_tensors_are_tracked_but_do_not_require_grad() {
    let y = leaf().mul_scalar(3.0).unwrap();
    assert!(y.is_tracked());
    assert!(!y.requires_grad());
}

#[test]
fn no_grad_is_observable_through_is_tracked() {
    let a = leaf();
    assert!(!no_grad(|| a.mul_scalar(3.0).unwrap()).is_tracked());
    assert!(a.mul_scalar(3.0).unwrap().is_tracked());
}

#[test]
fn constants_and_detached_tensors_are_not_tracked() {
    assert!(!Tensor::from_slice(&[1.0, 2.0], [2]).unwrap().is_tracked());
    assert!(!leaf().detach().is_tracked());
    assert!(
        leaf().is_tracked(),
        "a leaf that requires grad is on the tape"
    );
}
