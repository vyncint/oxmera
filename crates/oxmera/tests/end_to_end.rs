//! The umbrella crate's public surface, end to end: operators, autograd,
//! a training step, and device movement — everything a consumer's first
//! ten minutes touches.

use oxmera::nn::{Linear, MSELoss, Module, Sequential};
use oxmera::optim::{Optimizer, Sgd};
use oxmera::{Device, Tensor};

#[test]
fn operator_overloads_work_for_tensors_and_scalars() {
    let a = Tensor::from_slice(&[1.0, 2.0], [2]).unwrap();
    let b = Tensor::from_slice(&[10.0, 20.0], [2]).unwrap();
    assert_eq!((&a + &b).to_vec_f32().unwrap(), vec![11.0, 22.0]);
    assert_eq!((&b - &a).to_vec_f32().unwrap(), vec![9.0, 18.0]);
    assert_eq!((&a * 3.0).to_vec_f32().unwrap(), vec![3.0, 6.0]);
    assert_eq!((1.0 / &a).to_vec_f32().unwrap(), vec![1.0, 0.5]);
    assert_eq!((-&a).to_vec_f32().unwrap(), vec![-1.0, -2.0]);
}

#[test]
fn one_training_step_reduces_the_loss() {
    let model = Sequential::new().push(Linear::new(2, 1, 3));
    let x = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], [4, 2]).unwrap();
    let y = Tensor::from_slice(&[1.0, 2.0, 3.0, 0.0], [4, 1]).unwrap();
    let mut opt = Sgd::new(model.parameters(), 0.1);

    let before = MSELoss.forward(&model.forward(&x).unwrap(), &y).unwrap();
    let before_v = before.get_f32(&[]).unwrap();
    before.backward().unwrap();
    opt.step().unwrap();
    let after = MSELoss
        .forward(&model.forward(&x).unwrap(), &y)
        .unwrap()
        .get_f32(&[])
        .unwrap();
    assert!(
        after < before_v,
        "one SGD step must reduce the loss: {before_v} -> {after}"
    );
}

#[test]
fn init_and_device_movement() {
    let devices = oxmera::init();
    assert!(devices.contains(&Device::Cpu));
    let t = Tensor::randn_with_seed([4, 4], 5);
    for &device in &devices {
        let roundtrip = t.to_device(device).unwrap().to_device(Device::Cpu).unwrap();
        assert_eq!(
            roundtrip.to_vec_f32().unwrap(),
            t.to_vec_f32().unwrap(),
            "{device:?}"
        );
    }
}
