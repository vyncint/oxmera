//! Parameter groups: each group gets its own learning rate and weight
//! decay inside one optimizer step, and the single-group constructors are
//! exactly the one-group case.

use oxmera_nn::Param;
use oxmera_optim::{Adam, AdamW, Optimizer, ParamGroup, RmsProp, Sgd};
use oxmera_tensor::tensor::Tensor;

fn leaf(values: &[f32]) -> Param {
    Param::new(Tensor::from_slice(values, [values.len()]).unwrap())
}

/// A loss whose gradient on every parameter is 1: sum of everything.
fn backward_ones(params: &[Param]) {
    let mut total: Option<Tensor> = None;
    for p in params {
        let s = p.value().sum(&[]).unwrap();
        total = Some(match total {
            Some(t) => t.add(&s).unwrap(),
            None => s,
        });
    }
    total.unwrap().backward().unwrap();
}

fn values(p: &Param) -> Vec<f32> {
    p.value().to_vec_f32().unwrap()
}

#[test]
fn sgd_groups_apply_their_own_lr_and_decay() {
    let fast = leaf(&[1.0, 2.0]);
    let slow = leaf(&[1.0, 2.0]);
    let frozen = leaf(&[1.0, 2.0]);
    backward_ones(&[fast.clone(), slow.clone(), frozen.clone()]);
    let mut opt = Sgd::with_groups(
        vec![
            ParamGroup::new(vec![fast.clone()], 0.1, 0.0),
            ParamGroup::new(vec![slow.clone()], 0.01, 1.0), // grad += 1.0 * w
            ParamGroup::new(vec![frozen.clone()], 0.0, 0.0),
        ],
        0.0,
    );
    opt.step().unwrap();
    // fast: w - 0.1 * 1
    assert_eq!(values(&fast), vec![0.9, 1.9]);
    // slow: w - 0.01 * (1 + w)
    let s = values(&slow);
    assert!((s[0] - (1.0 - 0.01 * 2.0)).abs() < 1e-6, "{s:?}");
    assert!((s[1] - (2.0 - 0.01 * 3.0)).abs() < 1e-6, "{s:?}");
    // frozen: lr 0 leaves it untouched
    assert_eq!(values(&frozen), vec![1.0, 2.0]);
}

#[test]
fn adamw_decoupled_decay_is_per_group() {
    let decayed = leaf(&[1.0, -1.0]);
    let plain = leaf(&[1.0, -1.0]);
    backward_ones(&[decayed.clone(), plain.clone()]);
    let mut opt = AdamW::with_groups(vec![
        ParamGroup::new(vec![decayed.clone()], 0.1, 0.5),
        ParamGroup::new(vec![plain.clone()], 0.1, 0.0),
    ]);
    opt.step().unwrap();
    // First Adam step with grad 1 everywhere: m_hat = 1, v_hat = 1, so the
    // update is lr * 1/(1+eps) ≈ lr for every element. AdamW applies
    // decay to the weights first: w * (1 - lr*wd) = 0.95 w.
    let d = values(&decayed);
    let p = values(&plain);
    assert!(
        (p[0] - 0.9).abs() < 1e-6 && (p[1] - (-1.1)).abs() < 1e-6,
        "{p:?}"
    );
    assert!(
        (d[0] - (0.95 - 0.1)).abs() < 1e-6 && (d[1] - (-0.95 - 0.1)).abs() < 1e-6,
        "{d:?}"
    );
}

#[test]
fn adam_coupled_decay_is_per_group_and_the_step_count_is_shared() {
    let a = leaf(&[2.0]);
    let b = leaf(&[2.0]);
    backward_ones(&[a.clone(), b.clone()]);
    let mut opt = Adam::with_groups(vec![
        ParamGroup::new(vec![a.clone()], 0.1, 1.0), // effective grad 1 + 2 = 3
        ParamGroup::new(vec![b.clone()], 0.1, 0.0), // grad 1
    ]);
    opt.step().unwrap();
    // Bias-corrected first step: m_hat/sqrt(v_hat) = g/|g| = 1 for both,
    // regardless of the gradient's magnitude — so both move by lr. That
    // the decayed group does not move further is the point: coupled decay
    // changes the direction estimate, not the normalized step size.
    assert!((values(&a)[0] - 1.9).abs() < 1e-6, "{:?}", values(&a));
    assert!((values(&b)[0] - 1.9).abs() < 1e-6, "{:?}", values(&b));
    // A second step advances the shared count for both groups: the
    // second update is smaller than lr because v accumulates.
    opt.zero_grad();
    backward_ones(&[a.clone(), b.clone()]);
    opt.step().unwrap();
    let a2 = values(&a)[0];
    let b2 = values(&b)[0];
    assert!(a2 < 1.9 && b2 < 1.9 && a2 > 1.75 && b2 > 1.75, "{a2} {b2}");
}

#[test]
fn single_group_constructors_match_explicit_groups() {
    for (seed, wd) in [(1u64, 0.0f32), (2, 0.3)] {
        let init = Tensor::randn_with_seed([5], seed);
        let x1 = Param::new(init.clone());
        let x2 = Param::new(init.clone());
        for p in [&x1, &x2] {
            p.value()
                .mul(&p.value())
                .unwrap()
                .sum(&[])
                .unwrap()
                .backward()
                .unwrap();
        }
        let mut plain = AdamW::new(vec![x1.clone()], 0.05, wd);
        let mut grouped = AdamW::with_groups(vec![ParamGroup::new(vec![x2.clone()], 0.05, wd)]);
        plain.step().unwrap();
        grouped.step().unwrap();
        assert_eq!(values(&x1), values(&x2));
    }
}

#[test]
fn rmsprop_groups_use_their_own_lr() {
    let a = leaf(&[1.0]);
    let b = leaf(&[1.0]);
    backward_ones(&[a.clone(), b.clone()]);
    let mut opt = RmsProp::with_groups(vec![
        ParamGroup::with_lr(vec![a.clone()], 0.1),
        ParamGroup::with_lr(vec![b.clone()], 0.01),
    ]);
    opt.step().unwrap();
    // update = g / sqrt(0.01 g²) = 10 for g = 1.
    assert!((values(&a)[0] - 0.0).abs() < 1e-5, "{:?}", values(&a));
    assert!((values(&b)[0] - 0.9).abs() < 1e-5, "{:?}", values(&b));
}

#[test]
fn groups_mut_lets_a_schedule_change_the_lr_between_steps() {
    let w = leaf(&[1.0]);
    backward_ones(std::slice::from_ref(&w));
    let mut opt = Sgd::new(vec![w.clone()], 0.1);
    opt.step().unwrap();
    assert!((values(&w)[0] - 0.9).abs() < 1e-6);
    opt.groups_mut()[0].lr = 0.5;
    opt.zero_grad();
    backward_ones(std::slice::from_ref(&w));
    opt.step().unwrap();
    assert!((values(&w)[0] - 0.4).abs() < 1e-6, "{:?}", values(&w));
}

#[test]
fn zero_grad_clears_every_group() {
    let a = leaf(&[1.0]);
    let b = leaf(&[1.0]);
    backward_ones(&[a.clone(), b.clone()]);
    let opt = Adam::with_groups(vec![
        ParamGroup::with_lr(vec![a.clone()], 0.1),
        ParamGroup::with_lr(vec![b.clone()], 0.1),
    ]);
    assert!(a.grad().is_some() && b.grad().is_some());
    opt.zero_grad();
    assert!(a.grad().is_none() && b.grad().is_none());
}
