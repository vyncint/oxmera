//! Finite-difference validation of every backward pass: unary ops, binary
//! ops with broadcasting, matmul, reductions, views, indexing, and
//! representative composites (softmax, a layer-norm expression, losses).

use oxmera_autograd::gradcheck;
use oxmera_core::{Result, Shape};
use oxmera_tensor::tensor::Tensor;

const EPS: f32 = 1e-3;
const TOL: f32 = 2e-2; // f32 central differences are noisy; relative tolerance

fn positive(seed: u64, shape: impl Into<Shape>) -> Tensor {
    // abs(randn) + 0.5: keeps ln/sqrt/pow well-conditioned and away from
    // non-differentiable points.
    let t = Tensor::randn_with_seed(shape, seed);
    t.abs().unwrap().add_scalar(0.5).unwrap().detach()
}

fn check1(f: impl Fn(&Tensor) -> Result<Tensor>, input: Tensor) {
    gradcheck(|xs| f(&xs[0])?.sum(&[]), &[input], EPS, TOL).unwrap();
}

#[test]
fn unary_backward_passes() {
    check1(|x| x.neg(), Tensor::randn_with_seed([2, 3], 1));
    check1(|x| x.exp(), Tensor::randn_with_seed([2, 3], 2));
    check1(|x| x.ln(), positive(3, [2, 3]));
    check1(|x| x.sqrt(), positive(4, [2, 3]));
    check1(|x| x.sin(), Tensor::randn_with_seed([2, 3], 5));
    check1(|x| x.cos(), Tensor::randn_with_seed([2, 3], 6));
    check1(|x| x.tanh(), Tensor::randn_with_seed([2, 3], 7));
    check1(|x| x.sigmoid(), Tensor::randn_with_seed([2, 3], 8));
    check1(|x| x.gelu(), Tensor::randn_with_seed([2, 3], 9));
    // relu and abs away from the kink at zero.
    check1(|x| x.relu(), positive(10, [2, 3]));
    check1(|x| x.abs(), positive(11, [2, 3]));
}

#[test]
fn binary_backward_passes_with_broadcasting() {
    let a = Tensor::randn_with_seed([2, 3], 20);
    let b = Tensor::randn_with_seed([3], 21);
    gradcheck(
        |xs| xs[0].add(&xs[1])?.sum(&[]),
        &[a.clone(), b.clone()],
        EPS,
        TOL,
    )
    .unwrap();
    gradcheck(
        |xs| xs[0].sub(&xs[1])?.sum(&[]),
        &[a.clone(), b.clone()],
        EPS,
        TOL,
    )
    .unwrap();
    gradcheck(
        |xs| xs[0].mul(&xs[1])?.sum(&[]),
        &[a.clone(), b.clone()],
        EPS,
        TOL,
    )
    .unwrap();

    let bp = positive(22, [3]);
    gradcheck(
        |xs| xs[0].div(&xs[1])?.sum(&[]),
        &[a.clone(), bp.clone()],
        EPS,
        TOL,
    )
    .unwrap();

    let base = positive(23, [2, 3]);
    let expo = Tensor::randn_with_seed([3], 24).detach();
    gradcheck(|xs| xs[0].pow(&xs[1])?.sum(&[]), &[base, expo], EPS, TOL).unwrap();

    // maximum/minimum away from ties.
    let m1 = positive(25, [2, 3]);
    let m2 = m1.add_scalar(1.0).unwrap().detach();
    gradcheck(
        |xs| xs[0].maximum(&xs[1])?.sum(&[]),
        &[m1.clone(), m2.clone()],
        EPS,
        TOL,
    )
    .unwrap();
    gradcheck(|xs| xs[0].minimum(&xs[1])?.sum(&[]), &[m1, m2], EPS, TOL).unwrap();
}

#[test]
fn matmul_backward_passes() {
    let a = Tensor::randn_with_seed([3, 4], 30);
    let b = Tensor::randn_with_seed([4, 2], 31);
    gradcheck(|xs| xs[0].matmul(&xs[1])?.sum(&[]), &[a, b], EPS, TOL).unwrap();

    let ab = Tensor::randn_with_seed([2, 3, 4], 32);
    let bb = Tensor::randn_with_seed([2, 4, 2], 33);
    gradcheck(|xs| xs[0].matmul(&xs[1])?.sum(&[]), &[ab, bb], EPS, TOL).unwrap();
}

#[test]
fn reduction_backward_passes() {
    // weight the reduced outputs so the gradient is not uniformly 1.
    let w = Tensor::from_vec_f32(vec![0.3, -1.7], Shape::from([2])).unwrap();
    let x = Tensor::randn_with_seed([2, 3], 40);
    gradcheck(
        |xs| xs[0].sum(&[1])?.mul(&w)?.sum(&[]),
        std::slice::from_ref(&x),
        EPS,
        TOL,
    )
    .unwrap();
    gradcheck(
        |xs| xs[0].mean(&[0])?.sum(&[]),
        std::slice::from_ref(&x),
        EPS,
        TOL,
    )
    .unwrap();
    // max/min: seeds chosen tie-free.
    gradcheck(
        |xs| xs[0].max_keepdim(&[1], false)?.sum(&[]),
        std::slice::from_ref(&x),
        EPS,
        TOL,
    )
    .unwrap();
    gradcheck(|xs| xs[0].min(&[0])?.sum(&[]), &[x], EPS, TOL).unwrap();
}

#[test]
fn view_backward_passes() {
    let x = Tensor::randn_with_seed([2, 6], 50);
    gradcheck(
        |xs| {
            xs[0]
                .reshape([3, 4])?
                .sum(&[1])?
                .mul(&Tensor::from_vec_f32(
                    vec![1.0, -2.0, 0.5],
                    Shape::from([3]),
                )?)?
                .sum(&[])
        },
        std::slice::from_ref(&x),
        EPS,
        TOL,
    )
    .unwrap();
    gradcheck(
        |xs| xs[0].permute(&[1, 0])?.narrow(0, 1, 3)?.sum(&[]),
        std::slice::from_ref(&x),
        EPS,
        TOL,
    )
    .unwrap();
    gradcheck(
        |xs| {
            let b = xs[0]
                .unsqueeze(0)?
                .broadcast_to(Shape::from([3, 2, 6]))?
                .contiguous()?;
            b.mul_scalar(0.5)?.sum(&[])
        },
        &[x],
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn indexing_backward_passes() {
    let x = Tensor::randn_with_seed([5, 3], 60);
    let idx = Tensor::from_vec_i64(vec![4, 0, 2, 0], Shape::from([4])).unwrap();
    gradcheck(
        |xs| {
            let sel = xs[0].index_select(0, &idx)?;
            sel.mul(&sel)?.sum(&[])
        },
        &[x],
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn composite_backward_passes() {
    let x = Tensor::randn_with_seed([4, 5], 70);
    gradcheck(
        |xs| {
            let p = xs[0].softmax(1)?;
            // A non-trivial functional of the softmax.
            p.mul(&p)?.sum(&[])
        },
        std::slice::from_ref(&x),
        EPS,
        TOL,
    )
    .unwrap();
    gradcheck(
        |xs| xs[0].log_softmax(1)?.mean(&[]),
        std::slice::from_ref(&x),
        EPS,
        TOL,
    )
    .unwrap();

    // A layer-norm-shaped expression, end to end.
    gradcheck(
        |xs| {
            let mu = xs[0].mean_keepdim(&[1], true)?;
            let c = xs[0].sub(&mu)?;
            let var = c.mul(&c)?.mean_keepdim(&[1], true)?;
            c.div(&var.add_scalar(1e-5)?.sqrt()?)?.mul(&xs[0])?.sum(&[])
        },
        &[x],
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn accumulation_no_grad_and_zero_grad() {
    let x = Tensor::randn_with_seed([3], 80).requires_grad_(true);

    // A value used twice accumulates both contributions.
    let y = x.mul(&x).unwrap().sum(&[]).unwrap();
    y.backward().unwrap();
    let g1 = x.grad().unwrap().to_vec_f32().unwrap();
    let want: Vec<f32> = x.to_vec_f32().unwrap().iter().map(|v| 2.0 * v).collect();
    for (a, b) in g1.iter().zip(&want) {
        assert!((a - b).abs() < 1e-5);
    }

    // Second backward accumulates on top.
    let y2 = x.sum(&[]).unwrap();
    y2.backward().unwrap();
    let g2 = x.grad().unwrap().to_vec_f32().unwrap();
    for (g, w) in g2.iter().zip(&want) {
        assert!((g - (w + 1.0)).abs() < 1e-5, "{g} vs {}", w + 1.0);
    }

    // zero_grad clears; no_grad records nothing.
    x.zero_grad();
    assert!(x.grad().is_none());
    let z = oxmera_tensor::no_grad(|| x.mul(&x).unwrap().sum(&[]).unwrap());
    assert!(
        z.backward().is_ok(),
        "backward on an untracked value is a no-op"
    );
    assert!(x.grad().is_none(), "no_grad must record nothing");
}
