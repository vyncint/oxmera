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

/// Broadcast matmul (issue #22): gradients for a batch-1 or rank-2
/// operand are summed over the broadcast batch axis.
#[test]
fn matmul_batch_broadcast_gradients_check() {
    use oxmera_tensor::tensor::Tensor;
    let a = Tensor::randn_with_seed([2, 2, 3], 21);
    let b1 = Tensor::randn_with_seed([1, 3, 2], 22); // batch-1 broadcast
    oxmera_autograd::gradcheck(
        |i| i[0].matmul(&i[1])?.sum(&[0, 1, 2]),
        &[a.clone(), b1],
        1e-3,
        2e-2,
    )
    .expect("batch-1 operand");
    let b2 = Tensor::randn_with_seed([3, 2], 23); // rank-2 operand
    oxmera_autograd::gradcheck(
        |i| i[0].matmul(&i[1])?.sum(&[0, 1, 2]),
        &[a.clone(), b2],
        1e-3,
        2e-2,
    )
    .expect("rank-2 operand");
    let a2 = Tensor::randn_with_seed([2, 3], 24); // rank-2 on the left
    let b3 = Tensor::randn_with_seed([4, 3, 2], 25);
    oxmera_autograd::gradcheck(
        |i| i[0].matmul(&i[1])?.sum(&[0, 1, 2]),
        &[a2, b3],
        1e-3,
        2e-2,
    )
    .expect("rank-2 left operand");
}

/// Rank-4 matmul (issue #30): the lowering is composed from recorded view
/// ops, so a broadcast batch operand's gradient must sum over the batches
/// it was repeated for, and a rank-2 operand's over every batch.
#[test]
fn matmul_rank_four_broadcast_gradients_check() {
    use oxmera_tensor::tensor::Tensor;
    let a = Tensor::randn_with_seed([2, 1, 3, 4], 31);
    let b = Tensor::randn_with_seed([1, 2, 4, 2], 32);
    oxmera_autograd::gradcheck(
        |i| i[0].matmul(&i[1])?.sum(&[0, 1, 2, 3]),
        &[a.clone(), b],
        1e-3,
        2e-2,
    )
    .expect("rank-4 x rank-4 with broadcast batches");
    let w = Tensor::randn_with_seed([4, 3], 33);
    oxmera_autograd::gradcheck(
        |i| i[0].matmul(&i[1])?.sum(&[0, 1, 2, 3]),
        &[a.clone(), w],
        1e-3,
        2e-2,
    )
    .expect("rank-4 x rank-2");
    let w3 = Tensor::randn_with_seed([2, 4, 3], 34); // right-aligned rank-3
    oxmera_autograd::gradcheck(
        |i| i[0].t()?.matmul(&i[1])?.sum(&[0, 1, 2, 3]),
        &[Tensor::randn_with_seed([2, 2, 4, 3], 35), w3],
        1e-3,
        2e-2,
    )
    .expect("transposed rank-4 x rank-3");
}

/// einsum is permute/sum/matmul underneath, so every contraction shape
/// must be differentiable in both operands.
#[test]
fn einsum_gradients_check() {
    use oxmera_tensor::{einsum, tensor::Tensor};
    let x = Tensor::randn_with_seed([2, 2, 3, 4], 41);
    let w = Tensor::randn_with_seed([2, 4, 3], 42);
    oxmera_autograd::gradcheck(
        |i| einsum("rbhd,rdo->rbho", &[&i[0], &i[1]])?.sum(&[0, 1, 2, 3]),
        &[x, w],
        1e-3,
        2e-2,
    )
    .expect("rbhd,rdo->rbho");
    let a = Tensor::randn_with_seed([3, 4], 43);
    let b = Tensor::randn_with_seed([4, 5], 44);
    oxmera_autograd::gradcheck(
        |i| {
            einsum("ij,jk->ki", &[&i[0], &i[1]])?
                .mul(&i[0].sum(&[])?)?
                .sum(&[0, 1])
        },
        &[a.clone(), b],
        1e-3,
        2e-2,
    )
    .expect("ij,jk->ki");
    oxmera_autograd::gradcheck(
        |i| {
            einsum("ij->j", &[&i[0]])?
                .mul(&einsum("ij->j", &[&i[0]])?)?
                .sum(&[0])
        },
        &[a],
        1e-3,
        2e-2,
    )
    .expect("ij->j");
    let u = Tensor::randn_with_seed([5], 45);
    let v = Tensor::randn_with_seed([5], 46);
    oxmera_autograd::gradcheck(|i| einsum("i,i->", &[&i[0], &i[1]]), &[u, v], 1e-3, 2e-2)
        .expect("i,i->");
}

/// Linear algebra (issue #26): diag/trace are composites; cholesky carries
/// Murray's VJP and logdet's gradient must come out as A⁻¹ (symmetrized).
#[test]
fn linalg_gradients_check() {
    use oxmera_tensor::tensor::Tensor;
    let m = Tensor::randn_with_seed([2, 4, 4], 51);
    let spd = m
        .matmul(&m.t().unwrap())
        .unwrap()
        .add(&Tensor::eye(4).mul_scalar(4.0).unwrap())
        .unwrap();
    // The check perturbs single elements, which breaks symmetry; feed the
    // input through a symmetrizer so every probe stays SPD and the
    // analytic gradient is compared on the symmetric manifold.
    let sym = |a: &Tensor| a.add(&a.t()?)?.mul_scalar(0.5);
    oxmera_autograd::gradcheck(
        |i| {
            sym(&i[0])?
                .cholesky()?
                .mul(&sym(&i[0])?.cholesky()?)?
                .sum(&[0, 1, 2])
        },
        std::slice::from_ref(&spd),
        1e-2,
        3e-2,
    )
    .expect("cholesky");
    oxmera_autograd::gradcheck(
        |i| sym(&i[0])?.logdet()?.sum(&[0]),
        std::slice::from_ref(&spd),
        1e-2,
        3e-2,
    )
    .expect("logdet");
    // d logdet / dA = A⁻¹ for a symmetric A: check against eigh.
    let a = sym(&spd.narrow(0, 0, 1).unwrap().reshape([4, 4]).unwrap())
        .unwrap()
        .detach()
        .requires_grad_(true);
    a.logdet().unwrap().backward().unwrap();
    let g = a.grad().unwrap();
    let (w, v) = a.eigh().unwrap();
    let inv_w = Tensor::from_vec_f32(
        w.to_vec_f32().unwrap().iter().map(|x| 1.0 / x).collect(),
        [4],
    )
    .unwrap();
    let inv = v
        .mul(&inv_w.unsqueeze(0).unwrap())
        .unwrap()
        .matmul(&v.t().unwrap())
        .unwrap();
    for (x, y) in g
        .to_vec_f32()
        .unwrap()
        .iter()
        .zip(inv.to_vec_f32().unwrap())
    {
        assert!((x - y).abs() < 2e-3 * 1.0f32.max(y.abs()), "{x} vs {y}");
    }
    oxmera_autograd::gradcheck(
        |i| i[0].diag()?.mul(&i[0].trace()?.unsqueeze(1)?)?.sum(&[0, 1]),
        &[Tensor::randn_with_seed([3, 3, 3], 52)],
        1e-3,
        2e-2,
    )
    .expect("diag/trace");
    oxmera_autograd::gradcheck(
        |i| {
            i[0].diag_embed()?
                .matmul(&i[0].diag_embed()?)?
                .sum(&[0, 1, 2])
        },
        &[Tensor::randn_with_seed([2, 3], 53)],
        1e-3,
        2e-2,
    )
    .expect("diag_embed");
}
