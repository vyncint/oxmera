//! `einsum` (issue #30): the common one- and two-operand contractions,
//! checked against matmul, transpose, sum and explicit loops.
use oxmera_tensor::einsum;
use oxmera_tensor::tensor::Tensor;

fn seq(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * scale - 1.0).collect()
}

fn close(got: &[f32], want: &[f32], context: &str) {
    assert_eq!(got.len(), want.len(), "{context}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5 * 1.0f32.max(w.abs()),
            "{context}: element {i}: {g} vs {w}"
        );
    }
}

#[test]
fn matrix_product_matches_matmul() {
    let a = Tensor::from_slice(&seq(3 * 4, 0.1), [3, 4]).unwrap();
    let b = Tensor::from_slice(&seq(4 * 5, 0.2), [4, 5]).unwrap();
    let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[3, 5]);
    close(
        &c.to_vec_f32().unwrap(),
        &a.matmul(&b).unwrap().to_vec_f32().unwrap(),
        "ij,jk->ik",
    );
    // Transposed output order is a permute of the same product.
    let ct = einsum("ij,jk->ki", &[&a, &b]).unwrap();
    assert_eq!(ct.dims(), &[5, 3]);
    close(
        &ct.to_vec_f32().unwrap(),
        &a.matmul(&b).unwrap().t().unwrap().to_vec_f32().unwrap(),
        "ij,jk->ki",
    );
}

#[test]
fn batched_product_matches_batched_matmul() {
    let a = Tensor::from_slice(&seq(2 * 3 * 4, 0.05), [2, 3, 4]).unwrap();
    let b = Tensor::from_slice(&seq(2 * 4 * 5, 0.03), [2, 4, 5]).unwrap();
    let c = einsum("bij,bjk->bik", &[&a, &b]).unwrap();
    close(
        &c.to_vec_f32().unwrap(),
        &a.matmul(&b).unwrap().to_vec_f32().unwrap(),
        "bij,bjk->bik",
    );
}

#[test]
fn the_stacked_replica_contraction_from_oxmega() {
    // [R, B, H, d] x [R, d, o] -> [R, B, H, o]: one weight per replica,
    // shared across the batch and history axes.
    let (r, b, h, d, o) = (2, 3, 4, 5, 6);
    let x = Tensor::from_slice(&seq(r * b * h * d, 0.01), [r, b, h, d]).unwrap();
    let w = Tensor::from_slice(&seq(r * d * o, 0.02), [r, d, o]).unwrap();
    let y = einsum("rbhd,rdo->rbho", &[&x, &w]).unwrap();
    assert_eq!(y.dims(), &[r, b, h, o]);
    let (xv, wv, yv) = (
        x.to_vec_f32().unwrap(),
        w.to_vec_f32().unwrap(),
        y.to_vec_f32().unwrap(),
    );
    for ri in 0..r {
        for bi in 0..b {
            for hi in 0..h {
                for oi in 0..o {
                    let mut want = 0.0f32;
                    for di in 0..d {
                        want += xv[((ri * b + bi) * h + hi) * d + di] * wv[(ri * d + di) * o + oi];
                    }
                    let got = yv[((ri * b + bi) * h + hi) * o + oi];
                    assert!(
                        (got - want).abs() <= 1e-5 * 1.0f32.max(want.abs()),
                        "({ri},{bi},{hi},{oi}): {got} vs {want}"
                    );
                }
            }
        }
    }
    // The same contraction through rank-4 matmul broadcasting agrees.
    let via_matmul = x
        .matmul(&w.reshape([r, 1, d, o]).unwrap())
        .unwrap()
        .to_vec_f32()
        .unwrap();
    close(&yv, &via_matmul, "einsum vs rank-4 matmul");
}

#[test]
fn single_operand_transpose_sum_and_trace_like_reductions() {
    let a = Tensor::from_slice(&seq(3 * 4, 0.1), [3, 4]).unwrap();
    close(
        &einsum("ij->ji", &[&a]).unwrap().to_vec_f32().unwrap(),
        &a.t().unwrap().to_vec_f32().unwrap(),
        "transpose",
    );
    close(
        &einsum("ij->i", &[&a]).unwrap().to_vec_f32().unwrap(),
        &a.sum(&[1]).unwrap().to_vec_f32().unwrap(),
        "row sums",
    );
    close(
        &einsum("ij->j", &[&a]).unwrap().to_vec_f32().unwrap(),
        &a.sum(&[0]).unwrap().to_vec_f32().unwrap(),
        "column sums",
    );
    let total = einsum("ij->", &[&a]).unwrap();
    assert_eq!(total.dims(), &[] as &[usize]);
    close(
        &total.to_vec_f32().unwrap(),
        &a.sum(&[]).unwrap().to_vec_f32().unwrap(),
        "full sum",
    );
    let t = Tensor::from_slice(&seq(2 * 3 * 4, 0.1), [2, 3, 4]).unwrap();
    let p = einsum("abc->cab", &[&t]).unwrap();
    assert_eq!(p.dims(), &[4, 2, 3]);
    close(
        &p.to_vec_f32().unwrap(),
        &t.permute(&[2, 0, 1]).unwrap().to_vec_f32().unwrap(),
        "permute",
    );
}

#[test]
fn dot_outer_and_elementwise_products() {
    let u = Tensor::from_slice(&seq(4, 0.5), [4]).unwrap();
    let v = Tensor::from_slice(&seq(4, 0.25), [4]).unwrap();
    let dot = einsum("i,i->", &[&u, &v]).unwrap();
    let want: f32 = u
        .to_vec_f32()
        .unwrap()
        .iter()
        .zip(v.to_vec_f32().unwrap())
        .map(|(a, b)| a * b)
        .sum();
    close(&dot.to_vec_f32().unwrap(), &[want], "dot");
    let outer = einsum("i,j->ij", &[&u, &v]).unwrap();
    assert_eq!(outer.dims(), &[4, 4]);
    let (uv, vv) = (u.to_vec_f32().unwrap(), v.to_vec_f32().unwrap());
    let want: Vec<f32> = (0..4)
        .flat_map(|i| (0..4).map(move |j| (i, j)))
        .map(|(i, j)| uv[i] * vv[j])
        .collect();
    close(&outer.to_vec_f32().unwrap(), &want, "outer");
    // Hadamard product: a letter in both operands and in the output.
    let had = einsum("i,i->i", &[&u, &v]).unwrap();
    close(
        &had.to_vec_f32().unwrap(),
        &u.mul(&v).unwrap().to_vec_f32().unwrap(),
        "hadamard",
    );
    // A summed-only letter on one side: "ij,j->" sums a then dots.
    let a = Tensor::from_slice(&seq(2 * 4, 0.1), [2, 4]).unwrap();
    let s = einsum("ij,j->", &[&a, &v]).unwrap();
    let want: f32 = a
        .sum(&[0])
        .unwrap()
        .to_vec_f32()
        .unwrap()
        .iter()
        .zip(&vv)
        .map(|(x, y)| x * y)
        .sum();
    close(&s.to_vec_f32().unwrap(), &[want], "ij,j->");
}

#[test]
fn bad_specs_are_typed_errors() {
    let a = Tensor::zeros([2usize, 3]);
    let b = Tensor::zeros([3usize, 4]);
    for spec in [
        "ij,jk",      // implicit output
        "ij,jk->ik",  // wrong operand count for one operand (checked below)
        "ii->i",      // diagonal
        "ij,jk->il",  // output letter absent
        "iJ,jk->ik",  // not lowercase
        "ijk,jk->ik", // rank mismatch
    ] {
        let ops: Vec<&Tensor> = if spec == "ij,jk->ik" {
            vec![&a]
        } else {
            vec![&a, &b]
        };
        let e = einsum(spec, &ops).unwrap_err().to_string();
        assert!(e.contains("einsum"), "{spec}: {e}");
    }
    // Size conflict on a shared letter.
    let c = Tensor::zeros([5usize, 4]);
    let e = einsum("ij,jk->ik", &[&a, &c]).unwrap_err().to_string();
    assert!(e.contains("letter 'j'"), "{e}");
}
