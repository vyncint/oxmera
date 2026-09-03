//! Small batched linear algebra (issue #26): identity/diagonal/trace
//! constructors, Cholesky, log-determinant and the symmetric eigen-
//! decomposition, checked against hand-computed values and reconstruction
//! identities on random SPD batches.
use oxmera_tensor::tensor::Tensor;

fn close(got: &[f32], want: &[f32], tol: f32, context: &str) {
    assert_eq!(got.len(), want.len(), "{context}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * 1.0f32.max(w.abs()),
            "{context}: element {i}: {g} vs {w}"
        );
    }
}

/// A random SPD batch: A = M Mᵀ + n I.
fn spd(batch: usize, n: usize, seed: u64) -> Tensor {
    let m = Tensor::randn_with_seed([batch, n, n], seed);
    m.matmul(&m.t().unwrap())
        .unwrap()
        .add(&Tensor::eye(n).mul_scalar(n as f32).unwrap())
        .unwrap()
}

#[test]
fn eye_diag_diag_embed_and_trace() {
    let i3 = Tensor::eye(3);
    assert_eq!(
        i3.to_vec_f32().unwrap(),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3]).unwrap();
    assert_eq!(a.diag().unwrap().to_vec_f32().unwrap(), vec![1.0, 5.0, 9.0]);
    assert_eq!(a.trace().unwrap().to_vec_f32().unwrap(), vec![15.0]);
    assert_eq!(a.trace().unwrap().dims(), &[] as &[usize]);
    // Batched: [2, 3, 3] → diag [2, 3], trace [2].
    let b = Tensor::from_slice(&(0..18).map(|x| x as f32).collect::<Vec<_>>(), [2, 3, 3]).unwrap();
    assert_eq!(
        b.diag().unwrap().to_vec_f32().unwrap(),
        vec![0.0, 4.0, 8.0, 9.0, 13.0, 17.0]
    );
    assert_eq!(b.trace().unwrap().to_vec_f32().unwrap(), vec![12.0, 39.0]);
    // diag_embed puts a vector on the diagonal and is diag's inverse there.
    let v = Tensor::from_slice(&[2.0, 3.0, 4.0, 5.0], [2, 2]).unwrap();
    let d = v.diag_embed().unwrap();
    assert_eq!(d.dims(), &[2, 2, 2]);
    assert_eq!(
        d.to_vec_f32().unwrap(),
        vec![2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0]
    );
    assert_eq!(
        d.diag().unwrap().to_vec_f32().unwrap(),
        v.to_vec_f32().unwrap()
    );
    // Non-square is a typed error.
    let e = Tensor::zeros([2usize, 3]).diag().unwrap_err().to_string();
    assert!(e.contains("diag"), "{e}");
}

#[test]
fn cholesky_reconstructs_the_input_and_is_lower_triangular() {
    let a = spd(3, 6, 7);
    let l = a.cholesky().unwrap();
    assert_eq!(l.dims(), a.dims());
    let lv = l.to_vec_f32().unwrap();
    for b in 0..3 {
        for i in 0..6 {
            for j in i + 1..6 {
                assert_eq!(lv[b * 36 + i * 6 + j], 0.0, "upper triangle must be zero");
            }
        }
    }
    let back = l.matmul(&l.t().unwrap()).unwrap();
    close(
        &back.to_vec_f32().unwrap(),
        &a.to_vec_f32().unwrap(),
        1e-4,
        "L Lᵀ = A",
    );
    // Only the lower triangle is read: garbage above the diagonal is ignored.
    let a2 = Tensor::from_slice(&[4.0, 999.0, 2.0, 3.0], [2, 2]).unwrap();
    let l2 = a2.cholesky().unwrap().to_vec_f32().unwrap();
    close(&l2, &[2.0, 0.0, 1.0, 2f32.sqrt()], 1e-6, "lower read");
}

#[test]
fn cholesky_of_an_indefinite_matrix_is_a_typed_error_naming_the_batch() {
    let mut data = spd(2, 3, 11).to_vec_f32().unwrap();
    // Break batch 1: make it negative definite.
    for x in &mut data[9..] {
        *x = -*x;
    }
    let a = Tensor::from_vec_f32(data, [2, 3, 3]).unwrap();
    let e = a.cholesky().unwrap_err().to_string();
    assert!(e.contains("positive definite"), "{e}");
    assert!(e.contains("matrix 1"), "{e}");
    assert!(a.logdet().is_err());
}

#[test]
fn logdet_and_det_match_the_eigenvalue_reference() {
    let a = spd(4, 5, 13);
    let ld = a.logdet().unwrap();
    assert_eq!(ld.dims(), &[4]);
    let (w, _) = a.eigh().unwrap();
    let wv = w.to_vec_f32().unwrap();
    let want: Vec<f32> = (0..4)
        .map(|b| {
            wv[b * 5..(b + 1) * 5]
                .iter()
                .map(|x| (*x as f64).ln())
                .sum::<f64>() as f32
        })
        .collect();
    close(&ld.to_vec_f32().unwrap(), &want, 1e-4, "logdet vs Σ ln λ");
    let det = a.det().unwrap().to_vec_f32().unwrap();
    let want_det: Vec<f32> = want.iter().map(|x| x.exp()).collect();
    close(&det, &want_det, 1e-3, "det");
    // 2x2 by hand: det [[2,1],[1,2]] = 3.
    let m = Tensor::from_slice(&[2.0, 1.0, 1.0, 2.0], [2, 2]).unwrap();
    close(
        &m.det().unwrap().to_vec_f32().unwrap(),
        &[3.0],
        1e-5,
        "2x2 det",
    );
    close(
        &m.logdet().unwrap().to_vec_f32().unwrap(),
        &[3f32.ln()],
        1e-5,
        "2x2 logdet",
    );
}

#[test]
fn eigh_diagonalizes_symmetric_batches() {
    let a = spd(3, 5, 17);
    let (w, v) = a.eigh().unwrap();
    assert_eq!(w.dims(), &[3, 5]);
    assert_eq!(v.dims(), &[3, 5, 5]);
    let wv = w.to_vec_f32().unwrap();
    for b in 0..3 {
        for i in 1..5 {
            assert!(wv[b * 5 + i] >= wv[b * 5 + i - 1], "ascending: {wv:?}");
        }
        for i in 0..5 {
            assert!(wv[b * 5 + i] > 0.0, "SPD eigenvalues are positive");
        }
    }
    // A V = V Λ and Vᵀ V = I.
    let av = a.matmul(&v).unwrap();
    let vl = v.mul(&w.unsqueeze(1).unwrap()).unwrap(); // scale column j by λ_j
    close(
        &av.to_vec_f32().unwrap(),
        &vl.to_vec_f32().unwrap(),
        1e-3,
        "A V = V Λ",
    );
    let vtv = v.t().unwrap().matmul(&v).unwrap();
    let eye = Tensor::eye(5)
        .unsqueeze(0)
        .unwrap()
        .broadcast_to([3, 5, 5])
        .unwrap();
    close(
        &vtv.to_vec_f32().unwrap(),
        &eye.contiguous().unwrap().to_vec_f32().unwrap(),
        1e-4,
        "VᵀV = I",
    );
    // Trace and logdet agree with the spectrum.
    let tr = a.trace().unwrap().to_vec_f32().unwrap();
    let sums: Vec<f32> = (0..3)
        .map(|b| wv[b * 5..(b + 1) * 5].iter().sum())
        .collect();
    close(&tr, &sums, 1e-4, "trace = Σ λ");
}

#[test]
fn eigh_handles_an_indefinite_matrix_and_a_diagonal_one() {
    let m = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], [2, 2]).unwrap();
    let (w, _) = m.eigh().unwrap();
    close(&w.to_vec_f32().unwrap(), &[-1.0, 1.0], 1e-6, "±1");
    let d = Tensor::from_slice(&[3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0], [3, 3]).unwrap();
    let (w, v) = d.eigh().unwrap();
    close(&w.to_vec_f32().unwrap(), &[1.0, 2.0, 3.0], 1e-6, "sorted");
    // Eigenvectors are signed unit basis vectors in the sorted order.
    let vv = v.to_vec_f32().unwrap();
    for (col, row) in [(0usize, 1usize), (1, 2), (2, 0)] {
        assert!((vv[row * 3 + col].abs() - 1.0).abs() < 1e-6, "{vv:?}");
    }
}
