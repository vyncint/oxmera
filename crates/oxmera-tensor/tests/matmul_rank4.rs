//! Rank-4+ matmul (issue #30): every leading dimension is a batch
//! dimension and batch dimensions broadcast NumPy-style. The backends
//! still see the rank-2/3 contract; the lowering lives in the method
//! layer and is checked here against an explicit loop.
use oxmera_tensor::tensor::Tensor;

fn seq(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * scale - 1.0).collect()
}

fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k).map(|x| a[i * k + x] * b[x * n + j]).sum();
        }
    }
    out
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
fn rank_four_equal_batches_match_the_loop() {
    let (r, b, m, k, n) = (2, 3, 4, 5, 6);
    let a = Tensor::from_slice(&seq(r * b * m * k, 0.01), [r, b, m, k]).unwrap();
    let w = Tensor::from_slice(&seq(r * b * k * n, 0.02), [r, b, k, n]).unwrap();
    let c = a.matmul(&w).unwrap();
    assert_eq!(c.dims(), &[r, b, m, n]);
    let (av, wv, cv) = (
        a.to_vec_f32().unwrap(),
        w.to_vec_f32().unwrap(),
        c.to_vec_f32().unwrap(),
    );
    for i in 0..r * b {
        let want = reference(
            &av[i * m * k..(i + 1) * m * k],
            &wv[i * k * n..(i + 1) * k * n],
            m,
            k,
            n,
        );
        close(
            &cv[i * m * n..(i + 1) * m * n],
            &want,
            &format!("batch {i}"),
        );
    }
}

#[test]
fn batch_dimensions_broadcast_against_each_other() {
    // [2, 1, 3, 4] x [1, 5, 4, 6] -> [2, 5, 3, 6]
    let a = Tensor::from_slice(&seq(2 * 3 * 4, 0.05), [2, 1, 3, 4]).unwrap();
    let w = Tensor::from_slice(&seq(5 * 4 * 6, 0.03), [1, 5, 4, 6]).unwrap();
    let c = a.matmul(&w).unwrap();
    assert_eq!(c.dims(), &[2, 5, 3, 6]);
    let (av, wv, cv) = (
        a.to_vec_f32().unwrap(),
        w.to_vec_f32().unwrap(),
        c.to_vec_f32().unwrap(),
    );
    for i in 0..2 {
        for j in 0..5 {
            let want = reference(
                &av[i * 12..(i + 1) * 12],
                &wv[j * 24..(j + 1) * 24],
                3,
                4,
                6,
            );
            let at = (i * 5 + j) * 18;
            close(&cv[at..at + 18], &want, &format!("batch ({i},{j})"));
        }
    }
}

#[test]
fn a_rank_two_operand_multiplies_every_batch_of_a_rank_four_one() {
    let a = Tensor::from_slice(&seq(2 * 3 * 4 * 5, 0.01), [2, 3, 4, 5]).unwrap();
    let w = Tensor::from_slice(&seq(5 * 2, 0.1), [5, 2]).unwrap();
    let c = a.matmul(&w).unwrap();
    assert_eq!(c.dims(), &[2, 3, 4, 2]);
    let (av, wv, cv) = (
        a.to_vec_f32().unwrap(),
        w.to_vec_f32().unwrap(),
        c.to_vec_f32().unwrap(),
    );
    for i in 0..6 {
        let want = reference(&av[i * 20..(i + 1) * 20], &wv, 4, 5, 2);
        close(&cv[i * 8..(i + 1) * 8], &want, &format!("batch {i}"));
    }
    // And on the left: [k, n]-shaped weights times a rank-4 batch.
    let l = Tensor::from_slice(&seq(3 * 4, 0.1), [3, 4]).unwrap();
    let r = Tensor::from_slice(&seq(2 * 2 * 4 * 5, 0.01), [2, 2, 4, 5]).unwrap();
    let c = l.matmul(&r).unwrap();
    assert_eq!(c.dims(), &[2, 2, 3, 5]);
    let (lv, rv, cv) = (
        l.to_vec_f32().unwrap(),
        r.to_vec_f32().unwrap(),
        c.to_vec_f32().unwrap(),
    );
    for i in 0..4 {
        let want = reference(&lv, &rv[i * 20..(i + 1) * 20], 3, 4, 5);
        close(&cv[i * 15..(i + 1) * 15], &want, &format!("left batch {i}"));
    }
}

#[test]
fn a_shorter_batch_prefix_is_right_aligned() {
    // [2, 3, 4, 5] x [3, 5, 6]: the rank-3 operand's batch (3) aligns with
    // the trailing batch dim of the rank-4 operand.
    let a = Tensor::from_slice(&seq(2 * 3 * 4 * 5, 0.01), [2, 3, 4, 5]).unwrap();
    let w = Tensor::from_slice(&seq(3 * 5 * 6, 0.02), [3, 5, 6]).unwrap();
    let c = a.matmul(&w).unwrap();
    assert_eq!(c.dims(), &[2, 3, 4, 6]);
    let (av, wv, cv) = (
        a.to_vec_f32().unwrap(),
        w.to_vec_f32().unwrap(),
        c.to_vec_f32().unwrap(),
    );
    for i in 0..2 {
        for j in 0..3 {
            let want = reference(
                &av[(i * 3 + j) * 20..(i * 3 + j + 1) * 20],
                &wv[j * 30..(j + 1) * 30],
                4,
                5,
                6,
            );
            let at = (i * 3 + j) * 24;
            close(&cv[at..at + 24], &want, &format!("batch ({i},{j})"));
        }
    }
}

#[test]
fn rank_five_works_and_a_transposed_view_is_handled() {
    let a = Tensor::from_slice(&seq(2 * 2 * 2 * 3 * 4, 0.01), [2, 2, 2, 3, 4]).unwrap();
    let w = Tensor::from_slice(&seq(2 * 2 * 2 * 4 * 3, 0.02), [2, 2, 2, 4, 3]).unwrap();
    let c = a.matmul(&w).unwrap();
    assert_eq!(c.dims(), &[2, 2, 2, 3, 3]);
    // a.t() is a strided view: [.., 4, 3] x [.., 3, 4] -> [.., 4, 4].
    let ct = a.t().unwrap().matmul(&w.t().unwrap()).unwrap();
    assert_eq!(ct.dims(), &[2, 2, 2, 4, 4]);
    let (av, wv, ctv) = (
        a.to_vec_f32().unwrap(),
        w.to_vec_f32().unwrap(),
        ct.to_vec_f32().unwrap(),
    );
    // Batch 0 by hand: a0^T (4x3) times w0^T (3x4).
    let mut a0t = vec![0.0; 12];
    let mut w0t = vec![0.0; 12];
    for i in 0..3 {
        for j in 0..4 {
            a0t[j * 3 + i] = av[i * 4 + j];
        }
    }
    for i in 0..4 {
        for j in 0..3 {
            w0t[j * 4 + i] = wv[i * 3 + j];
        }
    }
    close(
        &ctv[..16],
        &reference(&a0t, &w0t, 4, 3, 4),
        "transposed batch 0",
    );
}

#[test]
fn incompatible_high_rank_operands_are_typed_errors() {
    let a = Tensor::zeros([2usize, 3, 4, 5]);
    // Inner dimension mismatch.
    let e = a.matmul(&Tensor::zeros([2usize, 3, 6, 7])).unwrap_err();
    assert!(e.to_string().contains("shape mismatch"), "{e}");
    // Batch dims that neither match nor are 1.
    let e = a.matmul(&Tensor::zeros([3usize, 3, 5, 7])).unwrap_err();
    assert!(e.to_string().contains("broadcast"), "{e}");
    // Rank 1 is still not a matrix.
    assert!(
        Tensor::zeros([5usize])
            .matmul(&Tensor::zeros([2usize, 3, 5, 7]))
            .is_err()
    );
}

#[test]
fn zero_sized_batches_and_matrices_are_fine() {
    let a = Tensor::zeros([0usize, 2, 3, 4]);
    let w = Tensor::zeros([1usize, 2, 4, 5]);
    let c = a.matmul(&w).unwrap();
    assert_eq!(c.dims(), &[0, 2, 3, 5]);
    assert_eq!(c.numel(), 0);
}
