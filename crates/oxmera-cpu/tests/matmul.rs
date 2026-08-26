//! Matrix multiplication: hand-computed products (originally the A4
//! spec), algebraic laws through views, batched GEMM, and the tiled path
//! against a naive triple loop.

use oxmera_core::{Error, Shape};
use oxmera_tensor::tensor::Tensor;

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec_f32(data.to_vec(), Shape::from(shape)).unwrap()
}

#[test]
fn hand_computed_2x2() {
    let c = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .matmul(&t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]))
        .unwrap();
    assert_eq!(c.to_vec_f32().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn identity_zero_and_rectangular() {
    let a = t(&[1.5, -2.0, 0.25, 4.0], &[2, 2]);
    let eye = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    assert_eq!(
        a.matmul(&eye).unwrap().to_vec_f32().unwrap(),
        a.to_vec_f32().unwrap()
    );

    let r = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .matmul(&t(&[1.0, 10.0, 100.0], &[3, 1]))
        .unwrap();
    assert_eq!(r.to_vec_f32().unwrap(), vec![321.0, 654.0]);
}

#[test]
fn matmul_reads_through_a_transposed_view() {
    let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b_t_in_memory = t(&[5.0, 7.0, 6.0, 8.0], &[2, 2]);
    let bt = b_t_in_memory.permute(&[1, 0]).unwrap();
    let c = a.matmul(&bt).unwrap();
    assert_eq!(c.to_vec_f32().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn inner_dimension_mismatch_is_typed() {
    let a = t(&[1.0; 6], &[2, 3]);
    let b = t(&[1.0; 8], &[4, 2]);
    assert!(matches!(a.matmul(&b), Err(Error::ShapeMismatch { .. })));
}

#[test]
fn batched_matmul_matches_per_batch_2d() {
    let a = Tensor::randn_with_seed([3, 4, 5], 11);
    let b = Tensor::randn_with_seed([3, 5, 2], 12);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.dims(), &[3, 4, 2]);
    for i in 0..3 {
        let ai = a.narrow(0, i, 1).unwrap().reshape([4, 5]).unwrap();
        let bi = b.narrow(0, i, 1).unwrap().reshape([5, 2]).unwrap();
        let ci = ai.matmul(&bi).unwrap().to_vec_f32().unwrap();
        let want = c.narrow(0, i, 1).unwrap().to_vec_f32().unwrap();
        for (x, y) in ci.iter().zip(&want) {
            assert!((x - y).abs() <= 1e-5, "batch {i}: {x} vs {y}");
        }
    }
}

#[test]
fn tiled_path_matches_a_naive_triple_loop() {
    let (m, k, n) = (37, 41, 29); // deliberately not multiples of any block
    let a = Tensor::randn_with_seed([m, k], 1);
    let b = Tensor::randn_with_seed([k, n], 2);
    let c = a.matmul(&b).unwrap().to_vec_f32().unwrap();

    let av = a.to_vec_f32().unwrap();
    let bv = b.to_vec_f32().unwrap();
    let mut want = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += av[i * k + kk] * bv[kk * n + j];
            }
            want[i * n + j] = acc;
        }
    }
    for (i, (x, y)) in c.iter().zip(&want).enumerate() {
        assert!((x - y).abs() <= 1e-3, "element {i}: {x} vs {y}");
    }
}
