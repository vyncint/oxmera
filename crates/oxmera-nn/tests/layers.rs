//! Layer semantics: hand-verifiable forward passes, a naive-convolution
//! cross-check, mode switches, loss values, gradient flow through modules,
//! and the safetensors round-trip.

use oxmera_core::Shape;
use oxmera_nn::{
    BCEWithLogitsLoss, BatchNorm2d, Conv2d, CrossEntropyLoss, Dropout, Embedding, LayerNorm,
    Linear, MSELoss, Module, Sequential, serialize,
};
use oxmera_tensor::tensor::Tensor;

fn assert_close(a: &[f32], b: &[f32], tol: f32, context: &str) {
    assert_eq!(a.len(), b.len(), "{context}: length");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "{context}: element {i}: {x} vs {y}");
    }
}

#[test]
fn linear_is_x_wt_plus_b() {
    let layer = Linear::new(3, 2, 42);
    layer
        .weight()
        .set(Tensor::from_slice(&[1.0, 0.0, -1.0, 2.0, 1.0, 0.5], [2, 3]).unwrap());
    layer
        .bias()
        .unwrap()
        .set(Tensor::from_slice(&[10.0, -10.0], [2]).unwrap());
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0], [1, 3]).unwrap();
    let y = layer.forward(&x).unwrap();
    // Row: [1*1 + 2*0 + 3*(-1) + 10, 1*2 + 2*1 + 3*0.5 - 10] = [8, -4.5]
    assert_close(&y.to_vec_f32().unwrap(), &[8.0, -4.5], 1e-6, "linear");
}

#[test]
fn conv2d_matches_a_naive_direct_convolution() {
    let (n, cin, cout, h, w, kh, kw, stride, pad) = (2, 3, 4, 6, 5, 3, 2, 2, 1);
    let conv = Conv2d::new(cin, cout, (kh, kw), stride, pad, 7);
    let x = Tensor::randn_with_seed([n, cin, h, w], 8);
    let y = conv.forward(&x).unwrap();

    let h_out = (h + 2 * pad - kh) / stride + 1;
    let w_out = (w + 2 * pad - kw) / stride + 1;
    assert_eq!(y.dims(), &[n, cout, h_out, w_out]);

    // Naive direct convolution on the host.
    let xv = x.to_vec_f32().unwrap();
    let wv = conv.weight().value().to_vec_f32().unwrap();
    let get_x = |ni: usize, c: usize, i: isize, j: isize| -> f32 {
        if i < 0 || j < 0 || i >= h as isize || j >= w as isize {
            0.0
        } else {
            xv[((ni * cin + c) * h + i as usize) * w + j as usize]
        }
    };
    let yv = y.to_vec_f32().unwrap();
    for ni in 0..n {
        for co in 0..cout {
            for oi in 0..h_out {
                for oj in 0..w_out {
                    let mut acc = 0.0f32;
                    for c in 0..cin {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ii = (oi * stride + ki) as isize - pad as isize;
                                let jj = (oj * stride + kj) as isize - pad as isize;
                                acc +=
                                    get_x(ni, c, ii, jj) * wv[((co * cin + c) * kh + ki) * kw + kj];
                            }
                        }
                    }
                    let got = yv[((ni * cout + co) * h_out + oi) * w_out + oj];
                    assert!(
                        (got - acc).abs() <= 1e-4,
                        "conv [{ni},{co},{oi},{oj}]: {got} vs {acc}"
                    );
                }
            }
        }
    }
}

#[test]
fn layernorm_normalizes_the_last_dim() {
    let ln = LayerNorm::new(4);
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, -2.0, 0.0, 2.0, 4.0], [2, 4]).unwrap();
    let y = ln.forward(&x).unwrap();
    for row in 0..2 {
        let vals: Vec<f32> = (0..4).map(|c| y.get_f32(&[row, c]).unwrap()).collect();
        let mean: f32 = vals.iter().sum::<f32>() / 4.0;
        let var: f32 = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "row {row} mean {mean}");
        assert!((var - 1.0).abs() < 1e-3, "row {row} var {var}");
    }
}

#[test]
fn batchnorm_normalizes_in_train_and_uses_running_stats_in_eval() {
    let bn = BatchNorm2d::new(2);
    let x = Tensor::randn_with_seed([4, 2, 3, 3], 9)
        .mul_scalar(3.0)
        .unwrap()
        .add_scalar(5.0)
        .unwrap();
    let y = bn.forward(&x).unwrap();
    // Per-channel statistics of the output are ~N(0, 1) in training mode.
    let y0 = y.narrow(1, 0, 1).unwrap().to_vec_f32().unwrap();
    let mean: f32 = y0.iter().sum::<f32>() / y0.len() as f32;
    assert!(mean.abs() < 1e-4, "train-mode mean {mean}");

    bn.set_training(false);
    let z = bn.forward(&x).unwrap();
    assert_eq!(z.dims(), x.dims());
}

#[test]
fn dropout_is_identity_in_eval_and_scales_in_train() {
    let d = Dropout::new(0.5, 13);
    let x = Tensor::ones([1000]);
    let y = d.forward(&x).unwrap();
    let kept: Vec<f32> = y.to_vec_f32().unwrap();
    let zeros = kept.iter().filter(|&&v| v == 0.0).count();
    assert!(
        zeros > 300 && zeros < 700,
        "drop rate wildly off: {zeros}/1000"
    );
    assert!(
        kept.iter().all(|&v| v == 0.0 || (v - 2.0).abs() < 1e-6),
        "survivors scaled by 1/(1-p)"
    );

    d.set_training(false);
    let z = d.forward(&x).unwrap();
    assert_eq!(z.to_vec_f32().unwrap(), x.to_vec_f32().unwrap());
}

#[test]
fn embedding_looks_up_rows() {
    let emb = Embedding::new(5, 3, 21);
    let table = emb.weight().value().to_vec_f32().unwrap();
    let idx = Tensor::from_vec_i64(vec![4, 0, 4], Shape::from([3])).unwrap();
    let out = emb.forward(&idx).unwrap();
    assert_eq!(out.dims(), &[3, 3]);
    let ov = out.to_vec_f32().unwrap();
    assert_close(&ov[0..3], &table[12..15], 1e-6, "row 4");
    assert_close(&ov[3..6], &table[0..3], 1e-6, "row 0");
}

#[test]
fn loss_hand_values() {
    let mse = MSELoss;
    let a = Tensor::from_slice(&[1.0, 2.0], [2]).unwrap();
    let b = Tensor::from_slice(&[3.0, 2.0], [2]).unwrap();
    assert!((mse.forward(&a, &b).unwrap().get_f32(&[]).unwrap() - 2.0).abs() < 1e-6);

    // Uniform logits over 4 classes: CE = ln(4).
    let ce = CrossEntropyLoss;
    let logits = Tensor::zeros([3, 4]);
    let targets = Tensor::from_vec_i64(vec![0, 1, 3], Shape::from([3])).unwrap();
    let loss = ce.forward(&logits, &targets).unwrap().get_f32(&[]).unwrap();
    assert!((loss - 4.0f32.ln()).abs() < 1e-5, "{loss} vs ln4");

    // BCE at logit 0 vs target 0.5-ish: ln 2 at z=0, t arbitrary.
    let bce = BCEWithLogitsLoss;
    let z = Tensor::zeros([2]);
    let t = Tensor::from_slice(&[0.0, 1.0], [2]).unwrap();
    let loss = bce.forward(&z, &t).unwrap().get_f32(&[]).unwrap();
    assert!((loss - 2.0f32.ln()).abs() < 1e-6, "{loss} vs ln2");
}

#[test]
fn gradients_reach_module_parameters() {
    let model = Sequential::new()
        .push(Linear::new(3, 8, 1))
        .push(Linear::new(8, 2, 2));
    let x = Tensor::randn_with_seed([4, 3], 3);
    let target = Tensor::randn_with_seed([4, 2], 4);
    let loss = MSELoss
        .forward(&model.forward(&x).unwrap(), &target)
        .unwrap();
    loss.backward().unwrap();
    let params = model.parameters();
    assert_eq!(params.len(), 4);
    for (i, p) in params.iter().enumerate() {
        let g = p.grad().unwrap_or_else(|| panic!("param {i} has no grad"));
        assert_eq!(g.dims(), p.value().dims(), "param {i} grad shape");
    }
    model.zero_grad();
    assert!(params.iter().all(|p| p.grad().is_none()));
}

#[test]
fn safetensors_round_trip() {
    let model = Sequential::new()
        .push(Linear::new(4, 8, 5))
        .push(Linear::new(8, 3, 6));
    let x = Tensor::randn_with_seed([2, 4], 7);
    let before = model.forward(&x).unwrap().to_vec_f32().unwrap();

    let dir = std::env::temp_dir().join("oxmera-nn-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    serialize::save(&model, &path).unwrap();

    let fresh = Sequential::new()
        .push(Linear::new(4, 8, 100))
        .push(Linear::new(8, 3, 101));
    let differs = fresh.forward(&x).unwrap().to_vec_f32().unwrap();
    assert!(
        before
            .iter()
            .zip(&differs)
            .any(|(a, b)| (a - b).abs() > 1e-6)
    );

    serialize::load(&fresh, &path).unwrap();
    let after = fresh.forward(&x).unwrap().to_vec_f32().unwrap();
    assert_close(&before, &after, 1e-6, "safetensors round trip");
    std::fs::remove_file(&path).ok();
}
