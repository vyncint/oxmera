//! Reference dense linear algebra for small matrices, in `f64` on the
//! host: batched Cholesky, its reverse-mode derivative, and the cyclic
//! Jacobi symmetric eigen-decomposition. Inputs and outputs are `f32`
//! slabs in row-major `[batch, n, n]` order; every intermediate is `f64`
//! so the `f32` result is correctly rounded rather than accumulated.
//!
//! These are the CPU backend's implementations and the fallback every
//! other backend reaches through a device round-trip; `n` is expected to
//! be small (tens), which is where determinantal models and covariance
//! blocks live.

use oxmera_core::{Error, Result};

/// Lower-triangular Cholesky factors of `batch` SPD matrices, `L Lᵀ = A`.
///
/// Reads the lower triangle of each input (the upper is ignored, so a
/// numerically asymmetric input is accepted). Errors with the batch index
/// and pivot when a matrix is not positive definite.
pub fn cholesky(a: &[f32], batch: usize, n: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; batch * n * n];
    let mut l = vec![0.0f64; n * n];
    for b in 0..batch {
        let m = &a[b * n * n..(b + 1) * n * n];
        l.iter_mut().for_each(|x| *x = 0.0);
        for j in 0..n {
            let mut d = m[j * n + j] as f64;
            for k in 0..j {
                d -= l[j * n + k] * l[j * n + k];
            }
            if d.is_nan() || d <= 0.0 || d.is_infinite() {
                return Err(Error::InvalidArgument {
                    op: "cholesky",
                    detail: format!(
                        "matrix {b} is not positive definite (pivot {j} is {d:e}); cholesky/logdet need an SPD input"
                    ),
                });
            }
            let ljj = d.sqrt();
            l[j * n + j] = ljj;
            for i in j + 1..n {
                let mut s = m[i * n + j] as f64;
                for k in 0..j {
                    s -= l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = s / ljj;
            }
        }
        for (o, &v) in out[b * n * n..(b + 1) * n * n].iter_mut().zip(&l) {
            *o = v as f32;
        }
    }
    Ok(out)
}

/// Reverse-mode derivative of Cholesky (Murray, "Differentiation of the
/// Cholesky decomposition", 2016, eq. 7): given `L` and the gradient `Ḹ`
/// of a scalar with respect to `L`, the gradient with respect to the
/// symmetric input is
///
/// `Ā = ½ · L⁻ᵀ · Φ(Lᵀ Ḹ) · L⁻¹`, symmetrized,
///
/// where `Φ` keeps the lower triangle and halves the diagonal. Only the
/// lower triangle of `Ḹ` is read, matching the forward pass.
pub fn cholesky_backward(l: &[f32], grad_l: &[f32], batch: usize, n: usize) -> Vec<f32> {
    let nn = n * n;
    let mut out = vec![0.0f32; batch * nn];
    let mut p = vec![0.0f64; nn];
    let mut tmp = vec![0.0f64; nn];
    for b in 0..batch {
        let lb = &l[b * nn..(b + 1) * nn];
        let gb = &grad_l[b * nn..(b + 1) * nn];
        // P = Φ(Lᵀ Ḹ): lower triangle, diagonal halved.
        for i in 0..n {
            for j in 0..=i {
                let mut s = 0.0f64;
                for k in i..n {
                    // (Lᵀ)[i][k] = L[k][i]; Ḹ[k][j] with k ≥ j (lower).
                    s += lb[k * n + i] as f64 * gb[k * n + j] as f64;
                }
                p[i * n + j] = if i == j { 0.5 * s } else { s };
            }
            for j in i + 1..n {
                p[i * n + j] = 0.0;
            }
        }
        // tmp = L⁻ᵀ P : solve Lᵀ X = P (back substitution, Lᵀ upper).
        for col in 0..n {
            for i in (0..n).rev() {
                let mut s = p[i * n + col];
                for k in i + 1..n {
                    s -= lb[k * n + i] as f64 * tmp[k * n + col];
                }
                tmp[i * n + col] = s / lb[i * n + i] as f64;
            }
        }
        // S = tmp L⁻¹ : solve Y L = tmp, i.e. Lᵀ Yᵀ = tmpᵀ, row by row.
        let mut s_mat = vec![0.0f64; nn];
        for row in 0..n {
            for j in (0..n).rev() {
                let mut s = tmp[row * n + j];
                for k in j + 1..n {
                    s -= s_mat[row * n + k] * lb[k * n + j] as f64;
                }
                s_mat[row * n + j] = s / lb[j * n + j] as f64;
            }
        }
        // Ā = ½ (S + Sᵀ).
        let ob = &mut out[b * nn..(b + 1) * nn];
        for i in 0..n {
            for j in 0..n {
                ob[i * n + j] = (0.5 * (s_mat[i * n + j] + s_mat[j * n + i])) as f32;
            }
        }
    }
    out
}

/// Eigen-decomposition of `batch` symmetric matrices by the cyclic Jacobi
/// method: eigenvalues ascending (`[batch, n]`) and orthonormal
/// eigenvectors as columns (`[batch, n, n]`, `A V = V Λ`). Reads the full
/// matrix and symmetrizes it first.
pub fn eigh(a: &[f32], batch: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let nn = n * n;
    let mut values = vec![0.0f32; batch * n];
    let mut vectors = vec![0.0f32; batch * nn];
    let mut m = vec![0.0f64; nn];
    let mut v = vec![0.0f64; nn];
    for b in 0..batch {
        let src = &a[b * nn..(b + 1) * nn];
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] = 0.5 * (src[i * n + j] as f64 + src[j * n + i] as f64);
                v[i * n + j] = if i == j { 1.0 } else { 0.0 };
            }
        }
        for _sweep in 0..100 {
            let mut off = 0.0f64;
            for i in 0..n {
                for j in i + 1..n {
                    off += m[i * n + j] * m[i * n + j];
                }
            }
            if off < 1e-24 {
                break;
            }
            for p in 0..n {
                for q in p + 1..n {
                    let apq = m[p * n + q];
                    if apq.abs() < 1e-300 {
                        continue;
                    }
                    let app = m[p * n + p];
                    let aqq = m[q * n + q];
                    let theta = 0.5 * (aqq - app) / apq;
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let t = if theta == 0.0 { 1.0 } else { t };
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for k in 0..n {
                        let mkp = m[k * n + p];
                        let mkq = m[k * n + q];
                        m[k * n + p] = c * mkp - s * mkq;
                        m[k * n + q] = s * mkp + c * mkq;
                    }
                    for k in 0..n {
                        let mpk = m[p * n + k];
                        let mqk = m[q * n + k];
                        m[p * n + k] = c * mpk - s * mqk;
                        m[q * n + k] = s * mpk + c * mqk;
                    }
                    for k in 0..n {
                        let vkp = v[k * n + p];
                        let vkq = v[k * n + q];
                        v[k * n + p] = c * vkp - s * vkq;
                        v[k * n + q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        // Sort ascending, carrying the columns.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| m[i * n + i].total_cmp(&m[j * n + j]));
        for (slot, &i) in order.iter().enumerate() {
            values[b * n + slot] = m[i * n + i] as f32;
            for k in 0..n {
                vectors[b * nn + k * n + slot] = v[k * n + i] as f32;
            }
        }
    }
    (values, vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_of_a_known_matrix() {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]] → L = [[2,0,0],[6,1,0],[-8,5,3]]
        let a = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        let l = cholesky(&a, 1, 3).unwrap();
        let want = [2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0];
        for (x, y) in l.iter().zip(want) {
            assert!((x - y).abs() < 1e-5, "{l:?}");
        }
    }

    #[test]
    fn non_positive_definite_is_a_typed_error() {
        let e = cholesky(&[1.0, 2.0, 2.0, 1.0], 1, 2)
            .unwrap_err()
            .to_string();
        assert!(e.contains("positive definite"), "{e}");
        assert!(e.contains("matrix 0"), "{e}");
    }

    #[test]
    fn jacobi_recovers_a_diagonalizable_matrix() {
        let a = [2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
        let (w, v) = eigh(&a, 1, 3);
        let want = [2.0 - 2f64.sqrt(), 2.0, 2.0 + 2f64.sqrt()];
        for (x, y) in w.iter().zip(want) {
            assert!((*x as f64 - y).abs() < 1e-5, "{w:?}");
        }
        // Columns orthonormal.
        for i in 0..3 {
            for j in 0..3 {
                let dot: f32 = (0..3).map(|k| v[k * 3 + i] * v[k * 3 + j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-5, "VᵀV[{i}][{j}] = {dot}");
            }
        }
    }
}
