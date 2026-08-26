//! Cache-friendly, rayon-parallel matrix multiplication: rank-2 GEMM and
//! batched rank-3.

use crate::tensor::Tensor;
use oxmera_core::{Error, Result, Shape};
use rayon::prelude::*;

/// Block size along the shared dimension: keeps a b-row stripe resident
/// in cache while a row of the output accumulates.
const K_BLOCK: usize = 64;

/// `[m, k] x [k, n]` or `[b, m, k] x [b, k, n]`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    match (a.ndim(), b.ndim()) {
        (2, 2) => {
            let (m, ka) = (a.dims()[0], a.dims()[1]);
            let (kb, n) = (b.dims()[0], b.dims()[1]);
            if ka != kb {
                return Err(Error::ShapeMismatch {
                    expected: Shape::from([ka, n]),
                    got: b.shape().clone(),
                    op: "matmul",
                });
            }
            let av = a.to_vec_f32()?;
            let bv = b.to_vec_f32()?;
            let mut out = vec![0.0f32; m * n];
            gemm(&av, &bv, &mut out, m, ka, n);
            Tensor::from_vec_f32(out, Shape::from([m, n]))
        }
        (3, 3) => {
            let (ba, m, ka) = (a.dims()[0], a.dims()[1], a.dims()[2]);
            let (bb, kb, n) = (b.dims()[0], b.dims()[1], b.dims()[2]);
            if ba != bb || ka != kb {
                return Err(Error::ShapeMismatch {
                    expected: Shape::from([ba, ka, n]),
                    got: b.shape().clone(),
                    op: "matmul",
                });
            }
            let av = a.to_vec_f32()?;
            let bv = b.to_vec_f32()?;
            let mut out = vec![0.0f32; ba * m * n];
            out.par_chunks_mut(m * n).enumerate().for_each(|(i, c)| {
                gemm_serial(
                    &av[i * m * ka..(i + 1) * m * ka],
                    &bv[i * ka * n..(i + 1) * ka * n],
                    c,
                    m,
                    ka,
                    n,
                );
            });
            Tensor::from_vec_f32(out, Shape::from([ba, m, n]))
        }
        (ra, rb) => Err(Error::InvalidArgument {
            op: "matmul",
            detail: format!("supported ranks are 2x2 and 3x3 (batched); got {ra}x{rb}"),
        }),
    }
}

/// Parallel blocked GEMM: rows of the output split across the pool, the
/// shared dimension walked in cache-sized blocks, the inner loop a
/// vectorizable row-axpy.
fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m * n * k >= 32 * 1024 {
        c.par_chunks_mut(n).enumerate().for_each(|(i, crow)| {
            gemm_row(&a[i * k..(i + 1) * k], b, crow, k, n);
        });
    } else {
        gemm_serial(a, b, c, m, k, n);
    }
}

fn gemm_serial(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        gemm_row(&a[i * k..(i + 1) * k], b, &mut c[i * n..(i + 1) * n], k, n);
    }
}

#[inline]
fn gemm_row(arow: &[f32], b: &[f32], crow: &mut [f32], k: usize, n: usize) {
    for kb in (0..k).step_by(K_BLOCK) {
        let kend = (kb + K_BLOCK).min(k);
        for kk in kb..kend {
            let av = arow[kk];
            if av == 0.0 {
                continue;
            }
            let brow = &b[kk * n..kk * n + n];
            for (cv, &bv) in crow.iter_mut().zip(brow) {
                *cv += av * bv;
            }
        }
    }
}
