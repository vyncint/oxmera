//! Cache-friendly, rayon-parallel matrix multiplication: rank-2 GEMM and
//! batched rank-3.

use crate::backend::{MatmulPlan, plan_matmul};
use crate::tensor::Tensor;
use oxmera_core::Result;
use rayon::prelude::*;

/// Block size along the shared dimension: keeps a b-row stripe resident
/// in cache while a row of the output accumulates.
const K_BLOCK: usize = 64;

/// `[m, k] x [k, n]` or `[b, m, k] x [b, k, n]`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let plan = plan_matmul(a.shape(), b.shape())?;
    let MatmulPlan {
        batch,
        m,
        k,
        n,
        a_batch_stride,
        b_batch_stride,
        out_shape,
    } = plan;
    let av = a.to_vec_f32()?;
    let bv = b.to_vec_f32()?;
    let mut out = vec![0.0f32; batch * m * n];
    if batch == 1 {
        gemm(&av[..m * k], &bv[..k * n], &mut out, m, k, n);
    } else {
        // One batch element per task; a broadcast operand has stride 0
        // and is simply re-read, never copied.
        out.par_chunks_mut(m * n).enumerate().for_each(|(i, c)| {
            let ao = i * a_batch_stride;
            let bo = i * b_batch_stride;
            gemm_serial(&av[ao..ao + m * k], &bv[bo..bo + k * n], c, m, k, n);
        });
    }
    Tensor::from_vec_f32(out, out_shape)
}

fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m * n * k >= 32 * 1024 {
        // Four output rows per task: each loaded b[kk, j] feeds four
        // accumulator rows, quartering the traffic through B.
        c.par_chunks_mut(ROWS * n)
            .enumerate()
            .for_each(|(blk, cblk)| {
                let i0 = blk * ROWS;
                let rows = cblk.len() / n;
                gemm_rows(&a[i0 * k..(i0 + rows) * k], b, cblk, rows, k, n);
            });
    } else {
        gemm_serial(a, b, c, m, k, n);
    }
}

fn gemm_serial(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i0 in (0..m).step_by(ROWS) {
        let rows = ROWS.min(m - i0);
        gemm_rows(
            &a[i0 * k..(i0 + rows) * k],
            b,
            &mut c[i0 * n..(i0 + rows) * n],
            rows,
            k,
            n,
        );
    }
}

/// Rows per register block.
const ROWS: usize = 4;

/// `rows` (≤ ROWS) consecutive output rows, K-blocked, with the four row
/// accumulations sharing each load of B.
#[inline]
fn gemm_rows(a: &[f32], b: &[f32], c: &mut [f32], rows: usize, k: usize, n: usize) {
    if rows < ROWS {
        for i in 0..rows {
            gemm_row(&a[i * k..(i + 1) * k], b, &mut c[i * n..(i + 1) * n], k, n);
        }
        return;
    }
    let (c0, rest) = c.split_at_mut(n);
    let (c1, rest) = rest.split_at_mut(n);
    let (c2, c3) = rest.split_at_mut(n);
    for kb in (0..k).step_by(K_BLOCK) {
        let kend = (kb + K_BLOCK).min(k);
        for kk in kb..kend {
            let (a0, a1, a2, a3) = (a[kk], a[k + kk], a[2 * k + kk], a[3 * k + kk]);
            let brow = &b[kk * n..kk * n + n];
            for j in 0..n {
                let bv = brow[j];
                c0[j] += a0 * bv;
                c1[j] += a1 * bv;
                c2[j] += a2 * bv;
                c3[j] += a3 * bv;
            }
        }
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
